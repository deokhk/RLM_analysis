
import os
import json
import logging
import argparse
import fasttext
import csv

from transformers import AutoTokenizer
import torch
import numpy as np
from vllm import LLM, SamplingParams

from rlm_analysis.util.misc import divide_reasoning_trace_from_solution, unload_vllm_model
from rlm_analysis.util.language_analysis import compute_language_id_statistics_self_english_and_other

from rlm_analysis.dataset import (
    PolyMathDataset,
    MMLUProXLiteDataset,
    LowResourcePolyMathDataset,
    TranslatedPolyMathDataset,
    TranslatedMMLUProXLiteDataset,
    MMLUProXLiteDatasetForCalibration,
    FilteredMGSMDatasetForCalibration,
    TranslatedThinkIntvPolyMathDataset,
    TranslatedThinkIntvMMLUProXLiteDataset
)
from rlm_analysis.eval.evaluator import (
    PolyMathEvaluator,
    MGSMEvaluator, 
    MMLUProXLiteEvaluator
)

# ---------------------------------------------------------------------------
# logging & env
# ---------------------------------------------------------------------------
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["VLLM_USE_V1"] = "0" # V0 API
logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# helper utilities
# ---------------------------------------------------------------------------

def get_model_input_text(datapoint, tokenizer, args):
    """Qwen‑3 전용 입력 포맷터."""
    prompt_dict = datapoint["prompt_dict"]
    formatted_question = prompt_dict[0]["content"]
    if "Qwen3" in args.model_name:
        if args.do_thinking_intervention or args.test_with_translated_data_as_ut:
            forced_gen_prefix_with_think = prompt_dict[1]["content"]
            return (
                f"<|im_start|>user\n{formatted_question}<|im_end|>\n<|im_start|>assistant\n{forced_gen_prefix_with_think}"
            )
        else:
            formatted_text = tokenizer.apply_chat_template(
                prompt_dict,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True
            )
            return formatted_text
    elif "gpt-oss" in args.model_name:
        if args.do_thinking_intervention or args.test_with_translated_data_as_ut:
            formatted_text_without_prefix = tokenizer.apply_chat_template(
                prompt_dict[:1],
                tokenize=False,
                add_generation_prompt=True
            )
            prefix = prompt_dict[1]["content"]
            formatted_text = f"{formatted_text_without_prefix}<|channel|>analysis<|message|>{prefix}"
            return formatted_text
        else:
            formatted_text = tokenizer.apply_chat_template(
                prompt_dict,
                tokenize=False,
                add_generation_prompt=True,
            )
            return formatted_text

    else:
        raise ValueError(f"Unsupported model for formatting: {args.model_name}")


def main(args):

    assert torch.cuda.is_available() and torch.cuda.device_count() > 0, "CUDA is not available or no GPU found."
    # -------------------- dataset ------------------------------
    logger.info("Loading dataset: %s…", args.dataset_type)
    if args.polymath_split == "mid":
        logger.warning("Polymath 'mid' split is an alias for 'medium' split.")
        args.polymath_split = "medium"
    
    if args.dataset_type == "polymath":
        if args.test_with_translated_data:
            dataset = TranslatedPolyMathDataset(args)
        elif args.test_with_translated_data_as_ut:
            dataset = TranslatedThinkIntvPolyMathDataset(args)
        elif args.low_resource_experiment:
            dataset = LowResourcePolyMathDataset(args)
        else:
            dataset = PolyMathDataset(args)
    elif args.dataset_type == "mmlu_prox_lite":
        if args.test_with_translated_data:
            dataset = TranslatedMMLUProXLiteDataset(args)
        elif args.test_with_translated_data_as_ut:
            dataset = TranslatedThinkIntvMMLUProXLiteDataset(args)
        else:
            dataset = MMLUProXLiteDataset(args)
    elif args.dataset_type == "mgsm_filtered":
        dataset = FilteredMGSMDatasetForCalibration(args)
    elif args.dataset_type == "mmlu_prox_lite_dev":
        dataset = MMLUProXLiteDatasetForCalibration(args)
    else:
        raise ValueError(f"Unsupported dataset type: {args.dataset_type}")
    dataset = dataset.get_test_data()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        use_fast=True
    )

    logger.info("Loading model %s with vLLM…", args.model_name)

    kwargs = {
                "tensor_parallel_size": args.tensor_parallel_size, #int(os.getenv("VLLM_N_GPUS", "1"))
                "trust_remote_code": True,
                "gpu_memory_utilization": args.gpu_memory_utilization,
                "max_model_len": 40960 if "phi-4" not in args.model_name.lower() else 32768
            }
    llm = LLM(
        model=args.model_name,
        **kwargs
    )
    logger.info("Model loaded with %d tensor parallel size", args.tensor_parallel_size)

    # -------------------- evaluation ------------------------------
    eval_langs = list(args.eval_langs.split(","))
    eval_langs = [lang.strip() for lang in eval_langs if lang.strip()]  # Remove empty strings

    logger.info("Evaluating languages: %s", ", ".join(eval_langs))
    task_eval_results = {}
    eval_only_results = {}
    pred_results_dict = {}
    

    # For computing language ID distribution 
    fasttext_model = fasttext.load_model(args.fasttext_model_path)
    logger.info("FastText model loaded from %s", args.fasttext_model_path)

    for lang in eval_langs:
        
        lang_data = dataset[lang]

        prompts = [get_model_input_text(dp, tokenizer, args) for dp in lang_data]

        sampling = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p, top_k=args.top_k,
            max_tokens=args.max_new_tokens,
            seed=args.seed,
        )

        logger.info("Generating for %s …", lang)
        vllm_out = llm.generate(prompts, sampling_params=sampling)

        
        predictions = []
        for out in vllm_out:
            if "gpt-oss" in args.model_name:
                # For vllm 0.10.1, by default it ignores special tokens in decode..
                # So we manually decode from tokens
                pred = tokenizer.decode(out.outputs[0].token_ids).strip()
            else:
                pred = out.outputs[0].text.strip()
            predictions.append(pred)

        logger.info("Generation for %s done", lang)

        if lang not in pred_results_dict:
            pred_results_dict[lang] = []
        for sample, pred_res in zip(lang_data, predictions):
            (reasoning_trace, solution) = divide_reasoning_trace_from_solution(pred_res, args.model_name)
            pred_results_dict[lang].append(
                {
                    "id": sample["id"],
                    "language_code": lang,
                    "question": sample["question"],
                    "prediction": solution,
                    "reasoning_trace": reasoning_trace,
                    "answer":sample["answer"],
                }
            )

    # Now, unload the model to free GPU memory for evaluation

    unload_vllm_model(llm)
    del llm

    # Load parser llm
    parser_kwargs = {
                "tensor_parallel_size": args.tensor_parallel_size, #int(os.getenv("VLLM_N_GPUS", "1"))
                "gpu_memory_utilization": args.gpu_memory_utilization,
                "max_model_len": 4096
            }
    parser_llm = LLM(
        model=args.llm_parser_model_name,
        **parser_kwargs
    )
    parser_tokenizer = AutoTokenizer.from_pretrained(
        args.llm_parser_model_name,
        trust_remote_code=True,
        use_fast=True
    )
    logger.info("Answer parser model loaded with %d tensor parallel size", args.tensor_parallel_size)


    for lang in eval_langs:
        lang_data = dataset[lang]
        pred_results = pred_results_dict[lang]
        if args.dataset_type == "polymath":
            Evaluator = PolyMathEvaluator(lang, lang_data, parser_llm, parser_tokenizer)
        elif args.dataset_type == "mgsm_filtered":
            Evaluator = MGSMEvaluator(lang, lang_data, parser_llm, parser_tokenizer)
        elif args.dataset_type == "mmlu_prox_lite" or args.dataset_type == "mmlu_prox_lite_dev":
            Evaluator = MMLUProXLiteEvaluator(lang, lang_data, parser_llm, parser_tokenizer)

        logger.info("Evaluating task performance for language: {}...".format(lang))
        eval_results = Evaluator.evaluate(pred_results)

        if lang not in task_eval_results:
            task_eval_results[lang] = {}

        for pred_res, eval_result in zip(pred_results, eval_results):
            assert pred_res["id"] == eval_result["id"], f"id mismatch: {pred_res['id']} vs {eval_result['id']}"
            id_ = eval_result["id"]
            lang_code = eval_result["language_code"]

            if lang_code not in task_eval_results:
                task_eval_results[lang_code] = {}

            task_eval_results[lang_code][id_] = {
                "correct": eval_result["score"],
                "prediction": pred_res["prediction"],
                "reasoning_trace": pred_res["reasoning_trace"],
                "answer": pred_res["answer"],
                "pred_answer": str(eval_result.get("pred_answer")),  # JSON 직렬화 안전
                "pred_answer_from_trace": str(eval_result.get("pred_answer_from_trace")),
                "correct_from_trace": eval_result.get("score_from_trace", 0.0),
            }
        all_scores = [er.get("score", 0.0) for er in eval_results]
        all_trace_scores = [er.get("score_from_trace", 0.0) for er in eval_results]

        avg_score = float(np.mean(all_scores)) if len(all_scores) > 0 else 0.0
        avg_score_from_trace = float(np.mean(all_trace_scores)) if len(all_trace_scores) > 0 else 0.0

        # eval_only_results에 두 점수를 같이 보관 (아래 3)에서 CSV/JSON에 쓰입니다)
        eval_only_results[lang] = {
            "eval_score": avg_score,
            "eval_score_from_trace": avg_score_from_trace,
            "n_samples": len(pred_results)
        }

        logger.info(f"[{lang}] #samples={len(all_scores)}, "
                    f"average_score={avg_score:.4f}, average_score_from_trace={avg_score_from_trace:.4f}")


    model_name_without_organization = args.model_name.split("/")[-1]
    save_base_dir = ""
    if args.save_base_dir is not None:
        save_base_dir = args.save_base_dir
    else:
        save_base_dir = "./task_eval_results"
        if args.do_thinking_intervention:
            save_base_dir ="./thinking_intv_eval_results"
        
        if args.test_with_translated_data:
            save_base_dir ="./translated_eval_results"
        
        if args.test_with_translated_data_as_ut:
            save_base_dir ="./translated_think_intv_eval_results"
    
    if not os.path.exists(save_base_dir):
        os.makedirs(save_base_dir)
    if args.dataset_type == "polymath":
        task_eval_save_dir = os.path.join(save_base_dir, model_name_without_organization, args.dataset_type, args.polymath_split)
    else:
        task_eval_save_dir = os.path.join(save_base_dir, model_name_without_organization, args.dataset_type)
    if not os.path.exists(task_eval_save_dir):
        os.makedirs(task_eval_save_dir) 

    if args.dataset_type == "polymath":
        save_postfix = f"{args.dataset_type}_{args.polymath_split}_{args.seed}"
    else:
        save_postfix = f"{args.dataset_type}_{args.seed}"

    
    if args.do_thinking_intervention:
        save_postfix += f"_thinking_intv_{args.thinking_intervention_lang}"
    if args.test_with_translated_data:
        save_postfix += f"_translated_{args.thinking_intervention_lang}"
    if args.test_with_translated_data_as_ut:
        save_postfix += f"_translated_think_intv_{args.thinking_intervention_lang}"

    task_eval_save_path = os.path.join(task_eval_save_dir, f"{model_name_without_organization}_task_eval_results_{save_postfix}.json")
    eval_res_only_save_path = os.path.join(task_eval_save_dir, f"eval_score_with_language_distribution_{model_name_without_organization}_{save_postfix}.json")

    with open(task_eval_save_path, "w") as f:
        json.dump(task_eval_results, f, indent=4, ensure_ascii=False)
        logger.info(f"Task evaluation results (with full predictions) saved to {task_eval_save_path}")

    lang2lid2vals = {}
    stat_types = ["total_percentages", "pred_percentages", "reasoning_percentages"]

    for lang in task_eval_results:
        lid2vals = {stype: {} for stype in stat_types}
        valid_stat_counts = {stype: 0 for stype in stat_types}

        # Self, English, and Other language ID statistics
        for idx in task_eval_results[lang]:
            stats = compute_language_id_statistics_self_english_and_other(task_eval_results[lang][idx], lang, fasttext_model)
            for stype in stat_types:
                if stats.get(stype):
                    valid_stat_counts[stype] += 1
                for ctype, val in stats[stype].items():
                    if ctype not in lid2vals[stype]:
                        lid2vals[stype][ctype] = []
                    lid2vals[stype][ctype].append(val)
        # 평균 계산 (except does with {})
        lid_mean = {}
        for stype in stat_types:
            num_entries = valid_stat_counts[stype]
            lid_mean[stype] = {ctype: (sum(lid2vals[stype][ctype]) / num_entries if num_entries > 0 else 0.0) for ctype in lid2vals[stype]}
        lang2lid2vals[lang] = lid_mean

    # Eval res update
    for lang in eval_only_results:
        prev = eval_only_results[lang]
        new_stats = lang2lid2vals.get(lang, {})
        eval_only_results[lang] = {
            **prev,  # 기존 모든 키 보존
            "total_percentages": new_stats.get("total_percentages", {}),
            "pred_percentages": new_stats.get("pred_percentages", {}),
            "reasoning_percentages": new_stats.get("reasoning_percentages", {}),
        }

    with open(eval_res_only_save_path, "w") as f:
        json.dump(eval_only_results, f, indent=4, ensure_ascii=False)
        logger.info(f"Evaluation results (score and langauge statistics) saved to {eval_res_only_save_path}")


    # CSV 저장 루틴 교체
    csv_save_path = os.path.join(
        task_eval_save_dir,
        f"eval_summary_{model_name_without_organization}_{save_postfix}.csv"
    )

    # metric → {lang: value} 형태로 pivot
    metrics = ["eval_score", "eval_score_from_trace", "reasoning_percent_en", "pred_percent_self"]
    langs = eval_langs

    # 미리 딕셔너리 형태로 가공
    metric2values = {m: {} for m in metrics}
    for lang in langs:
        row = eval_only_results.get(lang, {})
        metric2values["eval_score"][lang] = row.get("eval_score", 0.0)
        metric2values["eval_score_from_trace"][lang] = row.get("eval_score_from_trace", 0.0)
        metric2values["reasoning_percent_en"][lang] = row.get("reasoning_percentages", {}).get("en", 0.0)
        metric2values["pred_percent_self"][lang] = row.get("pred_percentages", {}).get("self", 0.0)

    # CSV 쓰기 (metric 기준 row, 언어별 column)
    with open(csv_save_path, "w", newline="", encoding="utf-8") as fcsv:
        writer = csv.writer(fcsv)
        # 헤더: metric + 언어들
        writer.writerow(["metric"] + langs)
        for m in metrics:
            row = [m] + [f"{metric2values[m].get(lang, 0.0):.6f}" for lang in langs]
            writer.writerow(row)

    logger.info(f"Evaluation CSV (pivoted) saved to {csv_save_path}")

# ---------------------------------------------------------------------------
# entry
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-4B")

    # For loading the dataset
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="polymath",
        choices=["mmlu_prox_lite_dev", "mgsm_filtered", "polymath", "mmlu_prox_lite"],
    )
    parser.add_argument(
        "--eval_langs",
        type=str,
        default="en,de,es,ar,ja,ko,th,bn,sw,te",
        help="Comma-separated list of languages to evaluate (e.g., 'en,ko,bn')",
    )
    parser.add_argument(
        "--polymath_split",
        type=str,
        default="low",
        choices=["low", "medium", "high", "mid"],
    )
    parser.add_argument(
        "--save_base_dir", default=None,
        type=str, help="Base directory to save evaluation results. If not provided, default directory in the script will be used based on dataset and model."
    )
    # sampling params
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--max_new_tokens", type=int, default=32768) #32768
    
    # vllm params
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)

    # misc
    parser.add_argument("--fasttext_model_path", type=str, default="./misc/lid.176.ftz")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")


    # Thinking intv params 
    parser.add_argument("--do_thinking_intervention", action="store_true",
                        help="Whether to apply thinking intervention to all samples, regardless of their understandability")

    parser.add_argument("--thinking_intervention_lang", type=str, default="en", choices=["en", "same"],
                        help="Language to use for thinking intervention. 'en' for English, 'same' for the same language as the question.")

    # inference with translated dataset 
    parser.add_argument("--test_with_translated_data", action="store_true",
                        help="Whether to test with translated dataset - as input.")
    parser.add_argument("--test_with_translated_data_as_ut", action="store_true",
                        help="Whether to test with translated dataset - thinking intervention.")
    parser.add_argument("--translated_dataset_json_path", type=str, default=None,
                        help="Path to the JSON file containing translated questions and optinally options for thinking intervention."
                        )

    # Low-resource experiment
    parser.add_argument("--low_resource_experiment", action="store_true",
                        help="Whether to run low-resource experiment on PolyMath dataset."
                        )
    

    # Parser llm model
    parser.add_argument("--llm_parser_model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                        help="Model name for parsing the final answer from the reasoning trace."
                        )


    args = parser.parse_args()
    main(args)
 