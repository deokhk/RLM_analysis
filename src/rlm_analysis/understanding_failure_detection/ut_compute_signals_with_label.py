import argparse
import copy
import json
import logging
import math
import os
from typing import Any, Dict, List, Tuple

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from rlm_analysis.dataset import UnderStandabilityEvalDataset


logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model_and_tokenizer(model_name: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    logger.info("Loading tokenizer %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    logger.info("Loading model %s", model_name)
    if torch.cuda.is_available():
        if "gpt-oss-20b" in model_name:
            print("Using flash attention 2 for gpt-oss-20b")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                dtype="auto",
                attn_implementation="flash_attention_2"
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype="auto",
                device_map="auto",
            )
    else:
        logger.warning("CUDA not available; loading model on CPU")
        model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    return model, tokenizer


def build_dataset_tag(dataset_type: str, polymath_split: str) -> str:
    if dataset_type == "polymath":
        return f"{dataset_type}_{polymath_split}"
    return dataset_type


def take_first_n_tokens(text: str, tokenizer, n: int) -> str:
    if n is None or n <= 0:
        return text
    ids = tokenizer.encode(text, add_special_tokens=False)
    head_ids = ids[:n] if len(ids) > n else ids
    return tokenizer.decode(head_ids, skip_special_tokens=True)


def chat_template_from_prompt(messages: List[Dict[str, str]], tokenizer, model_name: str) -> str:
    if not messages:
        raise ValueError("Empty message list for chat template")

    formatted_question = messages[0]["content"]
    if len(messages) == 1:
        if "Qwen3" in model_name:
            return f"<|im_start|>user\n{formatted_question}<|im_end|>"
        if "gpt-oss" in model_name:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
        raise ValueError(f"Unsupported model for chat templating: {model_name}")

    if "Qwen3" in model_name:
        assistant_content = messages[1]["content"]
        return (
            f"<|im_start|>user\n{formatted_question}<|im_end|>\n<|im_start|>assistant\n{assistant_content}"
        )
    if "gpt-oss" in model_name:
        formatted_question = tokenizer.apply_chat_template(
            messages[:1],
            tokenize=False,
            add_generation_prompt=True
        )
        assistant_content = messages[1]["content"]
        return f"{formatted_question}<|channel|>analysis<|message|>{assistant_content}"
    raise ValueError(f"Unsupported model for chat templating: {model_name}")


def prepare_samples(args, tokenizer) -> List[Dict[str, Any]]:
    eval_langs = [lang.strip() for lang in args.eval_langs.split(",") if lang.strip()]
    if not eval_langs:
        with open(args.task_eval_results_path, "r", encoding="utf-8") as f:
            normal = json.load(f)
        eval_langs = sorted(normal.keys())
        logger.info("Infered eval languages from JSON: %s", ", ".join(eval_langs))

    dataset_builder = UnderStandabilityEvalDataset(
        args,
        eval_langs
    )
    rows_by_lang = dataset_builder.get()

    samples: List[Dict[str, Any]] = []
    skipped = 0
    for lang, rows in rows_by_lang.items():
        for row in rows:
            reasoning_trace = row.get("reasoning_trace", "") or ""
            truncated_reasoning = take_first_n_tokens(
                reasoning_trace,
                tokenizer,
                args.max_token_to_look_from_reasoning_trace,
            )
            prompt_messages = copy.deepcopy(row.get("original_prompt_dict_input_and_reasoning_trace", []))
            if len(prompt_messages) < 2:
                skipped += 1
                continue
            prompt_messages[1]["content"] = truncated_reasoning
            try:
                full_text = chat_template_from_prompt(prompt_messages, tokenizer, args.model_name)
                prompt_messages_no_reasoning = copy.deepcopy(prompt_messages)
                prompt_messages_no_reasoning[1]["content"] = ""
                prompt_text = chat_template_from_prompt(prompt_messages_no_reasoning, tokenizer, args.model_name)
            except ValueError as exc:
                logger.warning("Skipping %s/%s: %s", lang, row.get("id"), exc)
                skipped += 1
                continue

            prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
            if isinstance(prompt_ids, list) and prompt_ids and isinstance(prompt_ids[0], list):
                prompt_ids = prompt_ids[0]
            label = 1 if row["understandable"] == False else 0

            samples.append({
                "language": lang,
                "id": str(row.get("id")),
                "full_text": full_text,
                "prompt_length": len(prompt_ids),
                "label": label,
            })
    logger.info("Prepared %d samples (skipped %d) across %d languages", len(samples), skipped, len(rows_by_lang))
    # Sort samples in descending order of full_text length for better padding efficiency
    samples.sort(key=lambda x: len(tokenizer.encode(x["full_text"], add_special_tokens=False)), reverse=True)
    return samples


def compute_signals(
    args,
    model,
    tokenizer,
    samples: List[Dict[str, Any]],
    batch_size: int = 8,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    results: Dict[str, Dict[str, Dict[str, Any]]] = {}
    if not samples:
        return results
    
    for start in tqdm(range(0, len(samples), batch_size), desc="Computing signals"):
        batch = samples[start:start + batch_size]
        texts = [b["full_text"] for b in batch]
        encoded = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

        logits = outputs.logits.detach().cpu()  # (B, L, V)
        hidden = outputs.hidden_states[-1].detach().cpu()
        
        torch.cuda.empty_cache()

        temperature = float(args.temperature_used_for_generation)
        if temperature and temperature > 0:
            logits.div_(temperature)

        log_probs = torch.log_softmax(logits, dim=-1) # (B, L, V)
        target_ids = input_ids[:, 1:].cpu()
        token_log_probs = torch.gather(log_probs[:, :-1, :], 2, target_ids.unsqueeze(-1)).squeeze(-1)

        token_log_probs = token_log_probs.detach()
        log_probs = log_probs.detach()
        attention_mask_cpu = attention_mask.detach().cpu()

        top_k = int(args.confidence_top_k)
        if top_k <= 0 or top_k > log_probs.shape[-1]:
            top_k = log_probs.shape[-1]
        topk_log_probs = torch.topk(log_probs[:, :-1, :], k=top_k, dim=-1).values

        for idx, sample in enumerate(batch):
            seq_len = int(attention_mask_cpu[idx].sum().item())
            prompt_len = int(sample["prompt_length"]) or 0
            seq_token_log_probs = token_log_probs[idx, :max(seq_len - 1, 0)] # (L-1,)
            seq_topk_log_probs = topk_log_probs[idx, :seq_token_log_probs.shape[0], :]

            prompt_cutoff = max(min(prompt_len - 1, seq_token_log_probs.shape[0]), 0)

            prompt_log_probs = seq_token_log_probs[:prompt_cutoff]
            reasoning_topk_log_probs = seq_topk_log_probs[prompt_cutoff:]
            if reasoning_topk_log_probs.numel() > 0:
                token_conf_vals = (-reasoning_topk_log_probs.mean(dim=-1)).float()
                avg_conf = float(token_conf_vals.mean().item())
                min_conf = float(token_conf_vals.min().item())
            else:
                avg_conf = float("nan")
                min_conf = float("nan")

            if prompt_log_probs.numel() > 0:
                prompt_neg_log = (-prompt_log_probs).float()
                prompt_ln_nll = float(prompt_neg_log.mean().item())
            else:
                prompt_ln_nll = float("nan")

            last_index = max(seq_len - 1, 0)
            last_hidden = hidden[idx, last_index, :].clone()

            lang_map = results.setdefault(sample["language"], {})
            lang_map[sample["id"]] = {
                "avg_confidence": avg_conf,
                "min_confidence": min_conf,
                "prompt_ln_nll": prompt_ln_nll,
                "last_hidden_state": last_hidden,
                "not_understood_label": int(sample["label"]),
                "max_token_to_look_from_reasoning_trace": args.max_token_to_look_from_reasoning_trace,
                "temperature_used_for_generation": args.temperature_used_for_generation,
                "confidence_top_k": args.confidence_top_k,
            }

        del logits
        del token_log_probs
        del outputs
        del log_probs
        del topk_log_probs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results


def save_results(results: Dict[str, Dict[str, Dict[str, Any]]], args) -> str:
    model_tag = args.model_name.split("/")[-1]
    dataset_tag = build_dataset_tag(args.dataset_type, args.polymath_split)
    head_tag = (
        f"head{args.max_token_to_look_from_reasoning_trace}"
        if args.max_token_to_look_from_reasoning_trace and args.max_token_to_look_from_reasoning_trace > 0
        else "headall"
    )
    seed = args.seed
    filename = f"{model_tag}_{dataset_tag}_{head_tag}_seed_{seed}_signals_with_label{args.custom_postfix}.pth"
    output_dir = os.path.join(args.save_dir, model_tag, dataset_tag)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    torch.save(results, output_path)
    logger.info("Saved signals to %s", output_path)
    return output_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--dataset_type", type=str, required=True,
                        choices=[
                            "mgsm_filtered",
                            "mmlu_prox_lite",
                            "mmlu_prox_lite_dev",
                            "polymath",
                        ])
    parser.add_argument("--task_eval_results_path", type=str, required=True,
                        help="Path to task eval results JSON")
    parser.add_argument("--thinking_intv_eval_results_path", type=str, required=True,
                        help="Path to thinking intervention eval results JSON")
    parser.add_argument("--eval_langs", type=str, default="en,de,es,ar,ja,ko,th,bn,sw,te",
                        help="Comma-separated languages to include")
    parser.add_argument("--polymath_split", type=str, default="low")
    parser.add_argument("--save_dir", type=str, default="./ut_model_signals_with_label")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_token_to_look_from_reasoning_trace", type=int, default=-1,
                        help="If >0, take only the last N tokens from the reasoning trace before encoding")
    parser.add_argument("--temperature_used_for_generation", type=float, default=0.6)
    parser.add_argument("--confidence_top_k", type=int, default=20,
                        help="Top-K tokens to consider for confidence calculation")
    parser.add_argument("--seed", type=str, required=True, help="Random seed used for generating the task eval results")
    parser.add_argument("--custom_postfix", type=str, default="",
                        help="Custom string to append to the output filename")

    # Low-resource experiment
    parser.add_argument("--low_resource_experiment", action="store_true",
                        help="Whether to run low-resource experiment on PolyMath dataset."
                        )
    parser.add_argument("--translated_dataset_json_path", type=str, default=None,
                        help="Path to the JSON file containing translated questions and optinally options for thinking intervention."
                        )
    return parser.parse_args()


def main():
    args = parse_args()
    # sanity checks
    assert args.seed in args.task_eval_results_path, (
        f"Seed {args.seed} not found in task eval results path {args.task_eval_results_path}"
    )
    assert args.seed in args.thinking_intv_eval_results_path, (
        f"Seed {args.seed} not found in thinking intervention eval results path {args.thinking_intv_eval_results_path}"
    )
    model, tokenizer = load_model_and_tokenizer(args.model_name)
    samples = prepare_samples(args, tokenizer)
    if not samples:
        logger.warning("No samples prepared; exiting without saving")
        return
    # Currently, this code only supports batch size 1 cause the padding handling is not done properly for now
    assert args.batch_size == 1, "Batch size must be 1 for models with very large memory usage"
    results = compute_signals(args,model, tokenizer, samples, batch_size=args.batch_size)
    if not results:
        logger.warning("No results to save")
        return
    save_results(results, args)


if __name__ == "__main__":
    main()
