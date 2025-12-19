"""Fine-tune jhu-clsp/mmBERT-base for CoT (reasoning trace) understanding monitoring.

Task: Given (question, reasoning_trace) predict whether the reasoning trace shows incorrect understanding.

Label definition:
  1 (NOT UNDERSTAND / YES): task evaluation result correct == 0.0
  0 (UNDERSTAND / NO): task evaluation result correct == 1.0


Expected inputs:
  --task_eval_results_path JSON: {lang: {id: {correct: 0/1, reasoning_trace: str, question:?, ...}, ...}, ...}
  --thinking_intv_eval_results_path JSON: same structure (for potential negative augmentation)

Output directory structure:
  save_dir/
     mmBERT-base/
         polymath_low/ (split)
             seed_42/
                 checkpoint-* (HF trainer outputs)

Usage example:
  python rlm_analysis.understanding_failure_detection.ut_bert_monitoring_train.py \
      --model_name jhu-clsp/mmBERT-base \
      --eval_langs en,de,es \
      --polymath_split low \
      --task_eval_results_path path/to/Qwen3-4B_polymath_low_task_eval.json \
      --thinking_intv_eval_results_path path/to/Qwen3-4B_polymath_low_intv_eval.json \
      --output_dir ./mmbert_ft_understandability \
      --num_train_epochs 3

The produced model directory can be referenced in ut_test.py with method ft_mmbert_monitoring.
"""

from __future__ import annotations

import os
import json
import math
import argparse
import random
import torch
import torch.nn as nn
from datasets import load_dataset

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    set_seed,
)
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from rlm_analysis.understanding_failure_detection.ut_compute_signals_with_label import take_first_n_tokens
from rlm_analysis.lang_libs import LANG_LIBS, LANG_SUBJECTS


# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------

def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, preds)
    pr, rc, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    ba = balanced_accuracy_score(labels, preds)  # 추가

    return {
        "accuracy": acc,
        "precision": pr,
        "recall": rc,
        "f1": f1,
        "balanced_accuracy": ba,  # 추가
    }


@dataclass
class Example:
    id: str
    lang: str
    formatted_question: str
    reasoning_trace: str
    label: int  # 1 not understood, 0 not

    def to_text(self, sep_token: str = "[SEP]") -> str:
        # Format: question <SEP> reasoning_trace
        return f"{self.formatted_question} {sep_token} {self.reasoning_trace}".strip()


def _load_source_rows(dataset_type: str, lang: str, polymath_split: str) -> Tuple[Dict[str, Dict[str, Any]], str]:
    """Return mapping from sample id string -> raw dataset row & inferred logical split name.
    Mirrors logic from UnderStandabilityEvalDataset._load for supported dataset types.
    """
    src_map: Dict[str, Dict[str, Any]] = {}
    split_name = None
    if dataset_type == "polymath":
        split_name = polymath_split
        ds = load_dataset("Qwen/PolyMath", lang, split=split_name)
        for dp in ds:
            sid = str(dp.get("id"))
            src_map[sid] = dp
    elif dataset_type in ("mmlu_prox_lite_dev", "mmlu_prox_lite_test", "mmlu_prox_lite"):
        split_name = "validation" if dataset_type.endswith("dev") else "test"
        ds = load_dataset("li-lab/MMLU-ProX-Lite", lang, split=split_name)
        for dp in ds:
            sid = str(dp.get("question_id"))
            src_map[sid] = dp
    elif dataset_type == "mgsm_filtered":
        split_name = "filtered"
        ds = load_dataset("deokhk/filtered_mgsm_with_ids", split=lang)
        for idx, dp in enumerate(ds):
            sid = str(idx)
            src_map[sid] = dp
    else:
        raise ValueError(f"Unsupported dataset_type: {dataset_type}")
    return src_map, split_name or "test"


def get_prompt_for_mmlu_prox(question, category, options, lang):
    lang_lib_template = LANG_LIBS.get(lang, "")
    question_formatter = lang_lib_template[0]
    option_formatter = lang_lib_template[1]

    ans_suffix = lang_lib_template[5].format("X")
    subject_in_lang = ""
    if category == "computer science":
        # Use "computer_science" as the key if "computer science" is not found
        try:
            subject_in_lang = LANG_SUBJECTS[lang][category]
        except KeyError:
            subject_in_lang = LANG_SUBJECTS[lang]["computer_science"]
    else:
        subject_in_lang = LANG_SUBJECTS[lang][category]

    assert subject_in_lang != "", f"Subject translation not found for language: {lang}, category: {category}"
    options_letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
    options_string = ""
    for idx, option in enumerate(options):
        options_string += f"({options_letters[idx]}) {option}\n"
    options_string = options_string.rstrip("\n")
    prompt = lang_lib_template[3].format(subject=subject_in_lang, ans_suffix=ans_suffix) + f"\n{question_formatter} {question}\n{option_formatter}\n" + options_string 
    return prompt 


def _format_question(dataset_type: str, src_row: dict, raw_question: str, lang: str) -> str:
    # Import formatting instruction lazily to avoid circulars
    try:
        from rlm_analysis.dataset import MATH_LANG_TO_FORMATTING_INSTRUCTION  # type: ignore
    except Exception:
        MATH_LANG_TO_FORMATTING_INSTRUCTION = {}
    if dataset_type in ("polymath", "mgsm_filtered"):
        instr = MATH_LANG_TO_FORMATTING_INSTRUCTION.get(lang, "")
        if instr:
            return raw_question + "\n" + instr
        else:
            raise ValueError(f"No formatting instruction found for language: {lang}")
    elif dataset_type in ("mmlu_prox_lite_dev", "mmlu_prox_lite_test", "mmlu_prox_lite"):
        category = src_row["category"]
        option_fields = [f"option_{i}" for i in range(0, 10)]  # option_0 to option_9
        options = [src_row[field] for field in option_fields if field in src_row and (src_row[field] != None and src_row[field] != "N/A")]
        
        formatted_question= get_prompt_for_mmlu_prox(raw_question, category, options, lang)
        return formatted_question
    return raw_question


def build_examples(
    task_eval: Dict[str, Dict[str, Any]],
    thinking_intv_eval: Dict[str, Dict[str, Any]],
    lang: str,
    dataset_type: str,
    polymath_split: str,
    max_token_to_look_from_reasoning_trace: Optional[int] = -1,
    llm_tokenizer=None,
) -> List[Example]:
    """Create examples using source dataset to recover the original question text.

    task_eval entries typically lack the original question; we look it up by id.
    """
    src_map, _ = _load_source_rows(dataset_type, lang, polymath_split)
    examples: List[Example] = []
    missing_q = 0
    for sid, rec in task_eval.items():
        sid_str = str(sid)
        src_row = src_map.get(sid_str)
        # We only keep samples that are either correct normally or correct with thinking intervals
        if not (rec.get("correct", 0) == 1 or thinking_intv_eval[sid].get("correct", 0) == 1):
            continue

        if not src_row:
            missing_q += 1
            continue
        # Extract question field per dataset type
        if dataset_type == "polymath":
            raw_question = src_row.get("question", "")
        elif dataset_type in ("mmlu_prox_lite_dev", "mmlu_prox_lite_test", "mmlu_prox_lite"):
            raw_question = src_row.get("question", "")
        elif dataset_type == "mgsm_filtered":
            raw_question = src_row.get("question", "")
        else:
            raw_question = ""
        formatted_question = _format_question(dataset_type, src_row, raw_question, lang)
        reasoning_trace = rec.get("reasoning_trace", "")
        truncated_reasoning_trace = take_first_n_tokens(
            reasoning_trace,
            llm_tokenizer,
            max_token_to_look_from_reasoning_trace, # If -1, use full trace (by default)
        ) # We should truncate using llm tokenizer, not mmBERT tokenizer
        correct = 1 if rec.get("correct", 0) == 1 else 0
        not_understandable = 1 - correct
        examples.append(Example(id=sid_str, lang=lang, formatted_question=formatted_question, reasoning_trace=truncated_reasoning_trace, label=not_understandable))
    if missing_q:
        print(f"[INFO] Lang {lang}: {missing_q} ids skipped due to missing source question")
    return examples


def split_train_eval(examples, eval_ratio, seed):
    labels = [ex.label for ex in examples]
    train_idx, eval_idx = train_test_split(
        np.arange(len(examples)),
        test_size=eval_ratio,
        random_state=seed,
        stratify=labels if len(set(labels))>1 and labels.count(1) > 1 else None
    )
    train_set = [examples[i] for i in train_idx]
    eval_set  = [examples[i] for i in eval_idx]
    return train_set, eval_set


def make_hf_dataset(examples: List[Example], tokenizer, max_length: int):
    texts = [ex.to_text(tokenizer.sep_token if tokenizer.sep_token else "[SEP]") for ex in examples]
    labels = [ex.label for ex in examples]
    enc = tokenizer(texts, truncation=True, padding=False, max_length=max_length)
    data = {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"], "labels": labels, "lang": [ex.lang for ex in examples], "id": [ex.id for ex in examples]}
    return Dataset.from_dict(data)

def filepath_sanity_check(args):
    llm_base_name = args.llm_model_name.split('/')[-1]
    assert llm_base_name in args.task_eval_results_path, f"llm_model_name '{args.llm_model_name}' not found in task_eval_results_path '{args.task_eval_results_path}'"
    assert llm_base_name in args.thinking_intv_eval_results_path, f"llm_model_name '{args.llm_model_name}' not found in thinking_intv_eval_results_path '{args.thinking_intv_eval_results_path}'"



def main():  
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="jhu-clsp/mmBERT-base")
    parser.add_argument("--llm_model_name", type=str, default="Qwen/Qwen3-4B", help="LLM model name for logging purposes only.")
    parser.add_argument("--task_eval_results_path", type=str, required=True)
    parser.add_argument("--thinking_intv_eval_results_path", type=str, required=True)
    parser.add_argument("--max_token_to_look_from_reasoning_trace", type=int, default=-1,
                        help="If >0, take only the last N tokens from the reasoning trace before encoding")
    parser.add_argument("--eval_langs", type=str, default="en,de,es,ar,ja,ko,th,bn,sw,te")
    parser.add_argument("--dataset_type", type=str, default="polymath",
                        choices=["polymath", "mmlu_prox_lite_dev", "mmlu_prox_lite_test", "mgsm_filtered"],
                        help="Source dataset type for building rows and matching ids.")

    parser.add_argument("--polymath_split", type=str, default="low")
    parser.add_argument("--output_dir", type=str, default="./mmbert_ft_understandability")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_ratio", type=float, default=0.1, help="Portion of data reserved for validation per language (after augmentation).")
    parser.add_argument("--max_length", type=int, default=8192)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=16)
    parser.add_argument("--num_train_epochs", type=int, default=30)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")

    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience (epochs) if >0.")
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--eval_strategy", type=str, default="epoch", choices=["epoch", "steps"])
    parser.add_argument("--save_strategy", type=str, default="epoch", choices=["epoch", "steps"])
    parser.add_argument("--custom_postfix", type=str, default=None, help="Custom string to append to save checkpoint.")
    args = parser.parse_args()

    filepath_sanity_check(args)
    set_seed(args.seed)

    llm_model_name = args.llm_model_name
    if "qwen3-4b" in args.task_eval_results_path.lower():
        llm_model_name = "Qwen3-4B"
    elif "gpt-oss-20b" in args.task_eval_results_path.lower():
        llm_model_name = "GPT-OSS-20B"
    elif "qwen3-1.7b" in args.task_eval_results_path.lower():
        llm_model_name = "Qwen3-1.7B"
    elif "qwen3-8b" in args.task_eval_results_path.lower():
        llm_model_name = "Qwen3-8B"
    elif "qwen3-14b" in args.task_eval_results_path.lower():
        llm_model_name = "Qwen3-14B"
    else:
        llm_model_name = "Unknown"
    langs = [l.strip() for l in args.eval_langs.split(',') if l.strip()]
    task_eval_all = read_json(args.task_eval_results_path)
    thinking_intv_eval_all = read_json(args.thinking_intv_eval_results_path)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    llm_tokenizer = AutoTokenizer.from_pretrained(args.llm_model_name, use_fast=True)
    # ---------------- Build corpus ----------------
    per_language_splits: Dict[str, Dict[str, List[str]]] = {}
    all_train, all_eval = [], []
    for lang in langs:
        task_eval = task_eval_all.get(lang, {})
        thinking_intv_eval = thinking_intv_eval_all.get(lang, {})
        if not task_eval:
            print(f"[WARN] No task eval entries for lang={lang}; skipping")
            continue
        if not thinking_intv_eval:
            print(f"[WARN] No thinking interval eval entries for lang={lang}; skipping")
            continue
        examples = build_examples(task_eval, thinking_intv_eval, lang, args.dataset_type, args.polymath_split, args.max_token_to_look_from_reasoning_trace, llm_tokenizer)
        if not examples:
            print(f"[WARN] No examples produced for lang={lang}; skipping")
            continue
        train_lang, eval_lang = split_train_eval(examples, eval_ratio=args.eval_ratio, seed=args.seed)
        all_train.extend(train_lang)
        all_eval.extend(eval_lang)
        lang_splits = per_language_splits.setdefault(lang, {"train": [], "eval": []})
        lang_splits["train"].extend(ex.id for ex in train_lang)
        lang_splits["eval"].extend(ex.id for ex in eval_lang)
        print(f"Lang {lang}: total={len(examples)} train={len(train_lang)} eval={len(eval_lang)} (pos={sum(e.label==1 for e in examples)}, neg={sum(e.label==0 for e in examples)})")

    if not all_train:
        raise RuntimeError("No training data constructed. Check inputs.")

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        problem_type="single_label_classification",
    )

    train_ds = make_hf_dataset(all_train, tokenizer, args.max_length)
    eval_ds = make_hf_dataset(all_eval, tokenizer, args.max_length)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    dataset_name = f"polymath_{args.polymath_split}" if args.dataset_type == "polymath" else args.dataset_type
    model_dir_name = args.model_name.split('/')[-1]
    training_output_dir = os.path.join(
        args.output_dir,
        llm_model_name,
        model_dir_name,
        dataset_name,
        f"seed_{args.seed}_{args.custom_postfix}" if args.custom_postfix else f"seed_{args.seed}",
    )
    os.makedirs(training_output_dir, exist_ok=True)

    normalised_splits: Dict[str, Dict[str, List[str]]] = {}
    for lang, splits in per_language_splits.items():
        normalised_splits[lang] = {
            "train": sorted({str(sample_id) for sample_id in splits["train"]}),
            "eval": sorted({str(sample_id) for sample_id in splits["eval"]}),
        }

    split_metadata = {
        "seed": args.seed,
        "eval_ratio": args.eval_ratio,
        "dataset_type": args.dataset_type,
        "polymath_split": args.polymath_split if args.dataset_type == "polymath" else None,
        "languages": sorted(normalised_splits.keys()),
        "total_train_examples": len(all_train),
        "total_eval_examples": len(all_eval),
        "splits": normalised_splits,
    }
    split_metadata_path = os.path.join(training_output_dir, "sample_id_splits.json")
    with open(split_metadata_path, "w", encoding="utf-8") as fh:
        json.dump(split_metadata, fh, indent=2)
    print(f"Saved sample id splits to {split_metadata_path}")

    # Metric for best model will be f1 (on positive class = understood). Depending on usage you may invert.
    training_args = TrainingArguments(
        output_dir=training_output_dir,
        run_name=f"{args.model_name.split('/')[-1]}_{args.dataset_type}_{args.polymath_split}_seed{args.seed}_from_{llm_model_name}",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        eval_strategy=args.eval_strategy,
        save_strategy=args.save_strategy,
        save_total_limit=args.save_total_limit,
        metric_for_best_model="balanced_accuracy", 
        load_best_model_at_end=True,
        greater_is_better=True,
        logging_steps=args.logging_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        fp16=args.fp16,
        bf16=args.bf16,
        report_to="wandb"
    )

    callbacks = []
    try:
        from transformers import EarlyStoppingCallback  # optional
        if args.patience and args.patience > 0 and args.eval_strategy:
            callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.patience))
    except Exception:
        pass

    # ---------------- Weighted loss (handle imbalance) ----------------
    # Compute class distribution on train set
    labels_np = np.array(train_ds["labels"], dtype=int)
    n_pos = int((labels_np == 1).sum())
    n_neg = int((labels_np == 0).sum())
    if n_pos == 0 or n_neg == 0:
        class_weights = None
        print("[WARN] One class has zero samples; skipping class weighting.")
    else:
        # Inverse frequency weighting: weight_c = N / (2 * count_c)
        total = n_pos + n_neg
        w_pos = total / (2 * n_pos)
        w_neg = total / (2 * n_neg)
        class_weights = torch.tensor([w_neg, w_pos], dtype=torch.float)
        print(f"Class weights: neg={w_neg:.4f} pos={w_pos:.4f} (neg_count={n_neg}, pos_count={n_pos})")

    # We'll recreate loss function inside compute_loss to ensure weights on correct device.
    class_weights_tensor = class_weights  # keep original reference

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):  # type: ignore
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.get("logits") if isinstance(outputs, dict) else outputs.logits
            loss_fct = nn.CrossEntropyLoss(weight=class_weights_tensor.to(logits.device))
            loss = loss_fct(logits, labels)
            return (loss, outputs) if return_outputs else loss

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    print("==== Train dataset size:", len(train_ds))
    print("==== Eval dataset size:", len(eval_ds))
    trainer.train()

    best_ckpt = trainer.state.best_model_checkpoint
    if best_ckpt is None:
        print("[WARN] No best_model_checkpoint found. Did you disable evaluation/saving?")
    else:
        import shutil
        dest = os.path.join(training_args.output_dir, "best_checkpoint")
        # 폴더 통째로 복사 (기존에 있으면 덮어쓰기)
        if os.path.exists(dest):
            shutil.rmtree(dest)
        shutil.copytree(best_ckpt, dest)
        print(f"[INFO] Best checkpoint (by balanced_accuracy) copied to: {dest}")

    # Simple evaluation report
    metrics = trainer.evaluate()
    with open(os.path.join(dest, "eval_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print("Eval metrics:", metrics)


if __name__ == "__main__":
    main()
