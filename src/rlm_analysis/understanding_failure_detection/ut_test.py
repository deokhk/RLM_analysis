import argparse
import asyncio
import copy
import json
import logging
import math
import os
import re
import pickle
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt 
from openai import AsyncOpenAI
from sklearn.metrics import auc, f1_score, precision_recall_curve, roc_curve
from torch.utils.data import DataLoader, TensorDataset
from tqdm.asyncio import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from rlm_analysis.understanding_failure_detection.ut_probe_model import load_model, hs_dict
from vllm import LLM, SamplingParams

from rlm_analysis.dataset import UnderStandabilityEvalDataset
from rlm_analysis.understanding_failure_detection.ut_compute_signals_with_label import take_first_n_tokens

logging.basicConfig(format="%(levelname)s - %(name)s - %(message)s", level=logging.INFO)
logger = logging.getLogger("ut_test")
logger.setLevel(logging.INFO)



###############################################################################
# Self-Reflection Understandability Test 
###############################################################################

def get_model_input_text_for_prompt_dict(prompt_dict, tokenizer, args):
    """Input formatter."""
    formatted_question = prompt_dict[0]["content"]

    if "Qwen3" in args.model_name:
        if args.understandability_test_method == "self-reflection":
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
        if args.understandability_test_method == "self-reflection":
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
        raise ValueError(f"Unsupported model for understandability test: {args.model_name}")


REFLECTION_INSTRUCTION = "Wait, before proceeding, I will reflect on my prior reasoning to assess my overall understanding of the problem. I will respond with <Understandability>: YES or NO (YES if I'm fully confident that I understood the problem correctly, NO otherwise). <Understandability>:"


def parse_understandability_answer(raw_text: str) -> str:
    """Return 'YES', 'NO', or 'UNKNOWN' from a single-line reflection output."""
    if not raw_text:
        return "UNKNOWN"
    if "yes" in raw_text.lower():
        return "YES"
    elif "no" in raw_text.lower():
        return "NO"
    return "UNKNOWN"

def truncate_trace_if_needed(reasoning_trace: str, tokenizer, max_tokens: int) -> str:
    if max_tokens and max_tokens > 0:
        return take_first_n_tokens(reasoning_trace, tokenizer, max_tokens)
    return reasoning_trace

def build_reflection_prompt(
    original_prompt_messages: List[Dict[str, str]],
    tokenizer,
    args
) -> str:
    assert len(original_prompt_messages)  == 2
    reasoning_trace = original_prompt_messages[1]["content"]
    rt = truncate_trace_if_needed(reasoning_trace or "", tokenizer, args.max_token_to_look_from_reasoning_trace)
    rt = rt.replace("</think>", "")  # in case of GPT-OSS
    original_prompt_messages[1]["content"] = rt + REFLECTION_INSTRUCTION

    prompt = get_model_input_text_for_prompt_dict(original_prompt_messages, tokenizer, args)
    return prompt


def run_self_reflection(
    args,
    dataset: Dict[str, List[Dict[str, Any]]],
    task_eval_results: Dict[str, Any],
    eval_langs: Sequence[str],
) -> Dict[str, Dict[str, Any]]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA must be available for self-reflection method")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        use_fast=True,
    )
    llm = LLM(
        model=args.model_name,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=40960 if "phi-4" not in args.model_name.lower() else 32768,
    )
    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.self_reflection_max_new_tokens,
        seed=args.seed,
    )

    lang2results: Dict[str, Dict[str, Any]] = {}
    for lang in eval_langs:
        if lang not in dataset:
            logger.warning("Language %s not in dataset; skipping", lang)
            continue
        lang_data = dataset[lang]
        lang_task = task_eval_results.get(lang, {})
        prompts: List[str] = []
        metas: List[Tuple[str, int]] = []
        for sample in lang_data:
            sid = sample["id"]
            task_res = lang_task.get(str(sid)) or lang_task.get(sid)
            if not task_res:
                continue
            original_msgs = copy.deepcopy(sample["original_prompt_dict_input_and_reasoning_trace"])
            prompt = build_reflection_prompt(original_msgs, tokenizer, args)
            prompts.append(prompt)
            metas.append((str(sid), int(task_res.get("correct", 0))))
        if not prompts:
            continue
        logger.info("Generating self-reflection outputs for %s (#%d)…", lang, len(prompts))
        outs = llm.generate(prompts, sampling_params=sampling)
        lang2results[lang] = {}
        for out, (sid, correct_label) in zip(outs, metas):
            raw = out.outputs[0].text.strip() if out.outputs and out.outputs[0].text else ""
            understood = parse_understandability_answer(raw)
            lang2results[lang][sid] = {
                "understood": understood,
                "raw_output": raw,
                "correct": correct_label,
            }
        logger.info("Done %s", lang)
    return lang2results

@dataclass(frozen=True)
class SampleScore:
    language: str
    sample_id: str
    score: float
    label: int  # 1 = not understood, 0 = understood
    prediction: Optional[int] = None


@dataclass(frozen=True)
class MethodConfig:
    loader: str  # 'pth', 'json_score', 'json_binary'
    score_field: Optional[str]
    higher_score_indicates_positive: bool
    requires_threshold: bool = True
    supports_curve_metrics: bool = True
    binary_field: Optional[str] = None
    binary_positive_values: Optional[Tuple[str, ...]] = None
    default_threshold: Optional[float] = None


SUPPORTED_METHODS: Dict[str, MethodConfig] = {
    "random_baseline": MethodConfig(
        loader="random",
        score_field="pos_probability",
        higher_score_indicates_positive=True,
        requires_threshold=False,
        supports_curve_metrics=False,
    ),
    "avg_confidence": MethodConfig(
        loader="pth",
        score_field="avg_confidence",
        higher_score_indicates_positive=False,
        requires_threshold = True
    ),
    "min_confidence": MethodConfig(
        loader="pth",
        score_field="min_confidence",
        higher_score_indicates_positive=False,
        requires_threshold = True
    ),
    "prompt_ln_nll": MethodConfig(
        loader="pth",
        score_field="prompt_ln_nll",
        higher_score_indicates_positive=True,
        requires_threshold = True
    ),
    "self-reflection": MethodConfig(
        loader="json_binary",
        score_field="predicted_not_understood",
        higher_score_indicates_positive=True,
        requires_threshold=False,
        supports_curve_metrics=False,
        binary_field="understood",
        binary_positive_values=("no", "unknown", "false", "0"),
    ),
    "gpt_monitoring": MethodConfig(
        loader="json_binary",
        score_field="predicted_not_understood",
        higher_score_indicates_positive=True,
        requires_threshold=False,
        supports_curve_metrics=False,
        binary_field="understood",
        binary_positive_values=("no", "false", "0"),
    ),
    "ft_mmbert_monitoring": MethodConfig(
        loader="json_score",
        score_field="prob_not_understood",
        higher_score_indicates_positive=True,
        requires_threshold=True,
        supports_curve_metrics=True,
        default_threshold=0.5,
    ),
    "ft_probe": MethodConfig(
        loader="json_score",
        score_field="prob_not_understood",
        higher_score_indicates_positive=True,
        requires_threshold=True,
        supports_curve_metrics=True,
        default_threshold=0.5,
    ),
}




def _to_float(value: object) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (float, int)):
        result = float(value)
    elif isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return None
        result = float(value.item())
    else:
        try:
            result = float(value)
        except (TypeError, ValueError):
            return None
    if not math.isfinite(result):
        return None
    return result


def _to_int(value: object) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, bool):
        return int(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _load_json_payload(path: str) -> Dict[str, Dict[str, Dict[str, object]]]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Unexpected JSON payload type: {type(payload).__name__}")
    return payload


def _extract_label(sample_payload: Dict[str, object]) -> Optional[int]:
    label = _to_int(sample_payload.get("not_understood_label"))
    if label in (0, 1):
        return label
    correct = _to_int(sample_payload.get("correct"))
    if correct in (0, 1):
        return 1 - correct
    return None


def _normalise_binary_value(value: object) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        return value.strip().lower()
    if isinstance(value, bool):
        return str(value).lower()
    try:
        return str(int(value)).lower()
    except (TypeError, ValueError):
        return str(value).lower()


def _parse_score_payload(
    payload: Dict[str, Dict[str, Dict[str, object]]],
    method_cfg: MethodConfig,
    allowed_langs: Optional[set[str]],
) -> Dict[str, List[SampleScore]]:
    per_language: Dict[str, List[SampleScore]] = {}
    for language, samples in payload.items():
        if allowed_langs and language not in allowed_langs:
            continue
        if not isinstance(samples, dict):
            logger.warning("Skipping language %s due to invalid payload type %s", language, type(samples).__name__)
            continue
        entries: List[SampleScore] = []
        for sample_id, sample_payload in samples.items():
            if not isinstance(sample_payload, dict):
                continue
            score_raw = sample_payload.get(method_cfg.score_field) if method_cfg.score_field else None
            score = _to_float(score_raw)
            if score is None:
                continue
            label = _extract_label(sample_payload)
            if label not in (0, 1):
                continue
            entries.append(SampleScore(
                language=language,
                sample_id=str(sample_id),
                score=score,
                label=label,
            ))
        if entries:
            per_language[language] = entries
    return per_language


def _parse_binary_payload(
    payload: Dict[str, Dict[str, Dict[str, object]]],
    method_cfg: MethodConfig,
    allowed_langs: Optional[set[str]],
) -> Dict[str, List[SampleScore]]:
    if not method_cfg.binary_field or not method_cfg.binary_positive_values:
        raise ValueError("Binary method configuration missing required fields")
    positive_values = {val.lower() for val in method_cfg.binary_positive_values}
    per_language: Dict[str, List[SampleScore]] = {}
    for language, samples in payload.items():
        if allowed_langs and language not in allowed_langs:
            continue
        if not isinstance(samples, dict):
            logger.warning("Skipping language %s due to invalid payload type %s", language, type(samples).__name__)
            continue
        entries: List[SampleScore] = []
        for sample_id, sample_payload in samples.items():
            if not isinstance(sample_payload, dict):
                continue
            raw_pred = sample_payload.get(method_cfg.binary_field)
            norm_pred = _normalise_binary_value(raw_pred)
            if norm_pred is None:
                continue
            prediction = 1 if norm_pred in positive_values else 0
            label = _extract_label(sample_payload)
            if label not in (0, 1):
                continue
            entries.append(SampleScore(
                language=language,
                sample_id=str(sample_id),
                score=float(prediction),
                label=label,
                prediction=prediction,
            ))
        if entries:
            per_language[language] = entries
    return per_language


def _collect_random_baseline_entries(
    payload: Dict[str, Dict[str, Dict[str, object]]],
    allowed_langs: Optional[set[str]],
) -> Dict[str, List[Tuple[str, int]]]:
    per_language_labels: Dict[str, List[Tuple[str, int]]] = {}
    for language, samples in payload.items():
        if allowed_langs and language not in allowed_langs:
            continue
        if not isinstance(samples, dict):
            logger.warning(
                "Skipping language %s due to invalid payload type %s", language, type(samples).__name__
            )
            continue
        entries: List[Tuple[str, int]] = []
        for sample_id, sample_payload in samples.items():
            if not isinstance(sample_payload, dict):
                continue
            label = _extract_label(sample_payload)
            if label not in (0, 1):
                continue
            entries.append((str(sample_id), int(label)))
        if entries:
            per_language_labels[language] = entries
    return per_language_labels


def _generate_random_baseline_samples(
    payload: Dict[str, Dict[str, Dict[str, object]]],
    allowed_langs: Optional[set[str]],
    seed: Optional[int],
    per_language_positive_class_ratio: None,
) -> Dict[str, List[SampleScore]]:
    if not isinstance(payload, dict):
        raise ValueError("Random baseline expects a mapping from language to samples")

    per_language_labels = _collect_random_baseline_entries(payload, allowed_langs)

    if not per_language_labels:
        raise ValueError("No usable samples were found to construct the random baseline")
    

    results: Dict[str, List[SampleScore]] = {}
    base_seed = int(seed) if seed is not None else None
    for idx, language in enumerate(sorted(per_language_labels)):
        entries = sorted(per_language_labels[language], key=lambda item: item[0])
        total = len(entries)
        positives = sum(label for _, label in entries)
        p: float
        if per_language_positive_class_ratio:
            if language in per_language_positive_class_ratio:
                p = float(per_language_positive_class_ratio[language])
            else:
                p = float(positives) / float(total) if total else 0.0
        else:
            p = float(positives) / float(total) if total else 0.0

        if base_seed is None:
            rng = np.random.default_rng()
        else:
            rng = np.random.default_rng(base_seed)

        if total == 0:
            continue
        if p <= 0.0:
            preds = np.zeros(total, dtype=int)
        elif p >= 1.0:
            preds = np.ones(total, dtype=int)
        else:
            preds = (rng.random(total) < p).astype(int)

        results[language] = [
            SampleScore(
                language=language,
                sample_id=sid,
                score=p,
                label=lbl,
                prediction=int(pred),
            )
            for (sid, lbl), pred in zip(entries, preds)
        ]
    return results


def _iter_signal_files(path: str):
    if os.path.isfile(path):
        yield path
        return
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Signal path not found: {path}")
    for root, _dirs, files in os.walk(path):
        for file_name in files:
            if file_name.endswith((".pth", ".pt", ".bin", ".json")):
                yield os.path.join(root, file_name)


def _load_signal_payload(path: str) -> Dict[str, Dict[str, Dict[str, object]]]:
    if path.endswith(".json"):
        return _load_json_payload(path)
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Unexpected payload type in {path}: {type(payload).__name__}")
    return payload


def _normalise_hidden_state(hidden_state: object) -> Optional[torch.Tensor]:
    if hidden_state is None:
        return None
    if isinstance(hidden_state, torch.Tensor):
        tensor = hidden_state.detach().cpu().to(torch.float32)
    else:
        tensor = torch.tensor(hidden_state, dtype=torch.float32)
    if tensor.ndim > 1:
        tensor = tensor.view(-1)
    return tensor


def load_probe_features(
    signal_path: str,
    *,
    languages: Optional[Sequence[str]] = None,
) -> Tuple[Dict[str, List[Tuple[str, torch.Tensor, int]]], int]:
    allowed_langs = {lang for lang in languages if lang} if languages else None
    per_language: Dict[str, List[Tuple[str, torch.Tensor, int]]] = {}
    feature_dim: Optional[int] = None

    for file_path in sorted(_iter_signal_files(signal_path)):
        payload = _load_signal_payload(file_path)
        for language in sorted(payload.keys()):
            if allowed_langs and language not in allowed_langs:
                continue
            samples = payload[language]
            if not isinstance(samples, dict):
                continue
            entries = per_language.setdefault(language, [])
            for sample_id in sorted(samples.keys(), key=lambda sid: str(sid)):
                sample_payload = samples[sample_id]
                if not isinstance(sample_payload, dict):
                    continue
                hidden_state = _normalise_hidden_state(sample_payload.get("last_hidden_state"))
                if hidden_state is None:
                    continue
                label = _extract_label(sample_payload)
                if label not in (0, 1):
                    continue
                if feature_dim is None:
                    feature_dim = hidden_state.shape[0]
                elif hidden_state.shape[0] != feature_dim:
                    raise ValueError("Inconsistent hidden-state dimensionality across samples.")
                entries.append((str(sample_id), hidden_state, int(label)))

    if feature_dim is None:
        raise ValueError("No usable hidden states were loaded for probe evaluation.")
    return per_language, feature_dim


def _load_random_baseline_payload_from_path(
    path: str,
) -> Dict[str, Dict[str, Dict[str, object]]]:
    payload = None
    load_errors: List[str] = []
    try:
        candidate = torch.load(path, map_location="cpu")
        if isinstance(candidate, dict):
            payload = candidate
    except Exception as exc:  # pragma: no cover - best effort fallback
        load_errors.append(f"torch load failed: {exc}")
    if payload is None:
        try:
            payload = _load_json_payload(path)
        except Exception as exc:  # pragma: no cover - best effort fallback
            load_errors.append(f"json load failed: {exc}")
    if payload is None:
        joined = "; ".join(load_errors) if load_errors else "unsupported format"
        raise ValueError(f"Failed to load input for random baseline from {path}: {joined}")
    if not isinstance(payload, dict):
        raise ValueError(
            f"Random baseline input must be a mapping, got {type(payload).__name__}"
        )
    return payload


def load_samples_for_method(
    args,
    path: str,
    method_cfg: MethodConfig,
    *,
    languages: Optional[Sequence[str]] = None,
    seed: Optional[int] = None
) -> Dict[str, List[SampleScore]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input path not found: {path}")

    if method_cfg.loader == "pth":
        payload = torch.load(path, map_location="cpu")
        if not isinstance(payload, dict):
            raise ValueError(f"Unexpected .pth payload type: {type(payload).__name__}")
    elif method_cfg.loader in {"json_score", "json_binary"}:
        payload = _load_json_payload(path)
    elif method_cfg.loader == "random":
        payload = _load_random_baseline_payload_from_path(path)
        per_language_positive_class_ratio = load_per_language_positive_class_ratio_from_calibration_set(args, "random_baseline")
    else:
        raise ValueError(f"Unsupported loader type: {method_cfg.loader}")

    allowed_langs = None if not languages else {lang for lang in languages}

    if method_cfg.loader == "json_binary":
        per_language = _parse_binary_payload(payload, method_cfg, allowed_langs)
    elif method_cfg.loader == "random":
        per_language = _generate_random_baseline_samples(
            payload,
            allowed_langs,
            seed,
            per_language_positive_class_ratio=per_language_positive_class_ratio,
        )
    else:
        per_language = _parse_score_payload(payload, method_cfg, allowed_langs)

    if not per_language:
        raise ValueError("No usable samples were loaded from the provided input file.")
    return per_language


def build_samples_from_results(
    lang2results: Dict[str, Dict[str, Any]],
    method_cfg: MethodConfig,
    *,
    languages: Optional[Sequence[str]] = None,
) -> Dict[str, List[SampleScore]]:
    allowed_langs = None if not languages else {lang for lang in languages}
    per_language: Dict[str, List[SampleScore]] = {}
    for language, samples in lang2results.items():
        if allowed_langs and language not in allowed_langs:
            continue
        entries: List[SampleScore] = []
        for sid, payload in samples.items():
            label = _extract_label(payload)
            if label not in (0, 1):
                continue
            if method_cfg.loader == "json_score":
                score_val = _to_float(payload.get(method_cfg.score_field)) if method_cfg.score_field else None
                if score_val is None:
                    continue
                entries.append(
                    SampleScore(
                        language=language,
                        sample_id=str(sid),
                        score=score_val,
                        label=label,
                    )
                )
            else:
                if method_cfg.binary_field:
                    norm = _normalise_binary_value(payload.get(method_cfg.binary_field))
                    if norm is None:
                        continue
                    positive_values = {
                        val.lower() for val in (method_cfg.binary_positive_values or ())
                    }
                    prediction = 1 if norm in positive_values else 0
                else:
                    prediction = _to_int(payload.get("predicted_not_understood"))
                    if prediction not in (0, 1):
                        continue
                entries.append(
                    SampleScore(
                        language=language,
                        sample_id=str(sid),
                        score=float(prediction),
                        label=label,
                        prediction=prediction,
                    )
                )
        if entries:
            per_language[language] = entries
    if not per_language:
        raise ValueError("Failed to convert inference results into samples")
    return per_language


def flatten_samples(per_language: Dict[str, List[SampleScore]], exclude_english=False) -> List[SampleScore]:
    merged: List[SampleScore] = []
    for lang, samples in per_language.items():
        if lang == "en" and exclude_english:
            continue
        merged.extend(samples)
    return merged


def determine_best_threshold(
    samples: Sequence[SampleScore],
    method_cfg: MethodConfig,
) -> float:
    if not samples:
        raise ValueError("Cannot determine threshold without samples")

    labels = np.array([s.label for s in samples], dtype=int)
    scores = np.array([s.score for s in samples], dtype=float)

    if np.unique(labels).size < 2:
        raise ValueError("Need at least one positive and one negative sample to estimate a threshold")

    adjusted_scores = scores if method_cfg.higher_score_indicates_positive else -scores
    precision, recall, thresholds = precision_recall_curve(labels, adjusted_scores, pos_label=1)

    
    epsilon = 1e-7
    f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + epsilon)

    best_idx = int(np.argmax(f1_scores))
    best_threshold = thresholds[best_idx]

    logger.info(
        "Selected threshold %.6f (F1=%.4f, Precision=%.4f, Recall=%.4f)",
        best_threshold if method_cfg.higher_score_indicates_positive else -best_threshold,
        f1_scores[best_idx],
        precision[best_idx],
        recall[best_idx],
    )
    return best_threshold if method_cfg.higher_score_indicates_positive else -best_threshold


def apply_threshold(scores: np.ndarray, threshold: float, higher_score_is_positive: bool) -> np.ndarray:
    if higher_score_is_positive:
        return (scores >= threshold).astype(int)
    return (scores <= threshold).astype(int)


def compute_tnr_at_fnr(labels: np.ndarray, adjusted_scores: np.ndarray, target_fnr: float) -> float:
    if np.unique(labels).size < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(labels, adjusted_scores, pos_label=1)
    fnr = 1.0 - tpr
    mask = fnr <= (target_fnr + 1e-6)
    if not np.any(mask):
        return float("nan")
    tnr_values = 1.0 - fpr[mask]
    return float(np.max(tnr_values))


def compute_metrics_for_samples(
    samples: Sequence[SampleScore],
    *,
    threshold: Optional[float],
    method_cfg: MethodConfig,
) -> Dict[str, float]:
    if not samples:
        return {
            "n": 0,
            "positives": 0,
            "negatives": 0,
            "tp": 0,
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "balanced_accuracy": float("nan"),
            "f1": float("nan"),
            "pr_auc": float("nan"),
            "tnr_at_fnr_0.10": float("nan"),
            "tnr_at_fnr_0.05": float("nan"),
            "tpr": float("nan"),
            "tnr": float("nan"),
        }

    scores = np.array([s.score for s in samples], dtype=float)
    labels = np.array([s.label for s in samples], dtype=int)

    if method_cfg.requires_threshold:
        if threshold is None:
            raise ValueError("Threshold is required for this method but was not provided")
        predictions = apply_threshold(scores, threshold, method_cfg.higher_score_indicates_positive)
    else:
        preds_list: List[int] = []
        for sample in samples:
            if sample.prediction in (0, 1):
                preds_list.append(int(sample.prediction))
            else:
                preds_list.append(int(round(sample.score)))
        predictions = np.array(preds_list, dtype=int)
    tp = int(np.sum((predictions == 1) & (labels == 1)))
    tn = int(np.sum((predictions == 0) & (labels == 0)))
    fp = int(np.sum((predictions == 1) & (labels == 0)))
    fn = int(np.sum((predictions == 0) & (labels == 1)))

    tpr = tp / (tp + fn) if (tp + fn) else float("nan")
    tnr = tn / (tn + fp) if (tn + fp) else float("nan")
    if math.isnan(tpr) or math.isnan(tnr):
        balanced_accuracy = float("nan")
    else:
        balanced_accuracy = 0.5 * (tpr + tnr)

    try:
        f1 = float(f1_score(labels, predictions, zero_division=0))
    except ValueError:
        f1 = float("nan")

    if method_cfg.supports_curve_metrics:
        adjusted_scores = scores if method_cfg.higher_score_indicates_positive else -scores
        if np.unique(labels).size < 2:
            pr_auc = float("nan")
        else:
            precision, recall, _ = precision_recall_curve(labels, adjusted_scores)
            if precision.size == 0 or recall.size == 0:
                pr_auc = float("nan")
            else:
                pr_auc = float(auc(recall, precision))

        tnr_at_10 = compute_tnr_at_fnr(labels, adjusted_scores, 0.10)
        tnr_at_05 = compute_tnr_at_fnr(labels, adjusted_scores, 0.05)
    else:
        pr_auc = float("nan")
        tnr_at_10 = float("nan")
        tnr_at_05 = float("nan")

    return {
        "n": int(labels.size),
        "positives": int(labels.sum()),
        "negatives": int(labels.size - labels.sum()),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "balanced_accuracy": balanced_accuracy,
        "f1": f1,
        "pr_auc": pr_auc,
        "tnr_at_fnr_0.10": tnr_at_10,
        "tnr_at_fnr_0.05": tnr_at_05,
        "tpr": tpr,
        "tnr": tnr,
    }


def _compute_confusion_counts(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    method_cfg: MethodConfig,
) -> Tuple[int, int, int, int]:
    preds = apply_threshold(scores, threshold, method_cfg.higher_score_indicates_positive)
    tp = int(np.sum((preds == 1) & (labels == 1)))
    tn = int(np.sum((preds == 0) & (labels == 0)))
    fp = int(np.sum((preds == 1) & (labels == 0)))
    fn = int(np.sum((preds == 0) & (labels == 1)))
    return tp, tn, fp, fn


def _balanced_accuracy_from_counts(
    tp: int,
    tn: int,
    fp: int,
    fn: int,
) -> Tuple[float, float, float, float]:
    tpr = tp / (tp + fn) if (tp + fn) else float("nan")
    tnr = tn / (tn + fp) if (tn + fp) else float("nan")
    ba = 0.5 * (tpr + tnr) if not (math.isnan(tpr) or math.isnan(tnr)) else float("nan")
    fpr = fp / (fp + tn) if (fp + tn) else float("nan")
    return ba, tpr, tnr, fpr


def _plot_boxplot_ax(
    ax,
    understood_values: Sequence[float],
    not_understood_values: Sequence[float],
    title: str,
    vline: Optional[float] = None,
) -> None:
    understood = np.array([v for v in understood_values if v == v], dtype=float)
    not_understood = np.array([v for v in not_understood_values if v == v], dtype=float)
    if understood.size == 0 and not_understood.size == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_title(title)
        return
    data = [not_understood, understood]
    bp = ax.boxplot(
        data,
        vert=False,
        patch_artist=True,
        showmeans=False,
        showfliers=True,
        medianprops=dict(linewidth=1.5),
        whiskerprops=dict(linewidth=1.0),
        capprops=dict(linewidth=1.0),
        boxprops=dict(linewidth=1.0),
    )
    colors = ["#FF6B6B", "#4D96FF"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    ax.set_yticks([1, 2])
    ax.set_yticklabels(["not understood (POS)", "understood (NEG)"], fontsize=12)
    if vline is not None and np.isfinite(vline):
        ax.axvline(vline, linestyle="--", linewidth=1.5, color="black")
        ylim = ax.get_ylim()
        ax.text(
            vline,
            ylim[1] - 0.05 * (ylim[1] - ylim[0]),
            f"τ={vline:.3f}",
            rotation=90,
            va="top",
            ha="right",
            fontsize=8,
        )
    ax.set_xlabel("Score", fontsize=12)
    ax.set_title(title, fontsize=12)


def _plot_box_overall(
    understood_values: Sequence[float],
    not_understood_values: Sequence[float],
    title: str,
    out_path: str,
    vline: Optional[float] = None,
) -> None:
    if plt is None:  # pragma: no cover - matplotlib optional
        return
    plt.figure(figsize=(8.0, 3.8), dpi=140)
    ax = plt.gca()
    _plot_boxplot_ax(ax, understood_values, not_understood_values, title, vline=vline)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0.02)  # pad_inches는 선택
    logger.info("Saved overall boxplot to %s", out_path)
    plt.close()


BPLOT_VERBOSE_METHOD_NAME = {
    "avg_confidence": "Average Confidence",
    "min_confidence": "Minimum Confidence",
    "prompt_ln_nll": "Prompt Log-Negative Likelihood",
    "self-reflection": "Self-Reflection",
    "gpt_monitoring": "GPT Monitoring",
    "ft_mmbert_monitoring": "Fine-tuned MMBERT Monitoring",
    "ft_probe": "Fine-tuned Probe",
    "random_baseline": "Random Baseline",
}

def _plot_by_language_boxplots(
    samples_by_language: Dict[str, List[SampleScore]],
    method_cfg: MethodConfig,
    method_name: str,
    threshold: float,
    out_path: str,
) -> None:
    if plt is None:  # pragma: no cover - matplotlib optional
        return
    languages = sorted(samples_by_language.keys())
    if not languages:
        logger.warning("No language data for confidence boxplots.")
        return
    n = len(languages)
    ncols = min(4, max(1, int(np.ceil(np.sqrt(n)))))
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 3.4 * nrows), dpi=140)
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]
    idx = -1
    for idx, language in enumerate(languages):
        samples = samples_by_language[language]
        scores = np.array([s.score for s in samples], dtype=float)
        labels = np.array([s.label for s in samples], dtype=int)
        understood_vals = [score for score, label in zip(scores, labels) if label == 0]
        not_understood_vals = [score for score, label in zip(scores, labels) if label == 1]
        tp, tn, fp, fn = _compute_confusion_counts(scores, labels, threshold, method_cfg)
        ba, tpr, tnr, _ = _balanced_accuracy_from_counts(tp, tn, fp, fn)
        title = (
            f"{language} • BA={ba:.3f} "
            f"(TPR={tpr:.2f}, TNR={tnr:.2f})"
        )
        _plot_boxplot_ax(axes[idx], understood_vals, not_understood_vals, title, vline=threshold)
    for j in range(idx + 1, len(axes)):
        axes[j].axis("off")
    fig.suptitle(f"Per-language boxplots • method={BPLOT_VERBOSE_METHOD_NAME[method_name]}", y=0.995, fontsize=12)
    fig.tight_layout(rect=[0, 0.0, 1, 0.97])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path)
    logger.info("Saved per-language boxplots to %s", out_path)
    plt.close(fig)


def generate_and_save_boxplots(
    samples_by_language: Dict[str, List[SampleScore]],
    method_cfg: MethodConfig,
    threshold: Optional[float],
    *,
    method_name: str,
    model_name: str,
    dataset_type: str,
    seed: int,
    output_dir: str,
    filename_suffix: str = "",
) -> None:
    if not method_cfg.requires_threshold:
        return
    if threshold is None or not np.isfinite(threshold):
        logger.warning("Skipping boxplots because a valid threshold was not available.")
        return
    if plt is None:
        logger.warning("matplotlib is not available; skipping boxplot generation.")
        return

    understood_values: List[float] = []
    not_understood_values: List[float] = []
    overall_scores: List[float] = []
    overall_labels: List[int] = []
    for samples in samples_by_language.values():
        for sample in samples:
            score_val = float(sample.score)
            understood_values.append(score_val) if sample.label == 0 else not_understood_values.append(score_val)
            overall_scores.append(score_val)
            overall_labels.append(int(sample.label))

    boxplot_dir = os.path.join(output_dir, "boxplots")
    overall_title: str
    if overall_scores:
        scores_np = np.array(overall_scores, dtype=float)
        labels_np = np.array(overall_labels, dtype=int)
        tp, tn, fp, fn = _compute_confusion_counts(scores_np, labels_np, threshold, method_cfg)
        ba, tpr, tnr, _ = _balanced_accuracy_from_counts(tp, tn, fp, fn)
        overall_title = (
            f"Overall • {BPLOT_VERBOSE_METHOD_NAME[method_name]} "
            f"• BA={ba:.3f} (TPR={tpr:.2f}, TNR={tnr:.2f})"
        )
    else:
        overall_title = (
            f"Overall • {BPLOT_VERBOSE_METHOD_NAME[method_name]} • model={model_name} • dataset={dataset_type}"
        )
    overall_path = os.path.join(
        boxplot_dir,
        f"ut_boxplot_overall_{method_name}_{seed}{filename_suffix}.pdf",
    )
    _plot_box_overall(understood_values, not_understood_values, overall_title, overall_path, vline=threshold)

    per_lang_path = os.path.join(
        boxplot_dir,
        f"ut_boxplot_per_language_{method_name}_{seed}{filename_suffix}.pdf",
    )
    _plot_by_language_boxplots(
        samples_by_language,
        method_cfg,
        method_name,
        threshold,
        per_lang_path,
    )


CALIBRATION_SUPPORTED_METHODS = {"avg_confidence", "min_confidence", "prompt_ln_nll", "ft_probe", "random_baseline"}
CALIBRATION_SUPPORTED_DATASETS = {"polymath", "mmlu_prox_lite"}

def load_calibration_set(
    args,
    method: str,
) -> Optional[float]:
    base_model_name = args.model_name.split("/")[-1].replace("/", "_")

    if args.dataset_type == "polymath":
        if getattr(args, "polymath_split", None) != "low":
            logger.warning(
                "Loading calibration threshold from mgsm_filtered, which is originally align with Polymath low split."
            )
        calibration_dataset = "mgsm_filtered"
    else:  # mmlu_prox_lite
        calibration_dataset = "mmlu_prox_lite_dev"

    calibration_path = os.path.join(
        args.save_dir,
        base_model_name,
        calibration_dataset,
        f"ut_metrics_{method}_{args.seed}.json",
    )

    if not os.path.exists(calibration_path):
        raise FileNotFoundError(
            f"Calibration metrics file not found: {calibration_path}"
        )

    with open(calibration_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return (calibration_path, payload)

def load_threshold_from_calibration_set(
    args,
    method: str,
) -> Optional[float]:
    if method not in CALIBRATION_SUPPORTED_METHODS:
        logger.warning(
            "Calibration threshold loading is not supported for method %s.",
            method,
        )
        return None
    if args.dataset_type not in CALIBRATION_SUPPORTED_DATASETS:
        logger.warning(
            "Calibration threshold loading is not supported for dataset %s.",
            args.dataset_type,
        )
        return None

    calibration_path, payload = load_calibration_set(args, method)
    threshold = payload.get("threshold")
    if threshold is None:
        raise ValueError(
            f"Calibration metrics file {calibration_path} does not contain a 'threshold' field."
        )

    logger.info(
        "Loaded calibration threshold %.6f from %s",
        float(threshold),
        calibration_path,
    )
    return float(threshold)

def load_per_language_positive_class_ratio_from_calibration_set(
    args,
    method: str,
) -> Optional[float]:
    if method not in CALIBRATION_SUPPORTED_METHODS:
        logger.warning(
            "Positive class ratio loading is not supported for method %s.",
            method,
        )
        return None
    if args.dataset_type not in CALIBRATION_SUPPORTED_DATASETS:
        logger.warning(
            "Positive class ratio loading is not supported for dataset %s.",
            args.dataset_type,
        )
        return None

    calibration_path, payload = load_calibration_set(args, method)
    per_language = payload["metrics"]["per_language"]

    per_language_positive_ratios = {
        lang: float(lang_metrics["positives"] / float(lang_metrics["n"]))
        for lang, lang_metrics in per_language.items()
    }

    logger.info(
        "Loaded positive class ratio dict %s from %s",
        per_language_positive_ratios,
        calibration_path,
    )
    return per_language_positive_ratios

def build_predictions(
    samples_by_language: Dict[str, List[SampleScore]],
    threshold: Optional[float],
    method_cfg: MethodConfig,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    results: Dict[str, Dict[str, Dict[str, float]]] = {}
    for language, samples in samples_by_language.items():
        lang_results: Dict[str, Dict[str, float]] = {}
        for sample in samples:
            if method_cfg.requires_threshold:
                assert threshold is not None
                predicted = int(apply_threshold(
                    np.array([sample.score], dtype=float),
                    threshold,
                    method_cfg.higher_score_indicates_positive,
                )[0])
            else:
                predicted = sample.prediction if sample.prediction in (0, 1) else int(round(sample.score))
            lang_results[sample.sample_id] = {
                (method_cfg.score_field or "score"): sample.score,
                "predicted_not_understood": predicted,
                "label_not_understood": int(sample.label),
            }
        results[language] = lang_results
    return results



GPT_MONITORING_PROMPT = """You are given a problem (question and possibly options) and a model’s reasoning trace.
Your task is to decide whether the model correctly understood the problem.
Do not solve the problem yourself.

Return the output strictly in the following JSON format, with no extra text. 
The "Reason" field should be one or two sentences.

{{
  "understood": true/false,
  "Reason": "<one or two sentences explanation of why you judged it this way>"
}}

Problem:
{problem_text}

Reasoning Trace:
{reasoning_trace}
"""



FENCE_RE_JSON = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL | re.IGNORECASE)

def parse_gpt_monitoring_json(output_text: str) -> Dict[str, Any]:
    """
    Parse {"understood": bool, "Reason": str} from model output.
    - Strips ```json fences if present.
    - Uses strict json.loads.
    - If parsing fails, returns {"__raw_output__": ...} for later inspection.
    """
    if not isinstance(output_text, str):
        return {"__raw_output__": str(output_text)}

    def _to_bool_like(v):
        if isinstance(v, bool): return v
        if isinstance(v, str): return v.strip().lower() in {"true","yes","1"}
        if isinstance(v, (int, np.integer)): return int(v) != 0
        return False

    cleaned = output_text.strip()
    m = FENCE_RE_JSON.match(cleaned)
    if m:
        cleaned = m.group(1).strip()

    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict) and "understood" in obj and "Reason" in obj:
            return obj
        # Gracefully handle lowercased keys
        if isinstance(obj, dict):
            lower = {k.lower(): v for k, v in obj.items()}
            if "understood" in lower and "reason" in lower:
                return {"understood": _to_bool_like(lower["understood"]), "Reason": str(lower["reason"])}
    except Exception:
        pass

    return {"__raw_output__": output_text}


async def _gpt_monitor_one_async(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    gpt_model_name: str,
    problem_text: str,
    reasoning_trace: str,
    temperature: float = 0.0
) -> Dict[str, Any]:
    """
    Send one GPT monitoring request using the JSON+Reason prompt.
    Returns a parsed dict: {"understood": bool, "Reason": str} or {"__raw_output__": "..."} on parse failure.
    """
    prompt = GPT_MONITORING_PROMPT.format(problem_text=problem_text, reasoning_trace=reasoning_trace)
    async with semaphore:
        resp = await client.chat.completions.create(
            model=gpt_model_name,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
    content = resp.choices[0].message.content if resp and resp.choices else ""
    return parse_gpt_monitoring_json(content)



async def _gpt_monitor_lang_async(
    samples: List[Dict[str, Any]],
    original_task_results: Dict[str, Any],
    gpt_model_name: str,
    model_name: str, # Model name used for generating reasoning trace
    max_concurrency: int,
    head_token_limit: int,
    temperature: float,
) -> Dict[str, Dict[str, Any]]:
    client = AsyncOpenAI()
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    semaphore = asyncio.Semaphore(max_concurrency)
    try:
        tasks = []
        metas: List[Tuple[str, int, str]] = []
        for sample in samples:
            sid = sample["id"]
            task_res = original_task_results.get(str(sid)) or original_task_results.get(sid)
            if not task_res:
                continue
            reasoning_trace = sample.get("reasoning_trace", "") or ""
            if head_token_limit and head_token_limit > 0:
                reasoning_trace = truncate_trace_if_needed(
                    reasoning_trace,
                    tokenizer,
                    head_token_limit,
                )
            problem_text = sample.get("formatted_question", "")
            tasks.append(
                _gpt_monitor_one_async(
                    client,
                    semaphore,
                    gpt_model_name,
                    problem_text,
                    reasoning_trace,
                    temperature=temperature,
                )
            )
            metas.append((str(sid), int(task_res.get("correct", 0)), reasoning_trace))

        results: Dict[str, Dict[str, Any]] = {}
        if not tasks:
            return results

        gathered = await tqdm.gather(*tasks)
        for entry, (sid, correct, reasoning_used) in zip(gathered, metas):
            understood_flag = entry.get("understood")
            understood = "YES" if understood_flag else "NO"
            results[sid] = {
                "understood": understood,
                "reasoning_used": reasoning_used,
                "gpt_reason": entry.get("Reason"),
                "raw_output": entry,
                "correct": correct,
            }
        return results
    finally:
        await client.close()


def run_gpt_monitoring(
    args,
    dataset: Dict[str, List[Dict[str, Any]]],
    task_eval_results: Dict[str, Any],
    eval_langs: Sequence[str],
) -> Dict[str, Dict[str, Any]]:
    # if args.monitored_file_path and os.path.exists(args.monitored_file_path):
    #     logger.info("Loading pre-computed GPT monitoring results from %s", args.monitored_file_path)
    #     with open(args.monitored_file_path, "r", encoding="utf-8") as f:
    #         return json.load(f)

    if not os.environ.get("OPENAI_API_KEY") and args.gpt_api_key_path and os.path.exists(args.gpt_api_key_path):
        with open(args.gpt_api_key_path, "r", encoding="utf-8") as f:
            os.environ["OPENAI_API_KEY"] = f.read().strip()
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not set. Provide via env or --gpt_api_key_path.")

    lang2results: Dict[str, Dict[str, Any]] = {}
    for lang in eval_langs:
        if lang not in dataset:
            logger.warning("Language %s not in dataset; skipping", lang)
            continue
        samples = dataset[lang]
        lang_task = task_eval_results.get(lang, {})

        if not lang_task:
            logger.warning("Missing task eval results for lang=%s; skipping", lang)
            continue
        logger.info("Submitting %d GPT calls for %s…", len(samples), lang)
        lang_results = asyncio.run(
            _gpt_monitor_lang_async(
                samples=samples,
                original_task_results=lang_task,
                gpt_model_name=args.gpt_monitoring_model,
                model_name=args.model_name,
                max_concurrency=args.gpt_max_concurrency,
                head_token_limit=args.max_token_to_look_from_reasoning_trace,
                temperature=args.temperature,
            )
        )
        lang2results[lang] = lang_results
        logger.info("Done %s (%d judged)", lang, len(lang_results))
    return lang2results


def run_ft_mmbert_monitoring(
    args,
    dataset: Dict[str, List[Dict[str, Any]]],
    task_eval_results: Dict[str, Any],
    eval_langs: Sequence[str],
) -> Dict[str, Dict[str, Any]]:
    from tqdm.auto import tqdm

    if not args.ft_mmbert_model_path or not os.path.exists(args.ft_mmbert_model_path):
        raise FileNotFoundError(
            "--ft_mmbert_model_path required for ft_mmbert_monitoring and must exist",
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    tokenizer_ft = AutoTokenizer.from_pretrained(args.ft_mmbert_model_path, use_fast=True)
    model_ft = AutoModelForSequenceClassification.from_pretrained(args.ft_mmbert_model_path)
    model_ft.to(device)
    model_ft.eval()
    logger.info("Loaded fine-tuned MMBERT model from %s", args.ft_mmbert_model_path)
    def build_text(formatted_question: str, reasoning: str) -> str:
        sep = tokenizer_ft.sep_token if tokenizer_ft.sep_token else "[SEP]"
        return f"{formatted_question} {sep} {reasoning}".strip()

    lang2results: Dict[str, Dict[str, Any]] = {}
    pbar_lang = tqdm(eval_langs, desc="Languages", unit="lang", position=0)

    for lang in pbar_lang:
        pbar_lang.set_description(f"Lang: {lang}")

        if lang not in dataset:
            logger.warning("Language %s not in dataset; skipping", lang)
            continue

        samples = dataset[lang]
        lang_task = task_eval_results.get(lang, {})
        texts: List[str] = []
        metas: List[Tuple[str, int]] = []

        # ---- inner tqdm #1: sample preprocessing (collect texts/metadata) ----
        for sample in tqdm(
            samples,
            desc=f"{lang}: collect",
            unit="sample",
            position=1,   # line below the outer loop
            leave=False,  # let the next bar overwrite this line
            mininterval=0.1,
        ):
            sid = sample["id"]
            task_res = lang_task.get(str(sid)) or lang_task.get(sid)
            if not task_res:
                continue
            rt = sample.get("reasoning_trace", "")
            if args.max_token_to_look_from_reasoning_trace > 0:
                rt = truncate_trace_if_needed(
                    rt,
                    tokenizer,
                    args.max_token_to_look_from_reasoning_trace,
                )
                sample["reasoning_trace"] = rt
            texts.append(build_text(sample.get("formatted_question", ""), sample.get("reasoning_trace", "")))
            metas.append((str(sid), int(task_res.get("correct", 0))))

        if not texts:
            continue

        enc = tokenizer_ft(
            texts,
            truncation=True,
            padding=True,
            max_length=args.ft_mmbert_max_length,
            return_tensors="pt",
        )
        ds = TensorDataset(enc["input_ids"], enc["attention_mask"])
        loader = DataLoader(ds, batch_size=args.ft_mmbert_batch_size)

        lang2results[lang] = {}
        with torch.no_grad():
            idx = 0

            # ---- inner tqdm #2: batch inference ----
            for batch in tqdm(
                loader,
                desc=f"{lang}: infer",
                unit="batch",
                total=len(loader),
                position=1,   # reuse the same line
                leave=False,
                mininterval=0.1,
            ):
                input_ids, attention_mask = [x.to(device) for x in batch]
                out = model_ft(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.softmax(out.logits, dim=-1)
                preds = torch.argmax(probs, dim=-1) # 1 = not understood, 0 = understood

                for prob_vec, pred, logit_vec in zip(probs.tolist(), preds.tolist(), out.logits.tolist()):
                    sid, corr = metas[idx]
                    understood_flag = (pred == 0)
                    lang2results[lang][sid] = {
                        "understood": "YES" if understood_flag else "NO",
                        "prob_understood": float(prob_vec[0]),
                        "prob_not_understood": float(prob_vec[1]),
                        "raw_logits": [float(x) for x in logit_vec],
                        "correct": corr,
                    }
                    idx += 1
    return lang2results


def run_ft_probe(
    args,
    eval_langs: Sequence[str],
) -> Dict[str, Dict[str, Any]]:
    if not args.ft_probe_model_dir or not os.path.exists(args.ft_probe_model_dir):
        raise FileNotFoundError("--ft_probe_model_dir is required for ft_probe and must point to an existing checkpoint.")
    if not args.signal_with_label_path or not os.path.exists(args.signal_with_label_path):
        raise FileNotFoundError("--signal_with_label_path is required for ft_probe and must exist.")

    checkpoint_path = os.path.join(args.ft_probe_model_dir, "best_probe.pt")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    best_configuration_path = os.path.join(args.ft_probe_model_dir, "best_configuration.json")
    with open(best_configuration_path, "r", encoding="utf-8") as f:
        best_cfg = json.load(f)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        raise ValueError("Probe checkpoint must contain a model state_dict under key 'model' or be a plain state_dict.")

    model_input_dim = hs_dict.get(args.model_name)
    pca_model = None
    pca_expected_input_dim: Optional[int] = None
    pca_original_dim: Optional[int] = None

    if args.use_pca:
        pca_path = args.pca_model_path or os.path.join(args.ft_probe_model_dir, "pca_model.pkl")
        if not os.path.exists(pca_path):
            raise FileNotFoundError(
                f"--use_pca was set but no PCA artifact found at {pca_path}. "
                "Pass --pca_model_path or ensure pca_model.pkl exists inside the probe directory."
            )
        with open(pca_path, "rb") as fh:
            loaded = pickle.load(fh)
        if isinstance(loaded, dict) and "pca" in loaded:
            pca_model = loaded["pca"]
            pca_expected_input_dim = int(loaded.get("pca_dim")) if loaded.get("pca_dim") is not None else None
            pca_original_dim = int(loaded.get("original_dim")) if loaded.get("original_dim") is not None else None
        else:
            pca_model = loaded
            pca_expected_input_dim = getattr(loaded, "n_components", None) or getattr(loaded, "n_components_", None)
            if pca_expected_input_dim is not None:
                pca_expected_input_dim = int(pca_expected_input_dim)
            pca_original_dim = getattr(loaded, "n_features_in_", None)
            if pca_original_dim is not None:
                pca_original_dim = int(pca_original_dim)

        if pca_model is None:
            raise ValueError(f"Failed to load PCA model from {pca_path}: unexpected object type {type(loaded).__name__}")

        if args.pca_dim is not None:
            pca_expected_input_dim = args.pca_dim if pca_expected_input_dim is None else pca_expected_input_dim
            if args.pca_dim != pca_expected_input_dim:
                raise ValueError(
                    f"--pca_dim ({args.pca_dim}) does not match the PCA artifact dimension ({pca_expected_input_dim})."
                )
        elif pca_expected_input_dim is None:
            raise ValueError(
                "--pca_dim must be provided when --use_pca is enabled and the PCA artifact does not include its dimensionality."
            )

        if pca_expected_input_dim is None:
            raise ValueError("Unable to determine PCA dimensionality; ensure the artifact stores it or pass --pca_dim.")

        model_input_dim = int(pca_expected_input_dim)
        logger.info("Loaded PCA model from %s with output dimension %d", pca_path, model_input_dim)
    else:
        if args.pca_dim is not None or args.pca_model_path:
            logger.warning("Ignoring --pca_dim/--pca_model_path because --use_pca is not set.")

    if model_input_dim is None:
        raise KeyError(f"Model name {args.model_name} not found in hs_dict and no PCA configuration provided.")

    hidden_size = int(best_cfg["hidden_size"])

    probe_model = load_model(model_input_dim, hidden_size, 2)
    probe_model.load_state_dict(state_dict)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    probe_model.to(device)
    probe_model.eval()

    per_language, feature_dim = load_probe_features(
        args.signal_with_label_path,
        languages=eval_langs,
    )

    if args.use_pca:
        if pca_original_dim is not None and feature_dim != pca_original_dim:
            logger.warning(
                "Loaded hidden-state dimension %d differs from PCA artifact expectation %d.",
                feature_dim,
                pca_original_dim,
            )
    elif feature_dim != model_input_dim:
        logger.warning(
            "Loaded probe input dim %d differs from checkpoint expectation %d; proceeding but results may be incorrect.",
            feature_dim,
            model_input_dim,
        )

    batch_size = max(1, int(getattr(args, "ft_probe_batch_size", 256)))
    results: Dict[str, Dict[str, Any]] = {}

    with torch.no_grad():
        for language, entries in per_language.items():
            if not entries:
                continue
            features = torch.stack([hidden for _, hidden, _ in entries], dim=0)
            if args.use_pca and pca_model is not None:
                transformed = pca_model.transform(features.cpu().numpy())
                features = torch.from_numpy(transformed).to(torch.float32)
            dataset = TensorDataset(features)
            loader = DataLoader(dataset, batch_size=batch_size)

            lang_results: Dict[str, Dict[str, Any]] = {}
            idx = 0
            for (batch_features,) in loader:
                batch_features = batch_features.to(device).to(torch.float32)
                logits = probe_model(batch_features)
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                logits_np = logits.cpu().numpy()
                for prob_vec, logit_vec in zip(probs, logits_np):
                    sample_id, _hidden, label = entries[idx]
                    lang_results[sample_id] = {
                        "prob_understood": float(prob_vec[0]),
                        "prob_not_understood": float(prob_vec[1]),
                        "raw_logits": [float(x) for x in logit_vec],
                        "predicted_not_understood": int(prob_vec[1] >= 0.5),
                        "not_understood_label": label,
                        "correct": 1 - label,
                    }
                    idx += 1
            results[language] = lang_results

    return results


def compute_metrics_summary(
    samples_by_language: Dict[str, List[SampleScore]],
    *,
    threshold: Optional[float],
    method_cfg: MethodConfig,
) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    for language, samples in samples_by_language.items():
        summary[language] = compute_metrics_for_samples(
            samples,
            threshold=threshold,
            method_cfg=method_cfg,
        )
    return summary

def file_path_sanity_check(args):
    assert str(args.seed) in args.task_eval_results_path, "Seed value is not identical in --seed and --task_eval_results_path"
    assert str(args.seed) in args.thinking_intv_eval_results_path, "Seed value is not identical in --seed and --thinking_intv_eval_results_path"


def main(args: argparse.Namespace) -> None:
    method = args.understandability_test_method
    if method not in SUPPORTED_METHODS:
        raise ValueError(
            f"Unsupported understandability_test_method '{method}'. Supported methods: {', '.join(sorted(SUPPORTED_METHODS))}"
        )
    method_cfg = SUPPORTED_METHODS[method]

    file_path_sanity_check(args)

    eval_langs = [lang.strip() for lang in args.eval_langs.split(",") if lang.strip()]

    eval_samples_by_language: Dict[str, List[SampleScore]]
    raw_results: Optional[Dict[str, Dict[str, Any]]] = None

    if method in {"self-reflection", "gpt_monitoring", "ft_mmbert_monitoring"}:
        if args.raw_results_output_path and os.path.exists(args.raw_results_output_path):
            logger.info(
                "Loading pre-computed %s results from %s",
                method,
                args.raw_results_output_path,
            )
            eval_samples_by_language = load_samples_for_method(
                args,
                args.raw_results_output_path,
                method_cfg,
                languages=eval_langs,
                seed=args.seed,
            )
        else:
            required_paths = [
                (args.task_eval_results_path, "--task_eval_results_path"),
                (args.thinking_intv_eval_results_path, "--thinking_intv_eval_results_path"),
            ]
            for path_value, flag in required_paths:
                if not path_value:
                    raise ValueError(f"{flag} is required for method {method}")
                if not os.path.exists(path_value):
                    raise FileNotFoundError(f"{flag} not found: {path_value}")

            if getattr(args, "polymath_split", None) == "mid":
                logger.warning("Polymath 'mid' split is an alias for 'medium'. Using 'medium'.")
                args.polymath_split = "medium"

            dataset_builder = UnderStandabilityEvalDataset(args, eval_langs)
            dataset_rows = dataset_builder.get()
            task_eval_results = dataset_builder.task_eval_results

            if not eval_langs:
                eval_langs = list(dataset_rows.keys())

            if method == "self-reflection":
                raw_results = run_self_reflection(args, dataset_rows, task_eval_results, eval_langs)
            elif method == "gpt_monitoring":
                raw_results = run_gpt_monitoring(args, dataset_rows, task_eval_results, eval_langs)
            else:
                raw_results = run_ft_mmbert_monitoring(args, dataset_rows, task_eval_results, eval_langs)

            eval_samples_by_language = build_samples_from_results(
                raw_results,
                method_cfg,
                languages=eval_langs,
            )
    elif method == "ft_probe":
        if args.raw_results_output_path and os.path.exists(args.raw_results_output_path):
            logger.info(
                "Loading pre-computed ft_probe results from %s",
                args.raw_results_output_path,
            )
            eval_samples_by_language = load_samples_for_method(
                args,
                args.raw_results_output_path,
                method_cfg,
                languages=eval_langs,
                seed=args.seed,
            )
        else:
            raw_results = run_ft_probe(args, eval_langs)
            eval_samples_by_language = build_samples_from_results(
                raw_results,
                method_cfg,
                languages=eval_langs,
            )
    else:
        eval_samples_by_language = load_samples_for_method(
            args,
            args.signal_with_label_path,
            method_cfg,
            languages=eval_langs,
            seed=args.seed
        )
    eval_samples_flat = flatten_samples(eval_samples_by_language, args.exclude_english)
    logger.info(
        "Loaded %d evaluation samples across %d languages",
        len(eval_samples_flat),
        len(eval_samples_by_language),
    )

    threshold: Optional[float]
    threshold_source: str
    if method_cfg.requires_threshold:
        threshold = None
        threshold_source = "auto_eval"
        if getattr(args, "user_threshold", None) is not None:
            threshold = float(args.user_threshold)
            threshold_source = "user"
            logger.info(
                "Using user-specified threshold %.6f for method %s",
                threshold,
                method,
            )
        elif getattr(args, "use_threshold_from_calibration_set", False):
            loaded_threshold = load_threshold_from_calibration_set(args, method)
            if loaded_threshold is not None:
                threshold = loaded_threshold
                threshold_source = "calibration"
        if threshold is None and method_cfg.default_threshold is not None:
            threshold = float(method_cfg.default_threshold)
            threshold_source = "default"
            logger.info(
                "Using default threshold %.6f for method %s",
                threshold,
                method,
            )
        if threshold is None:
            threshold = determine_best_threshold(eval_samples_flat, method_cfg)
            threshold_source = "auto_eval"
    else:
        threshold_source = "n/a"
        threshold = None
        if getattr(args, "use_threshold_from_calibration_set", False):
            logger.warning(
                "Calibration threshold requested but method '%s' does not use thresholds; ignoring.",
                method,
            )

    suffix_map = {
        "calibration": "_from_calibration_thr",
    }
    filename_suffix = suffix_map.get(threshold_source, "")

    metrics_per_language = compute_metrics_summary(
        eval_samples_by_language,
        threshold=threshold,
        method_cfg=method_cfg,
    )
    overall_metrics = compute_metrics_for_samples(
        eval_samples_flat,
        threshold=threshold,
        method_cfg=method_cfg,
    )
    predictions = build_predictions(
        eval_samples_by_language,
        threshold=threshold,
        method_cfg=method_cfg,
    )

    output = {
        "dataset_type": args.dataset_type,
        "model_name": args.model_name,
        "seed": args.seed,
        "method": method,
        "score_field": method_cfg.score_field,
        "higher_score_indicates_positive": method_cfg.higher_score_indicates_positive,
        "threshold": threshold,
        "threshold_source": threshold_source,
        "metrics": {
            "overall": overall_metrics,
            "per_language": metrics_per_language,
        },
    }

    overall_metrics = output["metrics"]["overall"]
    print(f"Overall metric | Balanced acc: {overall_metrics["balanced_accuracy"]:.4f} | F1: {overall_metrics["f1"]:.4f}")
    base_model_name = args.model_name.split("/")[-1].replace("/", "_")
    if args.dataset_type == "polymath" and getattr(args, "polymath_split", None):
        output_save_dir = os.path.join(
            args.save_dir,
            base_model_name,
            f"{args.dataset_type}_{args.polymath_split}"
        )
    else:
        output_save_dir = os.path.join(
            args.save_dir,
            base_model_name,
            args.dataset_type,
        )
    os.makedirs(output_save_dir, exist_ok=True)

    dataset_label = args.dataset_type
    if args.dataset_type == "polymath" and getattr(args, "polymath_split", None):
        dataset_label = f"{args.dataset_type}_{args.polymath_split}"

    generate_and_save_boxplots(
        eval_samples_by_language,
        method_cfg,
        threshold,
        method_name=method,
        model_name=args.model_name,
        dataset_type=dataset_label,
        seed=args.seed,
        output_dir=output_save_dir,
        filename_suffix=filename_suffix,
    )


    if raw_results:
        raw_save_path = os.path.join(
            output_save_dir,
            f"ut_raw_results_{method}_{args.seed}{filename_suffix}.json",
        )
        if args.raw_results_output_path:
            raw_save_path = args.raw_results_output_path

        os.makedirs(os.path.dirname(raw_save_path), exist_ok=True)
        with open(raw_save_path, "w", encoding="utf-8") as f:
            json.dump(raw_results, f, indent=2, ensure_ascii=False)
        logger.info("Saved raw results to %s", raw_save_path)

    # Save metrics summary and predictions if paths are provided, else print metrics
    metric_save_path = os.path.join(
        output_save_dir,
        f"ut_metrics_{method}_{args.seed}{filename_suffix}{args.custom_postfix}.json",
    )
    if args.metrics_output_path:
        metric_save_path = args.metrics_output_path
    
    os.makedirs(os.path.dirname(metric_save_path), exist_ok=True)
    with open(metric_save_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    logger.info("Saved metrics summary to %s", metric_save_path)

    prediction_save_path = os.path.join(
        output_save_dir,
        f"ut_predictions_{method}_{args.seed}{filename_suffix}{args.custom_postfix}.json",
    )
    if args.predictions_output_path:
        prediction_save_path = args.predictions_output_path
    os.makedirs(os.path.dirname(prediction_save_path), exist_ok=True)
    with open(prediction_save_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)
    logger.info("Saved per-sample predictions to %s", prediction_save_path)

def truncated_sanity_check(args):
    # If truncation is enabled, this must be reflected in the filenames
    if args.signal_with_label_path:
        assert f"_head{args.max_token_to_look_from_reasoning_trace}_" in args.signal_with_label_path, 'If --max_token_to_look_from_reasoning_trace is set, the value must be reflected in --signal_with_label_path filename (e.g. _head128_)'
    if args.ft_mmbert_model_path:
        assert f"{args.max_token_to_look_from_reasoning_trace}" in args.ft_mmbert_model_path, 'If --max_token_to_look_from_reasoning_trace is set, the value must be reflected in --ft_mmbert_model_path directory name'
    if args.ft_probe_model_dir:
        assert f"{args.max_token_to_look_from_reasoning_trace}" in args.ft_probe_model_dir, 'If --max_token_to_look_from_reasoning_trace is set, the value must be reflected in --ft_probe_model_dir directory name'

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Understandability test metrics scorer")
    parser.add_argument("--signal_with_label_path", type=str, default=None, required=False,
                        help="Path to input signals (.pth from ut_compute_signals_with_label.py or JSON from other methods)")
    parser.add_argument("--understandability_test_method", type=str, required=True,
                        choices=sorted(SUPPORTED_METHODS.keys()),
                        help="Scoring method to evaluate")

    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--dataset_type", type=str, required=True,
                        choices=[
                            "mgsm_filtered",
                            "mmlu_prox_lite",
                            "mmlu_prox_lite_dev",
                            "polymath",
                        ])
    parser.add_argument("--eval_langs", type=str,
                        default="en,de,es,ar,ja,ko,th,bn,sw,te",
                        help="Comma-separated list of languages to evaluate")
    parser.add_argument("--polymath_split", type=str, default="low",
                        choices=["low", "medium", "high", "mid"],
                        help="Polymath split to evaluate")
    parser.add_argument("--task_eval_results_path", type=str, required=True,
                        help="Path to task evaluation results JSON")
    parser.add_argument("--thinking_intv_eval_results_path", type=str, required=True,
                        help="Path to thinking intervention evaluation results JSON")

    parser.add_argument("--max_token_to_look_from_reasoning_trace", type=int, default=-1,
                        help="If >0 truncate reasoning trace tokens before scoring")
    parser.add_argument("--use_threshold_from_calibration_set", action="store_true",
                        help="If set, load the threshold from a calibration set instead of computing it")
    parser.add_argument("--user_threshold", type=float, default=None,
                        help="If set, use this threshold instead of computing it")

    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=-1)
    parser.add_argument("--self_reflection_max_new_tokens", type=int, default=128)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8)
    parser.add_argument("--seed", type=int, required=True)

    # GPT monitoring arguments
    parser.add_argument("--gpt_monitoring_model", type=str, default="gpt-4.1-mini")
    parser.add_argument("--gpt_max_concurrency", type=int, default=32)
    parser.add_argument("--gpt_api_key_path", type=str, default=None)

    # Fine-tuned mmBERT arguments
    parser.add_argument("--ft_mmbert_model_path", type=str, default=None,
                        help="Directory containing fine-tuned mmBERT model")
    parser.add_argument("--ft_mmbert_batch_size", type=int, default=4)
    parser.add_argument("--ft_mmbert_max_length", type=int, default=8192)

    # Fine-tuned probe arguments
    parser.add_argument("--ft_probe_model_dir", type=str, default=None,
                        help="Path to fine-tuned probe directory")
    parser.add_argument("--ft_probe_batch_size", type=int, default=256,
                        help="Batch size for probe inference")
    parser.add_argument("--use_pca", action="store_true",
                        help="Apply PCA-transform to hidden states before probe inference.")
    parser.add_argument("--pca_dim", type=int, default=None,
                        help="Expected PCA output dimensionality. If omitted, inferred from the PCA artifact.")
    parser.add_argument("--pca_model_path", type=str, default=None,
                        help="Optional path to a pickled PCA model. Defaults to <ft_probe_model_dir>/pca_model.pkl when --use_pca is set.")

    parser.add_argument("--save_dir", type=str, default="./ut_test_results",
                        help="Base directory for saving outputs")
    parser.add_argument("--metrics_output_path", type=str, default=None,
                        help="Optional path to save aggregated metrics JSON")
    parser.add_argument("--predictions_output_path", type=str, default=None,
                        help="Optional path to save per-sample prediction JSON")
    parser.add_argument("--raw_results_output_path", type=str, default=None,
                        help="Optional path to save raw method outputs (self-reflection/GPT/mmBERT). If set, we will skip method computation if the file exists.")

    parser.add_argument("--custom_postfix", type=str, default="",
                        help="Custom string to append to output filenames, e.g. '_v2'.")
    parser.add_argument("--exclude_english", action="store_true",
                        help="If set, exclude English samples from overall metrics and plots. This is required for unseen-language evaluation.")

    parser.add_argument("--low_resource_experiment", action="store_true",
                        help="Whether to run low-resource experiment on PolyMath dataset."
                        )
    parser.add_argument("--translated_dataset_json_path", type=str, default=None,
                        help="Path to the JSON file containing translated questions and optinally options for thinking intervention."
                        )

    args = parser.parse_args()

    if args.understandability_test_method not in {"self-reflection", "gpt_monitoring", "ft_mmbert_monitoring"}:
        if not args.signal_with_label_path:
            raise ValueError("--signal_with_label_path is required for signal-based methods")
    if args.max_token_to_look_from_reasoning_trace != -1:
        truncated_sanity_check(args)

    main(args)
