"""Selective translation metric aggregation utility."""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import os
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

LOGGER = logging.getLogger("selective_translation")

CALIBRATION_THRESHOLD_METHODS = {"avg_confidence", "min_confidence", "prompt_ln_nll"}
CALIBRATION_SUFFIX = "_from_calibration_thr"
SCENARIOS: Tuple[Tuple[str, Optional[float]], ...] = (
    ("default", None),
    ("fnr@0.05", 0.05),
    ("fnr@0.10", 0.10),
)


@dataclass
class MethodScoreConfig:
    score_field: Optional[str]
    higher_score_indicates_positive: bool = True
    supports_threshold_adjustment: bool = False


METHOD_SCORE_CONFIGS: Dict[str, MethodScoreConfig] = {
    "random_baseline": MethodScoreConfig(
        score_field="pos_probability",
        higher_score_indicates_positive=True,
        supports_threshold_adjustment=True,
    ),
    "avg_confidence": MethodScoreConfig(
        score_field="avg_confidence",
        higher_score_indicates_positive=False,
        supports_threshold_adjustment=True,
    ),
    "min_confidence": MethodScoreConfig(
        score_field="min_confidence",
        higher_score_indicates_positive=False,
        supports_threshold_adjustment=True,
    ),
    "prompt_ln_nll": MethodScoreConfig(
        score_field="prompt_ln_nll",
        higher_score_indicates_positive=True,
        supports_threshold_adjustment=True,
    ),
    "ft_mmbert_monitoring": MethodScoreConfig(
        score_field="prob_not_understood",
        higher_score_indicates_positive=True,
        supports_threshold_adjustment=True,
    ),
    "self-reflection": MethodScoreConfig(
        score_field=None,
        higher_score_indicates_positive=True,
        supports_threshold_adjustment=False,
    ),
    "gpt_monitoring": MethodScoreConfig(
        score_field=None,
        higher_score_indicates_positive=True,
        supports_threshold_adjustment=False,
    ),
    "ft_probe": MethodScoreConfig(
        score_field="prob_not_understood",
        higher_score_indicates_positive=True,
        supports_threshold_adjustment=True,
    ),
}


@dataclass
class SampleEntry:
    score: Optional[float]
    default_prediction: bool
    label: Optional[int]


def _normalise_eval_langs(eval_langs: Iterable[str] | str | None) -> List[str]:
    if eval_langs is None:
        return []
    if isinstance(eval_langs, str):
        return [lang.strip() for lang in eval_langs.split(",") if lang.strip()]
    return [lang.strip() for lang in eval_langs if lang and lang.strip()]


def _load_json(path: Path) -> Dict[str, Dict[str, Dict[str, object]]]:
    if not path.is_file():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Unexpected JSON structure in {path}: expected dict at top level")
    return payload


def _coerce_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return int(value) != 0
    if isinstance(value, str):
        normalised = value.strip().lower()
        if normalised in {"1", "true", "yes", "y", "t"}:
            return True
        if normalised in {"0", "false", "no", "n", "f"}:
            return False
    return bool(value)


def _coerce_float(value: object) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        result = float(value)
    elif isinstance(value, str):
        try:
            result = float(value)
        except ValueError:
            return None
    else:
        return None
    return result if math.isfinite(result) else None


def _coerce_int(value: object) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        value_int = int(value)
    elif isinstance(value, int):
        value_int = value
    elif isinstance(value, float):
        value_int = int(value)
    else:
        try:
            value_int = int(str(value).strip())
        except (TypeError, ValueError):
            return None
    if value_int in (0, 1):
        return value_int
    return None


def _extract_seed_from_stem(stem: str) -> Optional[int]:
    matches = re.findall(r"_(\d+)(?:_|$)", stem)
    if not matches:
        return None
    return int(matches[-1])


def _collect_seeds_from_paths(paths: Iterable[Path]) -> List[int]:
    seeds: List[int] = []
    for path in paths:
        if not path:
            continue
        candidate = _extract_seed_from_stem(path.stem)
        if candidate is not None:
            seeds.append(candidate)
    return seeds


def _list_available_prediction_seeds(
    predictions_dir: Path,
    method: str,
    suffix: str,
) -> List[int]:
    regex = re.compile(
        rf"^ut_predictions_{re.escape(method)}_(\d+){re.escape(suffix)}\.json$"
    )
    seeds: List[int] = []
    if not predictions_dir.is_dir():
        return seeds
    for path in predictions_dir.glob(f"ut_predictions_{method}_*{suffix or ''}.json"):
        match = regex.match(path.name)
        if match:
            seeds.append(int(match.group(1)))
    return sorted(set(seeds))


def _resolve_predictions_path(
    understandability_test_method: str,
    model_name: str,
    dataset_type: str,
    polymath_split: Optional[str],
    task_eval_path: Path,
    thinking_eval_path: Path,
    use_threshold_from_calibration_set: bool,
    ut_test_results_dir: str,
    custom_postfix: Optional[str] = None
) -> Path:
    base_model_name = model_name.split("/")[-1].replace("/", "_")
    if dataset_type == "polymath":
        if not polymath_split:
            raise ValueError("polymath_split must be provided when dataset_type is 'polymath'")
        dataset_dir_name = f"{dataset_type}_{polymath_split}"
    else:
        dataset_dir_name = dataset_type
    ut_test_results_dir = Path(ut_test_results_dir)
    predictions_dir = ut_test_results_dir / base_model_name / dataset_dir_name
    if not predictions_dir.is_dir():
        raise FileNotFoundError(f"Predictions directory not found: {predictions_dir}")

    suffix = ""
    if use_threshold_from_calibration_set and understandability_test_method in CALIBRATION_THRESHOLD_METHODS:
        suffix = CALIBRATION_SUFFIX

    if custom_postfix:
        suffix += custom_postfix
    inferred_seeds = sorted(set(_collect_seeds_from_paths([task_eval_path, thinking_eval_path])))
    candidate_seeds: List[int] = []
    if inferred_seeds:
        available = _list_available_prediction_seeds(predictions_dir, understandability_test_method, suffix)
        for seed in inferred_seeds:
            if seed in available:
                candidate_seeds.append(seed)
        if not candidate_seeds and suffix:
            LOGGER.warning(
                "Could not locate calibration predictions for seed(s) %s; trying without calibration suffix.",
                ", ".join(map(str, inferred_seeds)),
            )
            suffix = ""
            candidate_seeds = [
                seed
                for seed in inferred_seeds
                if seed in _list_available_prediction_seeds(
                    predictions_dir,
                    understandability_test_method,
                    suffix,
                )
            ]
    if not candidate_seeds:
        available = _list_available_prediction_seeds(predictions_dir, understandability_test_method, suffix)
        if not available and suffix:
            LOGGER.warning(
                "No predictions found with calibration suffix for method '%s'. Falling back to default predictions.",
                understandability_test_method,
            )
            suffix = ""
            available = _list_available_prediction_seeds(predictions_dir, understandability_test_method, suffix)
        candidate_seeds = available

    candidate_seeds = sorted(set(candidate_seeds))
    if not candidate_seeds:
        raise FileNotFoundError(
            f"Could not locate predictions for method '{understandability_test_method}' in {predictions_dir}"
        )
    if len(candidate_seeds) > 1:
        raise ValueError(
            "Multiple candidate seeds found for predictions; please disambiguate. "
            f"Seeds: {', '.join(map(str, candidate_seeds))}"
        )

    seed = candidate_seeds[0]
    prediction_path = predictions_dir / f"ut_predictions_{understandability_test_method}_{seed}{suffix}.json"
    print(f"Load predictions from: {prediction_path}")
    if not prediction_path.is_file():
        raise FileNotFoundError(f"Prediction file not found: {prediction_path}")
    return prediction_path


def _get_method_score_config(method: str) -> MethodScoreConfig:
    config = METHOD_SCORE_CONFIGS.get(method)
    if config is None:
        LOGGER.warning(
            "Method '%s' not found in METHOD_SCORE_CONFIGS. Threshold adjustments will be disabled.",
            method,
        )
        return MethodScoreConfig(score_field=None, higher_score_indicates_positive=True, supports_threshold_adjustment=False)
    return config


def _compute_threshold_for_target_fnr(
    scores: List[float],
    labels: List[int],
    target_fnr: float,
    config: MethodScoreConfig,
) -> Optional[float]:
    if not config.supports_threshold_adjustment:
        return None
    if not scores or len(scores) != len(labels):
        return None

    positives = sum(1 for label in labels if label == 1)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None

    unique_scores = sorted({score for score in scores if score is not None and math.isfinite(score)})
    if not unique_scores:
        return None

    if config.higher_score_indicates_positive:
        candidate_thresholds = sorted(unique_scores, reverse=True)
    else:
        candidate_thresholds = unique_scores

    tolerance = target_fnr + 1e-6

    for threshold in candidate_thresholds:
        tp = tn = fp = fn = 0
        for score, label in zip(scores, labels):
            if score is None or not math.isfinite(score):
                continue
            predicted_positive = score >= threshold if config.higher_score_indicates_positive else score <= threshold
            if label == 1:
                if predicted_positive:
                    tp += 1
                else:
                    fn += 1
            else:
                if predicted_positive:
                    fp += 1
                else:
                    tn += 1

        positives_count = tp + fn
        negatives_count = tn + fp
        if positives_count == 0 or negatives_count == 0:
            continue
        fnr = fn / positives_count
        if fnr <= tolerance:
            return threshold

    return None


def _apply_threshold(value: Optional[float], threshold: Optional[float], config: MethodScoreConfig) -> Optional[bool]:
    if threshold is None or value is None:
        return None
    if config.higher_score_indicates_positive:
        return value >= threshold
    return value <= threshold


def _build_scenario_prediction_map(
    samples_by_language: Dict[str, Dict[str, SampleEntry]],
    target_langs: List[str],
    config: MethodScoreConfig,
    threshold: Optional[float],
) -> Tuple[Dict[str, Dict[str, bool]], int]:
    scenario_predictions: Dict[str, Dict[str, bool]] = {}
    fallback_to_default = 0

    for lang in target_langs:
        entries = samples_by_language.get(lang, {})
        lang_predictions: Dict[str, bool] = {}
        for sample_id, entry in entries.items():
            if threshold is None:
                prediction = entry.default_prediction
            else:
                adjusted = _apply_threshold(entry.score, threshold, config)
                if adjusted is None:
                    fallback_to_default += 1
                    prediction = entry.default_prediction
                else:
                    prediction = bool(adjusted)
            lang_predictions[sample_id] = prediction
        scenario_predictions[lang] = lang_predictions

    return scenario_predictions, fallback_to_default


def _aggregate_metrics(
    prediction_map: Dict[str, Dict[str, bool]],
    task_eval: Dict[str, Dict[str, Dict[str, object]]],
    thinking_eval: Dict[str, Dict[str, Dict[str, object]]],
) -> Tuple[Dict[str, Dict[str, object]], Dict[str, object]]:
    per_language: Dict[str, Dict[str, object]] = {}
    overall_total = 0
    overall_selective = 0
    overall_usable = 0
    overall_correct_sum = 0.0
    overall_missing = 0

    for lang, lang_predictions in prediction_map.items():
        total_predictions = len(lang_predictions)
        selective_predictions = 0
        usable_samples = 0
        correct_sum = 0.0
        missing_samples = 0

        task_lang_payload = task_eval.get(lang)
        if not isinstance(task_lang_payload, dict):
            task_lang_payload = {}
        thinking_lang_payload = thinking_eval.get(lang)
        if not isinstance(thinking_lang_payload, dict):
            thinking_lang_payload = {}

        for sample_id, predicted_not_understood in lang_predictions.items():
            source_payload = thinking_lang_payload if predicted_not_understood else task_lang_payload
            if predicted_not_understood:
                selective_predictions += 1

            sample_payload = source_payload.get(sample_id)
            if sample_payload is None:
                missing_samples += 1
                LOGGER.warning(
                    "Sample '%s' for language '%s' not found in %s results.",
                    sample_id,
                    lang,
                    "thinking intervention" if predicted_not_understood else "task",
                )
                continue

            correct_value = _coerce_float(sample_payload.get("correct"))
            if correct_value is None:
                missing_samples += 1
                LOGGER.warning(
                    "Missing or invalid 'correct' value for sample '%s' (language '%s').",
                    sample_id,
                    lang,
                )
                continue

            usable_samples += 1
            correct_sum += correct_value

        accuracy = (correct_sum / usable_samples) if usable_samples else None
        selective_rate = (selective_predictions / total_predictions) if total_predictions else None

        per_language[lang] = {
            "total_predictions": total_predictions,
            "selective_predictions": selective_predictions,
            "selective_rate": selective_rate,
            "usable_samples": usable_samples,
            "missing_samples": missing_samples,
            "accuracy": accuracy,
        }

        overall_total += total_predictions
        overall_selective += selective_predictions
        overall_usable += usable_samples
        overall_correct_sum += correct_sum
        overall_missing += missing_samples

    overall_accuracy = (overall_correct_sum / overall_usable) if overall_usable else None
    overall_selective_rate = (overall_selective / overall_total) if overall_total else None

    overall_summary = {
        "total_predictions": overall_total,
        "selective_predictions": overall_selective,
        "selective_rate": overall_selective_rate,
        "usable_samples": overall_usable,
        "missing_samples": overall_missing,
        "accuracy": overall_accuracy,
    }

    return per_language, overall_summary


def _compute_label_confusion(
    samples_by_language: Dict[str, Dict[str, SampleEntry]],
    prediction_map: Dict[str, Dict[str, bool]],
) -> Dict[str, Optional[float]]:
    tp = tn = fp = fn = 0
    positives = negatives = 0

    for lang, lang_predictions in prediction_map.items():
        entries = samples_by_language.get(lang, {})
        for sample_id, predicted_not_understood in lang_predictions.items():
            entry = entries.get(sample_id)
            if entry is None or entry.label is None:
                continue
            label = entry.label
            if label == 1:
                positives += 1
                if predicted_not_understood:
                    tp += 1
                else:
                    fn += 1
            elif label == 0:
                negatives += 1
                if predicted_not_understood:
                    fp += 1
                else:
                    tn += 1

    fnr = (fn / positives) if positives else None
    tnr = (tn / negatives) if negatives else None

    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "positives": positives,
        "negatives": negatives,
        "fnr": fnr,
        "tnr": tnr,
    }


def selective_translation(
    understandability_test_method: str,
    model_name: str,
    dataset_type: str,
    eval_langs: Iterable[str] | str,
    polymath_split: Optional[str],
    task_eval_results_path: str | Path,
    thinking_intv_eval_results_path: str | Path,
    use_threshold_from_calibration_set: bool,
    ut_test_results_dir: str,
    custom_postfix: Optional[str] = None
) -> Dict[str, object]:
    task_eval_path = Path(task_eval_results_path).expanduser().resolve()
    thinking_eval_path = Path(thinking_intv_eval_results_path).expanduser().resolve()
    prediction_path = _resolve_predictions_path(
        understandability_test_method=understandability_test_method,
        model_name=model_name,
        dataset_type=dataset_type,
        polymath_split=polymath_split,
        task_eval_path=task_eval_path,
        thinking_eval_path=thinking_eval_path,
        use_threshold_from_calibration_set=use_threshold_from_calibration_set,
        ut_test_results_dir=ut_test_results_dir,
        custom_postfix=custom_postfix
    )

    predictions = _load_json(prediction_path)
    task_eval = _load_json(task_eval_path)
    thinking_eval = _load_json(thinking_eval_path)

    target_langs = _normalise_eval_langs(eval_langs)
    if not target_langs:
        languages: set[str] = set()
        if isinstance(predictions, dict):
            languages.update(predictions.keys())
        if isinstance(task_eval, dict):
            languages.update(task_eval.keys())
        if isinstance(thinking_eval, dict):
            languages.update(thinking_eval.keys())
        target_langs = sorted(languages)

    method_config = _get_method_score_config(understandability_test_method)

    samples_by_language: Dict[str, Dict[str, SampleEntry]] = {}
    threshold_scores: List[float] = []
    threshold_labels: List[int] = []

    for lang in target_langs:
        lang_predictions = predictions.get(lang) if isinstance(predictions, dict) else {}
        if not isinstance(lang_predictions, dict):
            if lang_predictions is not None:
                LOGGER.warning("Unexpected prediction payload for language '%s'; skipping predictions for this language.", lang)
            lang_predictions = {}

        task_lang_payload = task_eval.get(lang) if isinstance(task_eval, dict) else None
        if not isinstance(task_lang_payload, dict):
            task_lang_payload = {}

        thinking_lang_payload = thinking_eval.get(lang) if isinstance(thinking_eval, dict) else None
        if not isinstance(thinking_lang_payload, dict):
            thinking_lang_payload = {}

        all_sample_ids = set(lang_predictions.keys()) | set(task_lang_payload.keys()) | set(thinking_lang_payload.keys())
        if not all_sample_ids:
            samples_by_language[lang] = {}
            continue

        lang_entries: Dict[str, SampleEntry] = {}
        for sample_id in all_sample_ids:
            prediction_info = lang_predictions.get(sample_id) if lang_predictions else None
            if isinstance(prediction_info, dict):
                default_prediction = _coerce_bool(prediction_info.get("predicted_not_understood"))
                score = _coerce_float(prediction_info.get(method_config.score_field)) if method_config.score_field else None
                label = _coerce_int(prediction_info.get("label_not_understood"))
            else:
                default_prediction = False
                score = None
                label = None

            if score is not None and label is not None:
                threshold_scores.append(score)
                threshold_labels.append(label)

            lang_entries[sample_id] = SampleEntry(
                score=score,
                default_prediction=default_prediction,
                label=label,
            )

        samples_by_language[lang] = lang_entries

    scenario_results: Dict[str, Dict[str, object]] = {}

    for scenario_name, target_fnr in SCENARIOS:
        if target_fnr is None:
            threshold = None
        else:
            threshold = _compute_threshold_for_target_fnr(
                threshold_scores,
                threshold_labels,
                target_fnr,
                method_config,
            )
            if threshold is None:
                LOGGER.warning(
                    "Unable to compute threshold achieving FNR<=%.2f for method '%s'. Using default predictions.",
                    target_fnr,
                    understandability_test_method,
                )

        prediction_map, fallback_count = _build_scenario_prediction_map(
            samples_by_language,
            target_langs,
            method_config,
            threshold,
        )

        per_language_metrics, overall_metrics = _aggregate_metrics(
            prediction_map,
            task_eval,
            thinking_eval,
        )
        label_confusion = _compute_label_confusion(samples_by_language, prediction_map)

        scenario_results[scenario_name] = {
            "fnr_target": target_fnr,
            "threshold": threshold,
            "fallback_to_default_predictions": fallback_count if target_fnr is not None else 0,
            "per_language": per_language_metrics,
            "overall": overall_metrics,
            "label_metrics": label_confusion,
        }

    positives = int(sum(threshold_labels))
    negatives = int(len(threshold_labels) - positives)

    result: Dict[str, object] = {
        "predictions_path": str(prediction_path),
        "task_eval_results_path": str(task_eval_path),
        "thinking_intv_eval_results_path": str(thinking_eval_path),
        "method": understandability_test_method,
        "score_field": method_config.score_field,
        "supports_threshold_adjustment": method_config.supports_threshold_adjustment,
        "target_languages": target_langs,
        "threshold_dataset_stats": {
            "num_samples": len(threshold_scores),
            "positives": positives,
            "negatives": negatives,
        },
        "scenarios": scenario_results,
    }

    return result


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute selective translation metrics.")
    parser.add_argument("--understandability_test_method", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--dataset_type", required=True,
                        choices=[
                            "mmlu_prox_lite",
                            "polymath",
                        ])
    parser.add_argument("--eval_langs", default="en,de,es,ar,ja,ko,th,bn,sw,te", help="Comma-separated list of languages to include.")
    parser.add_argument("--polymath_split", default=None)
    parser.add_argument("--task_eval_results_path", required=True)
    parser.add_argument("--thinking_intv_eval_results_path", required=True)
    parser.add_argument(
        "--use_threshold_from_calibration_set",
        action="store_true",
        help="Use predictions generated with calibration-derived thresholds when available.",
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--ut_test_results_dir", type=str, default=None,
                        required=True, help="Path to the directory containing UT test results. Required for selective translation.")
    parser.add_argument("--save_dir", type=str, default="./selective_translation_outputs",
                        help="Base directory for saving outputs")
    parser.add_argument("--log_level", default="INFO")
    parser.add_argument("--custom_postfix", default=None, help="Custom postfix for output files.")
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    
    if args.use_threshold_from_calibration_set:
        # Sanity check
        assert args.understandability_test_method in CALIBRATION_THRESHOLD_METHODS, (
            f"Method '{args.understandability_test_method}' does not support calibration-based thresholds."
        )

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    results = selective_translation(
        understandability_test_method=args.understandability_test_method,
        model_name=args.model_name,
        dataset_type=args.dataset_type,
        eval_langs=args.eval_langs,
        polymath_split=args.polymath_split,
        task_eval_results_path=args.task_eval_results_path,
        thinking_intv_eval_results_path=args.thinking_intv_eval_results_path,
        use_threshold_from_calibration_set=args.use_threshold_from_calibration_set,
        ut_test_results_dir=args.ut_test_results_dir,
        custom_postfix=args.custom_postfix
    )

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

    suffix = "_from_calibration_thr" if args.use_threshold_from_calibration_set else ""
    if args.custom_postfix:
        suffix += args.custom_postfix
    results_save_path = os.path.join(
        output_save_dir,
        f"selective_translation_{args.understandability_test_method}_{args.seed}{suffix}.json"
    )
    with open(results_save_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=4)
    LOGGER.info("Results saved to %s", results_save_path)

    target_languages = results.get("target_languages", [])

    rows = []

    for scenario_key, scenario_result in results.get("scenarios", {}).items():
        overall_metrics = scenario_result["overall"]
        overall_accuracy = overall_metrics.get("accuracy")
        overall_selective_rate = overall_metrics.get("selective_rate")

        lang_accuracy_list = []
        lang_selective_rate_list = []
        for target_lang in target_languages:
            per_language_metrics = scenario_result["per_language"].get(target_lang, {})
            lang_accuracy = per_language_metrics.get("accuracy")
            lang_selective_rate = per_language_metrics.get("selective_rate")

            lang_accuracy_list.append(lang_accuracy)
            lang_selective_rate_list.append(lang_selective_rate)

        final_languages_list = target_languages + ["overall"]
        final_result_accuracy = lang_accuracy_list + [overall_accuracy]
        final_result_selective_rate = lang_selective_rate_list + [overall_selective_rate]

        # accuracy row
        rows.append({
            "scenario": scenario_key,
            "metric": "accuracy",
            **dict(zip(final_languages_list, final_result_accuracy))
        })
        # selective_rate row
        rows.append({
            "scenario": scenario_key,
            "metric": "selective_rate",
            **dict(zip(final_languages_list, final_result_selective_rate))
        })

    df = pd.DataFrame(rows)

    csv_save_path = os.path.join(
        output_save_dir,
        f"selective_translation_{args.understandability_test_method}_{args.seed}{suffix}.csv"
    )

    df.to_csv(csv_save_path, index=False)
    LOGGER.info("CSV summary saved to %s", csv_save_path)
    

if __name__ == "__main__":
    main()