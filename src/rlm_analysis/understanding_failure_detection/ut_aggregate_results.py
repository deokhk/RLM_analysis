#!/usr/bin/env python3
"""Aggregate UT metrics across seeds and languages."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import statistics
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence
from rlm_analysis.understanding_failure_detection.ut_test import SUPPORTED_METHODS

LOGGER = logging.getLogger("ut_aggregate_results")
logging.basicConfig(format="%(levelname)s - %(message)s", level=logging.INFO)
LOGGER.setLevel(logging.INFO)

METRIC_KEYS: Sequence[str] = (
    "balanced_accuracy",
    "f1",
    "pr_auc",
    "tnr_at_fnr_0.10",
    "tnr_at_fnr_0.05",
)

CALIBRATION_THRESHOLD_METHODS = {"avg_confidence", "min_confidence", "prompt_ln_nll"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate UT evaluation metrics across seeds and languages. "
            "The script expects files named like 'ut_metrics_<method>_<seed>.json' "
            "inside the results directory."
        )
    )
    parser.add_argument(
        "--results_dir",
        type=Path,
        help="Directory containing 'ut_metrics_<method>_<seed>.json' files",
    )
    parser.add_argument(
        "--metrics_using_threshold_from_calibration_set",
        action="store_true",
        help=(
            "Read metrics produced with calibration-derived thresholds for "
            "methods that support calibration (avg_confidence, min_confidence, prompt_ln_nll)."
        ),
    )
    parser.add_argument(
        "--custom_postfix",
        type=str,
        default="",
        help="Optional custom postfix for picking metric files."
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        help="Optional list of seeds to aggregate. Defaults to all seeds found per method.",
    )

    args = parser.parse_args()
    results_dir = args.results_dir
    if results_dir is None:
        parser.error("results_dir must be specified via --results_dir")
    return args




def load_supported_methods(ut_test_path: Path) -> List[str]:
    """Read method names from SUPPORTED_METHODS in ut_test.py without importing heavy deps."""
    if not ut_test_path.is_file():
        raise FileNotFoundError(f"Cannot locate ut_test.py at {ut_test_path}")

    content = ut_test_path.read_text(encoding="utf-8")
    match = re.search(r"SUPPORTED_METHODS\s*:\s*Dict\s*\[.*?\]\s*=\s*\{(.*?)\}\s*\n", content, flags=re.DOTALL)
    if not match:
        raise RuntimeError("Failed to locate SUPPORTED_METHODS definition in ut_test.py")

    body = match.group(1)
    method_names: List[str] = []
    for line in body.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key_match = re.match(r"['\"]([^'\"]+)['\"]\s*:", line)
        if key_match:
            method_names.append(key_match.group(1))
    if not method_names:
        raise RuntimeError("No method names found in SUPPORTED_METHODS")
    return method_names


def coerce_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if result != result:  # NaN guard
        return None
    return result


def pick_metric_files(
    results_dir: Path,
    method: str,
    seeds_filter: Optional[Iterable[int]],
    use_calibration_thresholds: bool,
    custom_postfix: str = "",
) -> Dict[int, Path]:
    suffix_pattern = "_from_calibration_thr" if (use_calibration_thresholds and method in CALIBRATION_THRESHOLD_METHODS) else ""
    if custom_postfix != "":
        suffix_pattern += f"{custom_postfix}"
    regex = re.compile(
        rf"^ut_metrics_{re.escape(method)}_(\d+){re.escape(suffix_pattern)}\.json$"
    )
    available: Dict[int, Path] = {}
    for path in results_dir.glob(f"ut_metrics_{method}_*.json"):
        
        match = regex.match(path.name)
        if not match:
            continue
        seed = int(match.group(1))
        if seeds_filter and seed not in seeds_filter:
            continue
        available[seed] = path
    return dict(sorted(available.items()))


def compute_mean_std(values: List[float]) -> tuple[Optional[float], Optional[float]]:
    if not values:
        return None, None
    mean = sum(values) / len(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, std


def aggregate_method(
    method: str,
    files: Dict[int, Path],
) -> tuple[
    Optional[Dict[str, str]],
    Optional[Dict[str, str]],
    Dict[str, Dict[str, str]],
    Dict[str, Dict[str, str]],
]:
    overall_metric_values: Dict[str, List[float]] = {metric: [] for metric in METRIC_KEYS}
    per_language_metric_values: Dict[str, Dict[str, List[float]]] = {}

    for seed, path in files.items():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            LOGGER.error("Failed to parse %s: %s", path, exc)
            continue
        metrics = payload.get("metrics", {})
        overall = metrics.get("overall", {})
        for key in METRIC_KEYS:
            value = coerce_float(overall.get(key))
            if value is not None:
                overall_metric_values[key].append(value * 100.0)

        per_language = metrics.get("per_language", {})
        if not isinstance(per_language, dict):
            continue
        for language, lang_metrics in per_language.items():
            if not isinstance(lang_metrics, dict):
                continue
            lang_store = per_language_metric_values.setdefault(
                language, {metric: [] for metric in METRIC_KEYS}
            )
            for key in METRIC_KEYS:
                value = coerce_float(lang_metrics.get(key))
                if value is not None:
                    lang_store[key].append(value * 100.0)

    if not any(overall_metric_values.values()):
        LOGGER.warning("No usable metrics found for method '%s'", method)
        overall_mean_summary = None
        overall_std_summary = None
    else:
        overall_mean_summary = {}
        overall_std_summary = {}
        for key, values in overall_metric_values.items():
            mean, std = compute_mean_std(values)
            overall_mean_summary[key] = f"{mean:.6f}" if mean is not None else ""
            overall_std_summary[key] = f"{std:.6f}" if std is not None else ""

    per_language_mean_summary: Dict[str, Dict[str, str]] = {}
    per_language_std_summary: Dict[str, Dict[str, str]] = {}
    for language, metric_lists in per_language_metric_values.items():
        mean_row: Dict[str, str] = {}
        std_row: Dict[str, str] = {}
        for metric, values in metric_lists.items():
            mean, std = compute_mean_std(values)
            mean_row[metric] = f"{mean:.6f}" if mean is not None else ""
            std_row[metric] = f"{std:.6f}" if std is not None else ""
        per_language_mean_summary[language] = mean_row
        per_language_std_summary[language] = std_row

    return (
        overall_mean_summary,
        overall_std_summary,
        per_language_mean_summary,
        per_language_std_summary,
    )


def write_overall_tables(
    output_dir: Path,
    mean_rows: List[Dict[str, str]],
    std_rows: List[Dict[str, str]],
) -> tuple[Optional[Path], Optional[Path]]:
    fieldnames = ["method", *METRIC_KEYS]
    mean_path = output_dir / "aggregated_overall_metrics_mean.csv"
    std_path = output_dir / "aggregated_overall_metrics_std.csv"

    if mean_rows:
        with mean_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in mean_rows:
                writer.writerow(row)
        LOGGER.info("Wrote overall mean metrics to %s", mean_path)
    else:
        LOGGER.warning("No rows to write for overall mean metrics; skipping %s", mean_path)
        mean_path = None

    if std_rows:
        with std_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in std_rows:
                writer.writerow(row)
        LOGGER.info("Wrote overall std metrics to %s", std_path)
    else:
        LOGGER.warning("No rows to write for overall std metrics; skipping %s", std_path)
        std_path = None

    return mean_path, std_path


def write_per_language_tables(
    output_dir: Path,
    method: str,
    per_language_mean_summary: Dict[str, Dict[str, str]],
    per_language_std_summary: Dict[str, Dict[str, str]],
) -> tuple[Optional[Path], Optional[Path]]:
    fieldnames = ["language", *METRIC_KEYS]
    mean_path = output_dir / f"per_language_metrics_{method}_mean.csv"
    std_path = output_dir / f"per_language_metrics_{method}_std.csv"

    if per_language_mean_summary:
        with mean_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for language in sorted(per_language_mean_summary):
                row = {"language": language}
                row.update(per_language_mean_summary[language])
                writer.writerow(row)
        LOGGER.info("Wrote per-language mean metrics for '%s' to %s", method, mean_path)
    else:
        LOGGER.warning("No per-language mean metrics to write for method '%s'", method)
        mean_path = None

    if per_language_std_summary:
        with std_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for language in sorted(per_language_std_summary):
                row = {"language": language}
                row.update(per_language_std_summary[language])
                writer.writerow(row)
        LOGGER.info("Wrote per-language std metrics for '%s' to %s", method, std_path)
    else:
        LOGGER.warning("No per-language std metrics to write for method '%s'", method)
        std_path = None

    return mean_path, std_path


def main() -> None:
    args = parse_args()

    results_dir: Path = args.results_dir
    if not results_dir.is_dir():
        raise NotADirectoryError(f"Results directory not found: {results_dir}")

    save_dir = f"aggregated_results_{args.custom_postfix}" if args.custom_postfix else "aggregated_results"
    output_dir = results_dir / save_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    methods = list(SUPPORTED_METHODS.keys())
    LOGGER.info("Discovered methods: %s", ", ".join(methods))

    seeds_filter = set(args.seeds) if args.seeds else None

    overall_mean_rows: List[Dict[str, str]] = []
    overall_std_rows: List[Dict[str, str]] = []

    for method in methods:
        files = pick_metric_files(
            results_dir,
            method,
            seeds_filter,
            args.metrics_using_threshold_from_calibration_set,
            args.custom_postfix
        )
        if not files:
            LOGGER.warning("No metric files found for method '%s'", method)
            continue
        if seeds_filter:
            missing = sorted(set(seeds_filter) - set(files))
            if missing:
                LOGGER.warning(
                    "Missing metric files for method '%s' and seeds: %s",
                    method,
                    ", ".join(str(seed) for seed in missing),
                )

        (
            overall_mean_summary,
            overall_std_summary,
            per_language_mean_summary,
            per_language_std_summary,
        ) = aggregate_method(method, files)

        if overall_mean_summary is not None and overall_std_summary is not None:
            mean_row = {"method": method}
            std_row = {"method": method}
            for metric in METRIC_KEYS:
                mean_row[metric] = overall_mean_summary.get(metric, "")
                std_row[metric] = overall_std_summary.get(metric, "")
            overall_mean_rows.append(mean_row)
            overall_std_rows.append(std_row)

        write_per_language_tables(
            output_dir,
            method,
            per_language_mean_summary,
            per_language_std_summary,
        )

    write_overall_tables(output_dir, overall_mean_rows, overall_std_rows)


if __name__ == "__main__":
    main()
