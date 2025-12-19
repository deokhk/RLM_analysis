#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional


# -----------------------------
# Helpers
# -----------------------------

DESIRED_LANG_ORDER = ["en", "de", "es", "ar", "ja", "ko", "th", "bn", "sw", "te", "Avg"]

def reorder_lang_dict(d: Dict[str, float]) -> Dict[str, float]:
    """
    If a key is in DESIRED_LANG_ORDER, follow that order.
    Otherwise, append remaining keys in alphabetical order.
    """
    keys = set(d.keys())
    ordered_keys: List[str] = []

    # 1) desired order (only those present)
    for k in DESIRED_LANG_ORDER:
        if k in keys:
            ordered_keys.append(k)
            keys.remove(k)

    # 2) leftover keys in alphabetical order
    ordered_keys.extend(sorted(keys))

    # build dict with insertion order
    return {k: d[k] for k in ordered_keys}

def safe_float(x: str) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def mean(xs: List[float]) -> float:
    if not xs:
        return float("nan")
    return sum(xs) / len(xs)


def std_population(xs: List[float]) -> float:
    """
    Population std (ddof=0). If you want sample std, change denom to (n-1).
    """
    if not xs:
        return float("nan")
    m = mean(xs)
    return math.sqrt(sum((v - m) ** 2 for v in xs) / len(xs))


def pretty_dataset_name(dataset_type: str, polymath_split: Optional[str]) -> str:
    if dataset_type == "polymath":
        if polymath_split is None:
            raise ValueError("polymath_split is required for dataset_type=polymath")
        return f"Polymath-{polymath_split.capitalize()}"
    if dataset_type == "mmlu_prox_lite":
        return "MMLU-ProX-Lite"
    # fallback
    if polymath_split:
        return f"{dataset_type}-{polymath_split}"
    return dataset_type


@dataclass(frozen=True)
class RunKey:
    model: str
    dataset_type: str
    split: Optional[str]  # polymath: low/medium/high, mmlu_prox_lite: None


# -----------------------------
# CSV parsing
# -----------------------------

def read_eval_summary_csv(path: Path) -> Dict[str, Dict[str, float]]:
    """
    Reads eval_summary_*.csv
    Returns:
      metric_to_lang_to_score: metric -> {lang: score_float}
    Example CSV:
      metric,en,de,...
      eval_score,0.77,0.75,...
      eval_score_from_trace,0.77,0.75,...
    """
    metric_to_lang: Dict[str, Dict[str, float]] = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        # DictReader will create keys: metric, en, de, ...
        for row in reader:
            metric = row.get("metric")
            if not metric:
                continue
            lang_to_val: Dict[str, float] = {}
            for k, v in row.items():
                if k == "metric":
                    continue
                fv = safe_float(v) if v is not None else None
                if fv is None:
                    continue
                lang_to_val[k] = fv
            metric_to_lang[metric] = lang_to_val
    return metric_to_lang


# -----------------------------
# File discovery
# -----------------------------

# We accept:
# task_eval_results:
#   eval_summary_<MODEL>_<DATASET>_<SPLIT>_<SEED>.csv
#   eval_summary_<MODEL>_<DATASET>_<SEED>.csv  (for mmlu_prox_lite)
#
# think_intv:
#   eval_summary_<MODEL>_<DATASET>_<SPLIT>_<SEED>_thinking_intv_en.csv
#   eval_summary_<MODEL>_<DATASET>_<SEED>_thinking_intv_en.csv

RE_BASE = re.compile(
    r"^eval_summary_(?P<model>.+?)_(?P<dataset>polymath|mmlu_prox_lite)"
    r"(?:_(?P<split>low|medium|high))?"
    r"_(?P<seed>\d+)\.csv$"
)

RE_U = re.compile(
    r"^eval_summary_(?P<model>.+?)_(?P<dataset>polymath|mmlu_prox_lite)"
    r"(?:_(?P<split>low|medium|high))?"
    r"_(?P<seed>\d+)_thinking_intv_en\.csv$"
)


def collect_eval_files(root_outputs: Path) -> Tuple[Dict[RunKey, Dict[int, Path]], Dict[RunKey, Dict[int, Path]]]:
    """
    Returns:
      base_files[RunKey][seed] = path  (from task_eval_results)
      u_files[RunKey][seed] = path     (from task_eval_results_think_intv)
    """
    base_dir = root_outputs / "task_eval_results"
    u_dir = root_outputs / "task_eval_results_think_intv"

    if not base_dir.exists():
        raise FileNotFoundError(f"Not found: {base_dir}")
    if not u_dir.exists():
        raise FileNotFoundError(f"Not found: {u_dir}")

    base_files: Dict[RunKey, Dict[int, Path]] = defaultdict(dict)
    u_files: Dict[RunKey, Dict[int, Path]] = defaultdict(dict)

    # Base
    for p in base_dir.rglob("eval_summary_*.csv"):
        m = RE_BASE.match(p.name)
        if not m:
            # skip eval_summary_both_* etc.
            continue
        model = m.group("model")
        dataset = m.group("dataset")
        split = m.group("split")
        seed = int(m.group("seed"))
        rk = RunKey(model=model, dataset_type=dataset, split=split)
        base_files[rk][seed] = p

    # U (thinking intervention)
    for p in u_dir.rglob("eval_summary_*.csv"):
        m = RE_U.match(p.name)
        if not m:
            continue
        model = m.group("model")
        dataset = m.group("dataset")
        split = m.group("split")
        seed = int(m.group("seed"))
        rk = RunKey(model=model, dataset_type=dataset, split=split)
        u_files[rk][seed] = p

    return base_files, u_files


# -----------------------------
# Aggregation logic
# -----------------------------

def aggregate_one_setting(
    seed_to_csv_path: Dict[int, Path],
    metric_name: str,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    For a given setting (e.g., Base) and a given metric_name (eval_score or eval_score_from_trace),
    compute mean/std across seeds per language.
    Returns:
      (lang_to_mean_percent, lang_to_std_percent) including "Avg"
    """
    # lang -> list[score]
    lang_to_scores: Dict[str, List[float]] = defaultdict(list)

    # read each seed file
    for seed, path in sorted(seed_to_csv_path.items()):
        metric_to_lang = read_eval_summary_csv(path)
        if metric_name not in metric_to_lang:
            raise KeyError(f"Metric '{metric_name}' not found in {path}")
        lang_to_val = metric_to_lang[metric_name]
        for lang, val in lang_to_val.items():
            lang_to_scores[lang].append(val * 100.0)  # convert to %
    # compute mean/std
    lang_to_mean: Dict[str, float] = {}
    lang_to_std: Dict[str, float] = {}
    langs = sorted(lang_to_scores.keys())

    for lang in langs:
        xs = lang_to_scores[lang]
        lang_to_mean[lang] = mean(xs)
        lang_to_std[lang] = std_population(xs)

    # Avg over languages (mean of per-language means; std over per-seed Avg is also possible,
    # but user asked: "based on eval_score*100 and std too". We'll compute std across seeds of per-seed Avg.)
    # So we compute seed-wise Avg first, then mean/std across seeds.
    seed_avgs: List[float] = []
    # seeds present may differ per language; use intersection approach per seed:
    # Build seed -> list of lang scores (only langs available in that seed)
    seed_to_langvals: Dict[int, List[float]] = defaultdict(list)
    for lang in langs:
        # we appended scores in seed iteration order; but seed list might be missing -> handle via explicit read again
        pass

    # robust seed-wise Avg: recompute by rereading per seed (small overhead, safer)
    for seed, path in sorted(seed_to_csv_path.items()):
        metric_to_lang = read_eval_summary_csv(path)
        lang_to_val = metric_to_lang[metric_name]
        vals = []
        for lang in langs:
            if lang in lang_to_val:
                vals.append(lang_to_val[lang] * 100.0)
        if vals:
            seed_avgs.append(mean(vals))

    lang_to_mean["Avg"] = mean(seed_avgs) if seed_avgs else float("nan")
    lang_to_std["Avg"] = std_population(seed_avgs) if seed_avgs else float("nan")

    lang_to_mean = reorder_lang_dict(lang_to_mean)
    lang_to_std = reorder_lang_dict(lang_to_std)
    return lang_to_mean, lang_to_std


def build_outputs_json(
    base_files: Dict[RunKey, Dict[int, Path]],
    u_files: Dict[RunKey, Dict[int, Path]],
) -> Tuple[Dict, Dict]:
    """
    Returns (mean_json, std_json) with structure:
      { model: { dataset_pretty: { "Base": {...}, "w/ T": {...}, "w/ U": {...}, "w/ U+T": {...} } } }
    """
    mean_out: Dict[str, Dict] = defaultdict(dict)
    std_out: Dict[str, Dict] = defaultdict(dict)

    # consider union of keys that exist in base (because Base/T require it)
    all_keys = set(base_files.keys()) | set(u_files.keys())

    for rk in sorted(all_keys, key=lambda x: (x.model, x.dataset_type, x.split or "")):
        model = rk.model
        dataset_pretty = pretty_dataset_name(rk.dataset_type, rk.split)

        # Base and w/ T require base_files
        if rk not in base_files or not base_files[rk]:
            # skip if no base results
            continue

        # U and U+T require u_files (optional; but usually exists)
        has_u = rk in u_files and bool(u_files[rk])

        # compute 4 configs
        base_mean, base_std = aggregate_one_setting(base_files[rk], "eval_score")
        t_mean, t_std = aggregate_one_setting(base_files[rk], "eval_score_from_trace")

        if has_u:
            u_mean, u_std = aggregate_one_setting(u_files[rk], "eval_score")
            ut_mean, ut_std = aggregate_one_setting(u_files[rk], "eval_score_from_trace")
        else:
            u_mean, u_std = {}, {}
            ut_mean, ut_std = {}, {}

        mean_out[model].setdefault(dataset_pretty, {})
        std_out[model].setdefault(dataset_pretty, {})

        mean_out[model][dataset_pretty]["Base"] = base_mean
        std_out[model][dataset_pretty]["Base"] = base_std

        mean_out[model][dataset_pretty]["w/ T"] = t_mean
        std_out[model][dataset_pretty]["w/ T"] = t_std

        if has_u:
            mean_out[model][dataset_pretty]["w/ U"] = u_mean
            std_out[model][dataset_pretty]["w/ U"] = u_std

            mean_out[model][dataset_pretty]["w/ U+T"] = ut_mean
            std_out[model][dataset_pretty]["w/ U+T"] = ut_std

    return mean_out, std_out


# -----------------------------
# CLI
# -----------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Aggregate multilingual reasoning results across seeds into mean/std JSON files."
    )
    parser.add_argument(
        "--outputs_dir",
        type=str,
        required=True,
        help="Path to outputs directory (contains task_eval_results/ and task_eval_results_think_intv/).",
    )
    parser.add_argument(
        "--save_mean_name",
        type=str,
        default="multilingual_reasoning_mean.json",
        help="Output filename for mean JSON (saved under outputs_dir).",
    )
    parser.add_argument(
        "--save_std_name",
        type=str,
        default="multilingual_reasoning_std.json",
        help="Output filename for std JSON (saved under outputs_dir).",
    )
    args = parser.parse_args()

    outputs_dir = Path(args.outputs_dir).expanduser().resolve()
    base_files, u_files = collect_eval_files(outputs_dir)
    mean_json, std_json = build_outputs_json(base_files, u_files)

    mean_path = outputs_dir / args.save_mean_name
    std_path = outputs_dir / args.save_std_name

    with mean_path.open("w", encoding="utf-8") as f:
        json.dump(mean_json, f, ensure_ascii=False, indent=2)

    with std_path.open("w", encoding="utf-8") as f:
        json.dump(std_json, f, ensure_ascii=False, indent=2)

    print(f"[OK] Saved mean: {mean_path}")
    print(f"[OK] Saved std : {std_path}")


if __name__ == "__main__":
    main()
