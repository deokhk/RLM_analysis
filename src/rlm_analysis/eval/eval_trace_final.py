#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Report agreement between answers extracted from:
  - prediction (first \boxed{...})
  - reasoning_trace (first \boxed{...})

Per model & per language, print:
  - N: total examples
  - N_valid: examples where BOTH sides have an extracted boxed answer
  - agree: count where math_equal(pred_box, trace_box) holds
  - agree_rate_valid: agree / N_valid
  - agree_rate_overall: agree / N
  - pred_only, trace_only, none: extraction diagnostics

JSON shape expected:
{
  "en": {
    "0": {
      "correct": true,
      "prediction": "...",
      "reasoning_trace": "...",
      "answer": 18,
      "pred_answer": "18"
    },
    ...
  },
  "ko": { ... },
  ...
}

Default path template:
  /home/deokhk/research/LRM_analysis/task_eval_results/{model}/mgsm/{model}_task_eval_results_mgsm_42.json
"""

import os
import json
import argparse
import logging
from eval.scripts import math_equal
from eval.evaluator import extract_last_boxed
import re
from typing import Optional, Union




# 숫자 본체만 캡처하고, 뒤에 단위가 와도 무시(매칭엔 영향 없음)
_NUM_RE = re.compile(r"""
    (?<!\w)                  # 숫자가 단어의 일부가 아닌 위치(붙은 영문자와 구분)
    [\$€£¥₩]?                # 선택적 통화기호(접두)
    \s*
    (?P<num>                 # <-- 숫자 본체만 캡처
        [+\-−]?
        (?:
            (?:\d{1,3}(?:,\d{3})+|\d+)   # 정수(천단위 콤마 허용)
        )
        (?:\.\d+)?                        # 선택적 소수부
        (?:[eE][+\-]?\d+)?                # 선택적 지수부
    )
    # 여기서 패턴은 종료되므로, 뒤에 어떤 단위/문자(%, 원, kg, m/s^2, ドル 등)가 와도 매칭에는 영향 없음
""", re.VERBOSE)





def _last_number_in(text: str) -> Optional[Union[int, float]]:
    """텍스트에서 마지막 숫자(뒤에 단위 무시)를 반환."""
    if not text:
        return None
    matches = list(_NUM_RE.finditer(text))
    if not matches:
        return None
    raw = matches[-1].group("num").replace(",", "")
    try:
        return int(raw) if re.fullmatch(r"[+\-−]?\d+", raw) else float(raw)
    except ValueError:
        return None

def process_file(model: str, path: str):
    """Compute per-language agreement stats for a single model file."""
    if not os.path.exists(path):
        logging.warning(f"[{model}] File not found: {path}")
        return {}

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    out = {}
    for lang, items in data.items():
        if lang == "zh" or lang == "ru":
            continue  # Skip unsupported languages
        n = 0
        n_valid = 0  # both extracted
        agree = 0
        pred_only = 0
        trace_only = 0
        none = 0

        for _id, item in items.items():
            n += 1
            pred_box = extract_last_boxed(item.get("prediction", ""))
            trace_box = extract_last_boxed(item.get("reasoning_trace", ""))

            if pred_box is not None and trace_box is not None:
                n_valid += 1
                try:
                    if math_equal(str(pred_box), str(trace_box)):
                        agree += 1
                except Exception:
                    pass
            elif pred_box is not None and trace_box is None:
                pred_only += 1
            elif pred_box is None and trace_box is not None:
                trace_only += 1
            else:
                none += 1

        out[lang] = {
            "N": n,
            "N_valid": n_valid,
            "agree": agree,
            "agree_rate_valid": (agree / n_valid) if n_valid > 0 else 0.0,
            "agree_rate_overall": (agree / n) if n > 0 else 0.0,
            "pred_only": pred_only,
            "trace_only": trace_only,
            "none": none,
        }
    return out

def micro_avg(stats_by_lang):
    """Micro-average across languages."""
    N = sum(s["N"] for s in stats_by_lang.values())
    N_valid = sum(s["N_valid"] for s in stats_by_lang.values())
    agree = sum(s["agree"] for s in stats_by_lang.values())
    pred_only = sum(s["pred_only"] for s in stats_by_lang.values())
    trace_only = sum(s["trace_only"] for s in stats_by_lang.values())
    none = sum(s["none"] for s in stats_by_lang.values())

    return {
        "N": N,
        "N_valid": N_valid,
        "agree": agree,
        "agree_rate_valid": (agree / N_valid) if N_valid > 0 else 0.0,
        "agree_rate_overall": (agree / N) if N > 0 else 0.0,
        "pred_only": pred_only,
        "trace_only": trace_only,
        "none": none,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--models",
        type=str,
        default="Qwen3-1.7B,Qwen3-4B,Qwen3-8B,Qwen3-14B,QwQ-32B-AWQ,DeepSeek-R1-Distill-Qwen-7B,DeepSeek-R1-Distill-Qwen-14B",
        help="Comma-separated model names."
    )
    ap.add_argument(
        "--eval_json_template",
        type=str,
        default="/home/deokhk/research/LRM_analysis/task_eval_results/{model}/mgsm/{model}_task_eval_results_mgsm_42.json",
        help='Template path containing "{model}".'
    )
    ap.add_argument(
        "--save_csv",
        type=str,
        default="",
        help="Optional: path to save per-language rows as CSV."
    )
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

    models = [m.strip() for m in args.models.split(",") if m.strip()]

    print("=" * 108)
    print(f"{'MODEL':28s} {'LANG':8s} {'N':>6s} {'N_valid':>8s} {'agree':>7s} {'agree@valid':>12s} {'agree@overall':>14s}  {'pred_only':>9s} {'trace_only':>10s} {'none':>6s}")
    print("-" * 108)

    rows = []
    for model in models:
        path = args.eval_json_template.format(model=model)
        stats = process_file(model, path)
        if not stats:
            continue

        # per-language
        for lang in sorted(stats.keys()):
            s = stats[lang]
            print(f"{model:28s} {lang:8s} {s['N']:6d} {s['N_valid']:8d} {s['agree']:7d} "
                  f"{s['agree_rate_valid']*100:11.2f}% {s['agree_rate_overall']*100:13.2f}%  "
                  f"{s['pred_only']:9d} {s['trace_only']:10d} {s['none']:6d}")
            rows.append({"model": model, "language": lang, **s})

        # micro-average
        m = micro_avg(stats)
        print(f"{model:28s} {'[ALL]':8s} {m['N']:6d} {m['N_valid']:8d} {m['agree']:7d} "
              f"{m['agree_rate_valid']*100:11.2f}% {m['agree_rate_overall']*100:13.2f}%  "
              f"{m['pred_only']:9d} {m['trace_only']:10d} {m['none']:6d}")
        print("-" * 108)
        rows.append({"model": model, "language": "[ALL]", **m})

    # optional CSV
    if args.save_csv:
        import csv
        os.makedirs(os.path.dirname(args.save_csv), exist_ok=True) if os.path.dirname(args.save_csv) else None
        with open(args.save_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["model","language","N","N_valid","agree","agree_rate_valid","agree_rate_overall","pred_only","trace_only","none"]
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved CSV to: {args.save_csv}")