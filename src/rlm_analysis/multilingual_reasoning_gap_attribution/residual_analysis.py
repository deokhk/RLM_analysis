#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute U/G/R shares (sum to 1) per (model, dataset, language) from mean/std JSONs,
with an option to restrict analysis to languages that differ significantly from the best-performing language (the "ceiling" language, usually English)
under Base via Welch's t-test (p < 0.05 by default).

Inputs (mean & std JSON structure):
{
  "<Model>": {
    "<Dataset>": {
      "Base":   { "en": ..., "de": ..., ..., "Avg": ... },
      "w/ U":   { ... },
      "w/ T":   { ... },
      "w/ U+T": { ... }
    },
    ...
  },
  ...
}

Outputs (shares JSON structure; mirrors input hierarchy):
{
  "<Model>": {
    "<Dataset>": {
      "per_lang": {
        "<lang>": {
          "shares": {"U": float, "G": float, "R": float},     # sum to 1 if H>0 else all 0
          "phi":    {"U": float, "G": float, "R": float},     # unnormalized contributions
          "H": float,
          "scores": {"Base": float, "U": float, "T": float, "UT": float,
                     "BaseMax": float, "Ceil": float}
        }, ...
      },
      "aggregate": {
        "Avg_unweighted": {"U": float, "G": float, "R": float},
        "Avg_headroom_weighted": {"U": float, "G": float, "R": float},
        "Total_headroom": float,
        "language_count": int,
        "significance_filter": {"enabled": bool, "alpha": float}
      }
    },
    ...
  }
}

Notes:
- By default, shares are computed from means only (no significance weighting).
- If std JSON + sample size n are provided with --weighting=hard|soft, Welch t-tests
  are used to (optionally) weight φ_U, φ_T. Sum-to-1 is preserved by defining φ_R = H - φ_U - φ_T.
- --compute_significant_language_only filters languages to those with p<alpha
  when comparing Base(lang) vs Base(ceiling_language) via Welch's t-test.
"""

import argparse
import json
import math
import os
from typing import Dict, Any, Optional, List, Tuple

# ---------- Optional Welch t-test helpers (scipy if available; else normal approx fallback) ----------
def try_scipy_cdf():
    try:
        from scipy.stats import t as student_t  # type: ignore
        return student_t
    except Exception:
        return None

_STUDENT_T = try_scipy_cdf()

def welch_t_pvalue(mean1, std1, n1, mean2, std2, n2) -> float:
    """Two-sided Welch's t-test p-value for difference in means.
    Falls back to normal approximation if SciPy is unavailable."""
    v1 = (std1 ** 2) / max(n1, 1)
    v2 = (std2 ** 2) / max(n2, 1)
    denom = math.sqrt(v1 + v2) if (v1 + v2) > 0 else float('inf')
    t = (mean1 - mean2) / denom if denom > 0 else 0.0
    # Welch–Satterthwaite df
    num = (v1 + v2) ** 2
    den = (v1 ** 2) / max(n1 - 1, 1) + (v2 ** 2) / max(n2 - 1, 1)
    df = num / den if den > 0 else 1e9  # very large df ~ normal

    if _STUDENT_T is not None:
        cdf = _STUDENT_T.cdf(abs(t), df)
        p = 2 * (1 - cdf)
        return float(max(min(p, 1.0), 0.0))
    else:
        # Normal approximation using error function
        z = abs(t)
        phi = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
        p = 2.0 * (1.0 - phi)
        return float(max(min(p, 1.0), 0.0))

# ---------- Core computation ----------

def get_base_ceiling_language(base_mean: Dict[str, Any], langs_all: List[str]) -> Tuple[str, float]:
    """
    Return (ceiling_language, ceiling_score) under Base, excluding 'Avg' by construction.
    If ties occur, picks the first one in sorted order for determinism.
    """
    candidates: List[Tuple[str, float]] = []
    for l in langs_all:
        v = base_mean.get(l, None)
        if isinstance(v, (int, float)):
            candidates.append((l, float(v)))
    if not candidates:
        raise ValueError("No valid Base scores to determine ceiling language.")
    # Sort by (-score, lang) for deterministic tie-breaking
    candidates.sort(key=lambda x: (-x[1], x[0]))
    return candidates[0][0], candidates[0][1]


def filter_langs_by_significance_to_ceiling(
    model: str,
    dset: str,
    langs_all: List[str],
    means: Dict[str, Any],
    stds: Dict[str, Any],
    n: int,
    alpha: float,
    ceiling_lang: str,
) -> List[str]:
    """
    Keep only languages with Base(lang) differing from Base(ceiling_lang) at p<alpha.
    Excludes 'Avg' by construction and uses the Base ceiling language as the reference.
    """
    base_mean = means[model][dset]["Base"]
    base_std  = stds[model][dset]["Base"]
    if ceiling_lang not in base_mean or ceiling_lang not in base_std:
        raise ValueError(f"[{model} / {dset}] ceiling language '{ceiling_lang}' missing mean/std in Base.")

    kept = []
    mean_ref = base_mean[ceiling_lang]
    std_ref  = base_std[ceiling_lang]

    for l in langs_all:
        if l == ceiling_lang:
            # Keep the reference in the returned list (useful for later ceiling computations),
            # though we typically skip analyzing it downstream.
            kept.append(l)
            continue
        if l not in base_std:
            # If std missing for a language, skip it under the strict filter
            continue
        p = welch_t_pvalue(base_mean[l], base_std[l], n, mean_ref, std_ref, n)
        if p < alpha:
            kept.append(l)
    return kept

def compute_shapley_and_shares(
    means: Dict[str, Any],
    stds: Optional[Dict[str, Any]],
    n: Optional[int],
    alpha: float,
    weighting: str,                 # "none" | "hard" | "soft"
    compute_sig_only: bool          # filter languages by Base(lang) vs Base(ceiling) significance
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    for model, dsets in means.items():
        out[model] = {}
        for dset, methods in dsets.items():
            try:
                base = methods["Base"]
                u = methods["w/ U"]
                t = methods["w/ T"]
                ut = methods["w/ U+T"]
            except KeyError:
                # Skip datasets with incomplete methods
                continue

            # Candidate language list (exclude "Avg")
            langs_all = [k for k in base.keys() if k.lower() != "avg"]

            # Determine ceiling language under Base (from all languages, not just non-English)
            try:
                ceil_lang, base_max = get_base_ceiling_language(base, langs_all)
            except ValueError:
                continue

            # Optional significance filter vs English on Base
            if compute_sig_only:
                if stds is None or n is None:
                    raise ValueError("--compute_significant_language_only requires --std_json and --n.")
                try:
                    langs = filter_langs_by_significance_to_ceiling(
                        model=model, dset=dset, langs_all=langs_all,
                        means=means, stds=stds, n=n, alpha=alpha,
                        ceiling_lang=ceil_lang,
                    )
                except ValueError as e:
                    # If filtering cannot be applied due to data issues, fall back to empty set
                    # (explicit: no languages analyzed if we cannot enforce the requested criterion)
                    print("Warning:", e)
                    langs = []
                # If nothing passes, we still produce an empty per_lang for transparency
            else:
                langs = langs_all

            per_lang = {}
            sum_phiU_Hw, sum_phiT_Hw, sum_phiR_Hw = 0.0, 0.0, 0.0
            sum_H = 0.0

            for l in langs:
                if l == ceil_lang:
                    # Actually, ceiling language does not have any impact since H=0. So we skip it here too.
                    continue

                S0  = base.get(l, None)
                SU  = u.get(l, None)
                ST  = t.get(l, None)
                SUT = ut.get(l, None)
                if not all(isinstance(x, (int, float)) for x in [S0, SU, ST, SUT]):
                    continue

                # Ceiling selection
                ceil_val = base_max

                H = max(0.0, ceil_val - S0)

                # Shapley-style U/G contributions (can be negative before clipping/weighting)
                phi_U = 0.5 * ((SU - S0) + (SUT - ST))
                phi_T = 0.5 * ((ST - S0) + (SUT - SU))

                # Optional significance weighting using stds + n
                if weighting != "none" and stds is not None and n is not None:
                    try:
                        s_base = stds[model][dset]["Base"][l]
                        s_u    = stds[model][dset]["w/ U"][l]
                        s_t    = stds[model][dset]["w/ T"][l]
                        s_ut   = stds[model][dset]["w/ U+T"][l]

                        # U is supported by (U vs Base) and (UT vs T)
                        p_u1 = welch_t_pvalue(SU, s_u, n, S0, s_base, n)
                        p_u2 = welch_t_pvalue(SUT, s_ut, n, ST, s_t, n)
                        # T is supported by (T vs Base) and (UT vs U)
                        p_t1 = welch_t_pvalue(ST, s_t, n, S0, s_base, n)
                        p_t2 = welch_t_pvalue(SUT, s_ut, n, SU, s_u, n)

                        if weighting == "hard":
                            wU = 1.0 if (p_u1 < alpha or p_u2 < alpha) else 0.0
                            wT = 1.0 if (p_t1 < alpha or p_t2 < alpha) else 0.0
                        else:  # "soft"
                            wU = max(0.0, min(1.0, 0.5 * ((1 - p_u1) + (1 - p_u2))))
                            wT = max(0.0, min(1.0, 0.5 * ((1 - p_t1) + (1 - p_t2))))

                        phi_U *= wU
                        phi_T *= wT
                    except Exception:
                        # If anything goes wrong, fall back to no weighting for this language
                        pass

                # Negative clipping
                phi_U = max(0.0, phi_U)
                phi_T = max(0.0, phi_T)

                # Define φ_R as the remainder to preserve φ_U + φ_T + φ_R = H
                phi_R = max(0.0, H - phi_U - phi_T)
                if phi_U + phi_T > H and (phi_U + phi_T) > 0:
                    scale = H / (phi_U + phi_T)
                    phi_U *= scale
                    phi_T *= scale
                    phi_R = 0.0

                # Shares (normalize by H)
                if H > 0:
                    U_share = phi_U / H
                    G_share = phi_T / H
                    R_share = phi_R / H
                else:
                    print("Warning: H=0 for", model, dset, l, "; setting U/G/R shares to 0.")
                    U_share = G_share = R_share = 0.0

                per_lang[l] = {
                    "shares": {"U": U_share, "G": G_share, "R": R_share},
                    "phi": {"U": phi_U, "G": phi_T, "R": phi_R},
                    "H": H,
                    "scores": {
                        "Base": S0, "U": SU, "T": ST, "UT": SUT,
                        "BaseMax": base_max, "Ceil": ceil_val
                    }
                }

                # Aggregation (headroom-weighted)
                sum_phiU_Hw += phi_U
                sum_phiT_Hw += phi_T
                sum_phiR_Hw += phi_R
                sum_H += H

            # Aggregates
            valid_langs = [l for l in per_lang if "shares" in per_lang[l]]
            if valid_langs:
                avg_U = sum(per_lang[l]["shares"]["U"] for l in valid_langs) / len(valid_langs)
                avg_G = sum(per_lang[l]["shares"]["G"] for l in valid_langs) / len(valid_langs)
                avg_R = sum(per_lang[l]["shares"]["R"] for l in valid_langs) / len(valid_langs)
            else:
                avg_U = avg_G = avg_R = 0.0

            if sum_H > 0:
                avgU_Hw = sum_phiU_Hw / sum_H
                avgG_Hw = sum_phiT_Hw / sum_H
                avgR_Hw = sum_phiR_Hw / sum_H
            else:
                avgU_Hw = avgG_Hw = avgR_Hw = 0.0

            out[model][dset] = {
                "per_lang": per_lang,
                "aggregate": {
                    "Avg_unweighted": {"U": avg_U, "G": avg_G, "R": avg_R},
                    "Avg_headroom_weighted": {"U": avgU_Hw, "G": avgG_Hw, "R": avgR_Hw},
                    "Total_headroom": sum_H,
                    "language_count": len(valid_langs),
                    "significance_filter": {"enabled": compute_sig_only, "alpha": alpha}
                }
            }

    return out

# ---------- CLI ----------
def main():
    p = argparse.ArgumentParser(description="Compute U/G/R failure shares from mean/std JSONs.")
    p.add_argument("--mean_json", required=True, help="Path to mean JSON")
    p.add_argument("--std_json", default=None, help="Path to std JSON (optional but required for weighting or significance filtering)")
    p.add_argument("--output_dir", required=True, help="Where to write the shares JSON")
    p.add_argument("--n", type=int, default=None, help="Per-cell sample size (e.g., 3). Needed for weighting/significance filtering.")
    p.add_argument("--alpha", type=float, default=0.05, help="Significance threshold (also used by --compute_significant_language_only)")
    p.add_argument("--weighting", choices=["none", "hard", "soft"], default="none",
                   help="Use Welch t-tests to weight φ_U, φ_T")
    p.add_argument("--compute_significant_language_only", action="store_true",
                   help="Analyze only languages with Base(lang) significantly different from Base(ceiling) at p<alpha.")
    args = p.parse_args()

    print("Loading means from", args.mean_json)
    with open(args.mean_json, "r", encoding="utf-8") as f:
        means = json.load(f)

    stds = None
    if args.std_json:
        with open(args.std_json, "r", encoding="utf-8") as f:
            stds = json.load(f)

    shares = compute_shapley_and_shares(
        means=means,
        stds=stds,
        n=args.n,
        alpha=args.alpha,
        weighting=args.weighting,
        compute_sig_only=args.compute_significant_language_only
    )
    print("Computed shares for", sum(len(dsets) for dsets in shares.values()), "datasets across", len(shares), "models.")
    os.makedirs(args.output_dir, exist_ok=True)
    postfix = "_siglangonly" if args.compute_significant_language_only else "_alllangs"

    save_path = os.path.join(args.output_dir, f"failure_attribution_shares{postfix}.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(shares, f, ensure_ascii=False, indent=2)

    print("Saved shares JSON to", save_path)
if __name__ == "__main__":
    main()
