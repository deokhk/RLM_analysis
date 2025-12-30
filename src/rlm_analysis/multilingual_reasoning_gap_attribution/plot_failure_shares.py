#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot stacked bar charts of Gap shares (U/R/G) by language from the shares JSON,
and aggregate dataset-level gap shares per (dataset, model).

Inputs:
- shares_json (required):
  {
    "<Model>": {
      "<Dataset>": {
        "per_lang": {
          "<lang>": {
            "shares": {"U": float, "G": float, "R": float},  # sum ~ 1
            "phi": {...},
            "H": float,
            "scores": {"Base": float, "U": float, "T": float, "UT": float, ...}
          }, ...
        },
        "aggregate": {
          "Avg_unweighted": {"U": float, "G": float, "R": float},
          "Avg_headroom_weighted": {"U": float, "G": float, "R": float},
          ...
        }
      }, ...
    }, ...
  }

This script:
  1) per-language stacked chart per (model,dataset)
  2) aggregated-by-dataset per model (language-level bars grouped by dataset)
  3) grid of aggregated-by-dataset for multiple models
  4) (UPDATED) dataset-grouped, per-model stacked bars using
     aggregate["Avg_headroom_weighted"] for each dataset,
     saved as gap_aggregated__dataset_grouped_models.{png,pdf}
"""

import argparse
import json
import os
import math
from typing import List, Dict, Any, Tuple, Optional

import matplotlib
matplotlib.use("Agg")  # for headless environments
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def sanitize_filename(s: str) -> str:
    keep = "-_.() "
    return "".join(ch if ch.isalnum() or ch in keep else "_" for ch in s)


def load_shares(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def pick_lang_order(per_lang: Dict[str, Any], prefer: List[str] = None) -> List[str]:
    langs = [l for l in per_lang.keys() if l.lower() != "avg"]
    if prefer:
        ordered = [l for l in prefer if l in per_lang]
        rest = sorted([l for l in langs if l not in ordered])
        return ordered + rest
    return sorted(langs)


def add_percent_labels(ax, bottoms, values, xs, fmt="{:.1f}%", min_display: float = 0.0, fontsize: int = 13):
    for x, b, v in zip(xs, bottoms, values):
        if v <= 0 or v < min_display:
            continue
        y = b + v / 2.0
        ax.text(
            x, y, fmt.format(100.0 * v),
            ha="center", va="center",
            color="white", fontsize=fontsize, fontweight="bold"
        )


def _renorm_urg(u: float, r: float, g: float) -> Tuple[float, float, float]:
    u, r, g = max(0.0, u), max(0.0, r), max(0.0, g)
    s = u + r + g
    if s > 0:
        return (u / s, r / s, g / s)
    return (0.0, 0.0, 0.0)


# -------------------- UPDATED: dataset-level aggregate using Avg_headroom_weighted --------------------

def get_dataset_aggregate_shares(
    shares_json: Dict[str, Any],
    model: str,
    dataset: str,
    agg_key: str = "Avg_headroom_weighted",
) -> Optional[Tuple[float, float, float]]:
    """
    Return (U, R, G) from shares_json[model][dataset]["aggregate"][agg_key].
    If missing, return None.
    """
    payload = ((shares_json.get(model, {}) or {}).get(dataset, {}) or {})
    agg = (payload.get("aggregate", {}) or {})
    vals = (agg.get(agg_key, None))
    if not isinstance(vals, dict):
        return None

    u = float(vals.get("U", 0.0))
    r = float(vals.get("R", 0.0))
    g = float(vals.get("G", 0.0))
    return _renorm_urg(u, r, g)


def prepare_gap_aggregate_series_by_model(
    shares_json: Dict[str, Any],
    models: List[str],
    datasets: List[str],
    agg_key: str = "Avg_headroom_weighted",
) -> Optional[List[Tuple[str, Dict[str, Any]]]]:
    """
    For each model, build series over datasets using aggregate[agg_key].
    """
    prepared = []
    for model in models:
        U_vals, R_vals, G_vals, labels = [], [], [], []
        for dset in datasets:
            tup = get_dataset_aggregate_shares(shares_json, model, dset, agg_key=agg_key)
            if tup is None:
                continue
            u, r, g = tup
            U_vals.append(u)
            R_vals.append(r)
            G_vals.append(g)
            labels.append(dset)

        if labels:
            xs = list(range(len(labels)))
            prepared.append((model, {
                "xs": xs,
                "U": U_vals,
                "R": R_vals,
                "G": G_vals,
                "labels": labels,
            }))

    return prepared if prepared else None


def plot_gap_aggregated_subplots(
    series_by_model: List[Tuple[str, Dict[str, Any]]],
    out_dir: str,
    title: str = "",
    y_label: str = "Share of Gap (%)",
    legend_labels=("Understanding", "Reasoning", "Generating in Input Language"),
    y_lim=(0.0, 1.0),
    dpi: int = 220,
    ncols: Optional[int] = None,
):
    """
    Save as:
      gap_aggregated__dataset_grouped_models.png/pdf
    """
    os.makedirs(out_dir, exist_ok=True)
    num_models = len(series_by_model)
    if num_models == 0:
        return

    if not ncols or ncols <= 0:
        ncols = min(3, num_models)
    nrows = math.ceil(num_models / ncols)

    base_width = 6.0
    base_height = 8.5
    figsize = (base_width * ncols, base_height * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharey=True)
    axes_list = list(axes.flat) if hasattr(axes, "flat") else [axes]

    bar_colors = ("#6c71d6", "#8cc671", "#d68c71")

    for idx, (model, series) in enumerate(series_by_model):
        ax = axes_list[idx]
        xs = series["xs"]
        U = series["U"]
        R = series["R"]
        G = series["G"]
        labels = series["labels"]

        ax.bar(xs, U, label=legend_labels[0], color=bar_colors[0])
        ax.bar(xs, R, bottom=U, label=legend_labels[1], color=bar_colors[1])
        bottoms_G = [u + r for u, r in zip(U, R)]
        ax.bar(xs, G, bottom=bottoms_G, label=legend_labels[2], color=bar_colors[2])

        add_percent_labels(ax, [0.0 for _ in U], U, xs, min_display=0.05, fontsize=17)
        add_percent_labels(ax, U, R, xs, min_display=0.05, fontsize=17)
        add_percent_labels(ax, bottoms_G, G, xs, min_display=0.05, fontsize=17)

        ax.set_title(model, fontsize=22, pad=20)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=18)
        ax.set_ylim(y_lim)

        yticks = [i / 10 for i in range(0, 11, 2)]
        ax.set_yticks(yticks)
        if idx % ncols == 0:
            ax.set_ylabel(y_label, fontsize=22)
            ax.set_yticklabels([f"{int(t*100)}" for t in yticks], fontsize=20)
        else:
            ax.set_ylabel("")
            ax.tick_params(axis="y", labelleft=False)

    for extra_ax in axes_list[len(series_by_model):]:
        fig.delaxes(extra_ax)

    legend_handles = [
        Patch(facecolor=bar_colors[0], label=legend_labels[0]),
        Patch(facecolor=bar_colors[1], label=legend_labels[1]),
        Patch(facecolor=bar_colors[2], label=legend_labels[2]),
    ]

    fig.legend(
        legend_handles,
        [h.get_label() for h in legend_handles],
        loc="lower center",
        ncol=3,
        fontsize=19,
        framealpha=0.9,
    )

    if title:
        fig.suptitle(title, fontsize=18, y=0.99)

    fig.tight_layout(rect=[0, 0.06, 1, 0.94], w_pad=0.05)

    base = "gap_aggregated__dataset_grouped_models"
    png_path = os.path.join(out_dir, base + ".png")
    pdf_path = os.path.join(out_dir, base + ".pdf")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {png_path}\nSaved: {pdf_path}")


# -------------------- Existing aggregated-by-dataset helpers --------------------

def prepare_aggregated_series(dset_to_per_lang: Dict[str, Dict[str, Any]],
                              languages_order_global: List[str] = None,
                              group_gap: float = 0.8):
    datasets = list(dset_to_per_lang.keys())

    xs, U, R, G, lang_labels = [], [], [], [], []
    group_bounds = []

    x = 0.0
    for dset in datasets:
        per_lang = dset_to_per_lang.get(dset) or {}
        if not per_lang:
            continue

        langs = pick_lang_order(per_lang, languages_order_global)
        if not langs:
            continue

        start_x = x
        added = False
        for l in langs:
            shares = per_lang.get(l, {}).get("shares", {})
            u = float(shares.get("U", 0.0))
            r = float(shares.get("R", 0.0))
            g = float(shares.get("G", 0.0))

            u, r, g = _renorm_urg(u, r, g)

            xs.append(x)
            U.append(u)
            R.append(r)
            G.append(g)
            lang_labels.append(l)
            x += 1.0
            added = True

        if not added:
            continue

        end_x = x - 1.0
        group_bounds.append((start_x, end_x, dset))
        x += group_gap

    if not xs:
        return None

    return {
        "xs": xs,
        "U": U,
        "R": R,
        "G": G,
        "lang_labels": lang_labels,
        "group_bounds": group_bounds,
        "total_bars": len(xs),
        "num_groups": len(group_bounds),
    }


def draw_aggregated_chart(ax,
                          series: Dict[str, Any],
                          model_title: str,
                          y_label: str,
                          y_lim,
                          legend_labels,
                          annotate: bool,
                          group_gap: float,
                          show_ylabel: bool = True,
                          show_legend: bool = True):
    xs = series["xs"]
    U = series["U"]
    R = series["R"]
    G = series["G"]
    lang_labels = series["lang_labels"]
    group_bounds = series["group_bounds"]

    bar_colors = ("#6c71d6", "#8cc671", "#d68c71")

    ax.bar(xs, U, label=legend_labels[0], color=bar_colors[0])
    ax.bar(xs, R, bottom=U, label=legend_labels[1], color=bar_colors[1])
    bottoms_G = [u + r for u, r in zip(U, R)]
    ax.bar(xs, G, bottom=bottoms_G, label=legend_labels[2], color=bar_colors[2])

    ax.set_xticks(xs)
    ax.set_xticklabels(lang_labels, fontsize=18)

    ax.set_ylim(y_lim)
    ax.set_title(model_title, fontsize=18)

    yticks = [i/10 for i in range(0, 11)]
    ax.set_yticks(yticks)
    if show_ylabel:
        ax.set_ylabel(y_label, fontsize=16)
        ax.set_yticklabels([f"{int(t*100)}" for t in yticks], fontsize=14)
    else:
        ax.set_ylabel("")
        ax.tick_params(axis="y", labelleft=False)

    for (start_x, end_x, dset) in group_bounds:
        center = (start_x + end_x) / 2.0
        ax.text(center, -0.1, dset, transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=13)
        if start_x != group_bounds[0][0]:
            ax.axvline(x=start_x - group_gap/2.0, color="k", linewidth=0.5, alpha=0.6)

    if annotate:
        add_percent_labels(ax, [0.0]*len(xs), U, xs, fmt="{:.1f}%", min_display=0.05, fontsize=14)
        add_percent_labels(ax, U, R, xs, fmt="{:.1f}%", min_display=0.05, fontsize=14)
        add_percent_labels(ax, bottoms_G, G, xs, fmt="{:.1f}%", min_display=0.05, fontsize=14)

    if show_legend:
        ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.18), framealpha=0.9, fontsize=15)

    return ax


def plot_stacked_shares(model: str,
                        dataset: str,
                        per_lang: Dict[str, Any],
                        out_dir: str,
                        languages_order: List[str] = None,
                        dpi: int = 200,
                        figsize=(12, 6),
                        title: str = None,
                        y_label: str = "Share of Gaps (%)",
                        legend_labels=("Understanding", "Reasoning", "Generating in Input Language"),
                        y_lim=(0, 1.0),
                        annotate=True):
    os.makedirs(out_dir, exist_ok=True)

    langs = pick_lang_order(per_lang, languages_order)
    if not langs:
        return

    U, R, G = [], [], []
    for l in langs:
        shares = (per_lang.get(l, {}) or {}).get("shares", {}) or {}
        u = float(shares.get("U", 0.0))
        r = float(shares.get("R", 0.0))
        g = float(shares.get("G", 0.0))
        u, r, g = _renorm_urg(u, r, g)
        U.append(u)
        R.append(r)
        G.append(g)

    x = list(range(len(langs)))

    plt.figure(figsize=figsize)
    ax = plt.gca()

    bar_colors = ("#6c71d6", "#8cc671", "#d68c71")

    ax.bar(x, U, label=legend_labels[0], color=bar_colors[0])
    ax.bar(x, R, bottom=U, label=legend_labels[1], color=bar_colors[1])
    bottoms_G = [u + r for u, r in zip(U, R)]
    ax.bar(x, G, bottom=bottoms_G, label=legend_labels[2], color=bar_colors[2])

    ax.set_xticks(x)
    ax.set_xticklabels(langs, fontsize=15)
    ax.set_ylim(y_lim)
    ax.set_ylabel(y_label, fontsize=16)
    ax.set_title(title or f"Gap Cause Breakdown by Language — {model} / {dataset}", fontsize=18)

    if annotate:
        add_percent_labels(ax, [0.0]*len(x), U, x, fmt="{:.1f}%", min_display=0.03, fontsize=15)
        add_percent_labels(ax, U, R, x, fmt="{:.1f}%", min_display=0.03, fontsize=15)
        add_percent_labels(ax, bottoms_G, G, x, fmt="{:.1f}%", min_display=0.03, fontsize=15)

    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.18), framealpha=0.9, fontsize=16)
    ax.set_yticks([i/10 for i in range(0, 11)])
    ax.set_yticklabels([f"{int(t*100)}" for t in ax.get_yticks()], fontsize=14)

    plt.tight_layout()

    base = f"{sanitize_filename(model)}__{sanitize_filename(dataset)}__Gap_shares_by_language"
    png_path = os.path.join(out_dir, base + ".png")
    pdf_path = os.path.join(out_dir, base + ".pdf")
    plt.savefig(png_path, dpi=dpi, bbox_inches="tight")
    plt.savefig(pdf_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"Saved: {png_path}\nSaved: {pdf_path}")


def plot_aggregated_by_dataset(model: str,
                               dset_to_per_lang: Dict[str, Dict[str, Any]],
                               out_dir: str,
                               languages_order_global: List[str] = None,
                               dpi: int = 200,
                               figsize=None,
                               title: str = None,
                               y_label: str = "Share of Gaps (%)",
                               legend_labels=("Understanding", "Reasoning", "Generating in Input Language"),
                               y_lim=(0, 1.0),
                               group_gap: float = 0.8,
                               annotate: bool = True):
    os.makedirs(out_dir, exist_ok=True)
    if not dset_to_per_lang:
        return

    series = prepare_aggregated_series(
        dset_to_per_lang=dset_to_per_lang,
        languages_order_global=languages_order_global,
        group_gap=group_gap,
    )
    if not series:
        return

    if figsize is None:
        total_bars = series["total_bars"]
        num_groups = series["num_groups"]
        width = max(12.0, 1.0 * total_bars + 0.6 * num_groups + 3.0)
        figsize = (width, 6.0)

    plt.figure(figsize=figsize)
    ax = plt.gca()

    draw_aggregated_chart(
        ax=ax,
        series=series,
        model_title=title or f"{model}",
        y_label=y_label,
        y_lim=y_lim,
        legend_labels=legend_labels,
        annotate=annotate,
        group_gap=group_gap,
        show_ylabel=True,
        show_legend=True,
    )
    plt.tight_layout()

    base = f"{sanitize_filename(model)}__aggregated_by_dataset"
    png_path = os.path.join(out_dir, base + ".png")
    pdf_path = os.path.join(out_dir, base + ".pdf")
    plt.savefig(png_path, dpi=dpi, bbox_inches="tight")
    plt.savefig(pdf_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"Saved: {png_path}\nSaved: {pdf_path}")


def plot_aggregated_models_subplots(model_to_dset: Dict[str, Dict[str, Dict[str, Any]]],
                                    out_dir: str,
                                    languages_order_global: List[str] = None,
                                    dpi: int = 200,
                                    y_label: str = "Share of Gaps (%)",
                                    legend_labels=("Understanding", "Reasoning", "Generating in Input Language"),
                                    y_lim=(0, 1.0),
                                    group_gap: float = 0.8,
                                    annotate: bool = False,
                                    ncols: int = None,
                                    figsize=None):
    os.makedirs(out_dir, exist_ok=True)
    if not model_to_dset:
        return

    prepared = []
    for model, dset_to_per_lang in model_to_dset.items():
        if not dset_to_per_lang:
            continue
        series = prepare_aggregated_series(
            dset_to_per_lang=dset_to_per_lang,
            languages_order_global=languages_order_global,
            group_gap=group_gap,
        )
        if not series:
            continue
        prepared.append((model, series))

    if not prepared:
        return

    num_models = len(prepared)
    if not ncols or ncols <= 0:
        ncols = min(3, num_models)
    nrows = math.ceil(num_models / ncols)

    if figsize is None:
        base_width = 8
        base_height = 4.5
        figsize = (base_width * ncols, base_height * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharey=True)
    axes_list = list(axes.flat) if hasattr(axes, "flat") else [axes]

    for idx, (model, series) in enumerate(prepared):
        ax = axes_list[idx]
        draw_aggregated_chart(
            ax=ax,
            series=series,
            model_title=model,
            y_label=y_label,
            y_lim=y_lim,
            legend_labels=legend_labels,
            annotate=annotate,
            group_gap=group_gap,
            show_ylabel=(idx % ncols == 0),
            show_legend=False,
        )

    for extra_ax in axes_list[len(prepared):]:
        fig.delaxes(extra_ax)

    fig.tight_layout(rect=[0, 0.18, 1, 1])

    base = "aggregated_by_dataset__models_subplots"
    png_path = os.path.join(out_dir, base + ".png")
    pdf_path = os.path.join(out_dir, base + ".pdf")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {png_path}\nSaved: {pdf_path}")


# -------------------- CLI --------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shares_json", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--languages", nargs="*", default=None)
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--figsize", nargs=2, type=float, default=[12.0, 6.0])
    ap.add_argument("--title_prefix", default="")
    ap.add_argument("--agg_auto_fig", action="store_true")
    ap.add_argument("--group_gap", type=float, default=0.8)
    ap.add_argument("--agg_subplot_cols", type=int, default=None)
    ap.add_argument("--gap_subplot_cols", type=int, default=None)

    # UPDATED: aggregate chart now uses shares_json["aggregate"]["Avg_headroom_weighted"]
    ap.add_argument(
        "--plot_gap_agg",
        action="store_true",
        help="Plot dataset-grouped, per-model stacked bars using aggregate['Avg_headroom_weighted'], "
             "saved as gap_aggregated__dataset_grouped_models.pdf"
    )

    args = ap.parse_args()

    shares = load_shares(args.shares_json)

    # Determine model/dataset lists
    all_models = list(shares.keys())
    models = args.models if args.models else all_models

    # datasets per model can differ; collect union for new chart, but preserve user subset order if given
    all_datasets = []
    for m in models:
        all_datasets.extend(list((shares.get(m, {}) or {}).keys()))
    dset_order = args.datasets if args.datasets else sorted(set(all_datasets))

    # ---------- Existing per-language & aggregated-by-dataset ----------
    model_to_dset_for_grid = {}

    for model in models:
        dsets = shares.get(model, {}) or {}
        dset_to_per_lang = {}
        for dset in dset_order:
            if dset not in dsets:
                continue
            payload = dsets[dset] or {}
            per_lang = payload.get("per_lang", {}) or {}
            if not per_lang:
                continue

            title = f"{args.title_prefix}Gap Cause Breakdown by Language ({model} / {dset})".strip()
            plot_stacked_shares(
                model=model,
                dataset=dset,
                per_lang=per_lang,
                out_dir=args.out_dir,
                languages_order=args.languages,
                dpi=args.dpi,
                figsize=(args.figsize[0], args.figsize[1]),
                title=title
            )
            dset_to_per_lang[dset] = per_lang

        if dset_to_per_lang:
            agg_title = f"{model}".strip()
            agg_figsize = None if args.agg_auto_fig else (16.0, 6.0)
            plot_aggregated_by_dataset(
                model=model,
                dset_to_per_lang=dset_to_per_lang,
                out_dir=args.out_dir,
                languages_order_global=args.languages,
                dpi=args.dpi,
                figsize=agg_figsize,
                title=agg_title,
                group_gap=args.group_gap,
                annotate=True
            )
            model_to_dset_for_grid[model] = dset_to_per_lang

    if model_to_dset_for_grid:
        plot_aggregated_models_subplots(
            model_to_dset=model_to_dset_for_grid,
            out_dir=args.out_dir,
            languages_order_global=args.languages,
            dpi=args.dpi,
            group_gap=args.group_gap,
            annotate=False,
            ncols=args.agg_subplot_cols,
        )

    # ---------- UPDATED: dataset-grouped aggregate chart (Avg_headroom_weighted) ----------
    if args.plot_gap_agg:
        preferred_order = [
            "Polymath-Low",
            "Polymath-Medium",
            "Polymath-High",
            "MMLU-ProX-Lite",
        ]
        gap_dset_order = [d for d in preferred_order if d in dset_order]
        for d in dset_order:
            if d not in gap_dset_order:
                gap_dset_order.append(d)

        series_by_model = prepare_gap_aggregate_series_by_model(
            shares_json=shares,
            models=models,
            datasets=gap_dset_order,
            agg_key="Avg_headroom_weighted",
        )
        if series_by_model:
            title = "" 
            plot_gap_aggregated_subplots(
                series_by_model=series_by_model,
                out_dir=args.out_dir,
                title=title,
                y_label="Share of Gap (%)",
                dpi=max(args.dpi, 220),
                ncols=args.gap_subplot_cols,
            )
        else:
            print("[WARN] No data available to plot gap aggregated chart. "
                  "Check if aggregate['Avg_headroom_weighted'] exists in shares_json.")


if __name__ == "__main__":
    main()
