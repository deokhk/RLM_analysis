# -*- coding: utf-8 -*-
"""
Performance Ratio vs. Translation ability scatter plot
- Color by language
- Marker by model
- Global linear trend line with Pearson r and slope in title
- Saves both PNG and PDF

Run: python plot_correlation.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Boost font sizes for better readability across plot elements
plt.rcParams.update({
    "font.size": 13,
    "axes.labelsize": 13,
    "axes.titlesize": 15,
    "legend.fontsize": 11,
    "legend.title_fontsize": 12,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
})

# ----------------------
# 1) Data (taken from Polymath-low)
# ----------------------
DATA = {
    ("Qwen3-1.7B", "de"): {"Performance Ratio": 87.3, "Translation ability": 91.6},
    ("Qwen3-1.7B", "es"): {"Performance Ratio": 95.0, "Translation ability": 95.2},
    ("Qwen3-1.7B", "ar"): {"Performance Ratio": 88.5, "Translation ability": 83.7},
    ("Qwen3-1.7B", "ja"): {"Performance Ratio": 79.3, "Translation ability": 86.2},
    ("Qwen3-1.7B", "ko"): {"Performance Ratio": 80.2, "Translation ability": 83.0},
    ("Qwen3-1.7B", "th"): {"Performance Ratio": 85.2, "Translation ability": 85.3},
    ("Qwen3-1.7B", "bn"): {"Performance Ratio": 66.9, "Translation ability": 63.2},
    ("Qwen3-1.7B", "sw"): {"Performance Ratio": 8.0,  "Translation ability": 10.8},
    ("Qwen3-1.7B", "te"): {"Performance Ratio": 45.6, "Translation ability": 59.2},

    ("Qwen3-4B", "de"): {"Performance Ratio": 91.2, "Translation ability": 95.0},
    ("Qwen3-4B", "es"): {"Performance Ratio": 97.2, "Translation ability": 95.7},
    ("Qwen3-4B", "ar"): {"Performance Ratio": 92.8, "Translation ability": 89.8},
    ("Qwen3-4B", "ja"): {"Performance Ratio": 88.4, "Translation ability": 94.3},
    ("Qwen3-4B", "ko"): {"Performance Ratio": 93.9, "Translation ability": 91.1},
    ("Qwen3-4B", "th"): {"Performance Ratio": 88.1, "Translation ability": 91.1},
    ("Qwen3-4B", "bn"): {"Performance Ratio": 86.2, "Translation ability": 86.3},
    ("Qwen3-4B", "sw"): {"Performance Ratio": 30.4, "Translation ability": 27.4},
    ("Qwen3-4B", "te"): {"Performance Ratio": 72.4, "Translation ability": 81.9},

    ("Qwen3-8B", "de"): {"Performance Ratio": 92.0, "Translation ability": 97.1},
    ("Qwen3-8B", "es"): {"Performance Ratio": 97.5, "Translation ability": 97.2},
    ("Qwen3-8B", "ar"): {"Performance Ratio": 94.5, "Translation ability": 94.3},
    ("Qwen3-8B", "ja"): {"Performance Ratio": 89.5, "Translation ability": 95.3},
    ("Qwen3-8B", "ko"): {"Performance Ratio": 93.9, "Translation ability": 94.7},
    ("Qwen3-8B", "th"): {"Performance Ratio": 94.2, "Translation ability": 94.7},
    ("Qwen3-8B", "bn"): {"Performance Ratio": 89.8, "Translation ability": 89.0},
    ("Qwen3-8B", "sw"): {"Performance Ratio": 63.7, "Translation ability": 46.4},
    ("Qwen3-8B", "te"): {"Performance Ratio": 79.8, "Translation ability": 87.5},

    ("Qwen3-14B", "de"): {"Performance Ratio": 94.1, "Translation ability": 97.1},
    ("Qwen3-14B", "es"): {"Performance Ratio": 98.3, "Translation ability": 97.4},
    ("Qwen3-14B", "ar"): {"Performance Ratio": 95.0, "Translation ability": 94.6},
    ("Qwen3-14B", "ja"): {"Performance Ratio": 93.8, "Translation ability": 95.7},
    ("Qwen3-14B", "ko"): {"Performance Ratio": 93.8, "Translation ability": 96.0},
    ("Qwen3-14B", "th"): {"Performance Ratio": 96.4, "Translation ability": 95.8},
    ("Qwen3-14B", "bn"): {"Performance Ratio": 93.8, "Translation ability": 91.9},
    ("Qwen3-14B", "sw"): {"Performance Ratio": 77.0, "Translation ability": 59.9},
    ("Qwen3-14B", "te"): {"Performance Ratio": 86.3, "Translation ability": 90.5},

    ("gpt-oss-20b", "de"): {"Performance Ratio": 89.2, "Translation ability": 97.4},
    ("gpt-oss-20b", "es"): {"Performance Ratio": 93.6, "Translation ability": 97.4},
    ("gpt-oss-20b", "ar"): {"Performance Ratio": 94.7, "Translation ability": 94.0},
    ("gpt-oss-20b", "ja"): {"Performance Ratio": 89.7, "Translation ability": 94.9},
    ("gpt-oss-20b", "ko"): {"Performance Ratio": 95.0, "Translation ability": 95.0},
    ("gpt-oss-20b", "th"): {"Performance Ratio": 94.2, "Translation ability": 95.0},
    ("gpt-oss-20b", "bn"): {"Performance Ratio": 93.9, "Translation ability": 95.5},
    ("gpt-oss-20b", "sw"): {"Performance Ratio": 81.9, "Translation ability": 74.7},
    ("gpt-oss-20b", "te"): {"Performance Ratio": 84.4, "Translation ability": 94.2},
}

def build_df(data: dict) -> pd.DataFrame:
    rows = []
    for (model, lang), v in data.items():
        rows.append({
            "model": model,
            "lang": lang,
            "Performance Ratio": v["Performance Ratio"],
            "Translation ability": v["Translation ability"],
        })
    return pd.DataFrame(rows)

def main(outbase: str = "perf_vs_trans"):
    df = build_df(DATA)

    # Languages & models
    languages = sorted(df["lang"].unique())
    models = list(df["model"].unique())

    # Color per language using Accent qualitative colormap
    cmap = plt.get_cmap("Accent")
    lang2color = {lg: cmap(i % cmap.N) for i, lg in enumerate(languages)}

    # Marker per model
    markers = ["o", "s", "^", "D", "P", "X", "*", "v", "<", ">"]
    model2marker = {m: markers[i % len(markers)] for i, m in enumerate(models)}

    # Figure
    plt.figure(figsize=(8, 6))

    ax = plt.gca()
    marker_size = 90  # enlarge scatter markers for better visibility
    for _, r in df.iterrows():
        ax.scatter(
            r["Translation ability"],
            r["Performance Ratio"],
            s=marker_size,
            c=[lang2color[r["lang"]]],
            marker=model2marker[r["model"]],
            edgecolors="black",
            linewidths=0.5,
            alpha=0.9,
        )

    # Global trend line & Pearson r
    x = df["Translation ability"].to_numpy()
    y = df["Performance Ratio"].to_numpy()
    coeff = np.polyfit(x, y, deg=1)
    slope, intercept = coeff[0], coeff[1]
    r = np.corrcoef(x, y)[0, 1]

    xs = np.linspace(x.min(), x.max(), 200)
    ys = slope * xs + intercept
    ax.plot(xs, ys, linewidth=2, color="red")

    # Legends (separate: languages/colors, models/markers)
    lang_handles = [
        plt.Line2D([0], [0], marker="o", linestyle="",
                   markerfacecolor=lang2color[lg], markeredgecolor="black",
                   markersize=8, label=lg)
        for lg in languages
    ]
    model_handles = [
        plt.Line2D([0], [0], marker=model2marker[m], linestyle="",
                   markerfacecolor="white", markeredgecolor="black",
                   markersize=8, label=m)
        for m in models
    ]
    leg1 = ax.legend(handles=lang_handles, title="Language (color)",
                     loc="lower right", frameon=True,
                     fontsize=13, title_fontsize=13)
    ax.add_artist(leg1)
    ax.legend(handles=model_handles, title="Model (marker)",
              loc="upper left", frameon=True, ncol=min(3, len(models)),
              fontsize=12, title_fontsize=13)

    # Labels & title
    ax.set_xlabel("GEMBA-DA (xx -> en)")
    ax.set_ylabel("Reasoning Performance Ratio (%)")
    ax.set_title("Correlation: Translation (xx->en) vs. Reasoning Performance")

    ax.text(
        0.85,
        0.7,
        f"r={r:.3f}",
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox={
            "boxstyle": "round,pad=0.3",
            "facecolor": "white",
            "edgecolor": "gray",
            "alpha": 0.8,
        },
    )

    ax.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()

    # Optional: per-model trend lines (uncomment to display)
    # for m, sub in df.groupby("model"):
    #     xm = sub["Translation ability"].to_numpy()
    #     ym = sub["Performance Ratio"].to_numpy()
    #     sm, bm = np.polyfit(xm, ym, deg=1)
    #     xs_m = np.linspace(xm.min(), xm.max(), 100)
    #     ys_m = sm * xs_m + bm
    #     ax.plot(xs_m, ys_m, linestyle="--", linewidth=1)

    # Save
    out_png = Path(f"{outbase}.png")
    out_pdf = Path(f"{outbase}.pdf")
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    print(f"Saved: {out_png.resolve()}")
    print(f"Saved: {out_pdf.resolve()}")

if __name__ == "__main__":
    main()
