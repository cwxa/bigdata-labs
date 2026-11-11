# -*- coding: utf-8 -*-
"""Regenerate the paper figures with English labels.

The experimental figures live in ../results and carry Chinese labels, which is
correct for the Chinese report and PPT but wrong for an IEEE paper. Both sets
read the same CSVs, so no number can drift between them.
"""

from __future__ import annotations

import csv
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter

PROJ = Path(__file__).resolve().parent
RES = PROJ.parent / "results"
OUT = PROJ / "figs"

matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9.5,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "savefig.dpi": 400,
    "savefig.bbox": "tight",
    "axes.unicode_minus": False,
})

C_AP = "#b5443a"
C_FP = "#1E4FA8"
C_GREY = "#8b97a8"


def read_csv(name):
    with open(RES / name, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def time_fig():
    rows = sorted(read_csv("summary_timing.csv"), key=lambda r: float(r["min_sup"]))
    xs = [100 * float(r["min_sup"]) for r in rows]
    ap = [float(r["apriori_seconds"]) for r in rows]
    fp = [float(r["fp_growth_seconds"]) for r in rows]

    # Ordinal x axis: the four thresholds are discrete settings, not a
    # continuum, and a log x axis produced overlapping minor-tick labels.
    pos = list(range(len(xs)))
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    ax.plot(pos, ap, "o-", color=C_AP, lw=1.4, ms=5, label="Apriori")
    ax.plot(pos, fp, "s-", color=C_FP, lw=1.4, ms=4.5, label="FP-growth")
    ax.set_yscale("log")
    ax.set_xticks(pos)
    ax.set_xticklabels([f"{x:g}" for x in xs])
    ax.set_xlim(-0.35, len(pos) - 0.65)
    ax.set_xlabel("Minimum support (%)")
    ax.set_ylabel("Running time (s)")
    ax.grid(True, which="both", ls="--", lw=0.5, alpha=0.35)
    ax.legend(frameon=False, loc="upper right")
    ax.margins(y=0.18)
    fig.savefig(OUT / "fig_timing.png")
    plt.close(fig)


def itemsets_fig():
    per = {}
    for sup, tag in ((0.02, "2%"), (0.01, "1%"), (0.005, "0.5%"), (0.002, "0.2%")):
        rows = read_csv(f"frequent_itemsets_sup{sup * 1000:.0f}permille.csv")
        per[tag] = Counter(int(r["k"]) for r in rows)
    tags = list(per)
    ks = sorted({k for m in per.values() for k in m})

    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    bottom = [0] * len(tags)
    colors = plt.get_cmap("tab10")
    for i, k in enumerate(ks):
        vals = [per[t].get(k, 0) for t in tags]
        ax.bar(tags, vals, bottom=bottom, color=colors(i % 10),
               edgecolor="white", lw=0.4, label=f"k = {k}")
        bottom = [b + v for b, v in zip(bottom, vals)]
    for i, total in enumerate(bottom):
        ax.text(i, total, f"{total}", ha="center", va="bottom", fontsize=7.5)
    ax.set_xlabel("Minimum support")
    ax.set_ylabel("Frequent itemsets")
    ax.margins(y=0.14)
    ax.grid(True, axis="y", ls="--", lw=0.5, alpha=0.35)
    ax.legend(title="Itemset length", frameon=False, fontsize=7,
              title_fontsize=7, ncol=3, loc="upper left")
    fig.savefig(OUT / "fig_itemsets.png")
    plt.close(fig)


def supp_hist_fig():
    rows = read_csv("frequent_itemsets_sup10permille.csv")
    ratios = [100 * float(r["support_ratio"]) for r in rows]
    med = statistics.median(ratios)

    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    ax.hist(ratios, bins=30, color=C_FP, alpha=0.85, edgecolor="white", lw=0.4)
    ax.axvline(1.0, color=C_AP, ls="--", lw=1.2, label="threshold (1%)")
    ax.axvline(med, color=C_GREY, ls=":", lw=1.2, label=f"median ({med:.2f}%)")
    ax.set_xlabel("Support (%)")
    ax.set_ylabel("Itemsets")
    ax.legend(frameon=False, fontsize=7)
    ax.grid(True, ls="--", lw=0.5, alpha=0.3)
    fig.savefig(OUT / "fig_supp_hist.png")
    plt.close(fig)


def pruning_fig():
    rows = read_csv("apriori_levels.csv")
    levels = sorted({float(r["min_sup"]) for r in rows})
    fig, axes = plt.subplots(2, 2, figsize=(3.5, 2.9))
    for idx, sup in enumerate(sorted(levels, reverse=True)):
        ax = axes[idx // 2][idx % 2]
        rs = [r for r in rows if abs(float(r["min_sup"]) - sup) < 1e-9]
        ks = [int(r["k"]) for r in rs]
        ax.bar([k - 0.19 for k in ks], [int(r["candidates"]) for r in rs],
               width=0.38, color=C_GREY, alpha=0.85, label="Candidates C(k)")
        ax.bar([k + 0.19 for k in ks], [int(r["frequent"]) for r in rs],
               width=0.38, color=C_AP, alpha=0.9, label="Retained L(k)")
        ax.set_yscale("log")
        ax.set_xticks(ks)
        ax.set_title(f"min_sup = {100 * sup:g}%", pad=3)
        ax.grid(True, axis="y", ls="--", lw=0.4, alpha=0.3)
        if idx // 2 == 1:
            ax.set_xlabel("Level k")
        if idx % 2 == 0:
            ax.set_ylabel("Count (log)")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False,
               fontsize=7, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(OUT / "fig_pruning.png")
    plt.close(fig)


def robust_fig():
    rows = sorted(read_csv("rule_robustness.csv"), key=lambda r: -float(r["lift"]))[:12]
    labels = [f"{r['antecedent']} → {r['consequent']}" for r in rows]
    est = [float(r["lift"]) for r in rows]
    lo = [float(r["lift"]) - float(r["lift_lo90"]) for r in rows]
    hi = [float(r["lift_hi90"]) - float(r["lift"]) for r in rows]
    y = list(range(len(rows)))

    fig, ax = plt.subplots(figsize=(3.4, 2.9))
    ax.barh(y, est, color=C_FP, alpha=0.85, height=0.62)
    ax.errorbar(est, y, xerr=[lo, hi], fmt="none", ecolor="#2c3e50",
                elinewidth=0.8, capsize=2, lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=6.5)
    ax.invert_yaxis()
    ax.axvline(1.0, color=C_AP, ls="--", lw=1.0, label="lift = 1")
    ax.set_xlabel("Lift (point estimate and 90% bootstrap CI)")
    ax.grid(True, axis="x", ls="--", lw=0.4, alpha=0.3)
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    fig.savefig(OUT / "fig_robust.png")
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    time_fig()
    itemsets_fig()
    supp_hist_fig()
    pruning_fig()
    robust_fig()
    for p in sorted(OUT.glob("*.png")):
        print(f"wrote {p.name} ({p.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
