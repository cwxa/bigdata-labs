# -*- coding: utf-8 -*-
"""实验二主实验脚本：Apriori / FP-growth 参数网格、一致性校验与结果导出.

大数据分析与处理 · 实验二
陈纬昕

运行
----
    python run_experiments.py                       # 默认网格
    python run_experiments.py --sup 0.02,0.01 --conf 0.5

产物（写入仓库内 lab2-association-rules/results/）
----------------------
CSV:
  summary_timing.csv            各支持度下两种算法的规模与耗时
  item_frequency.csv            全部 item 的频次与占比
  frequent_itemsets_*.csv       频繁项集明细
  rules_*.csv                   关联规则明细
  rules_summary.csv             规则数量随参数变化
  apriori_levels.csv            Apriori 每轮候选数 / 频繁数（剪枝效果）
PNG:
  fig1_support_vs_time.png      耗时—支持度曲线（对数纵轴）
  fig2_itemsets_by_k.png        各阶频繁项集数量
  fig3_rules_vs_conf.png        规则数量随最小置信度变化
  fig4_top_lift_rules.png       top-lift 规则
  fig5_support_hist.png         频繁项集支持度分布
  fig6_candidate_vs_frequent.png Apriori 剪枝效果
  fig7_fptree_stats.png         FP-tree 节点数与深度
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from apriori import (apriori, flat_itemsets, fmt_itemset, generate_rules,
                     load_transactions)
from fp_growth import fp_growth, summarize_by_k

# 路径基于脚本位置解析，保证从任意工作目录调用都能命中仓库内数据与产物目录
_HERE = Path(__file__).resolve().parent
DATA = str(_HERE.parent / "data" / "array_data.csv")
OUT = str(_HERE.parent / "results")

# 中文字体：Windows 优先微软雅黑，回退黑体/SimHei
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["figure.dpi"] = 120
matplotlib.rcParams["savefig.dpi"] = 300
matplotlib.rcParams["savefig.bbox"] = "tight"

# 统一配色（深色背景下的图表用浅底，保证打印/投影可读）
C_AP = "#c0392b"      # Apriori
C_FP = "#2471a3"      # FP-growth
C_GREY = "#7f8c8d"


def timed(fn, repeats: int = 3, min_repeats_time: float = 0.8):
    """计时：先跑一次；若耗时短于 min_repeats_time 再重复取最小值，否则只记单次."""
    t0 = time.perf_counter()
    result = fn()
    first = time.perf_counter() - t0
    if first >= min_repeats_time or repeats <= 1:
        return result, first
    best = first
    for _ in range(repeats - 1):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return result, best


def write_csv(path: str, header, rows) -> None:
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"  wrote {path} ({len(rows)} rows)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sup", default="0.02,0.01,0.005", help="逗号分隔的最小支持度")
    ap.add_argument("--conf", default="0.5,0.6,0.7", help="逗号分隔的最小置信度")
    ap.add_argument("--main-sup", type=float, default=0.01, help="细粒度分析用的支持度")
    args = ap.parse_args()

    sups = [float(x) for x in args.sup.split(",")]
    confs = [float(x) for x in args.conf.split(",")]
    os.makedirs(OUT, exist_ok=True)

    t_load = time.perf_counter()
    txns = load_transactions(DATA)
    n = len(txns)
    print(f"loaded {n} transactions in {time.perf_counter() - t_load:.3f}s; "
          f"distinct items = {len({i for t in txns for i in t})}")

    # ---------------- 数据基础统计 ---------------- #
    item_cnt = Counter(i for t in txns for i in t)
    write_csv(f"{OUT}/item_frequency.csv",
              ["item", "count", "ratio"],
              [(i, c, f"{c / n:.6f}")
               for i, c in sorted(item_cnt.items(), key=lambda kv: (-kv[1], kv[0]))])

    lens = [len(t) for t in txns]
    dataset_stats = {
        "transactions": n,
        "distinct_items": len(item_cnt),
        "total_item_occurrences": sum(item_cnt.values()),
        "avg_length": round(sum(lens) / n, 4),
        "min_length": min(lens),
        "max_length": max(lens),
        "items_ratio_ge_5pct": sum(1 for c in item_cnt.values() if c / n >= 0.05),
        "items_ratio_ge_1pct": sum(1 for c in item_cnt.values() if c / n >= 0.01),
    }

    # ---------------- 参数网格 ---------------- #
    summary = []
    rules_summary = []
    level_rows = []
    timing_curves = {"apriori": [], "fp_growth": []}
    itemsets_by_k = {}

    for min_sup in sups:
        print(f"\n== min_sup = {min_sup:.3%} ==")
        (freq_ap, level_stats), t_ap = timed(lambda: apriori(txns, min_sup))
        (freq_fp, tree), t_fp = timed(lambda: fp_growth(txns, min_sup))

        # 一致性校验：两套实现必须给出同一组频繁项集与相同支持计数
        ap_flat = {}
        for k in range(1, len(freq_ap)):
            ap_flat.update(freq_ap[k])
        assert ap_flat == freq_fp, f"Apriori/FP-growth mismatch at min_sup={min_sup}"

        total = len(ap_flat)
        max_k = max((len(s) for s in ap_flat), default=0)
        by_k = summarize_by_k(freq_fp)
        itemsets_by_k[min_sup] = by_k
        timing_curves["apriori"].append((min_sup, t_ap))
        timing_curves["fp_growth"].append((min_sup, t_fp))
        print(f"  itemsets={total} max_k={max_k} by_k={by_k} "
              f"| apriori {t_ap:.3f}s, fp-growth {t_fp:.3f}s, verified equal")

        summary.append({
            "min_sup": min_sup, "min_count": round(min_sup * n, 2),
            "frequent_itemsets": total, "max_k": max_k,
            "apriori_seconds": round(t_ap, 4), "fp_growth_seconds": round(t_fp, 4),
            "speedup_fp_over_apriori": round(t_ap / t_fp, 3) if t_fp else None,
            "fp_tree_nodes": tree.node_count(), "fp_tree_depth": tree.depth(),
        })

        # 明细：该支持度下的全部频繁项集
        write_csv(f"{OUT}/frequent_itemsets_sup{min_sup * 1000:.0f}permille.csv",
                  ["k", "itemset", "support_count", "support_ratio"],
                  [(len(s), fmt_itemset(s), c, f"{c / n:.6f}")
                   for s, c in [(s, ap_flat[s]) for s in sorted(ap_flat,
                                                                key=lambda x: (len(x), x))]])

        # Apriori 每轮候选/频繁/耗时
        for st in level_stats:
            level_rows.append({"min_sup": min_sup, "k": st.k,
                               "candidates": st.candidates, "frequent": st.frequent,
                               "seconds": round(st.seconds, 4)})

        # 规则：完整网格
        rules_all = generate_rules(freq_ap, n, min_conf=min(confs))
        for mc in confs:
            sel = [r for r in rules_all if r.confidence >= mc]
            n_lift = sum(1 for r in sel if r.lift > 1)
            rules_summary.append({
                "min_sup": min_sup, "min_conf": mc, "rules": len(sel),
                "lift_gt_1": n_lift, "lift_le_1": len(sel) - n_lift,
                "avg_conf": round(sum(r.confidence for r in sel) / len(sel), 4) if sel else 0.0,
                "avg_lift": round(sum(r.lift for r in sel) / len(sel), 4) if sel else 0.0,
                "max_lift": round(max((r.lift for r in sel), default=0.0), 4),
            })
            write_csv(f"{OUT}/rules_sup{min_sup * 1000:.0f}permille_conf{int(mc * 100)}.csv",
                      ["antecedent", "consequent", "support", "confidence", "lift"],
                      [(fmt_itemset(r.antecedent), fmt_itemset(r.consequent),
                        f"{r.support:.6f}", f"{r.confidence:.6f}", f"{r.lift:.6f}")
                       for r in sel])

    # ---------------- 主分析支持度 ---------------- #
    main_freq, _ = apriori(txns, args.main_sup)
    main_flat = {}
    for k in range(1, len(main_freq)):
        main_flat.update(main_freq[k])
    main_rules = generate_rules(main_freq, n, min_conf=min(confs))
    print(f"\nmain analysis: min_sup={args.main_sup} itemsets={len(main_flat)} "
          f"rules(conf>={min(confs)})={len(main_rules)}")

    # ---------------- CSV 汇总 ---------------- #
    write_csv(f"{OUT}/summary_timing.csv", list(summary[0].keys()),
              [[r[k] for k in summary[0]] for r in summary])
    write_csv(f"{OUT}/rules_summary.csv", list(rules_summary[0].keys()),
              [[r[k] for k in rules_summary[0]] for r in rules_summary])
    write_csv(f"{OUT}/apriori_levels.csv", ["min_sup", "k", "candidates", "frequent", "seconds"],
              [[r["min_sup"], r["k"], r["candidates"], r["frequent"], r["seconds"]]
               for r in level_rows])
    with open(f"{OUT}/dataset_stats.json", "w", encoding="utf-8") as f:
        json.dump(dataset_stats, f, ensure_ascii=False, indent=2)
    print(f"  wrote {OUT}/dataset_stats.json")

    # ---------------- 图 ---------------- #
    plot_time(timing_curves)
    plot_itemsets_by_k(itemsets_by_k)
    plot_rules_vs_conf(rules_summary)
    plot_top_lift(main_rules, args.main_sup)
    plot_support_hist(flat_itemsets(main_freq, n), n, args.main_sup)
    plot_pruning(level_rows, sups)
    plot_tree_stats(summary)


# ------------------------------------------------------------------ 绘图函数 #
def _style(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=13, pad=12)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.set_facecolor("#fbfbfb")


def plot_time(curves) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for algo, color, label in (("apriori", C_AP, "Apriori"),
                               ("fp_growth", C_FP, "FP-growth")):
        xs = [100 * s for s, _ in curves[algo]]
        ys = [max(t, 1e-3) for _, t in curves[algo]]
        ax.plot(xs, ys, "o-", color=color, label=label, linewidth=2, markersize=7)
        for x, y in zip(xs, ys):
            ax.annotate(f"{y:.2f}s", (x, y), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=8, color=color)
    ax.set_yscale("log")
    ax.set_xticks([100 * s for s, _ in curves["apriori"]])
    ax.margins(y=0.18)                       # 给顶部数值标签留白
    _style(ax, "运行耗时随最小支持度的变化", "最小支持度 min_sup (%)", "耗时 (秒，对数轴)")
    ax.legend()
    fig.savefig(f"{OUT}/fig1_support_vs_time.png")
    plt.close(fig)
    print(f"  wrote {OUT}/fig1_support_vs_time.png")


def plot_itemsets_by_k(by_k_map) -> None:
    sups = sorted(by_k_map, reverse=True)
    ks = sorted({k for m in by_k_map.values() for k in m})
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    bottom = [0] * len(sups)
    cmap = plt.get_cmap("tab10")
    for idx, k in enumerate(ks):
        vals = [by_k_map[s].get(k, 0) for s in sups]
        ax.bar([f"{100 * s:g}%" for s in sups], vals, bottom=bottom,
               label=f"{k}-项集", color=cmap(idx % 10), edgecolor="white", linewidth=0.4)
        bottom = [b + v for b, v in zip(bottom, vals)]
    for i, total in enumerate(bottom):
        ax.text(i, total, str(total), ha="center", va="bottom", fontsize=10)
    ax.margins(y=0.10)
    _style(ax, "不同支持度下频繁项集的数量与阶数分布", "最小支持度", "频繁项集数")
    ax.legend(title="项集阶数 k", fontsize=8, title_fontsize=9, ncol=4,
              loc="upper center", bbox_to_anchor=(0.5, -0.13), frameon=False)
    fig.savefig(f"{OUT}/fig2_itemsets_by_k.png")
    plt.close(fig)
    print(f"  wrote {OUT}/fig2_itemsets_by_k.png")
    fig.savefig(f"{OUT}/fig2_itemsets_by_k.png")
    plt.close(fig)
    print(f"  wrote {OUT}/fig2_itemsets_by_k.png")


def plot_rules_vs_conf(rules_summary) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    sups = sorted({r["min_sup"] for r in rules_summary}, reverse=True)
    palette = ["#c0392b", "#2471a3", "#48c9b0", "#f5b041"]
    for i, s in enumerate(sups):
        rows = [r for r in rules_summary if r["min_sup"] == s]
        ax.plot([100 * r["min_conf"] for r in rows], [r["rules"] for r in rows],
                "o-", color=palette[i % len(palette)], label=f"min_sup={100 * s:g}%")
    ax.set_yscale("symlog")
    _style(ax, "关联规则数量随最小置信度的变化", "最小置信度 min_conf (%)", "规则数（符号对数轴）")
    ax.legend()
    fig.savefig(f"{OUT}/fig3_rules_vs_conf.png")
    plt.close(fig)
    print(f"  wrote {OUT}/fig3_rules_vs_conf.png")


def plot_top_lift(rules, main_sup: float, top: int = 15) -> None:
    sel = [r for r in rules if r.lift > 1][:top] or rules[:top]
    fig, ax = plt.subplots(figsize=(8.4, 5.6))
    labels = [f"{fmt_itemset(r.antecedent)} → {fmt_itemset(r.consequent)}" for r in sel]
    lifts = [r.lift for r in sel]
    y = range(len(sel))
    bars = ax.barh(list(y), lifts, color=C_FP, alpha=0.85)
    for b, r in zip(bars, sel):
        ax.text(b.get_width(), b.get_y() + b.get_height() / 2,
                f" lift={r.lift:.2f} conf={r.confidence:.2f}", va="center", fontsize=8)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.axvline(1.0, color=C_GREY, linestyle="--", linewidth=1)
    _style(ax, f"提升度最高的 {len(sel)} 条关联规则（min_sup={100 * main_sup:g}%，"
               f"min_conf≥50%）", "提升度 lift", "")
    fig.savefig(f"{OUT}/fig4_top_lift_rules.png")
    plt.close(fig)
    print(f"  wrote {OUT}/fig4_top_lift_rules.png")


def plot_support_hist(flat, n: int, main_sup: float) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ratios = [s for _, s in flat]
    ax.hist(ratios, bins=30, color=C_FP, alpha=0.85, edgecolor="white")
    ax.axvline(main_sup, color=C_AP, linestyle="--", linewidth=1.5,
               label=f"min_sup={100 * main_sup:g}%")
    med = sorted(ratios)[len(ratios) // 2]
    ax.axvline(med, color=C_GREY, linestyle=":", linewidth=1.5,
               label=f"中位数={100 * med:.2f}%")
    _style(ax, "频繁项集的支持度分布", "支持度 (%)", "项集数")
    ax.xaxis.set_major_formatter(lambda v, _: f"{100 * v:g}")
    ax.legend()
    fig.savefig(f"{OUT}/fig5_support_hist.png")
    plt.close(fig)
    print(f"  wrote {OUT}/fig5_support_hist.png")


def plot_pruning(level_rows, sups) -> None:
    """分面展示：每个支持度档位内，各轮候选数 vs 剪枝后频繁数，并标注保留率."""
    sups_sorted = sorted(sups)
    cols = 2
    n_rows = (len(sups_sorted) + cols - 1) // cols
    fig, axes = plt.subplots(n_rows, cols, figsize=(9.4, 3.5 * n_rows), squeeze=False)
    for idx, s in enumerate(sups_sorted):
        ax = axes[idx // cols][idx % cols]
        rs = [r for r in level_rows if r["min_sup"] == s]
        ks = [r["k"] for r in rs]
        ax.bar([k - 0.2 for k in ks], [r["candidates"] for r in rs], width=0.4,
               color=C_GREY, alpha=0.8, label="候选数 C(k)")
        ax.bar([k + 0.2 for k in ks], [r["frequent"] for r in rs], width=0.4,
               color=C_AP, alpha=0.9, label="频繁项集数 L(k)")
        ax.set_yscale("log")
        ax.set_xticks(ks)
        ax.set_xlabel("项集阶数 k", fontsize=9)
        ax.set_ylabel("数量（对数轴）", fontsize=9)
        ax.set_title(f"min_sup = {100 * s:g}%", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.set_facecolor("#fbfbfb")
        tot_c = sum(r["candidates"] for r in rs)
        tot_f = sum(r["frequent"] for r in rs)
        ax.text(0.98, 0.95,
                f"候选合计 {tot_c:,}\n频繁合计 {tot_f:,}\n保留率 {100 * tot_f / tot_c:.1f}%",
                transform=ax.transAxes, ha="right", va="top", fontsize=8, color="#2c3e50")
        ax.legend(fontsize=8, loc="lower left")
    for j in range(len(sups_sorted), n_rows * cols):
        axes[j // cols][j % cols].set_visible(False)
    fig.suptitle("Apriori 各轮候选数与被剪枝后的频繁项集数", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(f"{OUT}/fig6_candidate_vs_frequent.png")
    plt.close(fig)
    print(f"  wrote {OUT}/fig6_candidate_vs_frequent.png")


def plot_tree_stats(summary) -> None:
    fig, ax1 = plt.subplots(figsize=(7.2, 4.4))
    xs = [100 * r["min_sup"] for r in summary]
    ax1.plot(xs, [r["fp_tree_nodes"] for r in summary], "o-", color=C_FP,
             linewidth=2, label="FP-tree 节点数")
    for x, r in zip(xs, summary):
        ax1.annotate(f'{r["fp_tree_nodes"]}', (x, r["fp_tree_nodes"]),
                     textcoords="offset points", xytext=(0, 9), ha="center",
                     fontsize=8, color=C_FP)
    ax1.set_yscale("log")
    _style(ax1, "FP-tree 规模（节点数 / 深度）随支持度的变化", "最小支持度 min_sup (%)", "节点数（对数轴）")
    ax2 = ax1.twinx()
    ax2.plot(xs, [r["fp_tree_depth"] for r in summary], "s--", color="#e67e22",
             linewidth=1.6, label="树深")
    ax2.set_ylabel("树深（层）", fontsize=11)
    ax1.set_xticks(xs)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    fig.savefig(f"{OUT}/fig7_fptree_stats.png")
    plt.close(fig)
    print(f"  wrote {OUT}/fig7_fptree_stats.png")


if __name__ == "__main__":
    main()
