# -*- coding: utf-8 -*-
"""关联规则的稳健性评估（bootstrap）.

大数据分析与处理 · 实验二
陈纬昕

问题：min_sup=1% 时最强的那批规则 lift 高达 40，但它们的绝对支持计数只有
约 20/1892。低支持度下 lift 是「小样本比值」，点估计极不稳定，直接当成
强关联结论不严谨。

做法：对 1892 条事务做有放回重采样（bootstrap）B 次，每次重算每条规则的
support / confidence / lift，取 5% 与 95% 分位数作为 90% 区间。
同时统计「置信度下界仍 >= min_conf」的规则数，作为稳健强规则的判据。

产物：
  results/rule_robustness.csv
  results/fig8_rule_robustness.png
"""

from __future__ import annotations

import argparse
import csv
import random
import time
from pathlib import Path
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from apriori import load_transactions, mine, fmt_itemset

# 路径基于脚本位置解析，保证从任意工作目录调用都能命中仓库内数据与产物目录
_HERE = Path(__file__).resolve().parent
DATA = str(_HERE.parent / "data" / "array_data.csv")
OUT = str(_HERE.parent / "results")

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["savefig.dpi"] = 300
matplotlib.rcParams["savefig.bbox"] = "tight"
matplotlib.rcParams["figure.dpi"] = 120


def percentile(sorted_vals: List[float], q: float) -> float:
    """线性插值分位数（不需要 numpy）."""
    if not sorted_vals:
        return float("nan")
    pos = q * (len(sorted_vals) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sup", type=float, default=0.01)
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--B", type=int, default=200, help="bootstrap 重采样次数")
    ap.add_argument("--seed", type=int, default=20261028)
    args = ap.parse_args()

    txns = load_transactions(DATA)
    n = len(txns)
    freq, rules, _ = mine(txns, args.sup, args.conf)
    print(f"min_sup={args.sup}, min_conf={args.conf}: {len(rules)} rules")

    # 需要反复计数的项集：每条规则的前件、后件、并集
    needed: List[Tuple[int, ...]] = sorted(
        {r.antecedent for r in rules} | {r.consequent for r in rules}
        | {tuple(sorted(r.antecedent + r.consequent)) for r in rules})
    needed_sets = [(s, frozenset(s)) for s in needed]
    print(f"distinct itemsets to recount: {len(needed)}")

    txn_sets = [frozenset(t) for t in txns]
    rng = random.Random(args.seed)
    idx_range = range(n)

    # 每个规则索引 → 对应三组计数的位置
    pos = {s: i for i, s in enumerate(needed)}
    rule_pos = [(pos[r.antecedent], pos[r.consequent],
                 pos[tuple(sorted(r.antecedent + r.consequent))]) for r in rules]

    samples: List[List[int]] = [[] for _ in rules]      # conf 样本
    lift_samples: List[List[float]] = [[] for _ in rules]
    t0 = time.perf_counter()
    for b in range(args.B):
        counts = [0] * len(needed)
        for _ in idx_range:                            # 有放回抽样 n 次
            fs = txn_sets[rng.randrange(n)]
            for i, (_, s) in enumerate(needed_sets):
                if s <= fs:
                    counts[i] += 1
        for ri, (ia, ic, iu) in enumerate(rule_pos):
            ca, cc, cu = counts[ia], counts[ic], counts[iu]
            if ca == 0 or cc == 0:                     # 分母为 0，该样本中规则不可定义
                continue
            conf = cu / ca
            samples[ri].append(conf)
            lift_samples[ri].append(conf / (cc / n))
        if (b + 1) % 50 == 0:
            print(f"  bootstrap {b + 1}/{args.B} ({time.perf_counter() - t0:.1f}s)")
    print(f"bootstrap done in {time.perf_counter() - t0:.1f}s")

    rows = []
    for ri, r in enumerate(rules):
        conf_list = sorted(samples[ri])
        lift_list = sorted(lift_samples[ri])
        rows.append({
            "antecedent": fmt_itemset(r.antecedent),
            "consequent": fmt_itemset(r.consequent),
            "support_count": round(r.support * n),
            "support": round(r.support, 6),
            "confidence": round(r.confidence, 4),
            "conf_lo90": round(percentile(conf_list, 0.05), 4),
            "conf_hi90": round(percentile(conf_list, 0.95), 4),
            "lift": round(r.lift, 4),
            "lift_lo90": round(percentile(lift_list, 0.05), 4),
            "lift_hi90": round(percentile(lift_list, 0.95), 4),
            "valid_samples": len(conf_list),
            "robust": percentile(conf_list, 0.05) >= args.conf,
        })

    with open(f"{OUT}/rule_robustness.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {OUT}/rule_robustness.csv ({len(rows)} rows)")

    robust_n = sum(1 for r in rows if r["robust"])
    ci_excludes_1 = sum(1 for r in rows if r["lift_lo90"] > 1)
    print(f"robust rules (conf 下界 >= {args.conf}): {robust_n}/{len(rows)}")
    print(f"rules whose lift 90% CI excludes 1: {ci_excludes_1}/{len(rows)}")

    # ---------------- 图：lift 点估计 + 90% 区间 ---------------- #
    top = sorted(rows, key=lambda r: -r["lift"])[:15]
    fig, ax = plt.subplots(figsize=(8.6, 5.8))
    labels = [f'{r["antecedent"]} → {r["consequent"]}' for r in top]
    y = list(range(len(top)))
    est = [r["lift"] for r in top]
    lo = [max(r["lift"] - r["lift_lo90"], 0) for r in top]
    hi = [max(r["lift_hi90"] - r["lift"], 0) for r in top]
    colors = ["#2471a3" if r["lift_lo90"] > 1 else "#b0b8bf" for r in top]
    ax.barh(y, est, color=colors, alpha=0.85)
    ax.errorbar(est, y, xerr=[lo, hi], fmt="none", ecolor="#2c3e50",
                elinewidth=1.2, capsize=3)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.axvline(1.0, color="#c0392b", linestyle="--", linewidth=1.2, label="lift = 1（独立）")
    ax.set_title(f"提升度最高的 15 条规则及其 90% bootstrap 区间\n"
                 f"（min_sup={100 * args.sup:g}%，B={args.B}；灰色表示区间覆盖 1）",
                 fontsize=12)
    ax.set_xlabel("提升度 lift", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.35, axis="x")
    ax.set_facecolor("#fbfbfb")
    ax.legend(fontsize=9)
    fig.savefig(f"{OUT}/fig8_rule_robustness.png")
    plt.close(fig)
    print(f"  wrote {OUT}/fig8_rule_robustness.png")


if __name__ == "__main__":
    main()
