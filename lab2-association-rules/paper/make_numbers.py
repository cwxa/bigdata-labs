# -*- coding: utf-8 -*-
"""Generate numbers.tex from the experiment outputs (single source of truth).

Every number that appears in the paper body, table captions and figure
captions is emitted here as a macro. The LaTeX source must not contain
literal experimental values -- see skill `latex-paper-delivery-guards` S5.
"""

from __future__ import annotations

import csv
import json
import re
import statistics
from pathlib import Path

PROJ = Path(__file__).resolve().parent
RES = PROJ.parent / "results"

LEVELS = [("A", 0.02), ("B", 0.01), ("C", 0.005), ("D", 0.002)]


def read_csv(name: str) -> list[dict]:
    with open(RES / name, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def fmt(x: float, nd: int = 2) -> str:
    """Fixed decimals; thousands separator for integers is handled by caller."""
    return f"{x:.{nd}f}"


def pct(ratio: float, nd: int = 2) -> str:
    return f"{100 * ratio:.{nd}f}"


def main() -> None:
    stats = json.loads((RES / "dataset_stats.json").read_text(encoding="utf-8"))
    timing = {round(float(r["min_sup"]), 6): r for r in read_csv("summary_timing.csv")}
    rules = read_csv("rules_summary.csv")
    levels = read_csv("apriori_levels.csv")
    robust = read_csv("rule_robustness.csv")

    m: dict[str, str] = {}

    # ---- dataset ----
    m["DSN"] = str(stats["transactions"])
    m["DSItems"] = str(stats["distinct_items"])
    m["DSOcc"] = f"{stats['total_item_occurrences']:,}"
    m["DSAvgLen"] = fmt(stats["avg_length"], 2)
    m["DSMinLen"] = str(stats["min_length"])
    m["DSMaxLen"] = str(stats["max_length"])
    m["DSItemsFive"] = str(stats["items_ratio_ge_5pct"])
    m["DSItemsOne"] = str(stats["items_ratio_ge_1pct"])

    item_freq = read_csv("item_frequency.csv")
    top_ratio = float(item_freq[0]["ratio"])
    m["DSMaxSupp"] = pct(top_ratio, 1)
    m["DSTopItemCount"] = item_freq[0]["count"]

    # ---- per support level ----
    for tag, sup in LEVELS:
        r = timing[round(sup, 6)]
        m[f"Items{tag}"] = f"{int(r['frequent_itemsets']):,}"
        m[f"MaxK{tag}"] = str(r["max_k"])
        m[f"ApTime{tag}"] = fmt(float(r["apriori_seconds"]), 3)
        m[f"FpTime{tag}"] = fmt(float(r["fp_growth_seconds"]), 3)
        m[f"Speedup{tag}"] = fmt(float(r["speedup_fp_over_apriori"]), 2)
        m[f"TreeNodes{tag}"] = f"{int(r['fp_tree_nodes']):,}"
        m[f"TreeDepth{tag}"] = str(r["fp_tree_depth"])
        m[f"MinCount{tag}"] = fmt(sup * stats["transactions"], 2)
        m[f"SupPct{tag}"] = pct(sup, 2).rstrip("0").rstrip(".")
        # candidate totals per level
        cand = sum(int(x["candidates"]) for x in levels
                   if abs(float(x["min_sup"]) - sup) < 1e-9)
        m[f"CandTotal{tag}"] = f"{cand:,}"

    # per-level per-round candidate detail goes to make_tables.py (table bodies)

    # overall keep rate of level D
    m["KeepTotalD"] = fmt(100 * int(timing[round(0.002, 6)]["frequent_itemsets"])
                          / sum(int(x["candidates"]) for x in levels
                                if abs(float(x["min_sup"]) - 0.002) < 1e-9), 2)
    m["CandOverTreeD"] = fmt(sum(int(x["candidates"]) for x in levels
                                 if abs(float(x["min_sup"]) - 0.002) < 1e-9)
                             / int(timing[round(0.002, 6)]["fp_tree_nodes"]), 1)

    # ---- rules ----
    for tag, sup in LEVELS:
        row = next(r for r in rules
                   if abs(float(r["min_sup"]) - sup) < 1e-9
                   and abs(float(r["min_conf"]) - 0.5) < 1e-9)
        m[f"Rules{tag}"] = f"{int(row['rules']):,}"
        m[f"RulesLift{tag}"] = f"{int(row['lift_gt_1']):,}"
        m[f"MaxLift{tag}"] = fmt(float(row["max_lift"]), 2)
    conf_word = {50: "Fifty", 60: "Sixty", 70: "Seventy"}
    for tag, sup in LEVELS:
        for mc, word in conf_word.items():
            row = next(r for r in rules
                       if abs(float(r["min_sup"]) - sup) < 1e-9
                       and abs(float(r["min_conf"]) - mc / 100) < 1e-9)
            m[f"Rules{tag}Conf{word}"] = f"{int(row['rules']):,}"

    # ---- level-wide retention and the k = 2 row (used in the prose) ----
    for tag, sup in LEVELS:
        rows = [x for x in levels if abs(float(x["min_sup"]) - sup) < 1e-9]
        tot_c = sum(int(x["candidates"]) for x in rows)
        keep = 100 * int(timing[round(sup, 6)]["frequent_itemsets"]) / tot_c
        m[f"KeepTotal{tag}"] = fmt(keep, 2)
        k2 = next(x for x in rows if x["k"] == "2")
        m[f"CandTwo{tag}"] = f"{int(k2['candidates']):,}"
        m[f"FreqTwo{tag}"] = f"{int(k2['frequent']):,}"
        m[f"KeepTwo{tag}"] = fmt(100 * int(k2["frequent"]) / int(k2["candidates"]), 2)

    # ---- robustness ----
    n_rules = len(robust)
    n_robust = sum(1 for r in robust if r["robust"] == "True")
    m["RobN"] = str(n_rules)
    m["RobPass"] = str(n_robust)
    m["RobShare"] = fmt(100 * n_robust / n_rules, 1)
    sup_counts = [int(r["support_count"]) for r in robust]
    m["RobSupMin"] = str(min(sup_counts))
    m["RobSupMax"] = str(max(sup_counts))
    m["RobSupMed"] = str(int(statistics.median(sup_counts)))

    widths = [float(r["lift_hi90"]) - float(r["lift_lo90"]) for r in robust]
    m["RobWidthMed"] = fmt(statistics.median(widths), 2)

    top = robust[0]
    m["TopLift"] = fmt(float(top["lift"]), 2)
    m["TopLiftLo"] = fmt(float(top["lift_lo90"]), 2)
    m["TopLiftHi"] = fmt(float(top["lift_hi90"]), 2)
    m["TopConf"] = fmt(float(top["confidence"]), 2)
    m["TopConfLo"] = fmt(float(top["conf_lo90"]), 2)
    m["TopConfHi"] = fmt(float(top["conf_hi90"]), 2)
    rev = next(r for r in robust
               if r["antecedent"] == top["consequent"] and r["consequent"] == top["antecedent"])
    m["RevConf"] = fmt(float(rev["confidence"]), 2)
    m["RevConfLo"] = fmt(float(rev["conf_lo90"]), 2)
    m["RevConfHi"] = fmt(float(rev["conf_hi90"]), 2)
    m["BootstrapB"] = "200"

    # ---- derived: lift lower bound under the chosen thresholds ----
    m["LiftBound"] = fmt(0.5 / top_ratio, 1)

    # ---- environment ----
    m["PyVersion"] = "3.13.14"

    # ---- checks: every macro must be a clean token ----
    bad = [k for k in m if not re.fullmatch(r"[A-Za-z]+", k)]
    if bad:
        raise SystemExit(f"bad macro names: {bad}")

    out = ["% Auto-generated by make_numbers.py -- do not edit by hand."]
    for k in sorted(m, key=lambda s: (len(s), s)):
        out.append(f"\\newcommand{{\\{k}}}{{{m[k]}}}")
    (PROJ / "numbers.tex").write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"wrote numbers.tex ({len(m)} macros)")


if __name__ == "__main__":
    main()
