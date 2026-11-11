# -*- coding: utf-8 -*-
"""Generate LaTeX table bodies whose contents are experimental values.

Rationale (skill `latex-paper-delivery-guards` S5): numbers inside generated
tables are the most "authoritative-looking" but least likely to be updated.
So the table bodies are emitted here from the result CSVs, never typed by hand.
"""

from __future__ import annotations

import csv
from pathlib import Path

PROJ = Path(__file__).resolve().parent
RES = PROJ.parent / "results"
TAB = PROJ / "tables"


def read_csv(name: str) -> list[dict]:
    with open(RES / name, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    TAB.mkdir(exist_ok=True)
    levels = read_csv("apriori_levels.csv")
    robust = read_csv("rule_robustness.csv")

    # ---- per-round candidate / frequent counts for the two key levels ----
    rows = []
    for tag, sup in (("B", 0.01), ("D", 0.002)):
        for r in levels:
            if abs(float(r["min_sup"]) - sup) > 1e-9:
                continue
            rows.append(
                f"{tag} & {r['k']} & {int(r['candidates']):,} & {int(r['frequent']):,} & "
                f"{float(r['seconds']):.3f} \\\\")
    body = "\n".join(rows)
    env = ("\n".join([
        "\\begin{tabular}{@{}llrrr@{}}",
        "\\toprule",
        "Level & $k$ & Candidates & Retained & Time (s)\\\\",
        "\\midrule",
        body,
        "\\bottomrule",
        "\\end{tabular}",
    ])) + "\n"
    (TAB / "tab_levels.tex").write_text(env, encoding="utf-8")
    print(f"wrote tables/tab_levels.tex ({len(rows)} rows)")

    # ---- top rules with bootstrap intervals ----
    top = sorted(robust, key=lambda r: -float(r["lift"]))[:8]
    rows = []
    for r in top:
        rows.append(
            f"${r['antecedent'].replace('{', '\\{').replace('}', '\\}')}$ & "
            f"${r['consequent'].replace('{', '\\{').replace('}', '\\}')}$ & "
            f"{float(r['support']):.4f} & {float(r['confidence']):.2f} & "
            f"[{float(r['conf_lo90']):.2f}, {float(r['conf_hi90']):.2f}] & "
            f"{float(r['lift']):.1f} & "
            f"[{float(r['lift_lo90']):.1f}, {float(r['lift_hi90']):.1f}] \\\\")
    body = "\n".join(rows)
    env = ("\n".join([
        "\\begin{tabular}{@{}ccrccrc@{}}",
        "\\toprule",
        "$Y$ & $X$ & supp & conf & 90\\% CI & lift & 90\\% CI\\\\",
        "\\midrule",
        body,
        "\\bottomrule",
        "\\end{tabular}",
    ])) + "\n"
    (TAB / "tab_rules.tex").write_text(env, encoding="utf-8")
    print(f"wrote tables/tab_rules.tex ({len(rows)} rows)")


if __name__ == "__main__":
    main()
