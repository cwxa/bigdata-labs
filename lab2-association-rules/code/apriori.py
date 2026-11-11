# -*- coding: utf-8 -*-
"""Apriori 频繁项集挖掘与关联规则生成.

大数据分析与处理 · 实验二
陈纬昕

实现要点
--------
1. 项集内部表示统一为 ``tuple(sorted(items))``：既是可哈希键，又便于按字典序
   做 k-1 项集自连接（Apriori-gen）。
2. 支持度以「事务数占比」计：supp(X) = count(X) / |D|，最小支持度阈值换算为
   最小计数 min_count = ceil(min_sup * |D|)。
3. 支持度计数提供两种策略，按估算代价自动选择：
   - SUBSET：逐事务枚举其 k-子集，只在候选哈希表里累加（稀疏、候选多时优）；
   - SCAN  ：遍历候选，逐事务判断是否包含（候选少、事务长时优）。
4. 规则方向严格遵循讲义定义 Conf(X <= Y) = P(X|Y) = supp(XY)/supp(Y)，
   即 Y 为前件（条件）、X 为后件（结论）。多数第三方库方向相反，不可混用。
"""

from __future__ import annotations

import csv
import itertools
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Set, Tuple

Itemset = Tuple[int, ...]


# --------------------------------------------------------------------------- #
# 数据读入
# --------------------------------------------------------------------------- #
def load_transactions(path: str) -> List[Itemset]:
    """读取每行一条事务的 CSV（逗号分隔的整数 item id），返回去重排序后的事务集合.

    - 行内重复 item 去重（本项目数据无重复，但保证一般性）；
    - 空行跳过；
    - 返回每条事务为已排序 tuple，保证后续子集枚举稳定可复现。
    """
    transactions: List[Itemset] = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.reader(f):
            items = {int(x) for x in row if x.strip() != ""}
            if items:
                transactions.append(tuple(sorted(items)))
    if not transactions:
        raise ValueError(f"no transaction parsed from {path}")
    return transactions


# --------------------------------------------------------------------------- #
# 候选生成：Apriori-gen（自连接 + 子集剪枝）
# --------------------------------------------------------------------------- #
def apriori_gen(prev_freq: Sequence[Itemset], k: int) -> Set[Itemset]:
    """由 L(k-1) 生成 C(k)：前 k-2 项相同的两个项集连接，再剪掉含非频繁 k-1 子集者.

    反单调性：若某 k 项集频繁，则其全部 (k-1) 子集必频繁。故任一 (k-1) 子集
    不在 L(k-1) 中的候选可直接删除，无需扫描数据库即可判定其不频繁。
    """
    prev_freq = sorted(prev_freq)          # 自连接必须依赖字典序，避免漏配
    prev_set = set(prev_freq)
    n = len(prev_freq)
    candidates: Set[Itemset] = set()
    for i in range(n):
        a = prev_freq[i]
        for j in range(i + 1, n):
            b = prev_freq[j]
            # 要求前 k-2 项相同，末项不同 —— 这样的两项集并集恰好 k 项
            if a[:-1] != b[:-1]:
                continue
            cand = a + (b[-1],)
            # 子集剪枝：检查全部 k 个 (k-1)-子集是否都频繁
            if all(cand[:t] + cand[t + 1:] in prev_set for t in range(k)):
                candidates.add(cand)
    return candidates


def _count_subsets(transactions: Sequence[Itemset], candidates: Set[Itemset],
                   k: int) -> Dict[Itemset, int]:
    """SUBSET 策略：对每条事务枚举其 k-子集，命中候选则计数."""
    counts: Dict[Itemset, int] = {}
    get = counts.get
    for t in transactions:
        if len(t) < k:
            continue
        for c in itertools.combinations(t, k):
            if c in candidates:            # 哈希查找，O(1) 期望
                counts[c] = get(c, 0) + 1
    return counts


def _count_scan(transactions: Sequence[Itemset], candidates: Set[Itemset]
                ) -> Dict[Itemset, int]:
    """SCAN 策略：遍历候选，逐事务判断包含关系."""
    txn_sets = [set(t) for t in transactions]
    counts: Dict[Itemset, int] = {}
    for c in candidates:
        cs = set(c)
        counts[c] = sum(1 for ts in txn_sets if cs <= ts)
    return counts


def _choose_strategy(transactions: Sequence[Itemset], candidates: Set[Itemset],
                     k: int) -> str:
    """按操作次数粗估选择计数策略（常数因子不精算，只做量级判断）."""
    subset_ops = sum(math.comb(len(t), k) for t in transactions if len(t) >= k)
    scan_ops = len(candidates) * len(transactions)
    return "SUBSET" if subset_ops <= scan_ops else "SCAN"


# --------------------------------------------------------------------------- #
# Apriori 主循环
# --------------------------------------------------------------------------- #
@dataclass
class LevelStat:
    """记录每一轮迭代的候选数、剪枝数与频繁项集数（用于报告中的剪枝效果分析）."""
    k: int
    candidates: int
    frequent: int
    seconds: float


def apriori(transactions: Sequence[Itemset], min_sup: float,
            keep_stats: bool = True) -> Tuple[List[Dict[Itemset, int]], List[LevelStat]]:
    """挖掘全部频繁项集.

    参数
    ----
    transactions : 事务集合，每条为已排序 tuple
    min_sup      : 最小支持度（占比，0 < min_sup <= 1）
    keep_stats   : 是否收集每轮统计

    返回
    ----
    freq_by_k : freq_by_k[k] 为 k 项频繁项集 → 支持计数；freq_by_k[0] 恒为 {}
    stats     : 每轮候选数/频繁数/耗时
    """
    if not 0 < min_sup <= 1:
        raise ValueError("min_sup must be in (0, 1]")

    n = len(transactions)
    # 支持计数阈值：supp(X) >= min_sup  <=>  count(X) >= ceil(min_sup * n)
    min_count = max(1, math.ceil(min_sup * n - 1e-9))

    freq_by_k: List[Dict[Itemset, int]] = [{}]
    stats: List[LevelStat] = []

    # --- k = 1：直接扫描一次数据库 ---
    import time
    t0 = time.perf_counter()
    counts1: Dict[Itemset, int] = {}
    get = counts1.get
    for t in transactions:
        for item in t:
            counts1[(item,)] = get((item,), 0) + 1
    L1 = {c: v for c, v in counts1.items() if v >= min_count}
    freq_by_k.append(L1)
    if keep_stats:
        stats.append(LevelStat(1, len(counts1), len(L1), time.perf_counter() - t0))
    if not L1:
        return freq_by_k, stats

    prev = sorted(L1)                      # 排序保证输出与运行完全可复现
    k = 2
    while prev:
        t0 = time.perf_counter()
        candidates = apriori_gen(prev, k)
        if not candidates:
            break
        if keep_stats:
            strategy = _choose_strategy(transactions, candidates, k)
        else:
            strategy = "SUBSET"
        counts = (_count_subsets(transactions, candidates, k) if strategy == "SUBSET"
                  else _count_scan(transactions, candidates))
        cur = {c: v for c, v in counts.items() if v >= min_count}
        if keep_stats:
            stats.append(LevelStat(k, len(candidates), len(cur), time.perf_counter() - t0))
        if not cur:
            break
        freq_by_k.append(cur)
        prev = sorted(cur)
        k += 1

    return freq_by_k, stats


# --------------------------------------------------------------------------- #
# 关联规则
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Rule:
    """一条关联规则 antecedent => consequent.

    对应讲义记号 Consequent <= Antecedent，即 confidence = P(consequent|antecedent)。
    """
    antecedent: Itemset
    consequent: Itemset
    support: float        # supp(A ∪ C)
    confidence: float     # supp(A ∪ C) / supp(A)
    lift: float           # confidence / supp(C)

    @property
    def key(self) -> Tuple[Itemset, Itemset]:
        return (self.antecedent, self.consequent)


def generate_rules(freq_by_k: List[Dict[Itemset, int]], n: int,
                   min_conf: float) -> List[Rule]:
    """从频繁项集导出满足最小置信度的关联规则.

    仅枚举非空真子集作为前件，避免 A => ∅ / ∅ => A 这类空洞规则。
    lift > 1 表示前件与后件正相关（讲义判据：有效强关联规则）。
    """
    if not 0 < min_conf <= 1:
        raise ValueError("min_conf must be in (0, 1]")

    # 支持度查找表：项集 → supp
    supp: Dict[Itemset, float] = {}
    for k in range(1, len(freq_by_k)):
        for itemset, cnt in freq_by_k[k].items():
            supp[itemset] = cnt / n

    rules: List[Rule] = []
    for k in range(2, len(freq_by_k)):
        for itemset in freq_by_k[k]:
            s_all = supp[itemset]
            items = list(itemset)
            for r in range(1, len(items)):          # 前件大小 1..k-1
                for ante in itertools.combinations(items, r):
                    cons = tuple(x for x in items if x not in ante)
                    s_ante = supp.get(ante)
                    if s_ante is None:              # 理论上不会发生（子集必频繁）
                        continue
                    conf = s_all / s_ante
                    if conf < min_conf:
                        continue
                    s_cons = supp[cons]
                    rules.append(Rule(ante, cons, s_all, conf, conf / s_cons))
    rules.sort(key=lambda r: (-r.lift, -r.confidence, r.antecedent, r.consequent))
    return rules


# --------------------------------------------------------------------------- #
# 便捷封装
# --------------------------------------------------------------------------- #
def mine(transactions: Sequence[Itemset], min_sup: float, min_conf: float
         ) -> Tuple[List[Dict[Itemset, int]], List[Rule], List[LevelStat]]:
    freq_by_k, stats = apriori(transactions, min_sup)
    rules = generate_rules(freq_by_k, len(transactions), min_conf)
    return freq_by_k, rules, stats


def flat_itemsets(freq_by_k: List[Dict[Itemset, int]], n: int) -> List[Tuple[Itemset, float]]:
    """展平为 (项集, 支持度) 列表，便于写表/画图."""
    out: List[Tuple[Itemset, float]] = []
    for k in range(1, len(freq_by_k)):
        for itemset, cnt in freq_by_k[k].items():
            out.append((itemset, cnt / n))
    out.sort(key=lambda x: (len(x[0]), -x[1], x[0]))
    return out


def fmt_itemset(itemset: Iterable[int]) -> str:
    return "{" + ",".join(str(i) for i in itemset) + "}"
