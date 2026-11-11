# -*- coding: utf-8 -*-
"""apriori.py 的正确性验证.

大数据分析与处理 · 实验二
陈纬昕

三层验证，由强到弱：
1. 讲义示例（Apriori to students Homework Readme.pdf 第 2-3 页）逐轮断言：
   D = {1,3,4}, {2,3,5}, {1,2,3,5}, {2,5}，min_sup = 50%
   → L1 = {1,2,3,5}, L2 = {1,3},{2,3},{2,5},{3,5}, L3 = {2,3,5}
2. 小规模随机数据与「暴力全枚举」对拍（穷举全部 2^n-1 个子集分别计数）。
3. 性质测试：反单调性（频繁项集的子集必频繁）、规则置信度/lift 独立复算。

运行：python test_apriori.py
"""

from __future__ import annotations

import itertools
import random
import unittest
from pathlib import Path

from apriori import (apriori, apriori_gen, generate_rules, flat_itemsets,
                     load_transactions)

# 真实数据集路径（基于脚本位置解析，不依赖调用时的工作目录）
DATA_CSV = str(Path(__file__).resolve().parent.parent / "data" / "array_data.csv")

# 讲义示例数据（Readme 第 2 页）
TEXTBOOK_D = [(1, 3, 4), (2, 3, 5), (1, 2, 3, 5), (2, 5)]


class TestTextbookExample(unittest.TestCase):
    """与讲义逐轮结果对齐——这是判断实现是否忠于定义的第一道关."""

    def setUp(self):
        self.freq, self.stats = apriori(TEXTBOOK_D, min_sup=0.5)

    def test_L1(self):
        L1 = set(self.freq[1])
        self.assertEqual(L1, {(1,), (2,), (3,), (5,)})
        # item 4 支持度 25% 应被剪掉
        self.assertNotIn((4,), L1)

    def test_L2(self):
        L2 = set(self.freq[2])
        self.assertEqual(L2, {(1, 3), (2, 3), (2, 5), (3, 5)})
        # {1,2} 与 {1,5} 支持度 25%，应被剪掉
        self.assertNotIn((1, 2), L2)
        self.assertNotIn((1, 5), L2)

    def test_L3_and_termination(self):
        self.assertEqual(set(self.freq[3]), {(2, 3, 5)})
        # 无法生成 4 项候选，迭代应在此终止
        self.assertEqual(len(self.freq), 4)          # freq[0] + L1..L3

    def test_supports(self):
        self.assertAlmostEqual(self.freq[1][(2,)], 3)
        self.assertAlmostEqual(self.freq[2][(1, 3)], 2)
        self.assertAlmostEqual(self.freq[3][(2, 3, 5)], 2)

    def test_rules_direction_and_values(self):
        """Conf(X<=Y)=supp(XY)/supp(Y)：前件是条件，后件是结论."""
        rules = generate_rules(self.freq, len(TEXTBOOK_D), min_conf=0.5)
        d = {(r.antecedent, r.consequent): r for r in rules}
        # 讲义第 3 页：买 1 的人必买 3（conf=1），买 3 的人只有 2/3 买 1
        self.assertIn(((1,), (3,)), d)               # 1 => 3
        self.assertAlmostEqual(d[((1,), (3,))].confidence, 1.0)
        self.assertIn(((3,), (1,)), d)               # 3 => 1
        self.assertAlmostEqual(d[((3,), (1,))].confidence, 2 / 3)
        # {2,5} 支持度 3/4=0.75，supp(2)=0.75, supp(5)=0.75 → conf=1, lift=4/3
        r25 = d[((2,), (5,))]
        self.assertAlmostEqual(r25.support, 0.75)
        self.assertAlmostEqual(r25.confidence, 1.0)
        self.assertAlmostEqual(r25.lift, 0.75 / 0.75 / 0.75)


class TestAgainstBruteForce(unittest.TestCase):
    """暴力法：枚举全部非空子集，直接数事务包含数。作为独立参照实现."""

    @staticmethod
    def brute_force(transactions, min_sup):
        n = len(transactions)
        txn_sets = [set(t) for t in transactions]
        universe = sorted({i for t in transactions for i in t})
        result = {}
        for r in range(1, len(universe) + 1):
            for combo in itertools.combinations(universe, r):
                cs = set(combo)
                cnt = sum(1 for ts in txn_sets if cs <= ts)
                if cnt / n >= min_sup:
                    result[combo] = cnt
        return result

    def test_random_datasets(self):
        rng = random.Random(20261028)
        for trial in range(30):
            n_items = rng.randint(4, 9)
            n_txn = rng.randint(3, 14)
            max_len = rng.randint(2, min(6, n_items))
            txns = []
            for _ in range(n_txn):
                k = rng.randint(1, max_len)
                txns.append(tuple(sorted(rng.sample(range(1, n_items + 1), k))))
            for min_sup in (0.1, 0.25, 0.4, 0.6):
                with self.subTest(trial=trial, min_sup=min_sup):
                    freq, _ = apriori(txns, min_sup)
                    got = {}
                    for k in range(1, len(freq)):
                        got.update(freq[k])
                    want = self.brute_force(txns, min_sup)
                    self.assertEqual(got, want,
                                     msg=f"mismatch trial={trial} min_sup={min_sup}")

    def test_rules_match_definition(self):
        rng = random.Random(7)
        txns = [tuple(sorted(rng.sample(range(1, 8), rng.randint(2, 5))))
                for _ in range(20)]
        freq, _ = apriori(txns, 0.2)
        rules = generate_rules(freq, len(txns), 0.5)
        supp = {c: v / len(txns) for k in range(1, len(freq)) for c, v in freq[k].items()}
        self.assertTrue(rules)
        for r in rules:
            whole = tuple(sorted(r.antecedent + r.consequent))
            self.assertAlmostEqual(r.confidence, supp[whole] / supp[r.antecedent])
            self.assertAlmostEqual(r.lift, r.confidence / supp[r.consequent])
            self.assertGreaterEqual(r.confidence, 0.5)


class TestProperties(unittest.TestCase):
    """反单调性与边界条件."""

    def test_anti_monotonicity(self):
        rng = random.Random(11)
        txns = [tuple(sorted(rng.sample(range(1, 13), rng.randint(3, 7))))
                for _ in range(60)]
        freq, _ = apriori(txns, 0.1)
        for k in range(2, len(freq)):
            for itemset in freq[k]:
                for sub in itertools.combinations(itemset, k - 1):
                    self.assertIn(sub, freq[k - 1],
                                  msg=f"{itemset} 的子集 {sub} 不频繁，违反反单调性")

    def test_gen_prunes_correctly(self):
        """apriori_gen 不应生成含非频繁子集的候选."""
        L2 = [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4)]   # (3,4) 缺失
        c3 = apriori_gen(L2, 3)
        self.assertIn((1, 2, 3), c3)
        self.assertIn((1, 2, 4), c3)
        self.assertNotIn((1, 3, 4), c3)    # 子集 (3,4) 非频繁 → 剪掉
        self.assertNotIn((2, 3, 4), c3)    # 子集 (3,4) 非频繁 → 剪掉

    def test_min_sup_boundary(self):
        """count/n >= min_sup 的边界：4 事务 min_sup=0.5 时应保留 count=2 的项."""
        freq, _ = apriori(TEXTBOOK_D, 0.5)
        self.assertEqual(freq[2][(1, 3)], 2)
        # min_sup 略高于 0.5 时 count=2（占比 0.5）应被剔除
        freq2, _ = apriori(TEXTBOOK_D, 0.51)
        self.assertNotIn((1, 3), freq2[2])

    def test_empty_and_single(self):
        freq, _ = apriori([(5,)], 0.5)
        self.assertEqual(set(freq[1]), {(5,)})
        self.assertEqual(len(freq), 2)
        rules = generate_rules(freq, 1, 0.5)
        self.assertEqual(rules, [])

    def test_invalid_params(self):
        with self.assertRaises(ValueError):
            apriori(TEXTBOOK_D, 0)
        with self.assertRaises(ValueError):
            generate_rules(apriori(TEXTBOOK_D, 0.5)[0], 4, 1.5)


class TestRealData(unittest.TestCase):
    """真实数据的基本健全性（不锁定数值结果，只查结构性质）."""

    def test_array_data(self):
        txns = load_transactions(DATA_CSV)
        self.assertEqual(len(txns), 1892)
        self.assertTrue(all(t == tuple(sorted(t)) for t in txns))
        self.assertTrue(all(len(set(t)) == len(t) for t in txns))
        freq, _ = apriori(txns, 0.01)
        total = sum(len(freq[k]) for k in range(1, len(freq)))
        self.assertEqual(total, 335)       # 单线程纯 Python 基线实测值
        self.assertEqual(len(freq), 4)     # 最大频繁项集阶数 k = 3


if __name__ == "__main__":
    unittest.main(verbosity=2)
