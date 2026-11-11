# -*- coding: utf-8 -*-
"""fp_growth.py 的验证：树结构自检 + 与 Apriori 输出逐一比对.

大数据分析与处理 · 实验二
陈纬昕

三个层次的检验：
1. 课件示例（PPT 第 42 页，min_support = 3）建树守恒：节点计数之和 =
   过滤后事务长度之和；前缀确实被共享（节点数 < 总计数）。
2. 与 Apriori 在同一数据、同一 min_sup 下，频繁项集集合与支持计数
   必须逐个相等——两套独立实现互证。
3. 真实数据 array_data.csv 上的一致性（2% / 1% / 0.5%）。

运行：python test_fp_growth.py
"""

from __future__ import annotations

import random
import unittest
from pathlib import Path

from apriori import apriori, load_transactions
from fp_growth import FPTree, fp_growth, summarize_by_k

# 真实数据集路径（基于脚本位置解析，不依赖调用时的工作目录）
DATA_CSV = str(Path(__file__).resolve().parent.parent / "data" / "array_data.csv")

# 课件第 42 页（Ch04 PatternMiningBasic, slide 42）示例
SLIDE_D = [
    ("f", "a", "c", "d", "g", "i", "m", "p"),
    ("a", "b", "c", "f", "l", "m", "o"),
    ("b", "f", "h", "j", "o", "w"),
    ("b", "c", "k", "s", "p"),
    ("a", "f", "c", "e", "l", "p", "m", "n"),
]


def _slide_txns():
    """把课件里的字母 item 映射成整数，复用同一套整数实现."""
    mapping = {ch: idx + 1 for idx, ch in enumerate(sorted({c for t in SLIDE_D for c in t}))}
    return [tuple(sorted(mapping[c] for c in t)) for t in SLIDE_D], mapping


class TestSlideExample(unittest.TestCase):
    """课件 min_support = 3 的示例：单项频次与树守恒."""

    def setUp(self):
        self.txns, self.mapping = _slide_txns()
        # min_support=3 / 5 条事务 = 0.6
        self.itemsets, self.tree = fp_growth(self.txns, 3 / 5)
        self.inv = {v: k for k, v in self.mapping.items()}

    def test_frequent_singletons(self):
        """f:4, a:3, c:4, b:3, m:3, p:3（课件标红的 F-list: f-c-a-b-m-p）."""
        singles = {self.inv[s[0]]: c for s, c in self.itemsets.items() if len(s) == 1}
        self.assertEqual(singles, {"f": 4, "a": 3, "c": 4, "b": 3, "m": 3, "p": 3})

    def test_count_conservation(self):
        """节点计数之和 = 各事务过滤后长度之和（建树不丢不重）."""
        expected = 0
        for t in self.txns:
            expected += sum(1 for i in t if (i,) in self.itemsets)
        self.assertEqual(self.tree.total_counts(), expected)
        self.assertEqual(expected, 20)

    def test_prefix_sharing(self):
        """五个事务只有 20 个 item 位，共享前缀后节点数应严格少于 20."""
        self.assertLess(self.tree.node_count(), 20)
        self.assertGreaterEqual(self.tree.depth(), 3)

    def test_root_children(self):
        """root 下只应有 f（4 条含 f）与 c（1 条不含 f 的 {b,c,k,s,p}）."""
        roots = {self.inv[node.name] for node in self.tree.root.children.values()}
        self.assertEqual(roots, {"f", "c"})


class TestAgainstApriori(unittest.TestCase):
    """两套独立实现互证：频繁项集与计数必须完全一致."""

    @staticmethod
    def _as_dict(freq_by_k):
        out = {}
        for k in range(1, len(freq_by_k)):
            out.update(freq_by_k[k])
        return out

    def test_random_datasets(self):
        rng = random.Random(20261028)
        for trial in range(25):
            n_items = rng.randint(4, 9)
            n_txn = rng.randint(3, 14)
            txns = [tuple(sorted(rng.sample(range(1, n_items + 1),
                                            rng.randint(1, min(6, n_items)))))
                    for _ in range(n_txn)]
            for min_sup in (0.1, 0.25, 0.4, 0.6):
                with self.subTest(trial=trial, min_sup=min_sup):
                    ap, _ = apriori(txns, min_sup)
                    fp, _ = fp_growth(txns, min_sup)
                    self.assertEqual(self._as_dict(ap), fp)

    def test_real_data_three_levels(self):
        txns = load_transactions(DATA_CSV)
        for min_sup in (0.02, 0.01, 0.005):
            with self.subTest(min_sup=min_sup):
                ap, _ = apriori(txns, min_sup)
                fp, tree = fp_growth(txns, min_sup)
                self.assertEqual(self._as_dict(ap), fp)
                # 附带记录规模，供报告引用
                print(f"min_sup={min_sup}: itemsets={len(fp)} "
                      f"by_k={summarize_by_k(fp)} tree_nodes={tree.node_count()} "
                      f"depth={tree.depth()}")


class TestEdgeCases(unittest.TestCase):
    def test_single_item_transactions(self):
        txns = [(1,), (1,), (2,)]
        itemsets, _ = fp_growth(txns, 0.5)
        self.assertEqual(itemsets, {(1,): 2})

    def test_no_frequent(self):
        itemsets, tree = fp_growth([(1, 2), (3, 4)], 0.9)
        self.assertEqual(itemsets, {})
        self.assertEqual(tree.node_count(), 0)

    def test_all_identical(self):
        txns = [(1, 2, 3)] * 5
        itemsets, _ = fp_growth(txns, 1.0)
        self.assertEqual(len(itemsets), 7)       # 3 单 + 3 双 + 1 三 = 2^3-1
        for c in itemsets.values():
            self.assertEqual(c, 5)

    def test_invalid(self):
        with self.assertRaises(ValueError):
            fp_growth([(1,)], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
