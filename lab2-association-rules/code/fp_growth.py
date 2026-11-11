# -*- coding: utf-8 -*-
"""FP-growth：基于 FP-tree 的频繁项集挖掘（Apriori 的对比算法）.

大数据分析与处理 · 实验二
陈纬昕

与 Apriori 的关键差异
--------------------
Apriori 每增加一个阶数就要完整扫描一次数据库，且候选集可能爆炸；
FP-growth 只扫描两次数据库：第一次统计单项频次，第二次把事务按全局频次
降序压缩进一棵前缀共享树（FP-tree）。之后递归地在「条件模式基」上建
「条件 FP-tree」，不再回读原始数据库，也不生成候选集。

两者输出等价（同一 min_sup 得到同一组频繁项集）。代价从「候选数 × 事务数」
变成「树节点数 × 递归深度」，在稀疏长事务数据上通常更快，代价是建树内存
与实现复杂度。
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

Itemset = Tuple[int, ...]


class FPNode:
    """FP-tree 节点：item 名、计数、父指针、子节点表、同名链指针."""

    __slots__ = ("name", "count", "parent", "children", "link")

    def __init__(self, name: Optional[int], parent: Optional["FPNode"] = None):
        self.name = name
        self.count = 0
        self.parent = parent
        self.children: Dict[int, "FPNode"] = {}
        self.link: Optional["FPNode"] = None     # header table 中同名节点的链指针

    def path_to_root(self) -> List[int]:
        """自身到根的 item 序列（不含 root），叶→根方向."""
        path, p = [], self.parent
        while p is not None and p.name is not None:
            path.append(p.name)
            p = p.parent
        return path


class FPTree:
    """FP-tree：root + header table（item → 支持计数 / 同名节点链）."""

    def __init__(self, min_count: int):
        self.min_count = min_count
        self.root = FPNode(None)
        self.header: Dict[int, int] = {}         # item → 支持计数
        self.head_links: Dict[int, FPNode] = {}  # item → 同名链首节点
        self.order: Dict[int, int] = {}          # item → 全局频次排名

    # -- 构建 -------------------------------------------------------------- #
    def _set_ranking(self, totals: Dict[int, int]) -> None:
        self.header = {i: c for i, c in totals.items() if c >= self.min_count}
        # 支持度降序，同频按 item id 升序 → 构建结果完全确定、可复现
        ranked = sorted(self.header.items(), key=lambda kv: (-kv[1], kv[0]))
        self.order = {item: idx for idx, (item, _) in enumerate(ranked)}

    def _insert(self, items: List[int], count: int = 1) -> None:
        """插入一条已按 self.order 排序的 item 序列（可带计数字段）."""
        node = self.root
        for item in items:
            child = node.children.get(item)
            if child is None:
                child = FPNode(item, node)
                node.children[item] = child
                head = self.head_links.get(item)
                if head is None:
                    self.head_links[item] = child
                else:                            # 挂到同名链尾
                    while head.link is not None:
                        head = head.link
                    head.link = child
            child.count += count
            node = child

    @classmethod
    def from_transactions(cls, transactions: Sequence[Itemset], min_count: int) -> "FPTree":
        """构造顶层 FP-tree：两遍扫描（统计频次 → 排序插入）."""
        tree = cls(min_count)
        totals: Counter = Counter()
        for t in transactions:
            totals.update(t)
        tree._set_ranking(dict(totals))
        for t in transactions:
            items = sorted((i for i in t if i in tree.header), key=tree.order.__getitem__)
            if items:
                tree._insert(items)
        return tree

    @classmethod
    def from_patterns(cls, patterns: Sequence[Tuple[List[int], int]],
                      min_count: int) -> "FPTree":
        """由条件模式基构造条件 FP-tree（每条路径带自己的计数）."""
        tree = cls(min_count)
        totals: Counter = Counter()
        for path, c in patterns:
            for x in path:
                totals[x] += c
        tree._set_ranking(dict(totals))
        for path, c in patterns:
            items = sorted((x for x in path if x in tree.header), key=tree.order.__getitem__)
            if items:
                tree._insert(items, c)
        return tree

    # -- 统计（报告与自检用） ---------------------------------------------- #
    def nodes(self) -> List[FPNode]:
        out, stack = [], list(self.root.children.values())
        while stack:
            node = stack.pop()
            out.append(node)
            stack.extend(node.children.values())
        return out

    def node_count(self) -> int:
        return len(self.nodes())

    def total_counts(self) -> int:
        """全部节点计数之和，应等于「过滤后各事务长度」之和（守恒校验）."""
        return sum(n.count for n in self.nodes())

    def depth(self) -> int:
        def _d(node: FPNode) -> int:
            return 1 + max((_d(c) for c in node.children.values()), default=0)
        return _d(self.root)

    def conditional_patterns(self, item: int) -> List[Tuple[List[int], int]]:
        """item 的条件模式基：(前缀路径, 节点计数) 列表."""
        patterns: List[Tuple[List[int], int]] = []
        node = self.head_links.get(item)
        while node is not None:
            patterns.append((node.path_to_root(), node.count))
            node = node.link
        return patterns


def _mine(tree: FPTree, suffix: Itemset, out: Dict[Itemset, int]) -> None:
    """在给定 FP-tree 上递归挖掘，suffix 为已确定的后缀项集."""
    # 处理顺序取支持度升序（教材做法，使条件模式基更短），不影响结果集合
    for item in sorted(tree.header, key=lambda i: (tree.header[i], i)):
        itemset = tuple(sorted(suffix + (item,)))
        out[itemset] = tree.header[item]        # supp(itemset) = 该项在当前条件树中的计数

        patterns = tree.conditional_patterns(item)
        if not patterns:
            continue
        cond = FPTree.from_patterns(patterns, tree.min_count)
        if cond.header:
            _mine(cond, itemset, out)


def fp_growth(transactions: Sequence[Itemset], min_sup: float
              ) -> Tuple[Dict[Itemset, int], FPTree]:
    """挖掘全部频繁项集.

    返回 (项集 → 支持计数, 顶层 FP-tree（供报告展示结构）)
    """
    if not 0 < min_sup <= 1:
        raise ValueError("min_sup must be in (0, 1]")
    min_count = max(1, math.ceil(min_sup * len(transactions) - 1e-9))
    tree = FPTree.from_transactions(transactions, min_count)
    found: Dict[Itemset, int] = {(i,): c for i, c in tree.header.items()}
    _mine(tree, (), found)
    return found, tree


def summarize_by_k(itemsets: Dict[Itemset, int]) -> Dict[int, int]:
    """各阶频繁项集数量."""
    c: Dict[int, int] = {}
    for s in itemsets:
        c[len(s)] = c.get(len(s), 0) + 1
    return dict(sorted(c.items()))
