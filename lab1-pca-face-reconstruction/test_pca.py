# -*- coding: utf-8 -*-
"""pca() 的数学正确性验证.

大数据分析与处理 · 实验一

不依赖 Olivetti 数据集与绘图后端，只用固定种子的合成数据验证四条数学性质：

1. **结构**：主成分正交归一（W·Wᵀ = I_d），特征值严格降序，方差贡献率之和与
   累计方差口径自洽；
2. **双路径等价**：SVD 与「协方差矩阵 + eigh」求出的前 L 个主成分张成同一子空间，
   重构结果逐元素一致（符号翻转不影响 x̂ = mean + Z·W）；
3. **误差口径**：丢弃 L 个主成分后的实测 MSE 等于 (1/d)·Σ_{i>L} λ_i，即实验报告里
   「理论预测」的公式来源；
4. **边界**：L = d 时重构误差趋零；非法 method 抛 ValueError。

运行：python -m unittest test_pca -v
"""

from __future__ import annotations

import unittest

import numpy as np

from pca_face_reconstruction import pca, reconstruct

# 合成数据规模：n > d，保证 SVD（full_matrices=False）能给出 d 个完整主成分
N_SAMPLES = 200
N_FEATURES = 64
L_PRINCIPAL = 20


def make_data(seed: int = 7) -> np.ndarray:
    """构造低秩高斯数据：3 个潜在因子 + 小噪声，模拟"少数主成分解释大部分方差"的场景。"""
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=(N_SAMPLES, 3))
    mixing = rng.normal(size=(3, N_FEATURES))
    noise = rng.normal(scale=0.1, size=(N_SAMPLES, N_FEATURES))
    return latent @ mixing + noise


class TestPCAStructure(unittest.TestCase):
    """主成分的代数结构：正交性、降序、贡献率。"""

    @classmethod
    def setUpClass(cls):
        # 全量分解（L = d）只需做一次，各用例共享
        cls.X = make_data()
        cls.Z, cls.W, cls.evals, cls.mean, cls.ratio = pca(cls.X, N_FEATURES)

    def test_components_orthonormal(self):
        """W 的行向量必须正交且单位长度：W·Wᵀ 应为单位矩阵。"""
        gram = self.W @ self.W.T
        np.testing.assert_allclose(gram, np.eye(N_FEATURES), atol=1e-10)

    def test_eigenvalues_descending(self):
        """特征值按降序排列，且全部非负（协方差矩阵半正定）。"""
        self.assertTrue(np.all(np.diff(self.evals) <= 0), "eigenvalues must be descending")
        self.assertTrue(np.all(self.evals >= -1e-12), "eigenvalues must be non-negative")

    def test_shapes_and_centering(self):
        """降维结果为 (n, L)，且投影系数在样本维度上近似零均值（中心化的直接推论）。"""
        self.assertEqual(self.Z.shape, (N_SAMPLES, N_FEATURES))
        self.assertEqual(self.W.shape, (N_FEATURES, N_FEATURES))
        self.assertEqual(self.mean.shape, (N_FEATURES,))
        np.testing.assert_allclose(self.Z.mean(axis=0), 0.0, atol=1e-10)

    def test_variance_ratio_sums_to_one(self):
        """L = d 时方差贡献率之和为 1；单独校验首项等于 λ1/Σλ。"""
        self.assertAlmostEqual(float(self.ratio.sum()), 1.0, places=10)
        self.assertAlmostEqual(float(self.ratio[0]),
                               float(self.evals[0] / self.evals.sum()), places=12)

    def test_truncated_matches_full_prefix(self):
        """L 截断的结果必须是全量分解的前 L 项（按特征值降序的一致性）。"""
        Z20, W20, ev20, mean20, _ = pca(self.X, L_PRINCIPAL)
        np.testing.assert_allclose(ev20, self.evals[:L_PRINCIPAL], rtol=1e-12)
        np.testing.assert_allclose(np.abs(W20), np.abs(self.W[:L_PRINCIPAL]), atol=1e-12)
        np.testing.assert_allclose(mean20, self.mean, atol=1e-12)
        self.assertEqual(Z20.shape, (N_SAMPLES, L_PRINCIPAL))


class TestReconstructionError(unittest.TestCase):
    """重构误差必须与特征值口径一致——这是实验报告里"理论 vs 观测"的数学依据。"""

    def test_mse_matches_sample_covariance_formula(self):
        """逐像素 MSE 必须严格等于 (n-1)/(n·d)·Σ_{j>L} λ_j（论文 Eq. 1）。

        注意归一化因子：pca() 返回的特征值来自样本协方差（除以 n-1），而逐像素 MSE
        除以 n·d，两者相差 (n-1)/n。日志里"理论预测"打印的是未修正的 Σλ/d，
        其相对偏差恒等于 1/n，属于口径差异而非求解误差——本用例把这两个口径都钉死。
        """
        X = make_data()
        n, d = X.shape
        _, _, ev_full, _, _ = pca(X, N_FEATURES)
        Z, W, _, mean, _ = pca(X, L_PRINCIPAL)

        X_hat = reconstruct(Z, W, mean)
        observed = float(((X - X_hat) ** 2).mean())        # 全部 n×d 元素上的平均
        uncorrected = float(ev_full[L_PRINCIPAL:].sum() / d)
        predicted = uncorrected * (n - 1) / n

        self.assertLess(abs(observed - predicted) / predicted, 1e-12,
                        f"observed={observed:.12f} predicted={predicted:.12f}")
        # 未修正口径的相对偏差应恰好落在 1/n 上（日志中 0.25% 的来源）
        self.assertAlmostEqual(abs(observed - uncorrected) / uncorrected, 1.0 / n, places=12)

    def test_full_rank_reconstruction_is_exact(self):
        """L = d 时不允许有信息损失：重构误差应落到浮点噪声量级。"""
        X = make_data()
        Z, W, _, mean, _ = pca(X, N_FEATURES)
        np.testing.assert_allclose(reconstruct(Z, W, mean), X, atol=1e-10)

    def test_zero_components_yields_mean_face(self):
        """L = 0（不取任何主成分）时重构结果退化为均值脸，误差最大。"""
        X = make_data()
        Z, W, _, mean, _ = pca(X, 0)
        self.assertEqual(Z.shape, (N_SAMPLES, 0))
        X_hat = reconstruct(Z, W, mean)
        np.testing.assert_allclose(X_hat, np.tile(mean, (N_SAMPLES, 1)), atol=1e-12)


class TestDualPathEquivalence(unittest.TestCase):
    """SVD 与协方差+eigh 是同一问题的两种解法，重构结果必须一致。"""

    def test_svd_and_cov_reconstruct_identically(self):
        X = make_data()
        Z_s, W_s, _, mean_s, _ = pca(X, L_PRINCIPAL, method="svd")
        Z_c, W_c, _, mean_c, _ = pca(X, L_PRINCIPAL, method="cov")

        # 特征向量可能整体差一个符号（特征向量方向不唯一），但 x̂ 不受符号影响
        np.testing.assert_allclose(reconstruct(Z_s, W_s, mean_s),
                                   reconstruct(Z_c, W_c, mean_c), atol=1e-9)

    def test_eigenvalues_agree(self):
        X = make_data()
        _, _, ev_s, _, _ = pca(X, L_PRINCIPAL, method="svd")
        _, _, ev_c, _, _ = pca(X, L_PRINCIPAL, method="cov")
        np.testing.assert_allclose(ev_s, ev_c, rtol=1e-8, atol=1e-12)


class TestInvalidArguments(unittest.TestCase):
    """错误的 method 必须显式失败，而不是静默返回看似合理的结果。"""

    def test_unknown_method_raises(self):
        with self.assertRaises(ValueError):
            pca(make_data(), L_PRINCIPAL, method="power-iteration")


if __name__ == "__main__":
    unittest.main(verbosity=2)
