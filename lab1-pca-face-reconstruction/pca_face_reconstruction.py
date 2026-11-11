# -*- coding: utf-8 -*-
"""
大数据分析与处理 —— 实验一：使用主成分分析(PCA)方法实现人脸重建
姓名：陈纬昕

实验步骤（严格对应实验指导）：
  1. 实现 PCA 函数（输入数据集 X、主成分个数 L；输出降维数据、L 个主成分及其特征值，按特征值降序）
  2. 基于 PCA 的人脸特征提取与人脸重建：加载 Olivetti faces 数据集（sklearn 读取）
  3. 选择部分面部，使用 matshow 函数将其可视化（3×6 = 18 张原始人脸）
  4. 利用 PCA 将人脸数据降到 20 维
  5. 使用 matshow 将特征脸可视化（2×10 = 20 个特征脸，含 PC01–PC20 编号）
  6. 随机选取一个人脸图片，观察其重构效果（原始人脸 vs 重构后人脸，标注样本 MSE/PSNR）
  7. 绘制样本的重构过程（原始人脸 + 各主成分的加权特征脸 系数×Face(i)）

扩展分析（7 步完成后自动运行，供报告与汇报的定量评估使用）：
  均值脸基线、k 扫描指标（MSE/RMSE/PSNR）、指标曲线（标注 k=20 选点）、
  累积重建（标注各 k 的 MSE）、残差可视化、分解计时（重复 3 次取中位数）、
  两路径等价性、身份可分离性（类内/类间距离比）、SSIM 与 1-NN 识别率。

运行环境：Python 3.13 + NumPy + Matplotlib + scikit-learn（Windows 10）
运行方式：python pca_face_reconstruction.py
输出：results/ 目录下 7 张结果图 + metrics_table.csv + 控制台运行结果
"""
import csv
import os
import time

import numpy as np
import matplotlib

# 无界面后端，保证脚本在命令行/服务器环境稳定保存图片；
# 如需在 PyCharm 中交互式查看图像，可注释掉下面这行、并取消文件末尾 plt.show() 的注释。
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager
from sklearn.datasets import fetch_olivetti_faces

# ---------------- 全局配置 ----------------
SEED = 0                 # 固定随机种子，保证结果可复现
N_COMPONENTS = 20        # 步骤4：降维目标维数 L = 20
IMG_SHAPE = (64, 64)     # 图像大小 64×64
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_HOME = os.path.join(ROOT, "data")      # 数据集目录（预置 olivettifaces.mat，可离线运行）
OUT_DIR = os.path.join(ROOT, "results")     # 结果图输出目录

np.random.seed(SEED)


def setup_chinese_font():
    """配置 matplotlib 中文字体与统一图表风格。"""
    candidates = ["Microsoft YaHei", "SimHei", "SimSun", "KaiTi",
                  "Noto Sans CJK SC", "Arial Unicode MS"]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            matplotlib.rcParams["font.sans-serif"] = [name] + \
                list(matplotlib.rcParams["font.sans-serif"])
            print("已启用中文字体:", name)
            break
    matplotlib.rcParams["axes.unicode_minus"] = False   # 正常显示负号
    # 统一图表风格：白底、统一标题/刻度字号
    matplotlib.rcParams.update({
        "figure.facecolor": "white",
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
    })


# =========================================================
# 步骤 1：实现 PCA 函数
# =========================================================
def pca(X, L, method="svd"):
    """
    主成分分析（PCA）。

    参数
    ----
    X : ndarray, 形状 (n_samples, n_features)，每行一个样本（一张人脸展平的向量）
    L : int，需要保留的主成分个数
    method : {"svd", "cov"}
        "svd" —— 对中心化数据矩阵做奇异值分解（数值更稳定、速度更快，推荐）；
        "cov" —— 教科书实现：构造协方差矩阵后做特征值分解。
        两种做法在数学上等价（λ_i = σ_i² / (n-1)），得到的主成分完全相同。

    返回
    ----
    Z : ndarray, (n_samples, L)，降维后的数据（样本在主成分方向上的投影/系数）
    components : ndarray, (L, n_features)，L 个主成分（每一行是一个主成分向量）
    eigenvalues : ndarray, (L,)，L 个特征值，按从大到小降序排列
    mean : ndarray, (n_features,)，训练数据的均值向量（用于重构）
    evr : ndarray, (L,)，各主成分的方差贡献率（以全部 400 个特征值之和为分母）
    """
    X = np.asarray(X, dtype=np.float64)
    n_samples, n_features = X.shape

    # ① 中心化：PCA 的前提是数据零均值
    mean = X.mean(axis=0)
    X_centered = X - mean

    if method == "svd":
        # ② 对中心化矩阵做经济型 SVD：X_c = U·S·V^T
        #    V 的列向量（即 Vt 的行向量）就是协方差矩阵的特征向量；
        #    特征值 λ_i = σ_i² / (n - 1)，且 S 天然按降序排列。
        _, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        eigenvalues_all = (S ** 2) / (n_samples - 1)
        components_all = Vt
    elif method == "cov":
        # ② 协方差矩阵 + 特征值分解
        C = (X_centered.T @ X_centered) / (n_samples - 1)
        eigvals, eigvecs = np.linalg.eigh(C)          # eigh 返回升序
        order = np.argsort(eigvals)[::-1]             # 转为降序
        eigenvalues_all = eigvals[order]
        components_all = eigvecs[:, order].T          # 每行一个主成分
    else:
        raise ValueError("method 只能是 'svd' 或 'cov'")

    # ③ 取前 L 个主成分（按特征值降序）
    components = components_all[:L]
    eigenvalues = eigenvalues_all[:L]

    # ④ 降维数据：样本在主成分上的投影系数 Z = X_c · W
    Z = X_centered @ components.T

    # ⑤ 各主成分的方差贡献率：λ_i / Σλ（分母为全部特征值之和，即数据总方差）
    explained_variance_ratio = eigenvalues / eigenvalues_all.sum()

    return Z, components, eigenvalues, mean, explained_variance_ratio


def reconstruct(Z, components, mean):
    """根据降维数据重构人脸图像：x̂ = mean + Z @ components（components 即 Wᵀ，形状 (L, d)）。"""
    return mean + Z @ components


# =========================================================
# 步骤 2：加载 Olivetti faces 数据集
# =========================================================
def load_faces():
    """加载 Olivetti 人脸数据集（400 张 64×64 灰度人脸，40 人 × 每人 10 张）。"""
    os.makedirs(DATA_HOME, exist_ok=True)
    faces = fetch_olivetti_faces(data_home=DATA_HOME, shuffle=False,
                                 random_state=SEED)
    return faces.data, faces.images, faces.target


# =========================================================
# 扩展分析（v2 新增，v3 重构为函数化结构）：定量评估与曲线
# =========================================================
def reconstruction_metrics(X, X_hat):
    """计算重构指标：平均 MSE、RMSE、逐图平均 PSNR（数据范围 [0,1]，峰值取 1）。"""
    diff = X - X_hat
    mse = float((diff ** 2).mean())
    rmse = mse ** 0.5
    mse_per_img = (diff ** 2).reshape(X.shape[0], -1).mean(axis=1)
    psnr = float((10 * np.log10(1.0 / mse_per_img)).mean())
    return mse, rmse, psnr


def evaluate_k_sweep(X, mean, components, evals, cum, ks, csv_path=None):
    """在给定 k 序列上计算重构指标（k=0 为均值脸基线），打印并保存 CSV。"""
    n = X.shape[0]
    Xc = X - mean
    print("=" * 60)
    print("扩展分析1  k 扫描指标（PSNR 为逐图平均口径；k=0 为均值脸基线）")
    print("   k        MSE      RMSE    PSNR(dB)   累计方差")
    rows = []
    for k in ks:
        if k == 0:
            X_hat = np.tile(mean, (n, 1))
            cv = 0.0
        else:
            W = components[:k]
            X_hat = mean + (Xc @ W.T) @ W
            cv = float(cum[k - 1])
        mse, rmse, psnr = reconstruction_metrics(X, X_hat)
        rows.append((k, mse, rmse, psnr, cv))
        print("  %-7s %8.5f  %7.5f  %8.3f  %8.2f%%"
              % (str(k) if k else "0(基线)", mse, rmse, psnr, cv * 100))
    if csv_path:
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            wr = csv.writer(f)
            wr.writerow(["k", "MSE", "RMSE", "PSNR_dB", "累计方差"])
            wr.writerows(rows)
    return rows


def plot_metric_curves(rows, cum, n, out_path):
    """图5：左——重构误差随 k 的变化（对数刻度，标注 k=20 选点）；右——全谱累计方差曲线。"""
    ks_plot = [r[0] for r in rows if r[0] > 0]
    mse_plot = [r[1] for r in rows if r[0] > 0]
    base_mse = rows[0][1]
    mse20 = next(r[1] for r in rows if r[0] == 20)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    axes[0].semilogy(ks_plot, mse_plot, "o-", color="#1E4FA8", lw=2, ms=5,
                     label="k 维重构 MSE")
    axes[0].axhline(base_mse, color="#D9534F", ls="--", lw=1.5,
                    label="均值脸基线 MSE = %.5f" % base_mse)
    axes[0].axvline(20, color="#8B97A8", ls=":", lw=1.5)
    axes[0].annotate("k=20：MSE %.5f\n（保留 76.34%% 方差）" % mse20,
                     xy=(20, mse20), xytext=(46, 0.012),
                     fontsize=9, color="#0E3F8C",
                     arrowprops=dict(arrowstyle="->", color="#0E3F8C", lw=1.2))
    axes[0].set_xlabel("主成分个数 k")
    axes[0].set_ylabel("平均重构 MSE（对数刻度）")
    axes[0].set_title("重构误差随 k 的变化")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=9, loc="upper right")
    axes[1].plot(np.arange(1, n + 1), cum * 100, color="#1E4FA8", lw=2)
    axes[1].axhline(90, color="#D9534F", ls="--", lw=1.2)
    axes[1].axvline(66, color="#D9534F", ls="--", lw=1.2)
    axes[1].plot([66], [cum[65] * 100], "o", color="#D9534F", ms=6)
    axes[1].annotate("k=66：累计方差 %.2f%%" % (cum[65] * 100),
                     xy=(66, cum[65] * 100), xytext=(96, 78),
                     fontsize=10, color="#D9534F")
    axes[1].set_xlabel("主成分个数 k")
    axes[1].set_ylabel("累计方差贡献率（%）")
    axes[1].set_title("全谱累计方差贡献率曲线")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_cumulative_reconstruction(X, mean, components, sample_idx, ks_show, out_path):
    """图6：同一张样本在不同 k 下的累积重建（每格标注该 k 的重构 MSE）。"""
    Xc = X - mean
    fig, axes = plt.subplots(1, len(ks_show) + 1, figsize=(17.5, 3.4))
    axes[0].matshow(X[sample_idx].reshape(IMG_SHAPE), cmap="gray")
    axes[0].set_title("原始人脸", fontsize=10, pad=6)
    axes[0].set_axis_off()
    for j, k in enumerate(ks_show, 1):
        Wk = components[:k]
        rec = mean + (Xc[sample_idx] @ Wk.T) @ Wk
        mse_k = float(((X[sample_idx] - rec) ** 2).mean())
        axes[j].matshow(rec.reshape(IMG_SHAPE), cmap="gray")
        axes[j].set_title("k=%d  MSE %.5f" % (k, mse_k), fontsize=8.5, pad=5)
        axes[j].set_axis_off()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_residual(X, mean, components, sample_idx, k=20, out_path=None):
    """图7：k 维重构与残差可视化（残差放大 3 倍显示），返回该样本的重构 MSE。"""
    Xc = X - mean
    W = components[:k]
    rec = mean + (Xc[sample_idx] @ W.T) @ W
    resid = np.abs(X[sample_idx] - rec)
    smse = float(((X[sample_idx] - rec) ** 2).mean())
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.6))
    axes[0].matshow(X[sample_idx].reshape(IMG_SHAPE), cmap="gray")
    axes[0].set_title("原始人脸", fontsize=10)
    axes[1].matshow(rec.reshape(IMG_SHAPE), cmap="gray")
    axes[1].set_title("k=%d 重构" % k, fontsize=10)
    axes[2].matshow(np.clip(resid * 3, 0, 1).reshape(IMG_SHAPE), cmap="gray")
    axes[2].set_title("残差 |原图−重构| ×3", fontsize=10)
    for ax in axes:
        ax.set_axis_off()
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return smse


def time_decompositions(Xc, repeats=3):
    """对 SVD 与协方差+eigh 两条路径各重复计时 repeats 次，返回计时列表与协方差特征对。"""
    n = Xc.shape[0]
    t_svd, t_cov = [], []
    w = v = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        np.linalg.svd(Xc, full_matrices=False)
        t_svd.append(time.perf_counter() - t0)
    for _ in range(repeats):
        t0 = time.perf_counter()
        C = (Xc.T @ Xc) / (n - 1)
        w, v = np.linalg.eigh(C)
        t_cov.append(time.perf_counter() - t0)
    return t_svd, t_cov, w, v


def verify_two_paths(X, mean, components, w, v, k=20):
    """验证 SVD 与协方差+eigh 两路径的 k 维重构数值等价性，返回最大绝对差。"""
    Xc = X - mean
    order = np.argsort(w)[::-1]
    Vt_cov = v[:, order][:, :k].T
    r_cov = mean + (Xc @ Vt_cov.T) @ Vt_cov
    r_svd = mean + (Xc @ components[:k].T) @ components[:k]
    return float(np.abs(r_cov - r_svd).max())


def extended_analysis(X, sample_idx):
    """扩展分析入口：基线/k 扫描指标 → 曲线 → 累积重建 → 残差 → 计时 → 两路径等价性。"""
    n, d = X.shape
    mean = X.mean(axis=0)
    _, components, evals, _, evr = pca(X, n)      # 全量主成分（单次 SVD，后续全部复用）
    cum = np.cumsum(evr)

    ks = [0, 1, 2, 3, 5, 10, 16, 20, 50, 66, 100, 150]
    rows = evaluate_k_sweep(X, mean, components, evals, cum, ks,
                            os.path.join(OUT_DIR, "metrics_table.csv"))

    pred20 = evals[20:].sum() / d
    obs20 = next(r[1] for r in rows if r[0] == 20)
    print("扩展分析2  理论预测 MSE(k=20) = %.6f｜观测 = %.6f｜相对偏差 %.2f%%"
          % (pred20, obs20, abs(pred20 - obs20) / pred20 * 100))
    W20 = components[:20]
    X_hat20 = mean + (X - mean) @ W20.T @ W20
    mse_i20 = ((X - X_hat20) ** 2).reshape(n, -1).mean(axis=1)
    print("           k=20 逐图 MSE：均值 %.5f ± 标准差 %.5f（400 张图像）"
          % (float(mse_i20.mean()), float(mse_i20.std())))

    plot_metric_curves(rows, cum, n, os.path.join(OUT_DIR, "fig5_metric_curves.png"))
    print("扩展分析3  指标曲线已保存 → results/fig5_metric_curves.png")

    plot_cumulative_reconstruction(X, mean, components, sample_idx, [1, 2, 5, 10, 20, 50],
                                   os.path.join(OUT_DIR, "fig6_cumulative_reconstruction.png"))
    print("扩展分析4  累积重建图已保存 → results/fig6_cumulative_reconstruction.png")

    smse = plot_residual(X, mean, components, sample_idx, k=20,
                         out_path=os.path.join(OUT_DIR, "fig7_residual.png"))
    print("扩展分析5  残差图已保存 → results/fig7_residual.png")
    print("           （样本索引 %d 的 k=20 重构 MSE = %.5f）" % (sample_idx, smse))

    t_svd, t_cov, w, v = time_decompositions(X - mean)
    print("扩展分析6  分解计时（重复 3 次取中位数）：")
    print("           SVD         中位 %.2f s（范围 %.2f–%.2f s）"
          % (float(np.median(t_svd)), min(t_svd), max(t_svd)))
    print("           协方差+eigh 中位 %.2f s（范围 %.2f–%.2f s）"
          % (float(np.median(t_cov)), min(t_cov), max(t_cov)))


    eq = verify_two_paths(X, mean, components, w, v)
    print("扩展分析7  两路径等价性：SVD 与协方差+eigh 的 k=20 重构最大绝对差 = %.2e" % eq)

    # ---------- 8. 身份可分离性（类内/类间距离比） ----------
    labels = np.repeat(np.arange(40), 10)          # 数据集结构：40 人 × 每人 10 张
    Z20 = (X - mean) @ components[:20].T
    sep = identity_separation(X, Z20, labels,
                              os.path.join(OUT_DIR, "fig8_identity_separation.png"))
    print("扩展分析8  身份可分离性（类内/类间平均距离比，越低越可分）：")
    for name, wm, bm, ratio in sep:
        print("           %s：类内 %.4f / 类间 %.4f → 比值 %.3f" % (name, wm, bm, ratio))
    print("           图8 已保存 → results/fig8_identity_separation.png")

    # ---------- 9. SSIM 与 1-NN 识别率 ----------
    ssim_rows = ssim_curve(X, mean, components, [1, 5, 10, 20, 50, 100])
    z_all = (X - mean) @ components.T              # 400×400 全量系数
    knn_rows = knn_accuracy(z_all, labels, [5, 10, 20, 30, 50, 66, 100, 150])
    plot_task_metrics(ssim_rows, knn_rows, os.path.join(OUT_DIR, "fig9_task_metrics.png"))
    if ssim_rows:
        print("扩展分析9  SSIM（全样本平均）：" +
              "  ".join("k=%d: %.3f" % (k, m) for k, m, _ in ssim_rows))
    print("扩展分析10 留一法 1-NN 识别率：" +
          "  ".join("k=%d: %.1f%%" % (k, a * 100) for k, a in knn_rows))
    print("           图9 已保存 → results/fig9_task_metrics.png")
    print("=" * 60)




def identity_separation(X, Z20, labels, out_path):
    """类内 / 类间平均距离（像素空间 vs PCA-20 空间），量化"身份可分离性"并绘制分布图（图8）。"""
    def pairwise_dist(A):
        sq = (A ** 2).sum(axis=1)
        D2 = sq[:, None] + sq[None, :] - 2 * (A @ A.T)
        return np.sqrt(np.maximum(D2, 0))

    n = X.shape[0]
    same = labels[:, None] == labels[None, :]
    off_diag = ~np.eye(n, dtype=bool)
    within_mask = same & off_diag
    between_mask = ~same
    stats = []
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.0))
    for ax, (name, A) in zip(axes, [("像素空间（4096 维）", X), ("PCA 空间（20 维）", Z20)]):
        d = pairwise_dist(A)
        w = d[within_mask]
        b = d[between_mask]
        stats.append((name, float(w.mean()), float(b.mean()), float(w.mean() / b.mean())))
        bins = np.linspace(0, np.percentile(b, 99), 60)
        ax.hist(w, bins=bins, alpha=0.65, label="同一人（类内）", color="#1E4FA8", density=True)
        ax.hist(b, bins=bins, alpha=0.55, label="不同人（类间）", color="#D9534F", density=True)
        ax.set_xlabel("两张人脸之间的 Euclidean 距离")
        ax.set_ylabel("概率密度")
        ax.set_title("%s：类内均值 %.4f / 类间均值 %.4f" % (name, w.mean(), b.mean()), fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return stats


def ssim_curve(X, mean, components, ks):
    """不同 k 下的平均结构相似性 SSIM（全样本平均，data_range=1.0）；未安装 scikit-image 时跳过。"""
    try:
        from skimage.metrics import structural_similarity as ssim_fn
    except ImportError:
        print("扩展分析9  跳过 SSIM：未安装 scikit-image（pip install scikit-image）")
        return []
    Xc = X - mean
    n = X.shape[0]
    rows = []
    for k in ks:
        W = components[:k]
        X_hat = mean + (Xc @ W.T) @ W
        s = [float(ssim_fn(X[i].reshape(IMG_SHAPE), X_hat[i].reshape(IMG_SHAPE), data_range=1.0))
             for i in range(n)]
        rows.append((k, float(np.mean(s)), float(np.std(s))))
    return rows


def knn_accuracy(z_all, labels, ks):
    """留一法 1-NN 识别率（PCA 特征空间，40 类分类任务）。"""
    rows = []
    for k in ks:
        Z = z_all[:, :k]
        sq = (Z ** 2).sum(axis=1)
        D2 = sq[:, None] + sq[None, :] - 2 * (Z @ Z.T)
        np.fill_diagonal(D2, np.inf)
        nn = D2.argmin(axis=1)
        rows.append((k, float((labels[nn] == labels).mean())))
    return rows


def plot_task_metrics(ssim_rows, knn_rows, out_path):
    """图9：左——SSIM 随 k 的变化；右——LOO 1-NN 识别率随 k 的变化。"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.0))
    if ssim_rows:
        axes[0].plot([r[0] for r in ssim_rows], [r[1] for r in ssim_rows], "o-",
                     color="#1E4FA8", lw=2, ms=5)
        axes[0].axvline(20, color="#8B97A8", ls=":", lw=1.5)
        axes[0].set_xlabel("主成分个数 k")
        axes[0].set_ylabel("平均 SSIM（40 张图像平均）")
        axes[0].set_title("结构相似性 SSIM 随 k 的变化")
        axes[0].grid(alpha=0.3)
    else:
        axes[0].text(0.5, 0.5, "未安装 scikit-image，SSIM 未计算", ha="center", va="center", fontsize=12)
        axes[0].set_axis_off()
    axes[1].plot([r[0] for r in knn_rows], [r[1] * 100 for r in knn_rows], "s-",
                 color="#0E3F8C", lw=2, ms=5)
    axes[1].axvline(20, color="#8B97A8", ls=":", lw=1.5)
    axes[1].set_xlabel("主成分个数 k")
    axes[1].set_ylabel("留一法 1-NN 识别率（%）")
    axes[1].set_title("最近邻识别率随 k 的变化（40 类）")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# =========================================================
# 主流程：严格对应实验指导 7 步（随后运行扩展分析）
# =========================================================
def main():
    setup_chinese_font()
    os.makedirs(OUT_DIR, exist_ok=True)

    # ---------- 步骤 2：加载数据集 ----------
    X, images, y = load_faces()
    print("=" * 60)
    print("步骤2  数据集加载完成")
    print("  data   :", X.shape, "（400 个样本，每个样本 64×64=4096 维）")
    print("  images :", images.shape)
    print("  target :", y.shape, " 共", len(np.unique(y)), "个受试者，每人", len(y) // len(np.unique(y)), "张")

    # ---------- 步骤 3：选择部分面部并可视化（3×6 = 18 张原始人脸） ----------
    # 每个受试者取第 1 张（索引 0,10,20,...），使 18 张人脸来自 18 个不同的人
    sample_idx = np.arange(0, 400, 10)[:18]
    fig, axes = plt.subplots(3, 6, figsize=(9, 4.6))
    for ax, i in zip(axes.ravel(), sample_idx):
        ax.matshow(images[i], cmap="gray")
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig1_original_faces.png"), dpi=200)
    plt.close(fig)
    print("=" * 60)
    print("步骤3  原始人脸可视化完成 → results/fig1_original_faces.png")

    # ---------- 步骤 4：利用 PCA 将人脸数据降到 20 维 ----------
    Z, components, eigenvalues, mean, evr = pca(X, N_COMPONENTS)
    explained = evr                                       # 各主成分方差贡献率
    cumulative = np.cumsum(explained)                    # 累计方差贡献率
    print("=" * 60)
    print("步骤4  PCA 降维完成：X", X.shape, "→ Z", Z.shape)
    print("  特征值（降序）:", np.round(eigenvalues, 3))
    print("  各主成分方差贡献率:", np.round(explained, 4))
    print("  前20个主成分累计方差贡献率: {:.2%}".format(cumulative[-1]))

    # ---------- 步骤 5：使用 matshow 将特征脸可视化（2×10 = 20 张，含编号） ----------
    # 每个主成分是长度为 4096 的向量，reshape 成 64×64 后即为一幅"特征脸"
    fig, axes = plt.subplots(2, 10, figsize=(13, 2.7))
    for pc, (ax, vec) in enumerate(zip(axes.ravel(), components), 1):
        ax.matshow(vec.reshape(IMG_SHAPE), cmap="gray")
        ax.set_title("PC{:02d}".format(pc), fontsize=9)
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig2_eigenfaces.png"), dpi=200)
    plt.close(fig)
    print("=" * 60)
    print("步骤5  特征脸可视化完成 → results/fig2_eigenfaces.png")

    # ---------- 步骤 6：随机选取一个人脸图片，观察其重构效果 ----------
    idx = int(np.random.randint(0, X.shape[0]))          # 随机下标（种子已固定，可复现）
    x = X[idx]
    x_hat = reconstruct(Z[idx:idx + 1], components, mean)[0]
    sample_mse = float(((x - x_hat) ** 2).mean())
    sample_psnr = float(10 * np.log10(1.0 / sample_mse))
    print("=" * 60)
    print("步骤6  随机选取样本：索引", idx, "（第", int(y[idx]), "号受试者）")

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.8))
    axes[0].matshow(x.reshape(IMG_SHAPE), cmap="gray")
    axes[0].set_title("原始人脸")
    axes[1].matshow(x_hat.reshape(IMG_SHAPE), cmap="gray")
    axes[1].set_title("重构后人脸（k=20）")
    fig.text(0.5, 0.045, "样本 MSE = %.5f    PSNR = %.1f dB" % (sample_mse, sample_psnr),
             ha="center", fontsize=10, color="#0E3F8C")
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    fig.savefig(os.path.join(OUT_DIR, "fig3_reconstruction_compare.png"), dpi=200)
    plt.close(fig)
    print("       重构效果对比图已保存 → results/fig3_reconstruction_compare.png")

    # ---------- 步骤 7：绘制样本的重构过程 ----------
    # 2×11 布局：第 1 格为原始人脸，其后 20 格为 "系数×特征脸"（即 x̂ = mean + Σ coef_i·v_i 的各项）
    fig, axes = plt.subplots(2, 11, figsize=(18, 3.6))
    axes = axes.ravel()
    axes[0].matshow(x.reshape(IMG_SHAPE), cmap="gray")   # 首格为原始人脸（保留坐标轴）
    coefs = Z[idx]
    for i in range(N_COMPONENTS):
        weighted = (coefs[i] * components[i]).reshape(IMG_SHAPE)
        axes[i + 1].matshow(weighted, cmap="gray")
        axes[i + 1].set_title("{:.2f}*Face({})".format(coefs[i], i), fontsize=8)
        axes[i + 1].set_axis_off()                       # 其余格只留标题，不显示坐标轴
    axes[-1].set_axis_off()                              # 第 22 格留空
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig4_reconstruction_process.png"), dpi=200)
    plt.close(fig)
    print("步骤7  重构过程图已保存 → results/fig4_reconstruction_process.png")
    print("  该样本在各主成分上的系数（前10个）:", np.round(coefs[:10], 3))
    print("=" * 60)
    print("全部 7 步实验完成，结果图目录：", OUT_DIR)

    # ==================== 扩展分析（v2 新增：定量评估与曲线） ====================
    extended_analysis(X, idx)


if __name__ == "__main__":
    main()
