# -*- coding: utf-8 -*-
"""
Big Data Analysis and Processing — Lab 1: Face Reconstruction using Principal Component Analysis (PCA)
Author: Weixin Chen

Lab steps (strictly following the lab manual):
  1. Implement a PCA function (input: dataset X and number of components L; output: reduced data,
     the L principal components and their eigenvalues, sorted in descending order of eigenvalue)
  2. PCA-based facial feature extraction and face reconstruction: load the Olivetti faces dataset (via sklearn)
  3. Select some faces and visualize them with matshow (3×6 = 18 original faces)
  4. Reduce the face data to 20 dimensions using PCA
  5. Visualize the eigenfaces with matshow (2×10 = 20 eigenfaces, labelled PC01–PC20)
  6. Randomly pick a face image and inspect its reconstruction (original vs reconstructed, with sample MSE/PSNR)
  7. Plot the reconstruction process of a sample (original face + weighted eigenfaces coef*Face(i))

Extended analysis (runs automatically after the 7 steps, used for the quantitative report/slides):
  Mean-face baseline, k-sweep metrics (MSE/RMSE/PSNR), metric curves (marking the k=20 choice),
  cumulative reconstruction (with per-k MSE), residual visualization, decomposition timing
  (median of 3 runs), two-path equivalence, identity separability (within/between distance ratio),
  SSIM and LOO 1-NN recognition accuracy.

Environment: Python 3.13 + NumPy + Matplotlib + scikit-learn (Windows 10)
Run: python pca_face_reconstruction_en.py
Output: 7 figures + metrics_table.csv + console log in the results_en/ directory
"""
import csv
import os
import time

import numpy as np
import matplotlib

# Headless backend so the script reliably saves figures from a terminal;
# for interactive viewing in PyCharm, comment out the line below and uncomment plt.show() at the end.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces

# ---------------- Global configuration ----------------
SEED = 0                    # fixed random seed for reproducibility
N_COMPONENTS = 20           # Step 4: target dimensionality L = 20
IMG_SHAPE = (64, 64)        # image size 64×64
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_HOME = os.path.join(ROOT, "data")          # dataset directory (pre-placed olivettifaces.mat, works offline)
OUT_DIR = os.path.join(ROOT, "results_en")      # output directory for figures (English version)

np.random.seed(SEED)


def setup_plot_style():
    """Unified figure style: white background, consistent title/tick font sizes."""
    matplotlib.rcParams["axes.unicode_minus"] = False   # proper minus signs
    matplotlib.rcParams.update({
        "figure.facecolor": "white",
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
    })


# =========================================================
# Step 1: implement the PCA function
# =========================================================
def pca(X, L, method="svd"):
    """
    Principal Component Analysis (PCA).

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features); each row is one sample (a flattened face image)
    L : int, number of principal components to keep
    method : {"svd", "cov"}
        "svd" — singular value decomposition of the centered data matrix (numerically more stable
                and faster; recommended);
        "cov" — textbook implementation: build the covariance matrix, then eigendecompose it.
        The two are mathematically equivalent (λ_i = σ_i² / (n-1)) and yield identical components.

    Returns
    -------
    Z : ndarray, (n_samples, L), the reduced data (projection coefficients of samples onto components)
    components : ndarray, (L, n_features), the L principal components (one per row)
    eigenvalues : ndarray, (L,), the L eigenvalues sorted in descending order
    mean : ndarray, (n_features,), the mean vector of the training data (needed for reconstruction)
    evr : ndarray, (L,), explained variance ratio of each component (denominator: sum of all 400 eigenvalues)
    """
    X = np.asarray(X, dtype=np.float64)
    n_samples, n_features = X.shape

    # (1) Center the data: PCA requires zero-mean input
    mean = X.mean(axis=0)
    X_centered = X - mean

    if method == "svd":
        # (2) Economy SVD of the centered matrix: X_c = U·S·V^T
        #     The rows of Vt are the eigenvectors of the covariance matrix;
        #     eigenvalues λ_i = σ_i² / (n - 1), and S is already in descending order.
        _, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        eigenvalues_all = (S ** 2) / (n_samples - 1)
        components_all = Vt
    elif method == "cov":
        # (2) Covariance matrix + eigendecomposition
        C = (X_centered.T @ X_centered) / (n_samples - 1)
        eigvals, eigvecs = np.linalg.eigh(C)          # eigh returns ascending order
        order = np.argsort(eigvals)[::-1]             # flip to descending
        eigenvalues_all = eigvals[order]
        components_all = eigvecs[:, order].T          # one component per row
    else:
        raise ValueError("method must be 'svd' or 'cov'")

    # (3) Keep the top-L components (already sorted by descending eigenvalue)
    components = components_all[:L]
    eigenvalues = eigenvalues_all[:L]

    # (4) Reduced data: projection coefficients Z = X_c · W
    Z = X_centered @ components.T

    # (5) Explained variance ratio: λ_i / Σλ (denominator: sum of ALL eigenvalues = total variance)
    explained_variance_ratio = eigenvalues / eigenvalues_all.sum()

    return Z, components, eigenvalues, mean, explained_variance_ratio


def reconstruct(Z, components, mean):
    """Reconstruct face images from reduced data: x̂ = mean + Z @ components (components is Wᵀ, shape (L, d))."""
    return mean + Z @ components


# =========================================================
# Step 2: load the Olivetti faces dataset
# =========================================================
def load_faces():
    """Load the Olivetti faces dataset (400 grayscale faces of 64×64, 40 subjects × 10 images each)."""
    os.makedirs(DATA_HOME, exist_ok=True)
    faces = fetch_olivetti_faces(data_home=DATA_HOME, shuffle=False,
                                 random_state=SEED)
    return faces.data, faces.images, faces.target


# =========================================================
# Extended analysis (v2; refactored into helpers in v3)
# =========================================================
def reconstruction_metrics(X, X_hat):
    """Compute reconstruction metrics: mean MSE, RMSE and per-image averaged PSNR (data in [0,1], peak = 1)."""
    diff = X - X_hat
    mse = float((diff ** 2).mean())
    rmse = mse ** 0.5
    mse_per_img = (diff ** 2).reshape(X.shape[0], -1).mean(axis=1)
    psnr = float((10 * np.log10(1.0 / mse_per_img)).mean())
    return mse, rmse, psnr


def evaluate_k_sweep(X, mean, components, evals, cum, ks, csv_path=None):
    """Compute reconstruction metrics over a k grid (k=0 is the mean-face baseline); print and save CSV."""
    n = X.shape[0]
    Xc = X - mean
    print("=" * 60)
    print("Extended 1  k-sweep metrics (PSNR = per-image average; k=0 is the mean-face baseline)")
    print("   k        MSE      RMSE    PSNR(dB)   cum.var")
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
        print("  %-8s %8.5f  %7.5f  %8.3f  %7.2f%%"
              % (str(k) if k else "0(base)", mse, rmse, psnr, cv * 100))
    if csv_path:
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            wr = csv.writer(f)
            wr.writerow(["k", "MSE", "RMSE", "PSNR_dB", "cum_var"])
            wr.writerows(rows)
    return rows


def plot_metric_curves(rows, cum, n, out_path):
    """Figure 5: left — error vs k (log scale, marking k=20); right — full-spectrum cumulative variance."""
    ks_plot = [r[0] for r in rows if r[0] > 0]
    mse_plot = [r[1] for r in rows if r[0] > 0]
    base_mse = rows[0][1]
    mse20 = next(r[1] for r in rows if r[0] == 20)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    axes[0].semilogy(ks_plot, mse_plot, "o-", color="#1E4FA8", lw=2, ms=5,
                     label="Reconstruction MSE")
    axes[0].axhline(base_mse, color="#D9534F", ls="--", lw=1.5,
                    label="Mean-face baseline MSE = %.5f" % base_mse)
    axes[0].axvline(20, color="#8B97A8", ls=":", lw=1.5)
    axes[0].annotate("k=20: MSE %.5f\n(keeps 76.34%% variance)" % mse20,
                     xy=(20, mse20), xytext=(46, 0.012),
                     fontsize=9, color="#0E3F8C",
                     arrowprops=dict(arrowstyle="->", color="#0E3F8C", lw=1.2))
    axes[0].set_xlabel("Number of components k")
    axes[0].set_ylabel("Average reconstruction MSE (log scale)")
    axes[0].set_title("Reconstruction error vs. k")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=9, loc="upper right")
    axes[1].plot(np.arange(1, n + 1), cum * 100, color="#1E4FA8", lw=2)
    axes[1].axhline(90, color="#D9534F", ls="--", lw=1.2)
    axes[1].axvline(66, color="#D9534F", ls="--", lw=1.2)
    axes[1].plot([66], [cum[65] * 100], "o", color="#D9534F", ms=6)
    axes[1].annotate("k=66: cumulative variance %.2f%%" % (cum[65] * 100),
                     xy=(66, cum[65] * 100), xytext=(96, 78),
                     fontsize=10, color="#D9534F")
    axes[1].set_xlabel("Number of components k")
    axes[1].set_ylabel("Cumulative explained variance (%)")
    axes[1].set_title("Cumulative explained variance (full spectrum)")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_cumulative_reconstruction(X, mean, components, sample_idx, ks_show, out_path):
    """Figure 6: cumulative reconstruction of the same sample across k (each panel annotated with its MSE)."""
    Xc = X - mean
    fig, axes = plt.subplots(1, len(ks_show) + 1, figsize=(17.5, 3.4))
    axes[0].matshow(X[sample_idx].reshape(IMG_SHAPE), cmap="gray")
    axes[0].set_title("Original", fontsize=10, pad=6)
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
    """Figure 7: k-dim reconstruction and residual visualization (residual amplified 3×); returns sample MSE."""
    Xc = X - mean
    W = components[:k]
    rec = mean + (Xc[sample_idx] @ W.T) @ W
    resid = np.abs(X[sample_idx] - rec)
    smse = float(((X[sample_idx] - rec) ** 2).mean())
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.6))
    axes[0].matshow(X[sample_idx].reshape(IMG_SHAPE), cmap="gray")
    axes[0].set_title("Original", fontsize=10)
    axes[1].matshow(rec.reshape(IMG_SHAPE), cmap="gray")
    axes[1].set_title("Reconstruction (k=%d)" % k, fontsize=10)
    axes[2].matshow(np.clip(resid * 3, 0, 1).reshape(IMG_SHAPE), cmap="gray")
    axes[2].set_title("Residual |orig−rec| ×3", fontsize=10)
    for ax in axes:
        ax.set_axis_off()
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return smse


def time_decompositions(Xc, repeats=3):
    """Time the SVD and covariance+eigh paths (repeats runs each); return timing lists and eigenvectors."""
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
    """Check numerical equivalence of the two paths; returns the max absolute reconstruction difference."""
    Xc = X - mean
    order = np.argsort(w)[::-1]
    Vt_cov = v[:, order][:, :k].T
    r_cov = mean + (Xc @ Vt_cov.T) @ Vt_cov
    r_svd = mean + (Xc @ components[:k].T) @ components[:k]
    return float(np.abs(r_cov - r_svd).max())


def extended_analysis(X, sample_idx):
    """Entry point: baseline/k-sweep → curves → cumulative reconstruction → residual → timing → equivalence."""
    n, d = X.shape
    mean = X.mean(axis=0)
    _, components, evals, _, evr = pca(X, n)      # all components (single SVD, reused throughout)
    cum = np.cumsum(evr)

    ks = [0, 1, 2, 3, 5, 10, 16, 20, 50, 66, 100, 150]
    rows = evaluate_k_sweep(X, mean, components, evals, cum, ks,
                            os.path.join(OUT_DIR, "metrics_table.csv"))

    pred20 = evals[20:].sum() / d
    obs20 = next(r[1] for r in rows if r[0] == 20)
    print("Extended 2  Theory MSE(k=20) = %.6f | Measured = %.6f | rel. deviation %.2f%%"
          % (pred20, obs20, abs(pred20 - obs20) / pred20 * 100))
    W20 = components[:20]
    X_hat20 = mean + (X - mean) @ W20.T @ W20
    mse_i20 = ((X - X_hat20) ** 2).reshape(n, -1).mean(axis=1)
    print("            per-image MSE at k=20: mean %.5f +/- std %.5f (400 images)"
          % (float(mse_i20.mean()), float(mse_i20.std())))

    plot_metric_curves(rows, cum, n, os.path.join(OUT_DIR, "fig5_metric_curves.png"))
    print("Extended 3  Metric curves saved → results_en/fig5_metric_curves.png")

    plot_cumulative_reconstruction(X, mean, components, sample_idx, [1, 2, 5, 10, 20, 50],
                                   os.path.join(OUT_DIR, "fig6_cumulative_reconstruction.png"))
    print("Extended 4  Cumulative reconstruction saved → results_en/fig6_cumulative_reconstruction.png")

    smse = plot_residual(X, mean, components, sample_idx, k=20,
                         out_path=os.path.join(OUT_DIR, "fig7_residual.png"))
    print("Extended 5  Residual figure saved → results_en/fig7_residual.png")
    print("            (sample index %d: k=20 reconstruction MSE = %.5f)" % (sample_idx, smse))

    t_svd, t_cov, w, v = time_decompositions(X - mean)
    print("Extended 6  Decomposition timing (median of 3 runs):")
    print("            SVD          median %.2f s (range %.2f–%.2f s)"
          % (float(np.median(t_svd)), min(t_svd), max(t_svd)))
    print("            Cov + eigh   median %.2f s (range %.2f–%.2f s)"
          % (float(np.median(t_cov)), min(t_cov), max(t_cov)))


    eq = verify_two_paths(X, mean, components, w, v)
    print("Extended 7  Two-path equivalence: max abs diff between SVD and cov+eigh k=20 reconstructions = %.2e" % eq)

    # ---------- 8. Identity separability (within/between distance ratio) ----------
    labels = np.repeat(np.arange(40), 10)          # dataset structure: 40 subjects × 10 images
    Z20 = (X - mean) @ components[:20].T
    sep = identity_separation(X, Z20, labels,
                              os.path.join(OUT_DIR, "fig8_identity_separation.png"))
    print("Extended 8  Identity separability (within/between mean distance, lower = better):")
    for name, wm, bm, ratio in sep:
        print("            %s: within %.4f / between %.4f -> ratio %.3f" % (name, wm, bm, ratio))
    print("            Figure 8 saved → results_en/fig8_identity_separation.png")

    # ---------- 9. SSIM and 1-NN accuracy ----------
    ssim_rows = ssim_curve(X, mean, components, [1, 5, 10, 20, 50, 100])
    z_all = (X - mean) @ components.T              # 400×400 full coefficients
    knn_rows = knn_accuracy(z_all, labels, [5, 10, 20, 30, 50, 66, 100, 150])
    plot_task_metrics(ssim_rows, knn_rows, os.path.join(OUT_DIR, "fig9_task_metrics.png"))
    if ssim_rows:
        print("Extended 9  SSIM (all-sample mean): " +
              "  ".join("k=%d: %.3f" % (k, m) for k, m, _ in ssim_rows))
    print("Extended 10 LOO 1-NN accuracy: " +
          "  ".join("k=%d: %.1f%%" % (k, a * 100) for k, a in knn_rows))
    print("            Figure 9 saved → results_en/fig9_task_metrics.png")
    print("=" * 60)




def identity_separation(X, Z20, labels, out_path):
    """Within/between-class mean distances (pixel space vs PCA-20), i.e. "identity separability" (Figure 8)."""
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
    for ax, (name, A) in zip(axes, [("Pixel space (4096-d)", X), ("PCA space (20-d)", Z20)]):
        d = pairwise_dist(A)
        w = d[within_mask]
        b = d[between_mask]
        stats.append((name, float(w.mean()), float(b.mean()), float(w.mean() / b.mean())))
        bins = np.linspace(0, np.percentile(b, 99), 60)
        ax.hist(w, bins=bins, alpha=0.65, label="Same subject (within)", color="#1E4FA8", density=True)
        ax.hist(b, bins=bins, alpha=0.55, label="Different subjects (between)", color="#D9534F", density=True)
        ax.set_xlabel("Euclidean distance between two faces")
        ax.set_ylabel("Probability density")
        ax.set_title("%s: within %.4f / between %.4f" % (name, w.mean(), b.mean()), fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return stats


def ssim_curve(X, mean, components, ks):
    """Mean SSIM at different k (all samples, data_range=1.0); skipped if scikit-image is unavailable."""
    try:
        from skimage.metrics import structural_similarity as ssim_fn
    except ImportError:
        print("Extended 9  SSIM skipped: scikit-image not installed (pip install scikit-image)")
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
    """Leave-one-out 1-NN recognition accuracy (PCA feature space, 40-class task)."""
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
    """Figure 9: left — SSIM vs k; right — LOO 1-NN accuracy vs k."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.0))
    if ssim_rows:
        axes[0].plot([r[0] for r in ssim_rows], [r[1] for r in ssim_rows], "o-",
                     color="#1E4FA8", lw=2, ms=5)
        axes[0].axvline(20, color="#8B97A8", ls=":", lw=1.5)
        axes[0].set_xlabel("Number of components k")
        axes[0].set_ylabel("Mean SSIM (average over 400 images)")
        axes[0].set_title("SSIM vs. k")
        axes[0].grid(alpha=0.3)
    else:
        axes[0].text(0.5, 0.5, "scikit-image not installed; SSIM not computed", ha="center", va="center", fontsize=12)
        axes[0].set_axis_off()
    axes[1].plot([r[0] for r in knn_rows], [r[1] * 100 for r in knn_rows], "s-",
                 color="#0E3F8C", lw=2, ms=5)
    axes[1].axvline(20, color="#8B97A8", ls=":", lw=1.5)
    axes[1].set_xlabel("Number of components k")
    axes[1].set_ylabel("LOO 1-NN accuracy (%)")
    axes[1].set_title("Nearest-neighbour accuracy vs. k (40 classes)")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# =========================================================
# Main pipeline: strictly follows the 7 lab steps (then runs the extended analysis)
# =========================================================
def main():
    setup_plot_style()
    os.makedirs(OUT_DIR, exist_ok=True)

    # ---------- Step 2: load the dataset ----------
    X, images, y = load_faces()
    print("=" * 60)
    print("Step 2  dataset loaded")
    print("  data   :", X.shape, "(400 samples, each 64×64 = 4096 dimensions)")
    print("  images :", images.shape)
    print("  target :", y.shape, " 40 subjects, 10 images each")

    # ---------- Step 3: select some faces and visualize (3×6 = 18 original faces) ----------
    # Take the first image of each subject (indices 0, 10, 20, ...) so that the 18 faces are 18 different people
    sample_idx = np.arange(0, 400, 10)[:18]
    fig, axes = plt.subplots(3, 6, figsize=(9, 4.6))
    for ax, i in zip(axes.ravel(), sample_idx):
        ax.matshow(images[i], cmap="gray")
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig1_original_faces.png"), dpi=200)
    plt.close(fig)
    print("=" * 60)
    print("Step 3  original faces visualized → results_en/fig1_original_faces.png")

    # ---------- Step 4: reduce the face data to 20 dimensions with PCA ----------
    Z, components, eigenvalues, mean, evr = pca(X, N_COMPONENTS)
    explained = evr                                       # per-component explained variance ratio
    cumulative = np.cumsum(explained)                     # cumulative explained variance ratio
    print("=" * 60)
    print("Step 4  PCA completed: X", X.shape, "→ Z", Z.shape)
    print("  eigenvalues (descending):", np.round(eigenvalues, 3))
    print("  explained variance ratio:", np.round(explained, 4))
    print("  cumulative explained variance of top 20: {:.2%}".format(cumulative[-1]))

    # ---------- Step 5: visualize the eigenfaces with matshow (2×10 = 20, labelled) ----------
    # Each component is a 4096-dim vector; reshaping to 64×64 yields an "eigenface"
    fig, axes = plt.subplots(2, 10, figsize=(13, 2.7))
    for pc, (ax, vec) in enumerate(zip(axes.ravel(), components), 1):
        ax.matshow(vec.reshape(IMG_SHAPE), cmap="gray")
        ax.set_title("PC{:02d}".format(pc), fontsize=9)
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig2_eigenfaces.png"), dpi=200)
    plt.close(fig)
    print("=" * 60)
    print("Step 5  eigenfaces visualized → results_en/fig2_eigenfaces.png")

    # ---------- Step 6: randomly pick one face and inspect its reconstruction ----------
    idx = int(np.random.randint(0, X.shape[0]))          # random index (seed fixed → reproducible)
    x = X[idx]
    x_hat = reconstruct(Z[idx:idx + 1], components, mean)[0]
    sample_mse = float(((x - x_hat) ** 2).mean())
    sample_psnr = float(10 * np.log10(1.0 / sample_mse))
    print("=" * 60)
    print("Step 6  random sample: index", idx, "(subject", int(y[idx]), ")")

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.8))
    axes[0].matshow(x.reshape(IMG_SHAPE), cmap="gray")
    axes[0].set_title("Original")
    axes[1].matshow(x_hat.reshape(IMG_SHAPE), cmap="gray")
    axes[1].set_title("Reconstructed (k=20)")
    fig.text(0.5, 0.045, "Sample MSE = %.5f    PSNR = %.1f dB" % (sample_mse, sample_psnr),
             ha="center", fontsize=10, color="#0E3F8C")
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    fig.savefig(os.path.join(OUT_DIR, "fig3_reconstruction_compare.png"), dpi=200)
    plt.close(fig)
    print("        comparison figure saved → results_en/fig3_reconstruction_compare.png")

    # ---------- Step 7: plot the reconstruction process of the sample ----------
    # 2×11 grid: first cell = original face, then 20 cells of "coef × eigenface" (terms of x̂ = mean + Σ coef_i·v_i)
    fig, axes = plt.subplots(2, 11, figsize=(18, 3.6))
    axes = axes.ravel()
    axes[0].matshow(x.reshape(IMG_SHAPE), cmap="gray")   # first cell: original face (axes kept)
    coefs = Z[idx]
    for i in range(N_COMPONENTS):
        weighted = (coefs[i] * components[i]).reshape(IMG_SHAPE)
        axes[i + 1].matshow(weighted, cmap="gray")
        axes[i + 1].set_title("{:.2f}*Face({})".format(coefs[i], i), fontsize=8)
        axes[i + 1].set_axis_off()                       # other cells: title only, no axes
    axes[-1].set_axis_off()                              # leave the 22nd cell empty
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig4_reconstruction_process.png"), dpi=200)
    plt.close(fig)
    print("Step 7  reconstruction-process figure saved → results_en/fig4_reconstruction_process.png")
    print("  coefficients of this sample (first 10):", np.round(coefs[:10], 3))
    print("=" * 60)
    print("All 7 steps completed. Output directory:", OUT_DIR)

    # ==================== Extended analysis ====================
    extended_analysis(X, idx)


if __name__ == "__main__":
    main()
