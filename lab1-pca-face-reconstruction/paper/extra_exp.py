# -*- coding: utf-8 -*-
"""Held-out sanity check + measured latency + baseline SSIM anchors.
Fit the PCA basis on 280 images (7 per subject), evaluate on the 120 unseen ones.
Prints a JSON block consumed by build.py (single source of numbers).

This experiment runs only when directly invoked or through build.py --refresh-data.
"""
import csv
import json
import logging
from pathlib import Path
import re
import sys
import time

import numpy as np
from sklearn.datasets import fetch_olivetti_faces
from skimage.metrics import structural_similarity as ssim_fn

BASE = Path(__file__).resolve().parent.parent
K = 20
IMG = (64, 64)
LOGGER = logging.getLogger('paper_extra_experiment')


def recorded_full_sample_metrics():
    # 复用全样本实验记录，保证 SSIM 增益的两端使用同一个评估协议。
    console_path = BASE / 'results' / 'console_output.txt'
    text = console_path.read_text(encoding='utf-8')
    line = re.search(r'扩展分析9[^\n]*', text)
    if not line:
        raise ValueError('Full-sample SSIM record not found: %s' % console_path)
    scores = {int(k): float(value) for k, value in re.findall(r'k=(\d+): ([\d.]+)', line.group(0))}
    csv_path = BASE / 'results' / 'metrics_table.csv'
    with csv_path.open(encoding='utf-8-sig', newline='') as stream:
        mse = next(float(row['MSE']) for row in csv.DictReader(stream) if int(row['k']) == K)
    return scores[K], mse


def main():
    started = time.perf_counter()
    LOGGER.info('Starting optional experiment base=%s k=%d image_shape=%s', BASE, K, IMG)
    faces = fetch_olivetti_faces(data_home=str(BASE / 'data'), shuffle=False, random_state=0)
    X = faces.data.astype(np.float64)
    y = faces.target
    n, d = X.shape
    LOGGER.info('Loaded faces samples=%d features=%d dtype=%s', n, d, X.dtype)

    # ---- stratified split: first 7 images of each subject train, last 3 test ----
    train_idx = np.array([s * 10 + i for s in range(40) for i in range(7)])
    test_idx = np.array([s * 10 + i for s in range(40) for i in range(7, 10)])
    Xtr, Xte, ytr, yte = X[train_idx], X[test_idx], y[train_idx], y[test_idx]
    LOGGER.info('Fixed per-subject split train=%d test=%d; no random split is applied', len(Xtr), len(Xte))

    fit_started = time.perf_counter()
    mean = Xtr.mean(axis=0)                      # mean face from TRAIN only
    Xc = Xtr - mean
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    W = Vt[:K]
    LOGGER.info('Training-only PCA fit completed components=%s elapsed=%.3f s', W.shape, time.perf_counter() - fit_started)

    # ---- held-out reconstruction quality ----
    rec = mean + (Xte - mean) @ W.T @ W
    mse = float(((Xte - rec) ** 2).mean())
    mse_i = ((Xte - rec) ** 2).reshape(len(Xte), -1).mean(axis=1)
    psnr = float((10 * np.log10(1.0 / mse_i)).mean())
    ssim_test = float(np.mean([ssim_fn(Xte[i].reshape(IMG), rec[i].reshape(IMG), data_range=1.0)
                               for i in range(len(Xte))]))

    # ---- held-out 1-NN accuracy (gallery = train projections) ----
    Ztr = Xc @ W.T
    Zte = (Xte - mean) @ W.T
    D = ((Zte[:, None, :] - Ztr[None, :, :]) ** 2).sum(axis=2)
    nn = D.argmin(axis=1)
    acc = float((ytr[nn] == yte).mean())
    # 保留原独立二项近似的输出键，仅供追溯，不作为受试者相关数据的严格区间。
    ci95 = float(1.96 * np.sqrt(acc * (1 - acc) / len(yte)) * 100)

    # ---- held-out separability ratio ----
    same = yte[:, None] == yte[None, :]
    off = ~np.eye(len(yte), dtype=bool)
    wm, bm = same & off, ~same

    def ratio(A):
        sq = (A ** 2).sum(axis=1)
        D2 = sq[:, None] + sq[None, :] - 2 * (A @ A.T)
        dist = np.sqrt(np.maximum(D2, 0))
        return float(dist[wm].mean() / dist[bm].mean())

    sep_test = ratio(Zte)                        # PCA-20 space, held-out projections
    sep_pix_test = ratio(Xte)                    # pixel space, same test subset
    LOGGER.info('Held-out metrics MSE=%.6f PSNR=%.3f SSIM=%.3f accuracy=%.1f%% separation=%.3f',
                mse, psnr, ssim_test, acc * 100, sep_test)

    # ---- mean-face SSIM anchor on all 400 images ----
    gmean = X.mean(axis=0)
    ssim_base = float(np.mean([ssim_fn(X[i].reshape(IMG), gmean.reshape(IMG), data_range=1.0)
                               for i in range(n)]))
    full_ssim, full_mse = recorded_full_sample_metrics()
    LOGGER.info('Full-sample SSIM anchors baseline=%.3f recorded_k20=%.3f gain=%.3f',
                ssim_base, full_ssim, full_ssim - ssim_base)

    # ---- measured encode + reconstruct latency ----
    # 保留原循环的均值、基底和逐图计时口径，避免刷新时改变既有实验定义。
    passes = 5
    times = []
    for pass_index in range(passes):
        t0 = time.perf_counter()
        for i in range(n):
            z = (X[i] - gmean) @ W.T
            _ = gmean + z @ W
        times.append((time.perf_counter() - t0) / n * 1000.0)
        LOGGER.info('Latency pass=%d/%d images=%d mean_ms_per_image=%.4f',
                    pass_index + 1, passes, n, times[-1])
    latency_ms = float(np.median(times))
    out = {
        'n_train': int(len(Xtr)), 'n_test': int(len(Xte)),
        'test_mse': round(mse, 6), 'test_psnr': round(psnr, 3),
        'test_ssim': round(ssim_test, 3), 'test_acc': round(acc * 100, 1),
        'test_acc_ci': round(ci95, 1), 'test_sep': round(sep_test, 3),
        'test_sep_pix': round(sep_pix_test, 3), 'ssim_base': round(ssim_base, 3),
        'ssim_gain': round(full_ssim - ssim_base, 3),
        'latency_ms': round(latency_ms, 2),
        'mse_rise_pct': round((mse / full_mse - 1) * 100, 1),
    }
    LOGGER.info('Optional experiment completed elapsed=%.2f s', time.perf_counter() - started)
    print(json.dumps(out, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    try:
        sys.exit(main())
    except Exception as exc:
        LOGGER.error('Optional experiment failed: %s', exc)
        sys.exit(3)
