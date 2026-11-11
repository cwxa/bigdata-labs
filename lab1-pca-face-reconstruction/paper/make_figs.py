# -*- coding: utf-8 -*-
"""Generate paper-native figures (column 3.5in / text 7in) from the experiment
console and the dataset, so every label stays legible after scaling.
Figures produced: cumulative (wide strip), residual, separation, task, efficiency.

统一导出 PNG 预览和含可编辑文字的 PDF；只读取既有计时结果，不重复性能测试。
"""
from __future__ import annotations

import io
import json
import logging
from pathlib import Path
import re
import sys
from time import perf_counter

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter
from scipy.io import loadmat

PROJ = Path(__file__).resolve().parent
BASE = PROJ.parent
FIG = PROJ / 'figs'
QA = FIG / 'qa'
CONSOLE = BASE / 'results' / 'console_output.txt'
FIG.mkdir(exist_ok=True)
QA.mkdir(exist_ok=True)

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
LOGGER = logging.getLogger(__name__)
logging.getLogger('fontTools').setLevel(logging.WARNING)
BLUE, RED, GRAY = '#1E4F8C', '#B55243', '#7D8895'
IMG = (64, 64)
matplotlib.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 7.5,
    'figure.facecolor': 'white',
    'axes.titlesize': 7.5,
    'axes.labelsize': 7.5,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.65,
    'lines.linewidth': 1.25,
    'legend.frameon': False,
    'pdf.fonttype': 42,
    'svg.fonttype': 'none',
    'savefig.facecolor': 'white',
})

# 本机审计工具仅用于量取最终坐标，不参与实验计算或图形绘制。
AUDIT_DIR = Path.home() / '.codex' / 'skills' / 'nature-figure' / 'scripts'
if AUDIT_DIR.exists():
    sys.path.insert(0, str(AUDIT_DIR))
    from audit_panel_alignment import require_matplotlib_panel_alignment
else:
    require_matplotlib_panel_alignment = None


def save_figure(fig, name, *, axes=None, row_groups=None, column_groups=None):
    """固定最终物理尺寸；审计坐标后导出，避免 tight 裁剪改变字号比例。"""
    fig.canvas.draw()
    if require_matplotlib_panel_alignment is not None:
        options = {'axes': axes} if axes is not None else {}
        if row_groups is not None:
            options['row_groups'] = row_groups
        if column_groups is not None:
            options['column_groups'] = column_groups
        require_matplotlib_panel_alignment(
            fig, json_out=QA / f'{name}.alignment.json',
            tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True, **options)
    else:
        LOGGER.warning('Panel alignment auditor unavailable; visual review is required for %s', name)
    fig.savefig(FIG / f'{name}.png', dpi=600)
    fig.savefig(FIG / f'{name}.pdf')
    fig.savefig(FIG / f'{name}.svg')
    LOGGER.info('Exported %s: %.2f x %.2f in', name, *fig.get_size_inches())
    plt.close(fig)


def panel_label(ax, label, *, x=0, y=1.13):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=8,
            fontweight='bold', va='bottom', ha='left', clip_on=False)


def photo(ax, image, title='', *, size=7.5):
    # 所有原图与重建图共用 [0, 1] 显示范围，避免逐图对比度造成误导。
    ax.imshow(image.reshape(IMG), cmap='gray', vmin=0, vmax=1,
              interpolation='nearest')
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=size, pad=3)


def style_axes(ax):
    ax.tick_params(length=2.5, width=0.6, pad=2)
    ax.grid(axis='y', alpha=0.18, lw=0.55)
    ax.set_axisbelow(True)


started = perf_counter()
txt = io.open(CONSOLE, encoding='utf-8').read()
svd_m, svd_lo, svd_hi = map(float, re.search(
    r'SVD\s+中位 ([\d.]+) s（范围 ([\d.]+)–([\d.]+) s）', txt).groups())
eig_m, eig_lo, eig_hi = map(float, re.search(
    r'协方差\+eigh 中位 ([\d.]+) s（范围 ([\d.]+)–([\d.]+) s）', txt).groups())
ssim = {int(k): float(v) for k, v in re.findall(
    r'k=(\d+): ([\d.]+)', re.search(r'扩展分析9[^\n]*', txt).group(0))}
acc = {int(k): float(v) for k, v in re.findall(
    r'k=(\d+): ([\d.]+)', re.search(r'扩展分析10[^\n]*', txt).group(0))}
sweep = {}
for match in re.finditer(r'^\s+(\d+)(\(基线\))?\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)%\s*$', txt, re.M):
    sweep[int(match.group(1))] = float(match.group(3))

# ---------- data ----------
# 复现 sklearn 缓存读取后的 float32 归一化与转置顺序，且全程离线。
raw = loadmat(BASE / 'data' / 'olivettifaces.mat')['faces'].T.copy()
faces = raw.astype(np.float32)
faces -= faces.min()
faces /= faces.max()
X = faces.reshape((400, 64, 64)).transpose(0, 2, 1).reshape(400, -1).astype(np.float64)
n, d = X.shape
mean = X.mean(axis=0)
Xc = X - mean
_, S, Vt = np.linalg.svd(Xc, full_matrices=False)
evals = (S ** 2) / (n - 1)
cum = np.cumsum(evals) / evals.sum() * 100
idx = int(re.search(r'随机选取样本：索引 (\d+)', txt).group(1))
ks_show = [1, 2, 5, 10, 20, 50]
reconstructions = {k: mean + (Xc[idx] @ Vt[:k].T) @ Vt[:k] for k in ks_show}
rec20 = reconstructions[20]
resid = np.abs(X[idx] - rec20)
LOGGER.info('Loaded %d faces, %d pixels; sample=%d; original timing medians: SVD=%.2fs, eigh=%.2fs',
            n, d, idx, svd_m, eig_m)

# 前 20 个特征脸采用紧凑 10×2 布局；每幅采用对称色阶，保留正负载荷。
fig, axes = plt.subplots(2, 10, figsize=(7, 1.75))
fig.subplots_adjust(left=0.015, right=0.985, bottom=0.025, top=0.87,
                    wspace=0.13, hspace=0.4)
for pc, ax in enumerate(axes.flat):
    eigenface = Vt[pc].reshape(IMG)
    limit = np.abs(eigenface).max()
    ax.imshow(eigenface, cmap='gray', vmin=-limit, vmax=limit, interpolation='nearest')
    ax.set_axis_off()
    ax.set_title(f'PC{pc + 1:02d}', fontsize=7.5, pad=2)
save_figure(fig, 'fig2_eigenfaces')

# ---------- fig: cumulative strip (text-width native, 7 in) ----------
fig, axes = plt.subplots(1, len(ks_show) + 1, figsize=(7, 1.23))
fig.subplots_adjust(left=0.01, right=0.99, top=0.72, bottom=0.02, wspace=0.12)
photo(axes[0], X[idx], 'Original')
for ax, k in zip(axes[1:], ks_show):
    mse_k = ((X[idx] - reconstructions[k]) ** 2).mean()
    photo(ax, reconstructions[k], f'k={k}\nMSE {mse_k:.4f}', size=7)
save_figure(fig, 'fig6_cumulative_reconstruction')

# ---------- fig: residual (column native, 3.5 in) ----------
fig, axes = plt.subplots(1, 3, figsize=(3.5, 1.35))
fig.subplots_adjust(left=0.015, right=0.985, top=0.81, bottom=0.03, wspace=0.12)
for ax, (image, title) in zip(axes, [(X[idx], 'Original'), (rec20, 'k=20'),
                                             (np.clip(resid * 3, 0, 1), 'Residual ×3')]):
    photo(ax, image, title)
save_figure(fig, 'fig7_residual')

# ---------- fig: separation (column native) ----------
labels = np.repeat(np.arange(40), 10)
same = labels[:, None] == labels[None, :]
off = ~np.eye(n, dtype=bool)
wm = same & off
bm = ~same


def pairwise_dist(A):
    sq = (A ** 2).sum(axis=1)
    D2 = sq[:, None] + sq[None, :] - 2 * (A @ A.T)
    return np.sqrt(np.maximum(D2, 0))


distances = [pairwise_dist(X), pairwise_dist(Xc @ Vt[:20].T)]


def separation_panel(ax, col, *, show_ylabel=True):
    dist = distances[col]
    w, b = dist[wm], dist[bm]
    bins = np.linspace(0, np.percentile(b, 99), 40)
    ax.hist(w, bins=bins, alpha=0.72, label='Within subject', color=BLUE, density=True)
    ax.hist(b, bins=bins, alpha=0.50, label='Between subjects', color=RED, density=True)
    ax.set_title(['Pixel space', 'PCA-20 space'][col], pad=7)
    ax.set_xlabel('Distance')
    if show_ylabel:
        ax.set_ylabel('Density')
    ax.set_xlim(0, 22)
    ax.set_xticks([0, 10, 20])
    style_axes(ax)


fig, axes = plt.subplots(1, 2, figsize=(3.5, 2.3))
fig.subplots_adjust(left=0.16, right=0.985, bottom=0.22, top=0.72, wspace=0.50)
for col, ax in enumerate(axes):
    separation_panel(ax, col, show_ylabel=col == 0)
    panel_label(ax, 'ab'[col], y=1.1)
handles, legend_labels = axes[0].get_legend_handles_labels()
fig.legend(handles, legend_labels, loc='upper center', ncol=1, bbox_to_anchor=(0.54, 1.0))
save_figure(fig, 'fig8_identity_separation')


def task_panel(ax, metric):
    values = ssim if metric == 'ssim' else acc
    ks = sorted(values)
    ax.plot(ks, [values[k] for k in ks], 'o-' if metric == 'ssim' else 's-',
            color=BLUE, ms=3)
    ax.axvline(20, color=GRAY, ls=':', lw=0.85)
    ax.set_xlabel('Components, k')
    ax.set_ylabel('SSIM' if metric == 'ssim' else 'Accuracy (%)')
    ax.set_title('Structural similarity' if metric == 'ssim' else '1-NN recognition', pad=7)
    if metric == 'ssim':
        ax.set_ylim(0.4, 0.95)
        ax.set_xticks([0, 50, 100])
    else:
        ax.set_ylim(70, 100)
        ax.set_xticks([0, 75, 150])
    style_axes(ax)


# ---------- fig: task metrics (column native) ----------
fig, axes = plt.subplots(1, 2, figsize=(3.5, 2.05))
fig.subplots_adjust(left=0.16, right=0.985, bottom=0.24, top=0.8, wspace=0.65)
for ax, metric, letter in zip(axes, ['ssim', 'acc'], 'ab'):
    task_panel(ax, metric)
    panel_label(ax, letter, y=1.1)
save_figure(fig, 'fig9_task_metrics')

# ---------- fig: efficiency (column native) ----------
DIM = d
fig, axes = plt.subplots(1, 2, figsize=(3.5, 2.25))
fig.subplots_adjust(left=0.17, right=0.985, bottom=0.25, top=0.73, wspace=0.64)
ax = axes[0]
med = [eig_m, svd_m]
assert np.all(np.asarray(med) > 0), 'Timing data must be strictly positive for log axes'
hi = [eig_hi, svd_hi]
ax.bar(['eigh', 'SVD'], med, width=0.5, color=[RED, BLUE],
       yerr=[[eig_m - eig_lo, svd_m - svd_lo], [eig_hi - eig_m, svd_hi - svd_m]],
       capsize=2.5, error_kw=dict(lw=0.8, ecolor='#333333'))
ax.set_yscale('log')
ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:g}'))
ax.set_ylabel('Time (s)')
ax.set_title('Decomposition time', pad=7)
for i, value in enumerate(med):
    ax.text(i, hi[i] * 1.35, f'{value:.2f}', ha='center', va='bottom', fontsize=7)
ax.set_ylim(min(med) / 3, max(hi) * 4)
style_axes(ax)
panel_label(ax, 'a', y=1.32)

ax = axes[1]
# 这里只估算 float64 数组的十进制 MB，不宣称是性能测试的峰值内存。
vals = [DIM * DIM * 8 / 1e6, n * DIM * 8 / 1e6, n * 20 * 8 / 1e6]
assert np.all(np.asarray(vals) > 0), 'Storage estimates must be strictly positive for log axes'
ax.bar(['Cov.', 'Data', 'Code'], vals, width=0.55, color=[RED, GRAY, BLUE])
ax.set_yscale('log')
ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:g}'))
ax.set_ylabel('Storage (MB)')
ax.set_title('Estimated float64\nstorage', pad=7)
for i, value in enumerate(vals):
    ax.text(i, value * 1.5, f'{value:.1f}' if value >= 1 else f'{value:.2f}',
            ha='center', va='bottom', fontsize=7)
ax.set_ylim(min(vals) / 4, max(vals) * 15)
style_axes(ax)
panel_label(ax, 'b', y=1.32)
save_figure(fig, 'fig10_efficiency')

LOGGER.info('Paper-native figures regenerated (7.0in strip + 3.5in columns)')

# ---------- merged figure A: cumulative (top) + residual (bottom), text width ----------
fig = plt.figure(figsize=(7, 2.5))
gs = GridSpec(2, 7, figure=fig, left=0.015, right=0.985, top=0.86,
              bottom=0.02, hspace=0.65, wspace=0.12)
top_axes = [fig.add_subplot(gs[0, j]) for j in range(7)]
photo(top_axes[0], X[idx], 'Original')
for ax, k in zip(top_axes[1:], ks_show):
    photo(ax, reconstructions[k], f'k={k}', size=7.5)
bottom_axes = [fig.add_subplot(gs[1, j]) for j in [2, 3, 4]]
for ax, (image, title) in zip(bottom_axes, [(X[idx], 'Original'), (rec20, 'k=20'),
                                                (np.clip(resid * 3, 0, 1), 'Residual ×3')]):
    photo(ax, image, title)
save_figure(fig, 'fig5_cumulative_residual')

# ---------- merged figure B: separability (top) + task metrics (bottom), column width ----------
fig, axes = plt.subplots(2, 2, figsize=(3.5, 3.4))
fig.subplots_adjust(left=0.16, right=0.985, bottom=0.12, top=0.84,
                    hspace=0.92, wspace=0.65)
for col in range(2):
    separation_panel(axes[0, col], col, show_ylabel=col == 0)
for col, metric in enumerate(['ssim', 'acc']):
    task_panel(axes[1, col], metric)
for ax, letter in zip(axes.flat, 'abcd'):
    panel_label(ax, letter, x=-0.26, y=1.17)
handles, legend_labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, legend_labels, loc='upper center', ncol=1,
           bbox_to_anchor=(0.55, 1.01), handlelength=1.4, labelspacing=0.2)
save_figure(fig, 'fig6_metrics_panels')

LOGGER.info('Merged figures written: fig5_cumulative_residual, fig6_metrics_panels')


def reconstruction_curves(axes):
    ax = axes[0]
    ks_sorted = sorted(k for k in sweep if k > 0)
    assert np.all(np.asarray(list(sweep.values())) > 0), 'MSE data must be strictly positive for log axes'
    ax.semilogy(ks_sorted, [sweep[k] for k in ks_sorted], 'o-', color=BLUE, ms=3)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:g}'))
    ax.axhline(sweep[0], color=RED, ls='--', lw=1,
               label=f'Mean-face baseline: {sweep[0]:.4f}')
    ax.axvline(20, color=GRAY, ls=':', lw=0.85)
    ax.set_xlabel('Components, k')
    ax.set_ylabel('MSE')
    ax.set_title('Reconstruction error', pad=8)
    ax.legend(fontsize=7, loc='upper right', handlelength=1.6)
    ax.set_ylim(0.0005, 0.07)
    ax.set_xticks([0, 50, 100, 150])
    style_axes(ax)
    panel_label(ax, 'a', y=1.15)
    ax = axes[1]
    ax.plot(np.arange(1, n + 1), cum, color=BLUE)
    ax.axhline(90, color=RED, ls='--', lw=0.85)
    ax.axvline(66, color=RED, ls=':', lw=0.85)
    ax.set_xlabel('Components, k')
    ax.set_ylabel('Cumulative variance (%)')
    ax.set_title('Retained variance', pad=8)
    ax.text(180, 28, f'k=66: {cum[65]:.2f}%', fontsize=7.5, color=RED)
    ax.set_ylim(0, 105)
    ax.set_xticks([0, 100, 200, 300, 400])
    style_axes(ax)
    panel_label(ax, 'b', y=1.15)


# 兼容既有指标曲线文件名，内容与主图定量面板完全一致。
fig, axes = plt.subplots(1, 2, figsize=(7, 2.05))
fig.subplots_adjust(left=0.08, right=0.985, bottom=0.24, top=0.78, wspace=0.30)
reconstruction_curves(axes)
save_figure(fig, 'fig5_metric_curves')

# ---------- results dashboard: curves + strip + residual (text width) ----------
# 四面板形成定量到实例的证据链：误差、方差、逐级重建、残差定位。
fig = plt.figure(figsize=(7, 4.1))
outer = GridSpec(2, 1, figure=fig, left=0.08, right=0.985, top=0.90,
                 bottom=0.025, height_ratios=[1, 1.30], hspace=0.56)
top_grid = outer[0].subgridspec(1, 2, wspace=0.30)
curve_axes = [fig.add_subplot(top_grid[0, j]) for j in range(2)]
reconstruction_curves(curve_axes)
bottom_grid = outer[1].subgridspec(2, 5, wspace=0.13, hspace=0.47)
image_axes = []
for j, k in enumerate(ks_show):
    ax = fig.add_subplot(bottom_grid[j // 3, j % 3])
    photo(ax, reconstructions[k], f'k={k} · {((X[idx] - reconstructions[k]) ** 2).mean():.4f}', size=7)
    image_axes.append(ax)
for row, col, image, title in [(0, 3, X[idx], 'Original'), (0, 4, rec20, 'k=20'),
                              (1, 3, np.clip(resid * 3, 0, 1), 'Residual ×3')]:
    ax = fig.add_subplot(bottom_grid[row, col])
    photo(ax, image, title, size=7)
    image_axes.append(ax)
note_ax = fig.add_subplot(bottom_grid[1, 4])
note_ax.set_axis_off()
note_ax.text(0.02, 0.85, f'Sample {idx}\nMSE {(resid ** 2).mean():.5f}\n\nAbsolute error\nscaled ×3',
             transform=note_ax.transAxes, fontsize=7, va='top', linespacing=1.1)
# c/d 是面板组标题，使用固定物理位置避免随图像纵横比发生漂移。
fig.canvas.draw()
cbox, dbox = image_axes[0].get_position(), image_axes[6].get_position()
fig.text(cbox.x0, cbox.y1 + 0.088, 'c', fontsize=8, fontweight='bold')
fig.text(cbox.x0 + 0.025, cbox.y1 + 0.088, 'Progressive reconstruction (k · MSE)', fontsize=7.5)
fig.text(dbox.x0, dbox.y1 + 0.088, 'd', fontsize=8, fontweight='bold')
fig.text(dbox.x0 + 0.025, dbox.y1 + 0.088, 'Reconstruction and residual', fontsize=7.5)
# 曲线面板与同一底排的等大图像分别形成可比较对齐组。
save_figure(fig, 'fig4_results_dashboard', axes=curve_axes + image_axes,
            row_groups=[{'id': 'quantitative', 'panels': ['a', 'b']},
                        {'id': 'images-upper', 'panels': ['c', 'd', 'e', 'i', 'j']},
                        {'id': 'images-lower', 'panels': ['f', 'g', 'h', 'k']}],
            column_groups=[{'id': 'progressive-1', 'panels': ['c', 'f']},
                           {'id': 'progressive-2', 'panels': ['d', 'g']},
                           {'id': 'progressive-3', 'panels': ['e', 'h']},
                           {'id': 'residual', 'panels': ['i', 'k']}])
LOGGER.info('Results dashboard written: fig4_results_dashboard')

# 审计说明记录显示变换、数据口径和结果溯源，便于后续编辑保持一致。
(QA / 'figure_notes.json').write_text(json.dumps({
    'backend': 'Python / Matplotlib',
    'data': {'samples': n, 'pixels': d, 'sample_index': idx, 'exclusions': 0},
    'statistics': {'timing': 'Existing console medians and min/max from 3 runs; not rerun',
                   'task_metrics': 'Existing console averages and leave-one-out accuracy',
                   'histograms': 'Ordered off-diagonal within-subject and all between-subject pairs; original 99th-percentile bin bound retained'},
    'images': {'original_and_reconstruction': 'Uncropped, grayscale range [0,1]',
               'residual': 'Absolute error multiplied by 3 and clipped to [0,1]',
               'eigenfaces': 'Per-component symmetric gray range; sign is SVD-dependent'},
    'storage': 'Decimal MB from float64 element counts; not measured peak memory',
    'font_floor_pt': 7,
    'exports': 'Editable PDF/SVG and 600 dpi PNG previews; no TIFF needed for the IEEE PDF workflow',
    'manual_review': 'All panels reviewed for clipping, readable titles, equivalent image scales, baseline placement and consistent colors',
}, ensure_ascii=False, indent=2), encoding='utf-8')
LOGGER.info('All figures completed in %.2f seconds', perf_counter() - started)
