# 实验一 · 使用 PCA 实现人脸重建

在 Olivetti faces 数据集上从零实现主成分分析，完成人脸特征提取、特征脸可视化与图像重建，并给出重构质量的定量评估。

## 1. 实验要求与代码对应

| 实验步骤 | 实现位置（`pca_face_reconstruction.py`） |
|---|---|
| ① 实现 PCA：输入数据集 X 与主成分个数 L，输出降维数据、L 个主成分及其特征值（降序） | `pca()`，`method="svd"` / `"cov"` 两条路径 |
| ② 加载 Olivetti faces 数据集 | `load_faces()` |
| ③ 选 18 张人脸用 `matshow` 可视化 | 步骤 3 → `results/fig1_original_faces.png` |
| ④ 降到 20 维 | 步骤 4，`N_COMPONENTS = 20` |
| ⑤ 用 `matshow` 可视化 20 个特征脸（含 PC01–PC20 编号） | 步骤 5 → `results/fig2_eigenfaces.png` |
| ⑥ 随机选一张人脸，观察重构效果 | 步骤 6 → `results/fig3_reconstruction_compare.png`（标注 MSE / PSNR） |
| ⑦ 绘制重构过程：原始人脸 + Σ 系数×特征脸 | 步骤 7 → `results/fig4_reconstruction_process.png` |

7 步之外还实现了 **10 项扩展分析**（均值脸基线、k 扫描、累积重建、残差、双路径等价性、分解计时、身份可分离性、SSIM、1-NN 识别率），对应 `fig5`–`fig9` 与 `metrics_table.csv`，是论文数据的唯一来源。

## 2. 文件说明

| 文件 | 作用 |
|---|---|
| `pca_face_reconstruction.py` | 中文版主脚本，图表标签为中文 |
| `pca_face_reconstruction_en.py` | 英文版，同一流水线，图表标签为英文（供英文论文使用） |
| `data/olivetti_py3.pkz` | sklearn 缓存格式的 Olivetti 数据集，保证离线可跑 |
| `results/` | 9 张结果图 + `metrics_table.csv` + `console_output.txt` |
| `results_en/console_output.txt` | 英文版运行日志；`paper/build.py` 用它交叉校验中英文两次运行的指标是否一致 |
| `paper/` | IEEEtran 论文源码、矢量图、构建脚本 |

## 3. 运行

```bash
python pca_face_reconstruction.py
```

- 环境：Python 3.12+、NumPy、Matplotlib、scikit-learn（见仓库根 `requirements.txt`）
- 耗时：约 40 s（其中分解计时会额外做 3×2 次重复，用于取中位数）
- 随机性：全局 `SEED = 0`，选中的样本恒为索引 172，结果完全可复现
- 输出：直接覆盖 `results/` 下同名文件；控制台内容可重定向保存为 `console_output.txt`

中文字体：脚本按 `Microsoft YaHei → SimHei → SimSun → KaiTi → Noto Sans CJK SC → Arial Unicode MS` 顺序探测，找不到任何中文字体时会回退到默认字体（中文标签显示为方框）。Linux/macOS 上建议先安装 Noto Sans CJK。

## 4. 关键结论

- **k=20 保留 76.34% 方差**，MSE 0.00456、PSNR 23.64 dB；k=0（仅用均值脸）的 MSE 为 0.01927，说明前 20 个主成分解释了约 76% 的重构误差下降。
- **理论与实现自洽**：用特征值之和预测 k=20 的 MSE = 0.004571，实测 0.004559，相对偏差 0.25%。
- **SVD 优于显式协方差矩阵**：4096 维下 SVD 中位 3.08 s，协方差 + `eigh` 中位 13.37 s，约 4.3 倍差距；两者重构结果最大绝对差 `1.97e-07`（浮点噪声级），可视为等价。
- **降维提升可分性**：类内/类间平均距离比 0.675（4096 维）→ 0.562（20 维）；1-NN 留一法识别率 76.0%（k=5）→ 94.8%（k=20）→ 95.5%（k≥50）。
- **SSIM 比 PSNR 更贴近人眼**：k=20 时 PSNR 已达 23.6 dB，但 SSIM 仅 0.698，说明结构细节仍有可见损失；要人眼满意需 k≥50（SSIM 0.812）。

## 5. 论文构建

```bash
cd paper
python build.py          # 需要 Tectonic，见仓库根 README
```

`build.py` 的数字来自 `results/console_output.txt`（不手抄），编译后还会跑交付守卫：缺字、超宽浮动体、浮动体跑到参考文献之后、孤立 `\label`，以及用 PyMuPDF 做的反向文本校验。
