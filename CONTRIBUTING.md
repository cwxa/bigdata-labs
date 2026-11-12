# 贡献指南

本仓库是《大数据分析与处理》课程实验的归档与整理版，主要目的是留存可复现的实验代码与论文源码。它不是一个活跃维护的库，但仍欢迎纠错与改进。

## 报告问题

- **数据或结论错误**：请开 Issue，并附上「复现命令 + 实际输出 + 期望输出」。涉及数字的，请说明是重跑脚本后仍不一致，还是仅阅读代码发现的。
- **安全相关问题**：不要开公开 Issue，请按 [SECURITY.md](SECURITY.md) 的流程处理。
- **论文表述问题**：请指出具体小节或 `\label`，便于定位。

## 分支与提交

分支模型（`main` 为保护分支，只接受 PR 合入）：

| 分支 | 用途 |
|---|---|
| `main` | 稳定状态，对应已发布 tag |
| `feature/<简述>` | 新增实验、新增分析维度 |
| `bugfix/<简述>` | 修正实现或数据错误 |
| `docs/<简述>` | 仅文档改动 |

提交信息遵循 [Conventional Commits](https://www.conventionalcommits.org/)，一条提交只做一件事：

```
feat(lab2): 增加 lift 的 Wilson 置信区间估计
fix(lab1): 修正 L=0 时重构结果未退化为均值脸的边界
docs: 补充实验二数据来源说明
test(lab1): 覆盖 L>min(n,d) 的截断行为
chore: 统一 ruff 检查配置到 pyproject.toml
```

`type` 取 `feat` / `fix` / `docs` / `test` / `refactor` / `chore`；`scope` 取 `lab1` / `lab2` / `paper` / 留空。

## 代码约定

1. **保持可独立运行**。`lab1` 的两个脚本刻意保持为自包含单文件（中文版/英文版），便于课程归档与逐行对照；不要为了 DRY 把它们拆成包——这是有意为之的权衡，不是疏漏。
2. **中文注释**。新增逻辑请写中文注释，说明「为什么」而不是复述「做了什么」；原有注释不得删除。控制台与图表标签沿用中文（这是课程交付形态），构建脚本与新增日志用英文。
3. **路径不写死**。任何数据与产物路径都必须基于 `Path(__file__)` 解析，禁止依赖调用者的工作目录，也不要把本机绝对路径提交进来。
4. **不做手动版本管理**。不要在任何文件里维护版本号或变更历史——版本由 Git tag 与 [CHANGELOG.md](CHANGELOG.md) 单点承载（SSOT）。
5. **不引入重量级依赖**。核心算法坚持手写实现（不调 `mlxtend`、`sklearn.decomposition`），这是本仓库的教学价值所在。

## 数据与隐私红线

以下内容**禁止**提交：

- 含姓名、学号的成绩交付物（Word 报告、汇报 PPT、已编译的署名 PDF 原件）；
- 他人作业、参考资料 PDF（版权与学术诚信）；
- 真实个人数据或任何未脱敏的标识符；
- 超过 10 MB 的二进制文件（数据集请改为提供下载脚本或缓存说明）。

## 本地验证清单

改动后请依次跑通以下三条，全部通过再提 PR：

```bash
# 1. 静态检查（与 CI 同一口径）
ruff check .

# 2. 单元测试：23 + 11 个用例
cd lab2-association-rules/code && python -m unittest test_apriori test_fp_growth -v && cd ../..
cd lab1-pca-face-reconstruction && python -m unittest test_pca -v && cd ..

# 3. 完整流水线（会重写 results/，提交前请确认差异可解释）
cd lab1-pca-face-reconstruction && python pca_face_reconstruction.py && cd ..
cd lab2-association-rules/code && python run_experiments.py && cd ../..
```

> CI 只跑第 1、2 项。第 3 项依赖绘图后端与数据集缓存，由贡献者在本地人工确认——这不是偷懒，而是避免在无头环境里用「没报错」冒充「结果正确」。

## 修改论文数字的唯一正确姿势

论文里的数字全部由脚本生成，**不要手改** `numbers.tex` 或 `tables/*.tex`：

```bash
cd lab1-pca-face-reconstruction/paper && python build.py    # 重读 results/ → 重生成宏 → 编译
cd lab2-association-rules/paper      && python build.py    # 重读 results/*.csv → 重生成宏与表格 → 编译
```

若实验结论发生变化，请同时更新对应 `README.md` 的「关键结论」小节，保持文档与数据一致。

## 新增实验

新建 `lab3-<主题>/`，目录结构沿用现有约定：

```
lab3-<主题>/
├── README.md                 # 目标、方法、运行方式、关键结论
├── <主脚本>.py               # 自包含、带 if __name__ == "__main__" 保护
├── test_<模块>.py            # 至少覆盖数学性质与边界条件
├── data/                     # 小型数据或缓存；大文件给下载说明
├── results/                  # 可提交的结果数据与图
└── paper/                    # 如需论文源码，沿用 build.py + 交付守卫
```

并在根 `README.md` 的实验一览表中补一行。
