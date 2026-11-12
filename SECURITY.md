# 安全策略

## 支持范围

本仓库是课程实验的归档项目，不发布生产软件，仅对最新提交提供安全响应：

| 版本 | 状态 |
|---|---|
| `main` 最新提交 | 接受报告 |
| 已发布 tag | 仅接受报告，不承诺回补 |

## 报告漏洞

**请勿通过公开 Issue 报告安全问题。** 请使用 GitHub 的私密报告通道：

1. 打开 <https://github.com/cwxa/bigdata-labs/security/advisories/new>；
2. 或在该仓库的 **Security → Advisories → Report a vulnerability** 提交。

报告中请包含：受影响的文件与行号、复现步骤、影响判断（例如"可导致内存占用翻倍"而非"有漏洞"）。

我会在 7 天内确认收到，并在修复后于 Advisory 中公开说明。由于这是个人课程项目，响应时间为尽力而为，不做 SLA 承诺。

## 已知的非风险项

以下情况属于预期行为，**不需要**报告：

- **`data/olivetti_py3.pkz` 是 pickle 格式**：它由 scikit-learn 的 `fetch_olivetti_faces` 生成，仅包含 NumPy 数组与字典。请勿加载来源不明的 `.pkz` / `.pkl` 文件——pickle 反序列化本身即可执行任意代码，这是格式特性而非本仓库的缺陷。
- **论文构建脚本会调用外部 `tectonic` 可执行文件**：路径来自 `TECTONIC` 环境变量或 `PATH`，请自行确认来源可信。
- **`lab2` 的 bootstrap 分析使用固定随机种子**：结果可复现是刻意设计，不是熵源缺陷。

## 依赖安全

依赖范围声明在 `requirements.txt`，均为广泛使用的科学计算库。请通过 `pip install --upgrade` 与 Dependabot 等常规手段跟踪上游 CVE；本仓库不另行维护依赖版本清单。
