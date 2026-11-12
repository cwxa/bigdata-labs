## 这个 PR 做了什么

<!-- 一句话结论：改了什么、为什么改。不要只贴 diff 摘要。 -->

## 关联 Issue

<!-- 例如：Closes #12。没有关联 Issue 的小改动可以写「无」并说明理由。 -->

## 改动类型

- [ ] `feat` 新增功能或分析维度
- [ ] `fix` 修正实现 / 数据错误
- [ ] `docs` 文档
- [ ] `test` 测试
- [ ] `refactor` 重构（不改变行为）
- [ ] `chore` 工程配置

## 验证结果

<!-- 请贴实际输出，不要只勾选。「跑过了」不是验证结果。 -->

```text
$ ruff check .
All checks passed!

$ cd lab2-association-rules/code && python -m unittest test_apriori test_fp_growth
Ran 23 tests in 0.860s
OK

$ cd lab1-pca-face-reconstruction && python -m unittest test_pca
Ran 11 tests in 0.055s
OK
```

## 对结果数据与论文的影响

- [ ] 不影响 `results/` 与论文数字
- [ ] 影响 `results/` 与论文数字，已重跑脚本并重建论文（`paper/build.py` 全守卫通过）
- [ ] 影响 `results/` 但论文无需改动（请说明原因）

## 检查清单

- [ ] 提交信息符合 [Conventional Commits](https://www.conventionalcommits.org/)，一条提交只做一件事
- [ ] 新增逻辑有中文注释说明「为什么」，原有注释未删除
- [ ] 未提交含姓名学号的交付物、第三方作业或超过 10 MB 的二进制文件
- [ ] 未手改 `numbers.tex` / `tables/*.tex` 等生成文件
- [ ] 文档（README / CHANGELOG）已同步更新
