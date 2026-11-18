[English](README.md) | 简体中文

# data

HotpotQA distractor dev 划分的固定、确定性子集。除 `hotpot/` 之外的每个文件都由
`src/peg_bench/build_hotpot.py` 或 `src/peg_bench/build_stability.py` 生成，因此这里没有任何手工编辑的内容。

上游 dump **不随本仓库分发**。请从 <https://hotpotqa.github.io/> 下载，放到
`data/hotpot/hotpot_dev_distractor_v1.json`（或将 `HOTPOT_DEV` 设为其他路径）。
预期 SHA-256：
`4e9ecb5c8d3b719f624d66b60f8d56bf227f03914f5f0753d6fa1b359d7104ea`。

| 路径 | 内容 |
|---|---|
| `hotpot/hotpot_dev_distractor_v1.json` | 上游 dump 的本地副本（被 git 忽略，经 SHA-256 校验） |
| `main/base.csv` | 300 个段落，30 个基础问题（每个 10 段） |
| `main/delta.csv` | 100 个段落，10 个增量问题，标题与 `base.csv` 不相交 |
| `main/queries_base.jsonl` / `queries_delta.jsonl` | 以 `supporting_titles` 作为标准答案的问题 |
| `main/edit.csv` | 10 个基础段落，改写并加入合成的 `ZX-nn` 验证码 |
| `main/delete.csv` | 5 个基础段落（与 `edit.csv` 不相交） |
| `main/edit_probes.jsonl` / `delete_probes.jsonl` | 用于改写/删除行为检查的检索探针 |
| `main/input_base_delta.csv` | GraphRAG update 输入：base + delta |
| `main/input_after_edit.csv` | 改写与删除之后的 GraphRAG update 输入 |
| `main/rename.csv` | 10 个已改写段落，以 `... (2026 revision)` 标题重新入库（对照组 B） |
| `main/input_rename.csv` | 对照组 B 的 GraphRAG update 输入：base + delta + 重命名副本 |
| `main/smoke.csv` | 用于流水线冒烟测试的两个段落 |
| `main/selection.json` | 选中了哪些 HotpotQA 问题，以及重叠断言 |
| `stability/base.csv` | 900 个段落，90 个基础问题 |
| `stability/delta.csv` | 300 个段落，30 个增量问题 |
| `stability/folds.json` | 120 个稳定性问题的 3 折划分 |
| `stability/*.csv`, `queries_*.jsonl` | 与 `main/` 相同的结构，用于稳定性集合 |

基础与增量的标题从不重叠。GraphRAG v3.3.0 依据文档*标题*检测增量输入
（`graphrag/index/update/incremental_index.py`），因此标题一旦重叠，"新"段落就会被静默丢弃。

HotpotQA 由 CMU 发布于 <https://hotpotqa.github.io/>。重新分发 dump 之前，请查看该站点上的当前许可；
本仓库仅将其用于基准测试。
