[English](README.md) | 简体中文

# results 索引

每个文件由对应脚本生成；`peg_gr_001` 与 `peg_gr_002` 不是两个版本，而是同一评测协议在两个索引实例上的结果。

| 文件 | 生成命令 | 内容 | 备注 |
|---|---|---|---|
| `peg_gr_001.json` | `evaluate`（update 实例） | 五臂检索、可见性探针、成本 | 主运行；宿主臂在该实例上向量表不完整 |
| `peg_gr_002.json` | `evaluate --update-root artifacts/runs/edit` | 同上，宿主臂改用向量表完整的实例 | 论文 5.7 的宿主数字来自此文件 |
| `stability_3fold.json` | `evaluate_stability` | 3 折、120 问 | 需稳定性索引 |
| `ablation.json` | `ablation` | 三种通道变体 | 复用已有宿主排名 |
| `baselines.json` | `baselines` | BM25、段落向量、文本单元向量、basic search | 含时间戳可行性结论 |
| `controls.json` | `controls --paragraph-results results/peg_gr_002.json` | 三种旁路、多跳覆盖、bootstrap 区间 | 零生成调用 |
| `reindex.json` | `reindex_control` | 换标题重入库（对照组 B） | 需 rename 索引 |
| `generation.json` | `generation` | 三臂答案 EM/F1 与探针 | 约 140 次生成调用 |
| `local_effects.json` | `local_effects` | 邻域 Jaccard、删除副作用、规模延迟 | 零生成调用 |
| `rebuild_variance.json` | `index_variance` | 三次重建的抽取密度与宿主指标 | 需 3 个重建实例 |
| `integration.json` | `integration_cost` | 读取宿主文档表并建立 PEG 状态 | 零生成调用 |

约定：涉及宿主检索的跨臂比较必须在同一次运行内完成；生成指标存在 ±1 题波动（论文局限 11）。
