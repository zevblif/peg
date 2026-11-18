[English](README.md) | 简体中文

# PEG：面向图 RAG 的非侵入增量可见性通道

本仓库在真实 Microsoft GraphRAG v3.3.0 宿主上，测量同标题改写与删除在索引中的实际行为，并给出一个不修改宿主的段落级可见性通道（PEG）。

能力边界：**PEG 不提升宿主的图归纳能力，也不提升常规检索精度。** 它保证的是"索引重建之前，改过的正文与删除状态在检索层可见"，并把这一保证写成操作契约与非侵入定理。

- 论文：[docs/PEG-visibility-zh.md](docs/PEG-visibility-zh.md)

## 1. 研究定位

| 项 | 内容 |
|---|---|
| 类型 | 系统 / 契约型实证研究 |
| 主张 | 段落级独立通道可在宿主重建前维持当前版本；操作满足五条行为性质；写入不触碰宿主 |
| 不主张 | 检索精度提升、图归纳增强、端到端答案质量普遍更优 |
| 宿主 | GraphRAG v3.3.0（源码安装，不重分发） |
| 模型 | DeepSeek-V4.1-Flash（关闭思考）+ `bge-m3-8k`（同权重，8k 上下文） |
| 数据 | HotpotQA 固定子集：30 + 10 问（300 + 100 段）；90 + 30 问（900 + 300 段） |

## 2. 快速开始

依赖 Python 3.12、一个 OpenAI 兼容 chat 端点、一个本地 Ollama 嵌入端点、GraphRAG v3.3.0 源码。

```powershell
# 0. 环境变量（只设一次；密钥不落盘）
$env:PEG_CHAT_PROVIDER = "openai"
$env:PEG_CHAT_MODEL    = "deepseek-ai/DeepSeek-V4.1-Flash"
$env:PEG_CHAT_API_BASE = "<https://.../v1>"
$env:PEG_CHAT_API_KEY  = "<api key>"
$env:PEG_CHAT_DISABLE_THINKING = "1"
$env:PEG_EMBED_MODEL = "bge-m3-8k"
$env:PEG_EMBED_BATCH = "8"
$env:PEG_CONCURRENCY = "24"
$env:PYTHONPATH = "src"
```

```powershell
# 1. 数据（确定性，无网络）
python -m peg_bench.build_hotpot
python -m peg_bench.build_stability
python -m peg_bench.make_inputs

# 2. 索引：基础 -> 增量 -> 修改/删除
python -m peg_bench.patch_settings --root artifacts/runs/base   --csv data/main/base.csv
python -m peg_bench.run_graphrag index  --root artifacts/runs/base
python -m peg_bench.patch_settings --root artifacts/runs/update --csv data/main/input_base_delta.csv
python -m peg_bench.run_graphrag update --root artifacts/runs/update
python -m peg_bench.patch_settings --root artifacts/runs/edit   --csv data/main/input_after_edit.csv
python -m peg_bench.run_graphrag update --root artifacts/runs/edit

# 2b. 稳定性索引（900 + 300 段；约 5,400 + 3,300 次生成调用，可跳过）
python -m peg_bench.patch_settings --root artifacts/runs/stab_base   --csv data/stability/base.csv
python -m peg_bench.run_graphrag index  --root artifacts/runs/stab_base
python -m peg_bench.patch_settings --root artifacts/runs/stab_update --csv data/stability/input_base_delta.csv
python -m peg_bench.run_graphrag update --root artifacts/runs/stab_update

# 3. PEG 状态与主评测
python -m peg_bench.build_peg_state
python -m peg_bench.evaluate `
  --base-root artifacts/runs/base --update-root artifacts/runs/update --edit-root artifacts/runs/edit `
  --peg-state artifacts/states/peg_base_delta --peg-behavior-state artifacts/states/peg_edit_delete `
  --output results/peg_gr_001.json

# 4. 补充实验（generation 与 index_variance 之外的步骤为零生成调用）
python -m peg_bench.ablation
python -m peg_bench.evaluate_stability --base-root artifacts/runs/stab_base --update-root artifacts/runs/stab_update `
  --peg-state artifacts/states/peg_stab_base_delta --data-dir data/stability
python -m peg_bench.controls --paragraph-results results/peg_gr_002.json
python -m peg_bench.baselines
python -m peg_bench.local_effects
python -m peg_bench.index_variance --roots artifacts/runs/rebuild_1 artifacts/runs/rebuild_2 artifacts/runs/rebuild_3
python -m peg_bench.generation --concurrency 8
```

一键入口：`scripts/reproduce.ps1`（主实验）、`scripts/smoke.ps1`（冒烟）。`artifacts/` 是被 `.gitignore` 排除的派生目录。

## 3. 主要结果

### 3.1 改写与删除的可见性

| 行为 | GraphRAG v3.3.0 | PEG |
|---|---|---|
| 改写标识串出现在返回正文 | 0 / 10 | **10 / 10** |
| 删除后陈旧标题仍被返回 | 5 / 5 | **0 / 5** |

宿主增量按标题判定：同标题改写不进新增集合，标题消失只记录不执行。PEG 用原地替换与墓碑补齐这两点，对应论文的 P1 与 P3。

### 3.2 检索与覆盖

| 通道 | 基础 | 增量 |
|---|---|---|
| 段落层 | 0.933 / 1.000 / 0.961 | 0.900 / 1.000 / 0.950 |

注：生成答案的指标在重跑时有 ±1 题波动，比较须在同一次运行内完成（论文局限 11）。

多跳覆盖：全部支撑段落同时进入 top-5 的比例 0.80、top-10 0.95（宿主图检索 0.40 / 0.575）。置信区间（2,000 轮）：段落层 Hit@1 0.925 [0.825, 1.000]，MRR 0.958 [0.904, 1.000]。

### 3.3 对照与基线

| 方法 | 基础 Hit@1/5/MRR | 增量 Hit@1/5/MRR |
|---|---|---|
| BM25 | 0.633 / 0.967 / 0.762 | 0.800 / 1.000 / 0.900 |
| 段落向量（无版本语义） | 0.933 / 1.000 / 0.961 | 0.900 / 1.000 / 0.950 |
| 宿主 local search | 0.567 / 0.833 / 0.678 | 0.600 / 0.900 / 0.708 |

三种更简单的旁路（外部删除集合、外部修订表、修订表 + 重嵌入）都能复现可见性；把改写文档换标题重入库需 170 次生成调用 / 303 s，且旧标题仍返回 10/10。因此 PEG 的差别在代数、性质与代价，不在"只有它能看见改动"。

### 3.4 生成答案

同一提示、同一模型、每题一次生成：PEG 上下文 EM 0.600 / F1 0.704，宿主 0.525 / 0.625；10 个改写探针中，PEG 答案包含标识串 10/10，宿主 0/10。

### 3.5 局部性、延迟与重建方差

- 改写后 top-10 邻居平均 Jaccard 0.845；删除后 38/40 查询返回完全不变。
- 检索 p50 在 300 / 400 / 1,200 段均为约 141 ms，延迟由查询嵌入主导。
- 同一 300 段语料重建三次：实体 2,192–2,218、社区 417–426、生成调用 1,410–1,430；宿主基础 Hit@1 0.700–0.767。PEG 指标在三次重建之间不变（0.933 / 1.000 / 0.961）。

### 3.6 成本

| 操作 | 生成调用 | 耗时 |
|---|---:|---:|
| PEG 编码 300 段 | 0（仅嵌入） | 6.7 s |
| PEG 追加 100 段 | 0（仅嵌入） | 2.4 s |
| 宿主基础索引 300 段 | 1,710 | 241 s |
| 宿主增量 100 段 | 647 | — |

## 4. 对结果的解读

**PEG可以做到这些**

- 宿主 `update` 只吸收新增标题；同标题改写与删除在索引中不可见（0/10 与 5/5）。
- 段落级独立通道（重新嵌入 + 墓碑）能补齐这两个缺口（10/10 与 0/5）。
- 五条行为性质、非侵入定理与操作代价上界成立，写入不改变宿主状态（论文第 4 节）。
- 可见性可以由更简单的旁路实现；PEG 的差别在代数、性质与代价。

**但是这些并不代表**

- PEG 提升常规检索精度：段落向量基线与其同分（0.925 / 1.000 / 0.958）。
- 融合优于 PEG 单独：多跳覆盖与答案质量都不支持。
- 端到端生产质量：只做单提示、单次生成与检索级评测。
- 大规模稳定性：最大 1,200 段、三次重建。

## 5. 局限

1. 粒度固定为段落；跨段落多跳需要宿主图结构。
2. 标题唯一；误用追加会静默忽略改写。
3. 每次替换各触发一次嵌入，未做窗口内合并（见存储设计文档）。
4. 只覆盖单提示、单次生成；答案指标随采样波动（EM 0.600–0.650）。
5. 单一嵌入模型；更换模型需重测。
6. 宿主索引有重建方差；单实例数字不应视为常数。
7. 宿主增量实例的文本单元向量表可能只含新增文档；宿主指标改在向量表完整的实例上报告。

## 6. 仓库结构

```
peg/
├── configs/            GraphRAG 配置模板与 prompts
├── data/               固定子集、探针、输入 CSV
├── docs/               论文
├── results/            11 个聚合结果 JSON
├── scripts/            reproduce.ps1 / smoke.ps1
├── src/peg_bench/      数据构建、索引包装、PEG 通道、评测与补充实验
└── tests/              通道、一致性、数据、配置与端点测试
```

关键模块：`peg_channel.py`（四元组状态与三种操作）、`evaluate.py`（五臂 + 行为探针）、`controls.py`（旁路对照）、`generation.py`（答案级）、`local_effects.py`（局部性/延迟）、`index_variance.py`（重建方差）。

## 7. 许可

- 代码 MIT，见 `LICENSE`。
- GraphRAG v3.3.0 为 MIT，本仓库不重分发其源码。
- HotpotQA 条款见 <https://hotpotqa.github.io/>；原始 dump 不随仓库分发，按 `data/README.md` 下载并校验 SHA-256，`data/` 下的子集仅用于基准测试。
