English | [简体中文](README-zh.md)

# PEG: A Non-Invasive Incremental Visibility Channel for Graph RAG

This repository measures how same-title rewrites and deletions actually behave in the index of a real Microsoft GraphRAG v3.3.0 host, and provides a paragraph-level visibility channel (PEG) that does not modify the host.

Capability boundary: **PEG does not improve the host's graph induction capability, nor does it improve conventional retrieval accuracy.** What it guarantees is that "before the index is rebuilt, modified text and deletion state are visible at the retrieval layer," and it states this guarantee as an operation contract with a non-invasiveness theorem.

- Paper: [docs/PEG-visibility.md](docs/PEG-visibility.md)

## 1. Research Positioning

| Item | Content |
|---|---|
| Type | Systems / contract-style empirical study |
| Claims | A paragraph-level independent channel can maintain the current version before the host rebuild; the operations satisfy five behavioral properties; writes do not touch the host |
| Non-claims | Retrieval accuracy improvement, graph induction enhancement, generally better end-to-end answer quality |
| Host | GraphRAG v3.3.0 (installed from source, not redistributed) |
| Models | DeepSeek-V4.1-Flash (thinking disabled) + `bge-m3-8k` (same weights, 8k context) |
| Data | Fixed HotpotQA subsets: 30 + 10 questions (300 + 100 paragraphs); 90 + 30 questions (900 + 300 paragraphs) |

## 2. Quick Start

Requires Python 3.12, an OpenAI-compatible chat endpoint, a local Ollama embedding endpoint, and the GraphRAG v3.3.0 source.

```powershell
# 0. Environment variables (set once; keys never touch disk)
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
# 1. Data (deterministic, no network)
python -m peg_bench.build_hotpot
python -m peg_bench.build_stability
python -m peg_bench.make_inputs

# 2. Indexing: base -> incremental -> edit/delete
python -m peg_bench.patch_settings --root artifacts/runs/base   --csv data/main/base.csv
python -m peg_bench.run_graphrag index  --root artifacts/runs/base
python -m peg_bench.patch_settings --root artifacts/runs/update --csv data/main/input_base_delta.csv
python -m peg_bench.run_graphrag update --root artifacts/runs/update
python -m peg_bench.patch_settings --root artifacts/runs/edit   --csv data/main/input_after_edit.csv
python -m peg_bench.run_graphrag update --root artifacts/runs/edit

# 2b. Stability index (900 + 300 paragraphs; about 5,400 + 3,300 generation calls, can be skipped)
python -m peg_bench.patch_settings --root artifacts/runs/stab_base   --csv data/stability/base.csv
python -m peg_bench.run_graphrag index  --root artifacts/runs/stab_base
python -m peg_bench.patch_settings --root artifacts/runs/stab_update --csv data/stability/input_base_delta.csv
python -m peg_bench.run_graphrag update --root artifacts/runs/stab_update

# 3. PEG state and main evaluation
python -m peg_bench.build_peg_state
python -m peg_bench.evaluate `
  --base-root artifacts/runs/base --update-root artifacts/runs/update --edit-root artifacts/runs/edit `
  --peg-state artifacts/states/peg_base_delta --peg-behavior-state artifacts/states/peg_edit_delete `
  --output results/peg_gr_001.json

# 4. Supplementary experiments (all steps except generation and index_variance make zero generation calls)
python -m peg_bench.ablation
python -m peg_bench.evaluate_stability --base-root artifacts/runs/stab_base --update-root artifacts/runs/stab_update `
  --peg-state artifacts/states/peg_stab_base_delta --data-dir data/stability
python -m peg_bench.controls --paragraph-results results/peg_gr_002.json
python -m peg_bench.baselines
python -m peg_bench.local_effects
python -m peg_bench.index_variance --roots artifacts/runs/rebuild_1 artifacts/runs/rebuild_2 artifacts/runs/rebuild_3
python -m peg_bench.generation --concurrency 8
```

One-command entry points: `scripts/reproduce.ps1` (main experiments) and `scripts/smoke.ps1` (smoke). `artifacts/` is a derived directory excluded by `.gitignore`.

## 3. Main Results

### 3.1 Visibility of Rewrites and Deletions

| Behavior | GraphRAG v3.3.0 | PEG |
|---|---|---|
| Rewrite identifier string appears in returned text | 0 / 10 | **10 / 10** |
| Stale title still returned after deletion | 5 / 5 | **0 / 5** |

The host's incremental update decides by title: a same-title rewrite does not enter the added set, and a disappeared title is only recorded, not executed. PEG closes these two gaps with in-place replacement and tombstones, corresponding to P1 and P3 in the paper.

### 3.2 Retrieval and Coverage

| Channel | Base | Incremental |
|---|---|---|
| Paragraph level | 0.933 / 1.000 / 0.961 | 0.900 / 1.000 / 0.950 |

Note: metrics for generated answers fluctuate by ±1 question across reruns; comparisons must be completed within the same run (paper limitation 11).

Multi-hop coverage: the proportion of queries where all supporting paragraphs enter top-5 is 0.80, top-10 0.95 (host graph retrieval 0.40 / 0.575). Confidence intervals (2,000 rounds): paragraph-level Hit@1 0.925 [0.825, 1.000], MRR 0.958 [0.904, 1.000].

### 3.3 Controls and Baselines

| Method | Base Hit@1/5/MRR | Incremental Hit@1/5/MRR |
|---|---|---|
| BM25 | 0.633 / 0.967 / 0.762 | 0.800 / 1.000 / 0.900 |
| Paragraph vector (no version semantics) | 0.933 / 1.000 / 0.961 | 0.900 / 1.000 / 0.950 |
| Host local search | 0.567 / 0.833 / 0.678 | 0.600 / 0.900 / 0.708 |

All three simpler bypasses (external deletion set, external revision table, revision table + re-embedding) can reproduce visibility; re-ingesting rewritten documents under new titles takes 170 generation calls / 303 s, and the old titles are still returned 10/10. PEG's difference therefore lies in the algebra, properties, and cost, not in being "the only approach that can see changes."

### 3.4 Generated Answers

Same prompt, same model, one generation per question: PEG context EM 0.600 / F1 0.704, host 0.525 / 0.625; among the 10 rewrite probes, PEG answers contain the identifier string 10/10, the host 0/10.

### 3.5 Locality, Latency, and Rebuild Variance

- After rewrites, top-10 neighbors have an average Jaccard of 0.845; after deletions, 38/40 queries return completely unchanged results.
- Retrieval p50 is about 141 ms at 300 / 400 / 1,200 paragraphs; latency is dominated by query embedding.
- Rebuilding the same 300-paragraph corpus three times: entities 2,192–2,218, communities 417–426, generation calls 1,410–1,430; host base Hit@1 0.700–0.767. PEG metrics are unchanged across the three rebuilds (0.933 / 1.000 / 0.961).

### 3.6 Cost

| Operation | Generation calls | Time |
|---|---:|---:|
| PEG encoding 300 paragraphs | 0 (embedding only) | 6.7 s |
| PEG appending 100 paragraphs | 0 (embedding only) | 2.4 s |
| Host base indexing 300 paragraphs | 1,710 | 241 s |
| Host incremental 100 paragraphs | 647 | — |

## 4. Reading the Results

**PEG can do the following**

- The host `update` only absorbs newly added titles; same-title rewrites and deletions are invisible in the index (0/10 and 5/5).
- A paragraph-level independent channel (re-embedding + tombstones) closes these two gaps (10/10 and 0/5).
- The five behavioral properties, the non-invasiveness theorem, and the operation cost upper bound hold; writes do not change host state (paper Section 4).
- Visibility can be implemented by simpler bypasses; PEG's difference lies in the algebra, properties, and cost.

**But these do not mean**

- PEG improves conventional retrieval accuracy: the paragraph vector baseline scores the same (0.925 / 1.000 / 0.958).
- Fusion outperforms PEG alone: neither multi-hop coverage nor answer quality supports it.
- End-to-end production quality: only single-prompt, single-generation, and retrieval-level evaluations were performed.
- Large-scale stability: at most 1,200 paragraphs and three rebuilds.

## 5. Limitations

1. Granularity is fixed at the paragraph level; cross-paragraph multi-hop requires the host graph structure.
2. Titles are unique; misuse with append silently ignores rewrites.
3. Each replacement triggers one embedding; no in-window batching (see the storage design document).
4. Covers only a single prompt and a single generation; answer metrics fluctuate with sampling (EM 0.600–0.650).
5. A single embedding model; changing models requires re-measurement.
6. The host index has rebuild variance; single-instance numbers should not be treated as constants.
7. The text-unit vector table of the host incremental instance may contain only newly added documents; host metrics are instead reported on an instance with a complete vector table.

## 6. Repository Structure

```
peg/
├── configs/            GraphRAG configuration templates and prompts
├── data/               Fixed subsets, probes, input CSVs
├── docs/               Paper
├── results/            11 aggregated result JSONs
├── scripts/            reproduce.ps1 / smoke.ps1
├── src/peg_bench/      Data construction, index wrapper, PEG channel, evaluation, and supplementary experiments
└── tests/              Channel, consistency, data, configuration, and endpoint tests
```

Key modules: `peg_channel.py` (quadruple state and three operations), `evaluate.py` (five arms + behavior probes), `controls.py` (bypass controls), `generation.py` (answer level), `local_effects.py` (locality/latency), `index_variance.py` (rebuild variance).

## 7. License

- Code is MIT, see `LICENSE`.
- GraphRAG v3.3.0 is MIT; this repository does not redistribute its source.
- HotpotQA terms at <https://hotpotqa.github.io/>; the original dump is not distributed with the repository, download and verify SHA-256 per `data/README.md`, and the subsets under `data/` are used only for benchmarking.
