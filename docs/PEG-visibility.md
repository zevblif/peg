English | [简体中文](PEG-visibility-zh.md)

# PEG: A Non-Invasive Incremental Visibility Channel for Graph RAG

## Abstract

The index of a graph retrieval-augmented generation system consists of entity extraction, description merging, and community summarization; its rebuild cost is far higher than a single-document write, so the system has a gap period between two rebuilds: knowledge has been changed, but the index is still old. This paper does not propose a new incremental mechanism; instead, it consolidates the scattered failure-handling practices found in engineering—title primary keys, in-place replacement, tombstones, skipping unchanged content—into a closed set of paragraph-level operations, and gives their behavioral contract, cost upper bound, and non-invasive theorem on top of the GraphRAG v3.3.0 host.

PEG's paragraph layer is a title-aligned quadruple: title, body text, unit vector, and visibility state; it absorbs increments through three operations—append, replace, and tombstone—and retrieval is the inner-product ranking of the query vector against visible paragraph vectors. The operations themselves are definitional, and this paper does not package them as new theorems; there are three substantive claims: any operation sequence does not change host state, the number of model calls for one incremental sequence is the sum of appends and replaces, and a premise checklist that tells practitioners which host designs break invalidation. The first two confine PEG to a visibility layer, not an index layer; the third gives readers a diagnostic they can apply to their own system.

On a real host with real models, on the same fixed HotpotQA subset, the host sees 0 of 10 same-title rewrites, and still returns stale paragraphs 5 times after 5 deletions; PEG gives 10/10 and 0/5 respectively. Paragraph-layer retrieval obtains $\mathrm{Hit@1}=0.933$, $\mathrm{Hit@5}=1.000$, and MRR $=0.961$, tied with ordinary paragraph vector retrieval; the proportion of multi-hop questions whose two supporting paragraphs both enter the top-5 is 0.80. Writes produce no generation calls. A controlled experiment shows that an external revision table can likewise make rewrites visible, so PEG's difference is not in "whether changes can be seen" but in operational closure, properties, and cost.

PEG does not provide exact deletion, does not repair the host's community reports or merged entity descriptions, and does not improve ordinary retrieval accuracy. It addresses different problems from approaches that require index rebuilding to guarantee deletion equivalence (Lightfoot) and approaches that place the graph and summaries in a single database (post-graph-rag): PEG's only exclusive condition is "does not modify the host."

**Keywords**: retrieval-augmented generation; graph RAG; incremental update; non-invasive; paragraph vector; visibility

## 1. Introduction

### 1.1 The Gap Period and the Failure of Premises

The index cost of graph RAG concentrates on the generation side. Taking GraphRAG as an example, indexing requires extracting entities and relations for each text unit, merging multiple descriptions of same-named entities, building hierarchical communities, and generating a summary for each community. We measured on a corpus of 300 paragraphs totaling 38,763 tokens: indexing produced 2,171 entities and 502 communities, and triggered 1,710 generation calls.

This determines that the index cannot be updated immediately with writes. Thus there necessarily exists a period between "knowledge has changed" and "the index has changed" during which the system presents old content; more seriously, this staleness cannot be identified from the retrieval results themselves.

The host's incremental capability determines the nature of the gap period. GraphRAG v3.3.0's incremental indexing uses document title as the criterion: same-title rewrites do not enter the added set, and disappearing titles are only recorded as a deletion set. Section 5.3 gives the corresponding behavioral experiment: 0 of 10 rewrites are visible, and stale paragraphs are still returned 5 times after 5 deletions.

Put another way: this kind of host does not lack the action of "update"; rather, the premises that "update" relies on do not hold—it assumes that a title change corresponds to a content change, and that deletions can be consumed downstream. Once the premises fail, it is meaningless to discuss incremental algorithms further.

### 1.2 Positioning of This Paper

This paper consolidates a set of practices that already exist in scattered engineering form into a closed operation set, and states the conditions under which it holds.

These practices are not new: paragraph primary keys derived from the source, in-place replacement, soft-delete markers, and skipping unchanged content by hash have all appeared repeatedly in retrieval systems. Recent deletion research also states plainly that its proposition is "on the edge of tautology" ([3]). Therefore this paper does not claim to invent an incremental mechanism, but claims three things:

1. Write "append, replace, tombstone" as quadruple operations on the same state, so that any legal interleaving yields a unique final state;
2. Give a premise checklist that readers can apply to their own host design;
3. Prove that this layer does not write host state, and give an upper bound on operation cost.

This positioning yields two direct conclusions: PEG can make modifications and deletions take effect at the retrieval layer before the host rebuilds; PEG's writes trigger no state change in the host; PEG's capability boundary is determined by "visibility" rather than "graph-structure induction."

### 1.3 Work and Boundaries of This Paper

This paper defines PEG's paragraph-layer state and three incremental operations; gives five behavioral properties that follow directly from the definitions, plus one non-invasive theorem and one cost proposition; completes retrieval, visibility, stability, mechanism verification, cost, baseline, and generated-answer experiments on a real host with real models; reports multi-hop coverage, bootstrap confidence intervals, and rebuild variance; and uses four simpler bypasses to test "whether visibility can only be provided by PEG."

This paper does not claim: retrieval accuracy improvement; exact deletion (vectors in host storage remain); repair of host community reports and entity descriptions; end-to-end production quality. Section 7 lists the boundaries one by one.

## 2. Related Work

### 2.1 Incremental Update and Deletion in Graph RAG

**GraphRAG** builds hierarchical communities with the Leiden algorithm and generates a summary for each community; its global induction capability is strong, but the generation volume during indexing is high and the incremental path is narrow; its community reports mix content from multiple documents, and deletions cannot be attributed to a source ([3] discusses this directly). **LightRAG** replaces community summaries with dual-level keywords, making indexing lighter, but it likewise merges entity descriptions with a model.

**Lightfoot** ([3]) requires that every derived record be a deterministic function of "extraction results from surviving paragraphs": descriptions stay at the paragraph level, no model judge is used, and synonym edges use thresholds rather than top-k neighbors. Deletion is therefore a transaction, and the post-deletion store equals, table by table, a rebuild that "never contained the document"; on an 11,402-paragraph corpus, deleting 10% took 14 seconds with no model calls. The cost is that it must build its own store, and it explicitly does not evaluate community-summary-type questions (its limitation 6).

**post-graph-rag** ([4]) places paragraphs, a canonical entity graph, and community summaries in the same PostgreSQL, uses bi-temporal records for relation validity time and system-belief time, and uses supersede to invalidate replaced old relations. It has both invalidation semantics and community summaries, but presupposes a single-database architecture.

### 2.2 Standard Practices in Engineering

In retrieval systems, the following practices have already been standardized: paragraph IDs derived from the source document and position, so that rewrites do not produce new IDs; soft deletion with an `is_active`-style marker, filtered at retrieval time; skipping unchanged documents by content hash; writing a tombstone before filling in the index. They overlap heavily in form with PEG's operations.

This paper's difference is not in these practices themselves, but in collecting them into an operation set with a final state, premises, and a cost bound, and in making "does not write the host" a verifiable condition rather than an implementation detail.

### 2.3 The Boundary of Soft Deletion

Soft deletion guarantees only that "retrieval cannot see it", not that "the data cannot be recovered". Ghost Vectors ([5]) verifies on three HNSW implementations: vectors marked deleted remain in the on-disk index and can be recovered by reading the raw index file to bypass the API; on structured sensitive data, the attribute recovery rate can reach 100%. This means tombstones cannot be used to satisfy compliance-level deletion requirements.

PEG's tombstone belongs to soft deletion: it clears the visibility state, not the storage. Section 7 lists this as a hard boundary.

### 2.4 Comparison

| | Paragraph primary key | Tombstone | Exact deletion | Bi-temporal | Community summaries | No host modification |
|---|---|---|---|---|---|---|
| Lightfoot [3] | Derived (paragraph-level) | — | Yes (transaction, rebuild-equivalent) | — | No | No (own store) |
| post-graph-rag [4] | Merged to canonical names | — | supersede (old relations invalidated, not document deletion) | Yes | Yes | No (single database) |
| Engineering practice | Common | Common | — | — | Implementation-dependent | Implementation-dependent |
| **PEG** | Title-unique | Yes | No (soft deletion) | No | Host-provided (stale) | **Yes** |

The reading of this table is: the first three columns compare capabilities, the last column compares applicable conditions. PEG is weaker than the former two in deletion capability; its only exclusive property is "acts on an existing host and does not require rebuilding." PEG does not improve the global layer; it merely does not destroy it. When global-layer synchronisation is required, PEG does not apply. Therefore this paper does not compare deletion correctness with Lightfoot, nor temporal modeling with post-graph-rag; it answers a different question: beside an already-running host, can changes be made visible with zero generation calls.

## 3. PEG's Structure and Operations

### 3.1 Paragraph-Layer State

The paragraph-layer state is the quadruple

$S = (T, X, V, \sigma)$

where $T = (t_1, \dots, t_N)$ is the paragraph title sequence, $X = (x_1, \dots, x_N)$ is the corresponding body text, the $i$-th row of $V \in \mathbb{R}^{N \times d}$ is the unit embedding vector of $x_i$, and $\sigma \in \{\text{active}, \text{deleted}\}^N$ is the visibility state. Each title also has an integer counter $c_i$, which records only the number of rewrites and does not participate in retrieval.

Titles are unique in $T$. This is PEG's core premise: it allows results to be aligned directly with document titles returned by the host, and it also gives the same paragraph only one position throughout its lifetime.

### 3.2 Incremental Operations

| Operation | Corresponding event | Precondition | State change | Model calls |
|---|---|---|---|---|
| Append $\mathrm{append}(t,x)$ | New paragraph enters the corpus | $t \notin T$ | Add one item at the end; vector obtained by embedding $x$; state active | 1 paragraph embedding |
| Replace $\mathrm{replace}(t,x)$ | Existing paragraph is rewritten | $t \in T$ | Overwrite the body text and vector of item $i$; increment counter by 1 | 1 paragraph embedding |
| Tombstone $\mathrm{delete}(A)$ | Paragraph is removed | $t \in T$ | Set the corresponding state to deleted; body text and vector retained | 0 |

Formally, let the position function be $\mathrm{pos}(t)$:

- $\mathrm{append}(t,x)$: if $t \notin T$, set $x_{N+1}=x$, $V_{N+1}=\mathrm{emb}(x)/\Vert\mathrm{emb}(x)\Vert$, $\sigma_{N+1}=\text{active}$, $c_{N+1}=1$; otherwise no field is modified.
- $\mathrm{replace}(t,x)$: set $i=\mathrm{pos}(t)$, $x_i=x$, $V_i=\mathrm{emb}(x)/\Vert\mathrm{emb}(x)\Vert$, $c_i=c_i+1$.
- $\mathrm{delete}(A)$: for each $t \in A \cap T$, set $\sigma_{\mathrm{pos}(t)}=\text{deleted}$; ignore $t \notin T$.

The difference among the three operations lies in whether re-embedding occurs: append and replace each require one paragraph embedding, while a tombstone changes only one state bit. The embedding function is computed over the entire paragraph text, so a rewrite must re-embed the whole paragraph; a new vector cannot be derived from the old vector and the changed fragment.

### 3.3 A Complete Incremental Sequence

Taking the experimental scenario as an example, the base corpus is 300 paragraphs, after which three kinds of changes occur:

| Step | Event | Operation | Result | Query visibility |
|---|---|---|---|---|
| 1 | 100 new documents enter the corpus | 100 appends | Paragraph count 300 → 400 | New paragraphs immediately retrievable |
| 2 | A note containing an identifier string is appended to the body of 10 paragraphs | 10 replaces | Corresponding body text and vectors updated | New body text returned, identifier string visible |
| 3 | 5 paragraphs are removed | 5 tombstones | 5 items set to deleted | Corresponding titles no longer returned |

Operation cost is 100 embeddings plus 10 embeddings; tombstones produce no embedding calls. When the same sequence is given to the host, steps 2 and 3 are not processed (Section 5.3).

### 3.4 Paragraph-Level Retrieval

Given a query $q$, let $r=\mathrm{emb}(q)/\Vert\mathrm{emb}(q)\Vert$ and similarity $s_i=\langle V_i,r\rangle$. The paragraph layer returns

$\mathrm{TopK}(q,k)=\arg\max_{i:\ \sigma_i=\text{active}} s_i \quad (\lvert\cdot\rvert=k)$

together with the corresponding title sequence and the current body text of these titles. Retrieval operates only on visible paragraphs, and ranking is determined solely by the vectors and the query.

## 4. Properties and Premises

### 4.1 Behavioral Properties Derived Directly from Definitions

The following five are not theorems, but behavioral contracts unfolded from the operation definitions. Writing them out is to give implementations and tests a clear object, not to add theoretical weight.

**P1 (Replacement is visible).** After executing $\mathrm{replace}(t,x)$ on $t$, any retrieval that returns $t$ returns the body text $x$. This follows directly from retrieval reading $X$ and from title uniqueness.

**P2 (Successive replacement takes the last one).** After executing $m$ replacements in sequence on the same title, the returned body text is $x^{(m)}$ from the last one. This follows directly from writing the same position and from later writes overwriting earlier ones.

**P3 (Tombstoned is invisible).** After executing a tombstone on $t$, no retrieval returns $t$. This follows directly from the retrieval feasible region being the active set.

**P4 (Append does not overwrite).** If $t \in T$, append changes no field. This follows directly from append's precondition. Its engineering meaning is that append is idempotent: repeated imports do not produce duplicates.

**P5 (Counter is irrelevant to retrieval).** Retrieval output does not depend on the counter $c$. This follows directly from the retrieval function's domain containing only $(T,X,V,\sigma)$. Therefore PEG does not provide "newer ranks higher" ordering.

### 4.2 The Non-Invasive Theorem

**Theorem 4.1 (Non-invasive).** Any operation sequence of PEG does not change the host's graph structure, community partition, community reports, document table, text-unit table, or vector table.

**Proof.** Enumerate the host's writable state surfaces and PEG's write targets:

| Host state surface | Does PEG write? | Basis |
|---|---|---|
| Entity and relation tables | No | PEG maintains only $(T,X,V,\sigma,c)$ |
| Community partition and community reports | No | PEG does not call community detection or summary generation |
| Document table and text-unit table | No | PEG does not write host storage, nor call the incremental indexing pipeline |
| Host vector table and cache | No | PEG's vectors exist only in its own state file |
| Host service process | No | PEG reads host artifacts in a read-only manner |

PEG's write set is its own state file; it is disjoint from every state surface in the table above, and it does not call the host's index update interface. Therefore after any operation sequence ends, the host state is the same as before the operations. $\square$

The value of this theorem is not in proof difficulty but in checkability: each row of the table above corresponds to a testable assertion, and Section 5.6 verifies the part related to retrieval through mechanism verification.

### 4.3 Operation Cost

**Proposition 4.2 (Operation cost).** If an incremental sequence contains $n_a$ appends, $n_r$ replaces, and $n_d$ tombstones, then the number of model calls is $n_a+n_r$, independent of paragraph length and of the size of the tombstone set.

**Proof.** Append and replace each require one paragraph embedding; tombstones call no model; batch processing does not change the count of each operation. $\square$

### 4.4 A Premise Checklist: Who Breaks Invalidation

This section states the paper's third substantive claim: it does not prove a property of PEG, but gives a premise checklist readers can apply to their own host design.

P1–P5 all rest on two premises: titles are unique, and operations are applied by title. When the premises fail, the properties fail accordingly. Pointing out who breaks the premises is more informative than repeating the proofs of the properties:

| Premise | Violator | Consequence |
|---|---|---|
| Title change corresponds to content change | GraphRAG v3.3.0's title criterion | Same-title rewrites do not enter the added set; old body text remains visible |
| Deletion can be consumed downstream | The same incremental path | Deletions are only recorded; stale paragraphs are still returned |
| Retrieval looks only at the visible set | Soft-deletion implementation ([5]) | Vectors remain in storage and can be recovered by bypassing the API |
| Derived structures do not depend on history | Merged entity descriptions, model judge, top-k neighbors ([3] identifies these) | After deletion, the structure is not rebuild-equivalent |

PEG's trade-off is: it gives up merged descriptions, judge, and top-k neighbors—structures that would break the premises; it never builds them in the first place, but only reads the host's artifacts. The cost is that these host structures remain stale after deletion. Section 7 states this cost explicitly.

## 5. Experiments

### 5.1 Setup

All experiments in this paper use the same dataset and the same model configuration.

| Item | Value |
|---|---|
| Reference host | GraphRAG v3.3.0 source code |
| Generation model | DeepSeek-V4.1-Flash, thinking process disabled |
| Embedding model | bge-m3, 1024 dimensions |
| Paragraph unit | HotpotQA single-paragraph body text |
| Corpus | 30 + 10 questions (300 + 100 paragraphs); 90 + 30 questions (900 + 300 paragraphs) |
| Metrics | Retrieval-level hit and MRR; answer-level EM and word-level F1 |

### 5.2 Retrieval Performance

Paragraph layer: base questions ($n=30$) $\mathrm{Hit@1}=0.933$, $\mathrm{Hit@5}=1.000$, MRR $=0.961$; incremental questions ($n=10$) $\mathrm{Hit@1}=0.900$, $\mathrm{Hit@5}=1.000$, MRR $=0.950$.

Multi-hop coverage: the proportion of questions whose supporting paragraphs all enter the top-5 is 0.80, and entering the top-10 is 0.95 ($n=40$); under the same definition, host graph retrieval gives 0.40 and 0.575 respectively.

Confidence intervals: 2,000 bootstrap resampling rounds on 40 questions (seed 17), paragraph layer $\mathrm{Hit@1}=0.925$[0.825, 1.000], MRR $=0.958$[0.904, 1.000]; because the incremental group has only 10 questions, $\mathrm{Hit@1}=0.900$[0.700, 1.000].

Fold-wise recheck (90 + 30 questions, 3 folds): paragraph-layer base $\mathrm{Hit@1}$ is 0.867, 0.967, 0.867; incremental questions are 1.000, 0.900, 1.000. The paragraph layer depends only on paragraph vectors and not on the host index, so the fold-wise numbers agree with the full set.

### 5.3 Visibility of Incremental Operations

Append a note containing a unique identifier string to the body of 10 base paragraphs (replace), execute a tombstone on another 5 paragraphs, then retrieve with probe questions:

| Behavior | Reference host | PEG |
|---|---|---|
| Identifier string after rewrite appears in returned body text | 0 / 10 | 10 / 10 |
| Deleted title is still returned | 5 / 5 | 0 / 5 |

This behavior of the host is the result of its incremental path being decided by title; PEG avoids both points through replacement and tombstone, corresponding to P1 and P3.

### 5.4 Score Change of a Single Rewrite

Rewriting also changes the relative position of that paragraph with respect to other paragraphs. Taking the query `What is the archive verification code for Ed Wood (film)?` as an example:

| Paragraph | Score / rank before | Score / rank after | Body text contains identifier string |
|---|---|---|---|
| Ed Wood (film) | 0.5719 / 1st | 0.6104 / 2nd | No → Yes |
| Ed Wood | 0.5338 / 2nd | 0.6283 / 1st | No → Yes |

The score of the rewritten paragraph rises, because the new body text contains phrases overlapping with the query; the rank may drop, because another paragraph rewritten at the same time rises more. Therefore "rewrite is visible" is not equal to "rank rises": PEG guarantees correct body text, not ranking gains.

The deletion side is equally intuitive: the 5 tombstoned paragraphs ranked 1st or 2nd before the change (scores 0.5587–0.6676) and no longer appear after the change.

### 5.5 Cost

| Operation | Model calls | Time |
|---|---:|---:|
| Encode 300 paragraphs | 0 (embedding only) | 6.7 s |
| Append 100 paragraphs | 0 (embedding only) | 2.4 s |
| Replace 10 paragraphs | 0 (embedding only) | Same order as 10 paragraph embeddings |
| Tombstone 5 paragraphs | 0 | Only changes state bits |

The 6.7 s for encoding 300 paragraphs is a batch average, about 22 ms per paragraph; the cost of replace and append grows linearly with the number of paragraphs. Reference host: base indexing 1,710 generation calls, 241 s; incremental update 647 generation calls.

### 5.6 Mechanism Verification

To confirm which step decides "rewrite visible" and "deletion visible" respectively, three controls are run on the same PEG state:

| Variant | Rewrite visible | Deletion residue | Duplicate slot rate |
|---|---|---:|---:|
| Replace + tombstone | 10/10 | 0/5 | 0.000 |
| Append duplicate + tombstone | 10/10 | 0/5 | 0.025 |
| Append duplicate, no tombstone | 10/10 | **5/5** | 0.025 |

Rewrite visibility holds under both write methods, showing that it comes from re-embedding; deletion visibility holds only with a tombstone, showing that it comes from the tombstone; appending duplicates causes the same title to appear repeatedly in the top-10, wasting 2.5% of the slots, which is why PEG adopts "title uniqueness + in-place replacement"; the ranking of the three variants is unaffected, consistent with P5.

### 5.7 Baseline Comparison

To judge whether "paragraph vector retrieval" itself is sufficient, three lower-cost methods are compared (the same batch of 40 questions, 400-paragraph corpus):

| Method | Base Hit@1 / Hit@5 / MRR | Incremental Hit@1 / Hit@5 / MRR | Time per question |
|---|---|---|---:|
| BM25 (lexical) | 0.633 / 0.967 / 0.762 | 0.800 / 1.000 / 0.900 | 1.2 ms |
| Paragraph vector (no version semantics) | 0.933 / 1.000 / 0.961 | 0.900 / 1.000 / 0.950 | 157 ms |
| Text-unit vector | 0.933 / 1.000 / 0.961 | 0.900 / 1.000 / 0.950 | 158 ms |
| Host local search | 0.567 / 0.833 / 0.678 | 0.600 / 0.900 / 0.708 | — |

There are three conclusions: paragraph-granularity vector retrieval is already near saturation on this corpus ($\mathrm{Hit@5}=1.000$), and PEG's retrieval quality comes from paragraph granularity and the embedding model; BM25 is markedly lower ($\mathrm{Hit@1}$ 0.633 vs. 0.933), indicating that the corpus relies on semantic matching; host graph retrieval is lower (0.575 combined vs. 0.933), because candidates are diluted by entities and relations. Therefore PEG's incremental value lies in visibility, not in ordinary retrieval accuracy.

This host-side row is measured on an index instance with a complete vector table. In this incremental index instance, the text-unit vector table contains only the 100 vectors of the newly added documents, and the base documents are not among them; measuring on that instance would underestimate the host. That instance is still used for the visibility experiment in Section 5.3, because version semantics are unrelated to the completeness of the vector table.

### 5.8 Simpler Bypasses

Can visibility be provided only by PEG? On the same 400-paragraph corpus and the same batch of 15 probes, we re-tested with three simpler bypasses, none of which modifies the host or produces generation calls:

| Bypass | Rewrite identifier string visible | Stale title residue after deletion | Re-embedding |
|---|---:|---:|---|
| Deletion filtering only | 0 / 10 | 0 / 5 | None |
| External revision table provides latest body text, ranking uses old vectors | 10 / 10 | 0 / 5 | None |
| External revision table + per-document re-embedding | 10 / 10 | 0 / 5 | 10 paragraphs |

The results show: deletion visibility requires only an external deletion set; rewrite visibility requires only an external revision table of "title → latest body text", and does not even necessarily require re-embedding. Therefore "making changes visible" is itself not a PEG-exclusive capability.

PEG's difference lies in the remaining four points. First, it defines changes as a closed algebra: title uniqueness, append idempotence, tombstone, version counter, so that multiple operations on the same title have a determinate final state, and any legal interleaving yields the same result. Second, these operations carry a checkable behavioral contract and cost upper bound (Section 4). Third, it does not depend on whether the host provides revision time; the host records only the indexing moment (Section 7, item 10), so the side table is itself an external channel, of the same kind as PEG. Fourth, it reads host artifacts only, and therefore introduces no half-updated state in the host.

The bypass without re-embedding still ranks with the pre-rewrite vectors. Among the 10 rewrites in this group, only 1 changes rank: after re-embedding, `Ed Wood (film)` drops from 1st to 2nd. This shows that on this corpus "body text visible" and "ranking change" can be separated: to let readers see the new body text, re-embedding is not necessarily needed; to change the ranking, it is. PEG chooses re-embedding to keep the vector consistent with the body text, not to improve ranking.

The fourth bypass is closer to engineering intuition: re-ingest the rewritten document under a new title. One incremental update on a 410-paragraph input (400 original paragraphs plus 10 renamed documents) measured 170 generation calls and 303 s; the indexed document count rises from 400 to 410, and the old and new titles are both retained. Among the 10 probes, the new body text is visible 10/10, but the old title is still returned 10/10, and each returned top-10 contains both the old and new titles (10/10 duplicate slots); the host does not process deletion, and the old title cannot be retired through the incremental path. Per document, this bypass is 17 generation calls per document, about 3 times the base index (1,710 calls / 300 paragraphs, i.e., 5.7 calls per document).

Thus the four bypasses present the same boundary: the more re-embedding is omitted, the easier it is to make the body text visible; the more the host is relied upon, the more generation calls and stale copies must be paid. PEG's choice is to make visibility an independent state, replacing the entire indexing pipeline with one paragraph embedding.

### 5.9 Generated Answers

Retrieval metrics only indicate whether the correct paragraph enters the list, not whether an answer can be produced from it. For this purpose, under the same prompt, the same model, and one generation per question, each of the three arms answers using the top 10 paragraphs of its own context; the reference answers are the original HotpotQA answers, and the metrics are exact match and word-level F1. A total of 140 generation calls, 113 s.

| Arm | Base EM / F1 | Incremental EM / F1 | Combined EM / F1 | Answers `unknown` |
|---|---|---|---|---:|
| PEG paragraphs | 0.667 / 0.770 | 0.400 / 0.507 | 0.600 / 0.704 | 0.150 |
| Host retrieval | 0.567 / 0.686 | 0.400 / 0.440 | 0.525 / 0.625 | 0.225 |
| Fusion of both | 0.633 / 0.775 | 0.400 / 0.507 | 0.575 / 0.708 | 0.125 |

The answer-side result of the rewrite probes is more direct: asking the model for the 10 identifier strings, under PEG context the answers are `ZX-00` through `ZX-09`, 10/10 correct; under host context, all 10 answers are `unknown`.

There are two conclusions. First, visibility at the retrieval layer propagates to the answer layer: rewrites that the host cannot see are likewise invisible in the answers. Second, PEG's answer quality is higher than the host's (combined EM 0.600 vs. 0.525), but the margin is limited, and fusion is not better than PEG alone (0.575 vs. 0.600), consistent with the conclusion at the retrieval layer.

### 5.10 Locality of Changes and Retrieval Latency

Is the effect of rewriting on the vector space local? For 10 rewrites, comparing the top-10 neighbors of the paragraph before and after the rewrite: the average Jaccard of the neighbor sets is 0.845, i.e., about 85% of neighbors remain unchanged, and the changes concentrate on the few paragraphs directly related to the old and new body text.

The impact scope of deletion is smaller: comparing the top-10 before and after tombstones for 40 queries, after removing the deleted titles the overlap of the two sets is 0.969; only 2 queries had the deleted title in their top 10 originally, and the returns of the other 38 queries are entirely unchanged. Deletion only frees the slots occupied by the deleted paragraphs and does not perturb the ranking of other paragraphs.

Retrieval latency and scale: the median time of query embedding plus inner-product ranking on 300, 400, and 1,200 paragraphs is 141.2, 141.1, and 141.0 ms respectively, with p95 of 147.0, 150.5, and 151.7 ms. Latency is dominated by query embedding; the ranking part is negligible at the thousand-paragraph scale.

### 5.11 Rebuild Variance of Host Indexing

Independently rebuilding the same 300-paragraph corpus three times, neither extraction density nor retrieval metrics is a constant:

| Rebuild | Entities | Communities | Generation calls | Base Hit@1 | Hit@5 | MRR |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 2,218 | 417 | 1,430 | 0.733 | 0.867 | 0.793 |
| 2 | 2,192 | 422 | 1,422 | 0.700 | 0.867 | 0.774 |
| 3 | 2,216 | 426 | 1,410 | 0.767 | 0.833 | 0.800 |

Across the three runs, the entity count differs by 26 (about 1.2%), the community count by 9, and the generation calls by 20; the $\mathrm{Hit@1}$ range on the 30 base questions is 0.067 (0.700 to 0.767), and the MRR range is 0.026. PEG's paragraph layer does not use the host's extraction results, and its metrics on the same question set are entirely unchanged across the three rebuilds ($\mathrm{Hit@1}=0.933$, $\mathrm{Hit@5}=1.000$, MRR $=0.961$), because it depends only on the embeddings of fixed body text.

Therefore host-side numbers should be read with intervals, and cross-instance differences should not be smaller than this fluctuation; all cross-arm comparisons in this paper are completed within the same run. The table contains no incremental documents, so the incremental questions are 0 on all three base indexes; this is determined by the corpus composition and is not part of the variance.

### 5.12 Integration Cost

The first five groups of experiments measure effectiveness, not integration cost; yet for a system already running, integration cost is the decision variable.

The existing costs are as follows: PEG encodes 300 paragraphs in 6.7 s and appends 100 paragraphs in 2.4 s, both with 0 generation calls; rebuilding the same corpus once on the host is 1,710 generation calls and 241 s. Lightfoot's deletion is a 14-second transaction, but it requires the index itself to be built from paragraph-level deterministic functions, so integration means replacing the store and re-indexing ([3]). PEG replaces nothing; it reads host artifacts only.

The integration cost of PEG can be measured directly: reading the host document table and building the paragraph state takes 11.5 s for 300 paragraphs and 12.7 s for 400 paragraphs; reading the table takes under 70 ms and the rest is embedding, and both runs make 0 generation calls. These are single runs after the embedding service was warm; cold and warm states change absolute seconds, so the magnitude and the call count matter, not the seconds themselves.

The key difference is not seconds but whether generation calls are needed: a rebuild-type approach spends 1,710 calls, while PEG's integration spends 0. It states the applicable boundary: when exact deletion and post-deletion equivalence are needed, a rebuild-type approach should be chosen; only when changes need to be made visible without touching the host does PEG come into play.

## 6. Discussion

### 6.1 What PEG Guarantees

PEG maintains the current visible version of paragraphs, making rewritten body text immediately retrievable; maintains deletion state, making deleted titles immediately unretrievable; absorbs increments with three operations, at a cost proportional to the number of changed paragraphs; multiple operations on the same title have a unique final state, and any legal interleaving yields the same result; outputs results by title, facilitating alignment with the host.

### 6.2 Boundaries

PEG does not enhance the host's graph induction capability, nor does it undertake community detection and summary generation; it does not provide "newer ranks higher" ordering (P5); it does not do fragment-level vector composition—a rewrite requires re-embedding the whole paragraph; it does not modify the host, nor perform index updates on the host's behalf.

Three boundaries need to be stated separately:

1. **No exact deletion.** The tombstone changes only the visibility state; vectors, entity descriptions, and community reports in host storage remain, and soft-deleted vectors can be recovered ([5]).
2. **No repair of the host's global layer.** After deletion, the host's community reports and merged entity descriptions still contain content from deleted documents until the host rebuilds. PEG retains this layer but does not repair it.
3. **Visibility can be implemented by simpler bypasses** (Section 5.8); what PEG provides is the closure, contract, and cost of this operation set, not "only it can make changes visible."

### 6.3 The Meaning of Non-Invasiveness

Non-invasiveness is not a stylistic choice but a deployability condition. The host's indexing pipeline involves external cost (generation calls), maintenance windows, and state-consistency constraints; any approach that rewrites host state must negotiate with these constraints. PEG limits itself to "read-only host artifacts + independent state", and can therefore be deployed and rolled back independently before the host rebuilds, without introducing a half-updated state in the host. Theorem 4.1 is the precise statement of this condition, and Section 5.6 gives the verification related to retrieval.

### 6.4 Scope of Applicability

PEG applies to paragraph-level corpora where body text changes frequently and changes need to become retrievable quickly, such as entry-style documents, fact cards, and question-answer pairs. It uses the paragraph as the retrieval unit and is not suitable for queries requiring cross-paragraph induction; such queries still need the host's graph and community structure. When compliance-level deletion, post-deletion equivalence, or temporal modeling is needed, the rebuild-type or single-database approaches in Section 2.4 should be adopted.

## 7. Limitations

1. Granularity is fixed at the paragraph: cross-paragraph induction is outside PEG's capability.
2. Title uniqueness: multiple rewrites of the same title must go through the replace path; if append is misused, the update is ignored without any indication (P4).
3. Successive rewrites are not merged: each replace triggers one embedding; no within-window merging is done.
4. Evaluation is limited to retrieval and single-generation: multi-turn dialogue, multiple prompts, and human review are not covered.
5. Single embedding model: retrieval metrics depend on bge-m3; changing the model requires re-evaluation.
6. Limited corpus: a fixed subset; indexing and retrieval behavior on very large corpora is not covered.
7. No temporal rollback implemented: the current state retains only the latest visible version and deletion state.
8. Embedding context: the default context length is smaller than the longest community report; the experiments use an equivalent embedding model with a larger context; this substitution affects only the host-side community report vectors, not PEG's paragraph vectors.
9. The host basic search's context builder reorders hits by text-unit table order, so the similarity order is lost in the returned results; in addition, the text-unit vector table of this incremental instance covers only the newly added documents. The host metrics in Section 5.7 are instead measured on an instance with a complete vector table.
10. A timestamp-filtering baseline cannot be constructed: the host records only the indexing moment, same-title rewrites do not update this field, and deleted documents leave no record.
11. Query embeddings are not cached: when scores are nearly tied, ranks swap between reruns, and metrics fluctuate by one question. All comparisons in the text are completed within the same run.
12. Section 5.8 tests only visibility, not interleaving of multiple writers, rollback equivalence, or concurrent writes.
13. Generated answers use only one prompt, one sample, and one model; no human review or second-model cross-scoring is done; the combined EM of two runs fluctuates between 0.600 and 0.650.
14. Host indexing has rebuild variance (Section 5.11): the base $\mathrm{Hit@1}$ range across three rebuilds of the same corpus is 0.067, and the entity count differs by about 1.2%.
15. **Storage-layer boundary of soft deletion**: the tombstone guarantees only that retrieval cannot see it; vectors in host storage can still be recovered ([5]), so PEG does not satisfy compliance-level deletion requirements.
16. **Precedents at the operation level**: title primary keys, tombstones, in-place replacement, and skipping unchanged content already have standardized practices in engineering; this paper's contribution is closure, contract, and cost, not invention.
17. **Staleness of the host's global layer**: community reports and merged entity descriptions are not updated by PEG's replacement or tombstone; when global-layer synchronization is needed, it must be rebuilt on the host side.

## 8. Conclusion

PEG consolidates and closes a set of paragraph-level failure-handling operations: it maintains the current version with title-aligned vectors, absorbs changes with append, replace, and tombstone, and returns only visible paragraphs at retrieval time. The five behavioral properties follow directly from the definitions; Theorem 4.1 proves that it does not touch the host's graph, communities, reports, document table, text-unit table, or vector table; Proposition 4.2 gives the upper bound on operation cost.

Experiments show that the reference host sees 0 of 10 rewrites and still returns stale paragraphs 5 times after 5 deletions, while PEG gives 10/10 and 0/5 respectively; PEG's writes produce no generation calls, and the cost is proportional to the number of changed paragraphs. The baseline comparison shows that paragraph vector retrieval is itself near saturation, and Section 5.8 further shows that simpler bypasses can also make changes visible, so PEG's defensible value lies in its closed operations, contract, and the condition of not touching the host.

## Appendix A: Proof Summary

**P1.** The retrieval body text is taken from $X$; a replace overwrites the unique position of that title; there is no second copy of the body text.

**P2.** Successive replacements write the same position, and later writes overwrite earlier ones; the title is unique, so no intermediate version remains.

**P3.** The retrieval feasible region is the set of visible paragraphs; after deletion the title is not in it, and there is no duplicate position.

**P4.** Append returns directly when the title already exists, writing no field.

**P5.** The retrieval function's domain is $(T,X,V,\sigma)$, which does not contain the counter $c$.

**Theorem 4.1.** PEG's write set is its own state file; it is disjoint from all six state surfaces of the host, and it does not call the host's index update interface.

**Proposition 4.2.** Append and replace take one embedding each, the tombstone zero.

## References

[1] Edge D, Trinh H, Cheng N, et al. From Local to Global: A Graph RAG Approach to Query-Focused Summarization. arXiv:2404.16130, 2024.

[2] Zhu Y, Xu W, Zhang J, et al. LightRAG: Simple and Fast Retrieval-Augmented Generation. arXiv:2410.05779, 2024.

[3] Allu U. Lightfoot: Graph Retrieval-Augmented Generation with Exact, Model-Free Document Deletion. arXiv:2611.11010, 2026.

[4] Rajah C. post-graph-rag: A PostgreSQL-Native Bi-Temporal Graph RAG Engine with Temporal Grounding at Synthesis. arXiv:2608.24921, 2026.

[5] Chakraborttii C, García Alvarado J, Abdulofizova S, Dwivedi S. Ghost Vectors: Soft-Deleted Embeddings Remain Reconstructible in HNSW Vector Databases. arXiv:2606.18497, 2026.

[6] Microsoft GraphRAG v3.3.0. https://github.com/microsoft/graphrag, 2024–2026.

[7] HotpotQA. https://hotpotqa.github.io/, 2018.
