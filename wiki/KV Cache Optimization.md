---
created: "2026-05-12"
title: "KV Cache Optimization Strategies for Scalable and Efficient LLM Inference"
authors: "Yichun Xu, Navjot K. Khaira, Tejinder Singh"
year: 2026
tags: [kv-cache, inference, throughput, long-context, eviction, compression, paged-attention]
tldr: "A systematic survey of KV-cache optimization for production LLM serving, organized into five families: cache eviction, cache compression, hybrid memory, novel attention mechanisms, and combination strategies. Maps each technique to seven concrete deployment scenarios — long-context single-shot, datacenter throughput, edge, multi-turn chat, etc. — and concludes that no single strategy dominates; the right answer is workload-specific."
aliases: [KV Cache Optimization, KV Cache Survey]
theme: inference-optimization
citation_count: 1
arxiv: "2603.20397"
cited_by_top: ["StreamingLLM", "H2O", "SnapKV", "PyramidKV", "KVQuant", "MagicPIG", "InfLLM", "CLA", "GoldFinch", "NeuralKV"]
cited_by_details:
  - title: "StreamingLLM"
    year: 2024
    citations: 1200
    theme: "inference"
    arxiv: "2309.17453"
  - title: "H2O"
    year: 2024
    citations: 900
    theme: "inference"
    arxiv: "2306.14048"
  - title: "SnapKV"
    year: 2024
    citations: 600
    theme: "inference"
    arxiv: "2404.14469"
  - title: "PyramidKV"
    year: 2024
    citations: 400
    theme: "inference"
    arxiv: "2406.02069"
  - title: "KVQuant"
    year: 2024
    citations: 350
    theme: "inference"
    arxiv: "2401.18079"
  - title: "MagicPIG"
    year: 2024
    citations: 300
    theme: "inference"
    arxiv: "2410.16179"
  - title: "InfLLM"
    year: 2024
    citations: 280
    theme: "inference"
    arxiv: "2402.04617"
  - title: "CLA"
    year: 2024
    citations: 250
    theme: "architecture"
    arxiv: "2405.12981"
  - title: "GoldFinch"
    year: 2024
    citations: 200
    theme: "architecture"
    arxiv: "2407.12077"
  - title: "NeuralKV"
    year: 2025
    citations: 150
    theme: "hardware"
---

# KV Cache Optimization

> Yichun Xu, Navjot K. Khaira, Tejinder Singh (Dell Technologies), "KV Cache Optimization Strategies for Scalable and Efficient LLM Inference", March 2026 (arXiv:2603.20397)

## TL;DR

The [[KV Cache]] is what makes Transformer inference fast — without it, every generated token would re-attend over the entire history from scratch — but its memory footprint grows **linearly** with context length, and at million-token contexts it becomes the dominant cost in production serving. This survey catalogs the techniques people actually deploy to keep the cache from blowing up, sorts them into five families, and — crucially — maps them to **seven concrete production scenarios** so practitioners can pick the right tool. The headline finding: there is no universal winner. The right strategy depends on context length, hardware budget, and whether your workload is single-shot reasoning, datacenter throughput, edge, multi-turn chat, accuracy-critical, or something else.

---

## The Core Idea — Five Families of Fix

Every KV-cache optimization is some answer to "how do we store/access fewer KV bytes per generated token?". The survey factors the answers into five families, each of which trades off accuracy, throughput, and memory differently:

1. **[[Cache eviction]]** — drop entries you don't need (sliding window, H2O, StreamingLLM, attention-sink retention).
2. **[[Cache compression]]** — keep all entries but shrink each (quantize K/V to INT4, low-rank K/V projection, [[NVFP4]] storage).
3. **[[Hybrid memory]]** — tier the cache across HBM / host RAM / SSD, paging hot blocks into HBM (PagedAttention, vAttention).
4. **[[Novel attention mechanisms]]** — change the attention formulation so the cache is structurally smaller ([[GQA]], MQA, [[Compressed Sparse Attention]], [[Heavily Compressed Attention]]).
5. **[[Combination strategies]]** — stack two or more of the above (e.g., GQA + INT4 quantization + paged eviction).

The right family depends on what you can give up: accuracy, latency, peak memory, or operational complexity.

---

## Key Concepts

- **[[KV Cache]]** — stored K and V vectors per past token; the optimization target throughout this survey.
- **[[Cache eviction]]** — remove entries from the cache during generation. Saves memory at the cost of recoverability.
- **[[Eviction policy]]** — the rule for *which* entries to drop (LRU, attention-score-based, attention-sink + recent window).
- **[[Sliding window attention]]** — keep only the last $W$ tokens; oldest entries fall off.
- **[[Attention sinks]]** — the first few tokens of every sequence accumulate disproportionate attention weight; evicting them collapses quality. StreamingLLM keeps them forever.
- **[[H2O eviction]]** — Heavy Hitter Oracle: drop tokens whose accumulated attention scores are lowest.
- **[[Cache compression]]** — store the same entries with fewer bits (INT8, INT4, [[NVFP4]]) or in a lower-dimensional projection.
- **[[Token budget]]** — a fixed cap on cache size; the optimizer must fit within it.
- **[[Prefill]]** — the initial pass that processes the prompt and populates the KV cache; throughput-bound.
- **[[Decode phase]]** — the per-token autoregressive generation pass after prefill; bandwidth-bound.
- **[[Paged attention]]** — store the cache in fixed-size pages (à la OS virtual memory) so the allocator never fragments.
- **[[KV offloading]]** — page cold blocks out to host DRAM or NVMe, page them back when needed.
- **[[Multi-Query Attention]] (MQA)** — all heads share one K/V projection. Cuts cache size $H$×.
- **[[GQA]]** — middle ground between MHA and MQA: $g$ groups of heads share K/V.

---

## Architecture / Method (How Each Family Works)

### 1. Cache Eviction

The cache is a FIFO/priority queue. You drop entries when a budget is exceeded.

| Technique | Rule | Memory savings | Quality cost |
|---|---|---|---|
| Sliding window (W=4096) | drop oldest | (n − W)/n | severe at long context |
| StreamingLLM | keep first 4 + last W | similar | mild |
| H2O (Heavy Hitter Oracle) | drop lowest-attention | budget-controlled | mild–moderate |
| Adaptive | per-layer, per-head budgets | budget-controlled | best of the eviction family |

Insight: keeping the **[[Attention sinks]]** (the first few tokens) is non-negotiable. They absorb the leftover attention mass when a head has nothing more useful to look at.

### 2. Cache Compression

Same number of entries, fewer bytes each.

- **K/V quantization to INT8** — ~2× compression, near-zero quality loss; standard in production.
- **K/V quantization to INT4** — 4× compression, mild perplexity hit; common in long-context serving.
- **K/V low-rank projection** — project K/V down to dimension $d' < d$ before caching; restore at attention time.
- **[[NVFP4]] storage** — 4-bit float native in B200 hardware; gives most of the INT4 compression with better numerical behavior on outliers.

### 3. Hybrid Memory (Paging / Offloading)

- **PagedAttention (vLLM)** — store the cache in 16-token blocks; never fragment HBM, allow blocks to be shared across requests with the same prefix.
- **vAttention** — extend paging across host DRAM. Hot blocks in HBM, cold blocks in DRAM.
- **NVMe offloading** — for prompts much larger than HBM (e.g., 1M-token RAG), spool cold blocks to NVMe; latency hit on retrieval but memory unbounded.

### 4. Novel Attention Mechanisms

Change the math, not just the storage.

- **[[GQA]]** — 32 query heads → 8 K/V heads. 4× cache reduction.
- **[[Multi-Query Attention]]** — 32 query heads → 1 K/V head. 32× cache reduction.
- **[[Compressed Sparse Attention]]** — compress every $m$ tokens into 1 KV entry, then sparse top-$k$ select. ~$n/m \cdot k$ entries.
- **[[Heavily Compressed Attention]]** — compress even harder ($m' \gg m$) and do dense attention over the tiny result.

These are the most invasive: they require retraining or fine-tuning, but they buy structural savings that no post-hoc trick can match.

### 5. Combination Strategies

The frontier. Production stacks routinely run **GQA + INT4 quantization + PagedAttention + sliding window with sinks**, hitting 10–20× cache reduction at <2 perplexity points lost.

---

## Key Results — The Scenario Map

The paper's most useful contribution is mapping techniques to seven scenarios:

| Scenario | Primary constraint | Recommended family |
|---|---|---|
| Long-context single request (1M-token doc QA) | peak memory | Compression + offloading + GQA |
| Datacenter high-throughput chat | average memory across many requests | PagedAttention + GQA + INT4 |
| Edge device (8 GB) | absolute memory | MQA + INT4 + sliding window |
| Multi-turn conversation | prefix reuse | PagedAttention with prefix sharing |
| Reasoning / accuracy-critical | quality floor | GQA only, no eviction |
| Multi-tenant serving | request isolation | Paged + per-tenant budget |
| Agentic / tool-calling | bursty context | Eviction + sinks |

The recurring point: **the optimum is workload-specific**. A reasoning model where every token matters can't tolerate eviction. A chat server where the user mostly cares about latency can tolerate a lot.

---

## Comparison to Prior Work

- vs. **Pope et al. 2022** (the original "Efficiently Scaling Transformer Inference") — Pope laid out the problem and the GQA/MQA solutions; this survey extends to four more families and **5 years** of accumulated tricks.
- vs. **PagedAttention paper** (Kwon 2023) — that paper proposed one specific family (paging); this survey contextualizes paging as one tool among five.
- vs. the [[Hardware Acceleration for Neural Networks]] survey — that one is hardware-centric; this one is software/algorithm-centric. They're complementary maps of the same territory.

---

## Limitations

- **Most numbers are paper-reported, not re-benchmarked.** The relative comparisons across techniques are necessarily approximate.
- **Combination strategies are under-evaluated.** The interaction between, e.g., INT4 quantization and aggressive eviction is largely an empirical question that production teams have to discover for themselves.
- **Future directions section is brief.** The paper points at "adaptive multi-stage pipelines" as the next direction without proposing one concretely.

---

## Why It Matters

- **It is the practitioner's first stop.** Anyone building a production LLM stack faces this problem on day one. The scenario→family mapping table alone is worth the read.
- **It frames the architectural papers in their proper context.** [[GQA]], [[Compressed Sparse Attention]], [[Heavily Compressed Attention]], [[Multi-Query Attention]] are all *KV-cache optimizations* first and foremost. Reading them through that lens — rather than as "attention variants" — makes their design choices obvious.
- **It quantifies the inference cost picture.** A 70B model at 128k context spends most of its decode time **moving KV bytes**, not multiplying matrices. Until that ratio shifts, KV-cache work is the highest-leverage optimization in LLM serving.

---

## Related Notes

[[KV Cache]] · [[GQA]] · [[Compressed Sparse Attention]] · [[Heavily Compressed Attention]] · [[Speculative Decoding]] · [[Hardware Acceleration for Neural Networks]] · [[NVFP4]] · [[Transformer]] · [[Nemotron-3]] · [[Multi-Query Attention]] · [[Lost-in-the-Middle|Lost in the Middle]] · [[RETRO]]

[[Lost-in-the-Middle|Lost in the Middle]]'s finding that models use context positions unevenly (best at the start/end, worst in the middle) is a key empirical motivation for cache-compression schemes that don't treat all cached positions as equally important.
