---
created: "2026-10-19"
title: "Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving"
authors: "Ruoyu Qin, Zheming Li, Weiran He, Mingxing Zhang, Yongwei Wu, Weimin Zheng, Xinran Xu (Moonshot AI / Tsinghua)"
year: 2024
arxiv: "2407.00079"
citation_count: 0
tags: [kv-cache, inference, serving, long-context, systems, disaggregation, prefill, decode-phase]
tldr: "Kimi's production LLM serving stack: split prefill and decode clusters, store KV cache across GPU HBM / CPU DRAM / SSD, and schedule on cache reuse + latency SLOs (with early reject when overloaded). Up to 525% throughput in simulated long-context stress; +75% real Kimi requests."
aliases: [Mooncake, KVCache-centric serving, Kimi serving]
theme: inference
---

# Mooncake

> Ruoyu Qin, Zheming Li, et al. (Moonshot AI), "Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving", 2024 (arXiv:2407.00079)

## The Problem / Motivation

This wiki’s systems theme: *decode is bandwidth-bound on the [[KV Cache]]*. Production chat (Kimi) makes that worse:

1. **Prefill vs decode want different hardware** — prefill is compute-heavy; decode is memory-bandwidth-heavy. One cluster does both poorly.
2. **Long context explodes KV** — multi-GB caches per request; GPU HBM is the scarce resource.
3. **Prefixes repeat** — system prompts, multi-turn history, shared documents. Recomputing or evicting that KV wastes the expensive part of the request.
4. **Overload is real** — unlike academic “assume we serve everything,” Mooncake must **reject** work or blow latency SLOs.

[[Paged attention|vLLM]] fixed *how* KV is stored on one node. Mooncake fixes **where** KV lives across a cluster and **who** gets scheduled.

## The Idea

Make **KV cache the center of the design**, not a side effect of model parallelism:

- **Disaggregate prefill and decode** into separate GPU pools.
- **Disaggregate the cache** across GPU HBM → host DRAM → SSD (use idle CPU/DRAM/SSD in the cluster).
- **Schedule** to maximize *effective* throughput under latency SLOs — reuse existing KV, pass prefill KV to decode, early-reject when predicted latency will fail.

## Architecture / Method

```
                ── KVCache-centric scheduler ──
                     │ SLO + reuse scores
     ┌───────────────┼───────────────┐
     ▼               ▼               ▼
 Prefill cluster  Decode cluster  Cache pool
 (compute-heavy)  (bandwidth)     HBM ↔ DRAM ↔ SSD
     │               ▲
     └── transfer KV ┘
```

- **Prefill cluster** — processes prompt, builds KV; can drop or reprioritize under load.
- **Decode cluster** — autoregressive steps on cached KV; bandwidth-bound ([[Decode phase]]).
- **Distributed KV store** — hierarchical: GPU memory (hot) → CPU DRAM (warm) → SSD (cold). Overlaps transfer with compute (same “fewer HBM bytes” logic as [[Flash Attention]]).
- **KVCache-centric scheduler** — scores requests by cache hit/reuse, transfer cost, and deadline/SLO; maximizes *goodput*, not raw tokens/s.
- **Prediction-based early rejection** — if queueing + transfer would miss SLO, reject/upstream-fail early instead of degrading everyone.

## Key Results

| Setting | Result |
|---|---|
| Long-context simulated overload | Up to **525%** throughput vs baseline **while meeting SLOs** |
| Real Kimi production workload | **+75%** more requests handled |
| Design win | Prefill/decode split + multi-tier KV + cache-aware scheduling + early reject |

## Comparison to Prior Work

- vs **[[vLLM]] / [[Paged attention]]** — paged KV on one server; Mooncake is **cluster-level** disaggregation of prefill/decode *and* cache tiers.
- vs **[[KV Cache Optimization]] (evict/compress)** — those shrink or drop KV; Mooncake **keeps** KV but moves and **reuses** it across tiers and machines.
- vs **[[KV offloading]] / [[VeriCache]]** — related “KV doesn’t live only in HBM” idea; Mooncake is the full production scheduler + SLO story for Kimi.
- vs **[[YOCO]] / [[Native Sparse Attention|NSA]]** — architectural ways to need less cache traffic; Mooncake assumes standard cache and optimizes the **serving system**.

## Limitations

- Tied to a large multi-cluster LLM service (Kimi); not a single-GPU recipe.
- Prediction-based rejection can refuse work that a simpler FIFO queue would eventually finish (goodput vs fairness).
- Hierarchical DRAM/SSD cache needs fast fabric and careful placement; not free on commodity boxes.
- Eval mixes simulated stress + one production workload — other traffic mixes may shift the 525%/75% numbers.
- Model-agnostic-ish, but co-design with [[GQA]]/long-context models is where it shines.

## Why It Matters

This is the **production counterpoint** to papers that only cut FLOPs: Mooncake is what happens when [[KV Cache]] is the product bottleneck at real QPS. It connects this wiki’s inference thread ([[Prefill]]/[[Decode phase]], [[Paged attention]], [[KV offloading]]) to **SLO-aware scheduling** and **cache reuse as a first-class resource**. If you care about serving LLMs at long context, Mooncake is the reference systems paper.

## Related Concepts

[[KV Cache]] · [[KV Cache Optimization]] · [[KV offloading]] · [[Paged attention]] · [[vLLM]] · [[Prefill]] · [[Decode phase]] · [[VeriCache]] · [[YOCO]] · [[Native Sparse Attention]] · [[GQA]] · [[HBM]] · [[GPU]] · [[Long Range Arena]] · [[Lost in the Middle]]