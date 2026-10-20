---
created: "2026-10-19"
title: "LMCache: An Efficient KV Cache Layer for Enterprise-Scale LLM Inference"
authors: "Yuhan Liu, Yihua Cheng, Jiayi Yao, Yuwei An, Xiaokun Chen, Shaoting Feng, Yuyang Huang, Samuel Shen, Rui Zhang, Kuntai Du, Junchen Jiang"
year: 2025
arxiv: "2510.09665"
citation_count: 0
tags: [kv-cache, inference, serving, long-context, systems, prefill, decode-phase, offloading]
tldr: "Open-source KV cache **layer** for vLLM/SGLang: extract KV out of GPU memory, share across engines/queries, support prefix reuse and PD disaggregation. Batched moves + I/O pipelining + connector API + control API. Up to **15×** throughput with vLLM on multi-round QA / document analysis. Enterprise lesson: remote KV can *hurt* TTFT if the tier is too slow; truncation halves prefix hit rate."
aliases: [LMCache, LMCACHE, enterprise KV cache]
theme: inference
---

# LMCache

> Yuhan Liu, et al., "LMCache: An Efficient KV Cache Layer for Enterprise-Scale LLM Inference", 2025 (arXiv:2510.09665)

## The Problem / Motivation

[[KV Cache]] used to live only in GPU HBM. Production reality:

- Multi-turn / RAG / agents **reuse prefixes** across queries and **across engines**
- Total cached KV **exceeds GPU memory**
- Need offload + transfer without stalling [[Prefill]] / [[Decode phase]]

There was no efficient open **KV layer** — only ad-hoc offload. LMCache is that layer (github.com/LMCache/LMCache), used with [[vLLM]] and SGLang.

## The Idea

Treat KV as a **shareable object** outside the engine:

1. **Extract** KV from modern engines (connectors — don’t fork vLLM every release).
2. **Store / share** across GPU, CPU, disk, and **network** (other engines / PD clusters).
3. **Control API** to orchestrate placement and reuse (not just “a cache blob”).

Optimized **data movement**: batched transfers, **compute ↔ I/O pipelining**.

## Architecture / Method

```
vLLM / SGLang
    │ connector (extract / inject KV)
    ▼
LMCache layer  ── L1 GPU  ── L2 CPU  ── L3 disk / remote
    │
    ├─ prefix reuse across queries
    └─ prefill↔decode (PD) disaggregation (cross-GPU/engine transfer)
```

- **Connectors** decouple LMCache from engine churn.
- **Control API** — flexible cache orchestration across the memory/network stack.
- High-performance movers — the same “fewer stalled bytes” idea as [[Flash Attention]] / [[Mooncake]].

## Key Results

| Axis | Result |
|---|---|
| vLLM + LMCache throughput | Up to **15×** on multi-round QA and document analysis |
| Adoption | Wide **enterprise** use (paper) — real stats, not only lab |
| Ops insight 1 | Fetching KV from **remote** storage can **hurt prefill delay** if the tier is too slow |
| Ops insight 2 | **Context truncation** (common in industry) can **halve** prefix-cache hit ratio |

So the paper is both a system and a **cost model warning**: offload is not free.

## Comparison to Prior Work

- vs **[[Paged attention]] / vLLM prefix cache** — those are **in-GPU** bookkeeping; LMCache is a **cross-engine / cross-tier** KV layer.
- vs **[[Mooncake]]** — Mooncake is a full **cluster scheduler** (prefill/decode split + multi-tier + SLOs); LMCache is the reusable **cache layer** those systems can sit on (and is compared to in later papers).
- vs **[[KV offloading]] / [[VeriCache]]** — concept/technique notes; LMCache is the **production OSS system**.
- vs **[[KV Cache Optimization]] survey** — taxonomy of eviction/compression; LMCache is **hybrid memory + reuse** as software.

## Limitations

- Hit rate depends on **prefix locality**; truncation and personalized prompts destroy reuse (their own finding).
- Remote tier bandwidth/latency can make “hits” slower than recompute (need admission / cost checks — later work: py-kvcache, Range, etc.).
- Extra CPU/DRAM/SSD and network — capacity is cheap, **restore path** is not.
- Correctness bugs in multi-replica / stream sync are real (later case studies on LMCache).
- Not a model-architecture win — systems only (pairs with [[YOCO]], [[Native Sparse Attention]]).

## Why It Matters

This wiki’s serving thread ([[Mooncake]], [[Paged attention]], [[KV Cache Optimization]]) needs a **named OSS layer** people actually deploy. LMCache is that reference: **KV as a first-class cache service** for enterprise LLM inference, with honest enterprise caveats (remote can hurt; truncation kills hits). Read it with Mooncake for the cluster story and the [[KV offloading]] concept note for the tiering math.

## Related Concepts

[[KV Cache]] · [[KV Cache Optimization]] · [[KV offloading]] · [[Paged attention]] · [[vLLM]] · [[Prefill]] · [[Decode phase]] · [[Mooncake]] · [[VeriCache]] · [[YOCO]] · [[GQA]] · [[HBM]] · [[Memory hierarchy]] · [[RAG]] · [[Lost in the Middle]]