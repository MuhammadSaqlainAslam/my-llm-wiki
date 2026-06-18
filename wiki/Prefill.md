---
created: "2026-05-12"
title: "Prefill (LLM Inference)"
tags: [inference, kv-cache, throughput]
tldr: "The first inference pass: process the entire input prompt at once and populate the [[KV Cache]]. Compute-bound and parallelizable; complementary to the [[Decode phase]] which is memory-bandwidth-bound and sequential."
---

# Prefill (LLM Inference)

An LLM inference request has two distinct phases. **Prefill** is the first pass: feed the prompt of length $L$ into the model in one parallel forward pass, computing and storing $K$ and $V$ for every prompt token. This is **compute-bound** — you're doing $O(L)$ work in parallel, the matmuls are large, tensor cores are saturated. Latency scales with prompt length but throughput is high. Then comes the **[[Decode phase]]**: generate tokens one at a time, reading from (and appending to) the KV cache. Decode is **memory-bandwidth-bound** — each step reads the full cache, the matmuls are tiny. Modern serving systems (vLLM, SGLang) batch prefill and decode separately because the optimal batch size, GPU utilization profile, and scheduling policy differ between the two.

## Where it appears

- [[KV Cache Optimization]]

---

*Related: [[Decode phase]] · [[KV Cache]] · [[Paged attention]]*
