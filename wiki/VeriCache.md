---
created: "2026-09-01"
title: "VeriCache: Turning Lossy KV Cache into Lossless LLM Inference"
authors: "Jiayi Yao, Samuel Shen, Kuntai Du, Shaoting Feng, Dongjoo Seo, Rui Zhang, Yuyang Huang, Yuhan Liu, Shan Lu, Junchen Jiang"
year: 2026
arxiv: "2605.17613"
tags: [kv-cache, lossless, speculative-decoding, cache-compression, inference-optimization]
citation_count: 0
tldr: "KV-cache compression (token dropping, quantization) is lossy and gets worse the longer you decode — fine for short outputs, catastrophic for code generation and tool calling. VeriCache drafts tokens using the cheap compressed cache, then verifies them against the full cache kept off-GPU, achieving byte-identical output to full-KV decoding at up to 4x the throughput."
aliases: ["VeriCache"]
---

# VeriCache: Turning Lossy KV Cache into Lossless LLM Inference

> Jiayi Yao, Samuel Shen, Kuntai Du, Shaoting Feng, Dongjoo Seo, Rui Zhang, Yuyang Huang, Yuhan Liu, Shan Lu, Junchen Jiang (University of Chicago, Tensormesh Inc., Samsung Semiconductor, Microsoft Research), "VeriCache: Turning Lossy KV Cache into Lossless LLM Inference", May 2026 (arXiv:2605.17613)

## The Problem / Motivation

As covered across this wiki's [[KV Cache Optimization]] survey, the KV cache is the dominant memory cost at long context, and the standard fixes — token dropping ([[Cache eviction]]) and quantization ([[Cache compression]]) — all trade accuracy for memory. That trade looks fine in short benchmarks: a compressed cache produces an output that's *close* to the full-cache output, and for a short answer, close is usually good enough. But decoding is autoregressive — every new token is generated conditioned on all the previous ones. A small divergence early in generation compounds: by token 500, a compressed-cache model's output can have drifted completely away from what the full cache would have produced. For most chat use cases this is a tolerable approximation. For **code generation and tool calling**, where a single wrong token can produce a syntax error or an invalid API call, it's a correctness bug, not a rounding error.

## The Idea

Don't choose between "fast and lossy" or "slow and exact." Use the fast, lossy compressed cache to **draft** tokens cheaply, then **verify** those drafts against the full, uncompressed cache — structurally the same draft-then-verify shape as [[Speculative Decoding]], except here the "draft model" and the "target model" are literally the same model; the only difference between draft and verify passes is which KV cache (compressed vs. full) the model attends over. If the drafted tokens match what the full cache would have produced, you keep them and move on at the compressed cache's speed. If they diverge, you fall back to the full cache and correct course. The output is byte-identical to full-KV decoding, because every token is checked against the full cache before being finalized — but most of the decoding happens at the compressed cache's speed.

The system engineering challenge this creates: the full KV cache has to exist *somewhere* to verify against, but keeping it resident in GPU memory the whole time defeats the entire point of compressing it in the first place. VeriCache's answer is to keep the full cache off-GPU (host DRAM) and swap it in only when verification needs it.

## Architecture / Method

```
                    ┌─────────────────────────────┐
                    │   Compressed KV cache (GPU)   │◀── HBM-bandwidth-bound
                    └─────────────────────────────┘
                              │
                              │ 1. Draft N tokens fast, using
                              │    only the compressed cache
                              ▼
                    draft tokens t_1 ... t_N
                              │
                              │ 2. Swap in full KV cache from
                              │    host DRAM (PCIe/network-bound —
   ┌──────────────────────────    happens IN PARALLEL with step 1
   │  Full KV cache (host DRAM)    for the *next* drafting window)
   └──────────────────────────┘
                              │
                              │ 3. Verify t_1...t_N against full cache
                              ▼
                    accept matching prefix, correct at first divergence
                              │
                              ▼
                    byte-identical output to full-KV decoding
```

Two insights make this fast rather than merely correct:

1. **The two costly operations are bound by different resources.** Compressed-cache decoding is HBM-bandwidth-bound (moving bytes within the GPU); swapping the full cache in from host DRAM is PCIe/network-bandwidth-bound (moving bytes between host and GPU). Because they bottleneck on different resources, they can run **in parallel** — the next window's full-cache swap-in happens while the current window's compressed-cache drafting is still running.
2. **Compressed output is usually close to full-cache output**, so a long drafting horizon (many tokens drafted before verification) amortizes the cost of each full-KV swap-in across many tokens, rather than paying the swap cost per-token.

VeriCache exposes compressors (token-dropping and quantization methods alike) through a **uniform compressor interface**, so it composes with a broad family of existing lossy KV-cache methods rather than requiring a bespoke compressor, and it also composes with traditional two-model speculative decoding on top.

## Key Results

| Metric | VeriCache | Lossy baseline (KVzip, compression 0.5) |
|---|---|---|
| Output correctness | Byte-identical to full-KV decoding | Diverges — accumulates ~14.4 nats KL per request on Llama-70B |
| Probability of reproducing Full-KV's exact output | Effectively guaranteed (by construction) | ~5×10⁻⁷ |
| KL divergence from Full KV | < 0.01 nats (attributable to hardware nondeterminism) | ~14.4 nats |
| Throughput vs. Full-KV inference | up to **4×** higher | — |
| Throughput on Llama-70B, Pipeline 1 | up to **3.82×** Full KV's throughput | — |
| Function-calling benchmark throughput at Full-KV accuracy | at least **59%** of the fastest KVzip configuration's throughput | KVzip loses up to ~30 accuracy points at matching throughput |

The function-calling result is the paper's sharpest illustration of the problem it's solving: KVzip can match VeriCache's throughput, but only by giving up ~30 points of accuracy on tool-calling correctness — exactly the failure mode (a single wrong token breaks a structured output) that motivates the paper.

## Comparison to Prior Work

- vs. **lossy KV-cache compression (token dropping, quantization — surveyed in [[KV Cache Optimization]])** — those methods trade accuracy for memory/throughput unconditionally; VeriCache uses the same compressors but adds a verification layer that recovers exactness, at a throughput cost much smaller than the accuracy the compressors would otherwise sacrifice.
- vs. **classic two-model [[Speculative Decoding]] (Leviathan 2023, Chen 2023)** — classic speculative decoding drafts with a smaller, separate draft model and verifies with the full target model; VeriCache drafts and verifies with the *same* model, using compressed vs. full KV cache as the speed/accuracy lever instead of model size. The two techniques compose: VeriCache explicitly supports layering on top of traditional speculative decoding.
- vs. **naive full-KV-cache retention** — full-KV inference is the accuracy ceiling VeriCache targets exactly, but VeriCache reaches most of the compressed cache's throughput by keeping the full cache off the critical GPU-memory path rather than paying its memory cost directly.

## Limitations

- Requires host DRAM (or equivalent off-GPU memory) capacity to hold the full KV cache and enough PCIe/network bandwidth for swap-ins to be amortizable — deployments without fast host-GPU interconnects will see less benefit.
- The throughput win depends on compressed-cache output actually being *close* to full-cache output most of the time; if a compressor produces frequent large divergences, verification triggers correction often and the amortization argument weakens.
- Evaluated primarily on function-calling and standard decoding benchmarks; the paper doesn't establish how the approach interacts with every existing eviction/compression scheme surveyed in [[KV Cache Optimization]], only that it exposes a uniform interface for them.

## Why It Matters

VeriCache resolves what looked like a fundamental tradeoff in KV-cache optimization — fast-but-approximate vs. exact-but-slow — by recognizing that the two operations it needs (fast compressed decoding, and full-cache verification) are bottlenecked by different hardware resources and can therefore overlap instead of competing. That reframing matters most exactly where the [[KV Cache Optimization]] survey's scenario map says accuracy is non-negotiable: code generation, tool/function calling, and agentic workflows, where a single dropped token is a correctness failure, not a quality degradation. It's also a clean example of applying speculative decoding's core insight (verify cheap drafts against an expensive ground truth) to a dimension — cache fidelity — rather than model size, which the [[Speculative Decoding]] literature had not previously targeted.

## Related Concepts

[[KV Cache]] · [[Speculative Decoding]] · [[Cache Compression]] · [[KV Cache Optimization]] · [[Hybrid memory|Hybrid Memory (KV-Cache Tiering)]]
