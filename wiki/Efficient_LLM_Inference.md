---
created: "2026-05-28"
title: "Efficient LLM Inference"
tags: [glossary, inference, systems, optimization]
tldr: "The set of techniques that reduce the time, memory, and compute cost of running a trained LLM at serving time without retraining."
---

## TL;DR
Getting a 70B model to respond in <100ms on affordable hardware requires aggressively co-optimizing memory layout, arithmetic precision, and batching strategy — inference is where the rubber meets the road.

## Intuition
A trained LLM is just a giant matrix multiply machine. At inference time, you're bottlenecked not by FLOPs but by **memory bandwidth** — for autoregressive decoding, each token requires loading all ~140GB of weights for a 70B model just to produce one token. This is called being *memory-bound*, and the arithmetic intensity (FLOPs/byte) is painfully low (~1 for batch=1 vs. ~300 for training).

The entire field of efficient inference is essentially: (1) reduce the bytes you move (quantization, pruning, sparsity), (2) amortize the cost over more tokens (batching, speculative decoding), and (3) cache and reuse computation (KV cache, prefix caching). Every technique attacks one or more of these axes. A useful mental model: a single A100 (80GB, 2TB/s bandwidth) can load its own memory ~25 times per second — that's your throughput ceiling for batch=1 decode.

## Why It Matters
- **KV cache is the central bottleneck** — storing keys/values for a 32k context window with a 70B model can require 32GB+ of VRAM, forcing architectural innovations like GQA, MQA, and paged attention (vLLM).
- **Quantization gives nearly free speedups** — INT8 inference loses <0.5% accuracy on most benchmarks while halving memory and doubling throughput; INT4 (GPTQ, AWQ) is viable for many tasks with ~1% degradation.
- **Speculative decoding breaks the serial bottleneck** — a small draft model proposes k=4–8 tokens in parallel, verified by the large model in one forward pass, achieving 2–3× speedup with *zero* quality loss.

## Key Formula or Mechanism
# Roofline model for inference throughput
arithmetic_intensity = FLOPs_per_token / bytes_moved_per_token
                     ≈ 2P / (2P) = 1  [batch=1, P=params, FP16]

# You're memory-bound when:
arithmetic_intensity < (peak_FLOPs / peak_bandwidth)
# A100: 312 TFLOPS / 2 TB/s = 156 → almost always memory-bound at decode

# Effective throughput ceiling (tokens/sec):
max_throughput ≈ peak_bandwidth / (bytes_per_param × num_params)
# A100 + 7B FP16: 2e12 / (2 × 7e9) ≈ 143 tokens/sec

## Where It Appears
- **FlashAttention** (Dao et al., 2022) — IO-aware exact attention, 2–4× faster
- **vLLM / PagedAttention** (Kwon et al., 2023) — virtual memory for KV cache
- **Speculative Decoding** (Leviathan et al., 2023; Chen et al., 2023)
- **GPTQ** (Frantar et al., 2022) — post-training INT4 quantization
- **Continuous Batching** (Orca, Yu et al., 2022) — iteration-level scheduling
- **AWQ** (Lin et al., 2023) — activation-aware weight quantization

## Related Concepts
[[KV Cache]]
[[NVFP4|Quantization]]
[[Speculative Decoding]]
[[FlashAttention]]
[[GQA|Grouped Query Attention]]
