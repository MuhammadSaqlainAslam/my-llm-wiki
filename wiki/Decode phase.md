---
created: "2026-05-12"
title: "Decode Phase (LLM Inference)"
tags: [inference, kv-cache, autoregressive, bandwidth]
tldr: "The autoregressive token-by-token generation phase that follows [[Prefill]]. Memory-bandwidth-bound: each step reads the full [[KV Cache]] from [[HBM]], the matmuls are tiny. Where speculative decoding and KV-cache optimizations pay off most."
---

# Decode Phase (LLM Inference)

After [[Prefill]] populates the cache, decode generates one token at a time. Each decode step: load the full model weights from HBM, load the entire KV cache from HBM, compute one new $(K_t, V_t)$ pair, append it to the cache, sample one output token. The matmuls per step are tiny — they don't saturate compute — and the binding constraint is *moving bytes from HBM*. Decode latency scales with model size + cache size + bandwidth. This is the regime where every KV-cache trick (compression, eviction, paging, GQA) matters most, and where [[Speculative Decoding]] gives its biggest wins by amortizing one HBM sweep across multiple generated tokens.

## Where it appears

- [[KV Cache Optimization]]

---

*Related: [[Prefill]] · [[KV Cache]] · [[Speculative Decoding]] · [[HBM]]*
