---
created: "2026-05-28"
title: "Efficient Transformer Architectures"
tags: [glossary, transformers, efficiency, architecture]
tldr: "Variants of the standard Transformer that reduce the O(n²) attention bottleneck in time, memory, or compute."
---

## TL;DR
Standard attention scales quadratically with sequence length; efficient transformers use sparse patterns, low-rank approximations, or linear kernels to push this toward O(n log n) or O(n).

## Intuition
The core problem: full self-attention on a sequence of length 4096 requires a 4096×4096 matrix (~67M entries, ~256MB in fp32 per layer per head). This becomes catastrophic at 100k+ tokens. The design space breaks into three camps: *sparse attention* (only attend to a structured subset of tokens — local windows, strided patterns, random hashes), *low-rank attention* (approximate the full matrix as a product of smaller ones, e.g. Linformer projects keys/values to k=256 dimensions regardless of n), and *kernel methods* (rewrite softmax(QKᵀ)V without materializing the full matrix, as in Performer's FAVOR+).

In practice, FlashAttention is the dominant practical win — not a mathematical approximation but a hardware-aware exact algorithm that tiles computation to stay in SRAM (~100x faster bandwidth), achieving ~3× wall-clock speedup on A100s while using O(n) memory instead of O(n²). The key insight is that the bottleneck is usually memory bandwidth, not FLOPs.

## Why It Matters
- **Long-context LLMs** (Claude 200k, Gemini 1M tokens) are only feasible because of IO-aware attention algorithms like FlashAttention-2/3 that keep memory linear in sequence length.
- **Training throughput** is directly bottlenecked by attention in long sequences; FlashAttention-2 achieves ~70% of theoretical A100 MFU vs ~35% for naive PyTorch attention.
- **Architectural search** for RNNs/SSMs (Mamba, RWKV) is partly motivated by making inference O(1) per token instead of O(n), critical for edge deployment.

## Key Formula or Mechanism
**Standard attention (expensive):**
O = softmax(QKᵀ / √d) · V    # materializes n×n matrix

**FlashAttention (IO-aware tiling — exact, not approximate):**
```python
# Tile Q, K, V into blocks that fit in SRAM (~20MB on A100)
for block_q in tiles(Q):
    m, l, acc = -inf, 0, 0          # running max, normalizer, output
    for block_k, block_v in tiles(K, V):
        s = block_q @ block_k.T / sqrt(d)
        m_new = max(m, rowmax(s))
        acc = acc * exp(m - m_new) + exp(s - m_new) @ block_v
        l   = l   * exp(m - m_new) + rowsum(exp(s - m_new))
        m   = m_new
    O[block_q] = acc / l            # numerically stable, never writes n×n
```

## Where It Appears
- **FlashAttention** — Dao et al. 2022, 2023 (FlashAttention-2), 2024 (FlashAttention-3)
- **Longformer / BigBird** — local + global sparse attention patterns
- **Linformer** — Wang et al. 2020, low-rank KV projection
- **Performer** — Choromanski et al. 2021, random feature kernel approximation
- **Mamba / S4** — Gu et al., SSM alternative avoiding attention entirely

## Related Concepts
[[Attention|Attention Mechanism]]
[[FlashAttention]]
[[State Space Model|State Space Models]]
[[Sliding window attention|Sparse Attention]]
[[KV Cache]]
