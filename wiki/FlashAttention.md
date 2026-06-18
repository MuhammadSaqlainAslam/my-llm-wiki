---
created: "2026-05-28"
title: "FlashAttention"
tags: [glossary, attention, efficiency, hardware-aware]
tldr: "IO-aware exact attention algorithm that tiles computation to avoid materializing the full N×N attention matrix in HBM, achieving 2-4x speedup with identical outputs."
---

## TL;DR
FlashAttention reorders the attention computation to keep data in SRAM (fast, ~20MB) instead of repeatedly reading/writing the full attention matrix to HBM (slow, ~40GB/s bandwidth bottleneck), producing bit-for-bit identical results to standard attention.

## Intuition
Standard attention computes `softmax(QKᵀ)V` and naively writes the full N×N score matrix to GPU HBM then reads it back — for N=2048 that's 2048²×2 bytes ≈ 16MB per layer per forward pass, and memory bandwidth is the bottleneck, not FLOPs. FlashAttention tiles Q, K, V into blocks that fit in SRAM (~100KB), fuses the softmax+matmul into one kernel, and streams through the computation without ever materializing the full matrix. The trick is online softmax: you can compute the running max and normalization factor incrementally as you process blocks, correcting previous outputs as new blocks arrive.

The memory savings are dramatic: O(N²) → O(N) HBM memory usage. For N=4096 on A100, standard attention uses ~1.3GB just for the attention matrix; FlashAttention uses ~1MB. This unlocks longer context (8k, 16k, 100k tokens) that was previously HBM-limited, not compute-limited.

## Why It Matters
- **Longer context windows**: Makes 32k–100k token contexts practical by breaking the O(N²) memory wall — e.g. GPT-4's long context, Llama 3's 128k window.
- **Training throughput**: 2-4x end-to-end speedup on GPT-2/BERT training benchmarks; FlashAttention-2 achieves ~72% of A100 peak FLOPs on attention.
- **Exact computation, no approximation**: Unlike sparse or linear attention approximations, gradients and outputs are numerically identical to standard attention — zero accuracy cost.

## Key Formula or Mechanism
```python
# Tiled online-softmax attention (pseudocode)
for block_j in blocks(K, V):           # iterate key/value blocks
    S_ij = Q_i @ block_j.K.T          # local scores
    m_new = max(m_prev, rowmax(S_ij)) # running max (numerically stable)
    P_ij = exp(S_ij - m_new)          # local softmax numerator
    # rescale previous output, accumulate new
    O_i = diag(exp(m_prev - m_new)) * O_i + P_ij @ block_j.V
    l_i = exp(m_prev - m_new) * l_i + rowsum(P_ij)  # running denom
O_i = O_i / l_i                        # final normalization
```

## Where It Appears
- **FlashAttention** (Dao et al., 2022) — original paper, MLSys 2022
- **FlashAttention-2** (Dao, 2023) — better parallelism, ~2x over v1
- **FlashAttention-3** (Shah et al., 2024) — H100-specific, asynchronous
- Adopted in: GPT-4, Llama 2/3, Mistral, Falcon, most modern transformer training stacks
- **Ring Attention** (Liu et al., 2023) — extends FlashAttention across multiple GPUs for million-token context

## Related Concepts
[[Attention|Scaled Dot-Product Attention]]
[[Online Softmax]]
[[FlashAttention|Memory-Efficient Attention]]
[[KV Cache]]
[[Multi-Head Attention]]
