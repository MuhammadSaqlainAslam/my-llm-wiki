---
created: "2026-05-12"
title: "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"
authors: "Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré"
year: 2022
arxiv: "2205.14135"
citation_count: 4404
tags: [attention, hardware, kernel-fusion, sram, gpu, io-awareness, efficiency]
tldr: "IO-aware exact attention that tiles computation into SRAM, never materializing the T×T attention matrix in HBM. 2–4× wall-clock speedup with zero approximation — the canonical hardware-aware algorithm and the template every serious attention kernel copies."
aliases: [FlashAttention, Flash Attention, FlashAttention v1]
theme: hardware
---

# FlashAttention

> Tri Dao, Daniel Fu, Stefano Ermon, Atri Rudra, Christopher Ré, "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness", NeurIPS 2022 (arXiv:2205.14135)

## TL;DR

Standard attention is not slow because it does a lot of arithmetic. It's slow because it moves a lot of bytes. For a sequence of length $T$, the softmax attention layer writes and reads a $T \times T$ matrix in [[HBM]] — that's $O(T^2)$ memory traffic, repeated on every forward and backward pass. On a modern GPU the HBM bandwidth (1–3 TB/s) is the binding constraint, not the number of multiplications.

**FlashAttention** restructures the exact same mathematical computation so that the $T \times T$ matrix is *never materialized in HBM*. The trick is tiling: divide $Q$, $K$, $V$ into blocks that fit in [[SRAM]] (~20 MB on an A100), fuse the softmax accumulation into the matmul, and write only the final output $O$ back to HBM. Exact result, 2–4× wall-clock speedup, up to 10–20× memory savings.

---

## The Core Idea — Bytes Are the Bottleneck

The standard attention recipe:

1. $S = QK^\top \in \mathbb{R}^{T \times T}$ — write to HBM
2. $P = \text{softmax}(S)$ — read $S$, write $P$
3. $O = PV$ — read $P$, write $O$

That's at minimum 3 round-trips through HBM just to move the attention matrix. For $T = 4096$, each layer produces a 64 MB matrix per head. With 32 heads and 96 layers of training, the bandwidth bill is staggering.

The GPU's SRAM (shared memory, ~20 MB) is 10–100× faster than HBM but tiny. FlashAttention's insight: keep all the intermediate math inside SRAM by tiling.

### Online Softmax

The core algorithmic challenge is that `softmax` requires the row maximum before dividing. FlashAttention implements an **online (numerically stable) softmax** that can accumulate one block at a time:

$$
m_i^{(j)} = \max(m_i^{(j-1)},\ \text{rowmax}(S_i^{(j)}))
$$

$$
O_i^{(j)} = \text{diag}(e^{m_i^{(j-1)} - m_i^{(j)}}) O_i^{(j-1)}
+ e^{S_i^{(j)} - m_i^{(j)}} V^{(j)}
$$

Each block of $K, V$ is streamed through in one pass. The running maximum $m_i^{(j)}$ and normalization factor are updated on the fly. No $T \times T$ matrix is ever written.

---

## Key Concepts

- **[[IO-awareness]]** — designing algorithms around reads/writes to slow memory (HBM), not just FLOPs
- **[[Kernel fusion]]** — merge the QKᵀ matmul + softmax + PV matmul into a single GPU kernel; intermediates stay in SRAM
- **[[SRAM]]** — on-chip shared memory (~20 MB on A100); 10–100× faster than HBM
- **[[HBM]]** — high-bandwidth off-chip DRAM (~1.5–3 TB/s); the slow tier that FlashAttention avoids
- **[[Online softmax]]** — a numerically stable running-sum formulation of softmax that works over tiled blocks
- **[[Tiling]]** — partitioning input matrices into SRAM-sized blocks to control memory residency
- **[[Recomputation]]** — in the backward pass, FlashAttention recomputes attention scores on the fly from Q, K, V rather than storing the $T \times T$ matrix; trades FLOPs for memory

---

## Architecture / Method

### Forward pass (conceptual sketch)

```
For each tile of Q (rows i to i+B):
  For each tile of K, V (columns j to j+B):
    Load Q_tile, K_tile, V_tile from HBM → SRAM
    S_tile = Q_tile @ K_tile.T          # in SRAM
    Update running max m, normalizer l
    Accumulate O_tile via online softmax formula
  Write O_tile to HBM
```

Total HBM writes: $O$ only ($T \times d$), plus logsumexp statistics for the backward pass.

### Backward pass

Recomputes attention scores tile by tile (reading only Q, K, V, not storing $P$), then computes gradients. This is more FLOPs than standard backprop but vastly fewer bytes moved.

### Block-sparse extension

The paper also proposes **block-sparse FlashAttention**: skip blocks known to be zero (e.g., outside a sliding window or a fixed sparsity pattern). The SRAM tile structure makes this natural — just skip the inner loop iteration.

---

## Key Results

| Model / Task | Baseline | FlashAttention |
|---|---|---|
| BERT-large (seq 512) | MLPerf 1.1 record | **+15% wall-clock** |
| GPT-2 (seq 1K) | HuggingFace impl | **3× speedup** |
| Long-range arena (1K–4K) | standard attention | **2.4× speedup** |
| Path-X (seq 16K) | all prior work: <50% | **61.4% accuracy** |
| Path-256 (seq 64K) | previously impossible | **63.1% accuracy** |

Memory: standard attention uses $O(T^2)$ HBM; FlashAttention uses $O(T)$ HBM (for the output and logsumexp).

---

## Comparison to Prior Work

- vs. **Approximate attention** (Linformer, Performer, Longformer) — those trade accuracy for asymptotic complexity. FlashAttention is exact but faster by being IO-aware, not approximation-aware.
- vs. **Standard PyTorch attention** — identical math, but PyTorch materializes the full $T \times T$ matrix in HBM; FlashAttention never does.
- vs. **[[FlashAttention-2]]** — v1's work partitioning was suboptimal (only 25–40% of peak FLOPS). v2 fixes thread-block and warp scheduling to reach 50–73%. v3 adds Hopper-specific TMA and warp specialization.

---

## Limitations

- **No context compression.** FlashAttention is exact attention — it doesn't reduce the $O(T^2)$ FLOPs, only the HBM traffic. For extremely long sequences, the compute cost still grows quadratically.
- **CUDA-specific at first.** The original implementation required hand-written CUDA kernels; porting to other backends (ROCm, TPU, CPU) was nontrivial (later addressed by community implementations and Triton ports).
- **Recomputation cost.** The backward pass recomputes attention on the fly, trading memory for extra FLOPs. On FLOP-bound workloads (small batches, long sequences) this can be noticeable.

---

## Why It Matters

- **It changed what sequences are practical.** Before FlashAttention, training on 8K+ sequences was memory-prohibitive. After, models like GPT-4 (32K) and Claude (100K) became tractable.
- **It is the canonical hardware-aware algorithm.** [[Hardware-Aware Scan]] in [[Mamba]] directly copies the FlashAttention playbook: fuse ops, keep intermediates in SRAM, minimize HBM round-trips.
- **Every serious attention kernel today is FlashAttention.** PyTorch's scaled_dot_product_attention, JAX's attention, cuDNN's attention — all incorporate these ideas.
- **It proves that exact methods beat approximations if you optimize the right thing.** Linear attention approximations were slower in wall-clock time despite lower FLOPs. FlashAttention is faster while being exact.

---

## Related Notes

[[FlashAttention-2]] · [[Transformer]] · [[Hardware Acceleration for Neural Networks]] · [[KV Cache Optimization]] · [[Hardware-Aware Scan]] · [[Kernel fusion]] · [[SRAM]] · [[HBM]] · [[Mamba]] · [[Speculative Decoding]]
