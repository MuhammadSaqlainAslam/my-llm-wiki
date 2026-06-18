---
created: "2026-05-20"
title: "Tiling"
tags: [glossary, hardware, attention, memory, kernel-fusion]
tldr: "Partitioning a large matrix computation into smaller blocks that fit in on-chip SRAM, so the GPU can process each block without touching HBM. The core technique behind FlashAttention, cuBLAS GEMM, and all efficient matrix kernels."
aliases: [blocked computation, memory tiling, tile-based computation]
---

## TL;DR

Tiling is the practice of breaking a large matrix computation into small rectangular blocks (tiles), each sized to fit in the GPU's on-chip SRAM. The GPU loads one tile, processes it completely, accumulates results, then moves on. Intermediate values never touch HBM — which is 10–100× slower than SRAM.

## Intuition

Think of SRAM as a small whiteboard next to your desk, and HBM as a filing cabinet across the room. Standard attention: read the whole $T \times T$ attention matrix from the cabinet, compute softmax, walk back, read it again for the value multiply. Tiled attention: draw a $B \times B$ block of the attention matrix on the whiteboard, compute softmax for that block only (using [[Online softmax]] to keep running stats), accumulate the output, erase, draw the next block. You never put the whole matrix in the cabinet.

In numbers: for $T = 4096$, $d = 128$, and an A100 with 40 MB SRAM, you can fit a tile of ~$B = 64$ rows of $K$ and $V$ in SRAM at once. The full $4096 \times 4096$ attention matrix (64 MB) is never materialized.

This same idea appears everywhere:
- **GEMM (matmul)**: cuBLAS tiles the operands into register-level blocks for Tensor Cores
- **FlashAttention**: tiles the $Q, K, V$ matrices + accumulates softmax in SRAM
- **Mamba's Hardware-Aware Scan**: tiles the SSM recurrence to avoid writing intermediate states

## Why It Matters

- **It's the reason FlashAttention is faster than the naive softmax despite more FLOPs.** Tiling eliminates the HBM round-trips that dominate wall-clock time.
- **It determines what sequence lengths are tractable.** You can handle $T = 8192$ because the tile-based algorithm uses $O(T)$ HBM, not $O(T^2)$.
- **It's a universal GPU optimization primitive.** Any algorithm that can be expressed as a sequence of tile operations benefits from this pattern.

## Where It Appears in This Wiki

- [[Flash Attention]] — tiles $Q, K, V$ to compute attention without materializing the $T \times T$ matrix
- [[FlashAttention-2]] — tiles more aggressively across thread blocks to increase parallelism
- [[Hardware-Aware Scan]] — tiles the SSM recurrence in [[Mamba]]
- [[IO-awareness]] — tiling is the primary tool for achieving IO-awareness

## Key Formula or Pseudocode

```python
# Tiled matrix multiply (conceptual)
for i in range(0, M, TILE):
    for j in range(0, N, TILE):
        # Load C_tile, A_tile row, B_tile col into SRAM
        for k in range(0, K, TILE):
            A_tile = load_sram(A[i:i+TILE, k:k+TILE])  # from HBM once
            B_tile = load_sram(B[k:k+TILE, j:j+TILE])  # from HBM once
            C_tile += A_tile @ B_tile                    # all in SRAM
        write_hbm(C[i:i+TILE, j:j+TILE], C_tile)       # write once
```

## Related Concepts

[[IO-awareness]] · [[SRAM]] · [[HBM]] · [[Online softmax]] · [[Flash Attention]] · [[Kernel fusion]]
