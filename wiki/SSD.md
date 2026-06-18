---
created: "2026-06-03"
title: "Structured State Space Duality (SSD)"
tags: [glossary, ssm, mamba, linear-attention, efficient-transformers]
tldr: "The mathematical equivalence between SSMs and linear attention, proved in Mamba-2: any N-semiseparable matrix can be computed either as a masked matrix product (attention-style) or as a recurrence (SSM-style)."
aliases: ["SSD", "State Space Duality"]
---

## TL;DR
SSD shows that Mamba-style SSMs and linear attention are two views of the same structured matrix computation. This duality lets you train with tensor-core matmuls (fast) and decode as a pure RNN (O(1) memory).

## Intuition
The Mamba recurrence $h_t = A_t h_{t-1} + B_t x_t$ and linear attention $O = (L \odot QK^\top)V$ both compute the same thing when the decay mask $L$ is structured in the right way — specifically when $A_t$ is constrained to scalar×identity. "Duality" means you can pick whichever formulation is more hardware-efficient for the moment: matrix multiply during training (attention view), recurrence during inference (SSM view).

The key constraint that enables the duality is **scalar-times-identity $A_t$** — a minimal restriction on Mamba-1's diagonal $A_t$ that simultaneously exposes the linear-attention structure. Under this constraint the state-space matrix becomes N-semiseparable, a well-studied class with efficient block-decomposition algorithms.

## Why It Matters
- **Training speed:** The SSD algorithm rewrites the SSM as block-diagonal matmuls, hitting GPU tensor cores — 2–8× faster throughput than Mamba-1's custom CUDA scan at matched quality.
- **State sizes scale up:** SSD-efficient training allows $N=64$–$128$ states (vs. $N=16$ in Mamba-1) without prohibitive compute cost — more representational capacity per layer.
- **Unified framework:** SSM research and linear-attention research can share algorithms, theory, and kernels (FlashAttention-style IO-aware tiling applies directly).

## Key Formula
$$Y = (L \odot CB^\top)\,X \qquad L_{ij} = \prod_{k=i+1}^{j} a_k \cdot \mathbf{1}[j \ge i]$$
$L$ is the structured causal mask; $\odot$ is elementwise multiply. Intra-chunk: compute $L \odot CB^\top$ as a dense matmul. Inter-chunk: carry a hidden state $h$ recurrently.

## Where It Appears
- **[[Transformers Are SSMs]]** — the Mamba-2 paper that introduces and proves SSD (arXiv:2405.21060)
- **[[Mamba2_SSD_Blog_Part1]]** — Tri Dao's blog post on the model view
- **[[Mamba2_SSD_Blog_Part3_Algorithm]]** — blog post on the algorithm / hardware efficiency

## Related Concepts
[[Transformers Are SSMs]] · [[Semiseparable matrix]] · [[Linear attention]] · [[Mamba]] · [[FlashAttention]] · [[Hardware-Aware Scan]]
