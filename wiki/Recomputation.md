---
created: "2026-05-20"
title: "Recomputation (Gradient Checkpointing)"
tags: [glossary, training, memory, efficiency, backprop]
tldr: "Trading extra FLOPs for less memory in the backward pass by recomputing intermediate activations on the fly rather than storing them. FlashAttention uses this to avoid saving the T×T attention matrix for backprop."
aliases: [recomputation, gradient checkpointing, activation recomputation]
---

## TL;DR

In training, the backward pass needs the forward activations to compute gradients. Standard practice: store every intermediate tensor during the forward pass, use them in the backward pass. Recomputation discards some of those tensors and recomputes them during backprop. More FLOPs, but dramatically less memory — sometimes the only way to fit a large model on GPU.

## Intuition

Consider training a 100-layer transformer. Standard backprop stores the activation at every layer (100 tensors × $T \times d$ floats each). For long sequences this is gigabytes of HBM. Recomputation says: instead of storing all of them, store only a subset (e.g., every 10th layer), then during the backward pass, rerun the forward pass from the nearest checkpoint to reconstruct whatever you need.

**FlashAttention's recomputation** is particularly elegant: instead of storing the $T \times T$ attention probability matrix $P = \text{softmax}(QK^\top)$ for the backward pass (which is the whole point — this matrix is huge), FlashAttention stores only the $Q, K, V$ tensors and the logsumexp statistics $L = m + \log \ell$. In the backward pass, it recomputes $S$ and $P$ tile-by-tile from $Q$ and $K$. The extra FLOPs are cheap compared to the HBM traffic saved.

The tradeoff: recomputation increases FLOPs by roughly 30% for FlashAttention-style attention (you compute $QK^\top$ twice). But it saves $O(T^2)$ memory, which is the dominant cost at long sequences.

## Why It Matters

- **It's what allows FlashAttention to be memory-efficient.** The $T \times T$ attention matrix is never stored in HBM — it's recomputed during the backward pass from $Q, K, V$.
- **Gradient checkpointing scales model depth.** Without it, training 100+ layer models on a single GPU would be impossible for long sequences.
- **It's a key lever in the memory-compute tradeoff.** More recomputation = less memory = can use larger batch sizes or longer sequences.

## Where It Appears in This Wiki

- [[Flash Attention]] — the backward pass recomputes attention tile-by-tile from $Q, K, V$; never stores $T \times T$ matrix
- [[FlashAttention-2]] — inherits the same recomputation approach, just with better scheduling
- [[Hardware Acceleration for Neural Networks]] — recomputation is listed as a standard memory optimization lever

## Key Formula or Pseudocode

```
# Standard attention (stores P during forward)
Forward:  S = Q @ K.T → P = softmax(S) → O = P @ V  [store P]
Backward: dV = P.T @ dO, dP = dO @ V.T, dS = softmax_backward(P, dP)

# FlashAttention (recomputes P during backward)
Forward:  compute O tile-by-tile; store only Q, K, V, L (logsumexp)
Backward: for each tile, recompute S = Q_tile @ K_tile.T from scratch
          recompute P = softmax(S - L), then proceed with gradient
```

## Related Concepts

[[Flash Attention]] · [[Tiling]] · [[IO-awareness]] · [[HBM]] · [[SRAM]]
