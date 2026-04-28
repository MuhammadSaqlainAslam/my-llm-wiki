---
title: "Grouped Query Attention (GQA)"
tags: [attention, memory, inference, kv-cache, efficiency]
tldr: "Partition Q heads into groups that share K and V heads. With 32 Q heads and 2 KV heads (as in Nemotron-3 Super), the KV cache shrinks 16× vs. standard MHA with minimal quality loss."
theme: efficiency
---

# Grouped Query Attention (GQA)

In standard multi-head attention (MHA) with $H$ heads, every head has its own Q, K, V projections. The [[KV Cache]] stores one key and one value tensor per head per past token: cache size $\propto H \times n \times d_{\text{head}}$. At $H = 32$, $n = 1{,}000{,}000$ tokens, and $d_{\text{head}} = 128$, with FP16, the KV cache alone is $2 \times 32 \times 10^6 \times 128 \times 2 \approx 16$ GB per layer — often larger than the model weights at long context. GQA partitions the $H$ query heads into $G$ groups ($G < H$). All query heads within a group share the same single pair of K and V heads. The grouped computation:

$$\text{head}_i = \text{Attention}(Q_i W_i^Q,\ K_g W_g^K,\ V_g W_g^V) \quad \text{where } g = \lfloor i / (H/G) \rfloor$$

With $G = H$ you recover standard MHA (one KV head per Q head). With $G = 1$ you get Multi-Query Attention (MQA) — one shared KV head for all Q heads, minimum memory, maximum quality compromise. GQA with $G = 2$ and $H = 32$ (as in [[Nemotron-3]] Super) stores only 2 K and 2 V tensors instead of 32 — a 16× reduction in KV cache size. Quality degradation from 32 → 2 KV heads is modest on most benchmarks because most of the information discriminated by multiple KV heads overlaps. The savings are enormous for long-context inference.

## Where it appears

- **[[Nemotron-3]]** — Super uses 32 Q heads and 2 KV heads throughout the (sparse) attention layers; this 16× cache reduction is part of what makes 1M-context inference practical

## Why it matters

- **It makes long-context inference fit in GPU memory.** Without GQA, serving Nemotron-3 Super at 1M context would require KV caches on the order of hundreds of GB per attention layer. With GQA and the Mamba-dominant layer mix (few attention layers), the total KV cache stays manageable.
- **It's a free lunch at the right $G$.** Empirically, going from MHA ($G = H$) to GQA with $G = H/4$ to $H/8$ causes minimal quality regression on most tasks — the redundancy across KV heads is high. Only at the extreme $G = 1$ (MQA) do you see consistent quality drops. GQA finds the Pareto knee.
- **It interacts with Mamba to double-down on memory savings.** In a hybrid model where 80%+ of layers are [[Mamba]] SSM layers (which have no KV cache at all) and only a few are attention layers, the attention layers' KV cache already dominates a smaller fraction of total memory. GQA on those few layers means the KV cache is negligible even at extreme sequence lengths.

---

*Related: [[KV Cache]] · [[Transformer]] · [[Nemotron-3]] · [[Mamba]]*
