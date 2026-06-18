---
created: "2026-05-28"
title: "State Space Duality (Mamba-2) Part I – The Model"
authors:
  - Tri Dao
  - Albert Gu
year: 2024
aliases:
  - "Mamba-2 Part I"
  - "SSD Model"
  - "State Space Duality"
  - "Mamba2 blog post"
tags:
  - state space model
  - SSM
  - mamba
  - attention
  - linear attention
  - sequence modeling
  - deep learning
  - language model
  - ssd
  - tensor-cores
  - recurrent-models
tldr: >
  Tri Dao and Albert Gu's blog post introducing Mamba-2 and the SSD (Structured State Space
  Duality) layer. Explains how a minor structural restriction on Mamba-1's diagonal-A SSM
  yields a scalar-times-identity form that is simultaneously a recurrent SSM and a form of
  linear attention, enabling tensor-core-friendly matrix-multiplication-based training.
  Mamba-2 achieves ~2–8× faster training than Mamba-1 and supports state sizes up to 128
  vs. 16 in Mamba-1.
wikilinks:
  - "[[Mamba]]"
  - "[[Transformers Are SSMs|Mamba-2]]"
  - "[[State Space Model|SSM]]"
  - "[[SSD]]"
  - "[[Attention]]"
  - "[[FlashAttention]]"
  - "[[RetNet]]"
  - "[[RWKV]]"
  - "[[Diagonal recurrence]]"
  - "[[Chunkwise recurrent]]"
  - "[[Exponential decay]]"
  - "[[Channel-mixing]]"
  - "[[Hardware-Aware Scan]]"
  - "[[FlashAttention-2]]"
  - "[[Sequence parallelism]]"
  - "[[KV cache]]"
---

# State Space Duality (Mamba-2) Part I – The Model

**Source:** [https://tridao.me/blog/2024/mamba2-part1-model/](https://tridao.me/blog/2024/mamba2-part1-model/)
**Authors:** Tri Dao (Princeton), Albert Gu (CMU)
**Published:** May 31, 2024
**Paper:** [arXiv 2405.21060](https://arxiv.org/abs/2405.21060)

---

## TL;DR

[[Transformers Are SSMs|Mamba-2]] introduces the **Structured State Space Duality ([[SSD]])** layer: a minimal restriction on [[Mamba]]'s diagonal-$A$ [[State Space Model|SSM]] — forcing $A_t$ to be scalar × identity — that simultaneously exposes a linear-[[Attention|attention]] interpretation. This duality enables training via tensor-core matrix multiplications, yielding **~2–8× faster training throughput** than [[Mamba]]'s parallel scan while supporting state sizes of $N=64$–$128$ (vs. $N=16$ in [[Mamba]]). The result is a model that is efficient as a recurrence at inference (constant-memory [[KV cache]] equivalent) and efficient as matrix operations during training.

---

## Motivation

Despite [[Mamba]]'s strong results, two key problems remained:

1. **Understanding:** SSMs felt disconnected from the dominant paradigm of **[[Attention|attention]]**. How are they related conceptually?
2. **Efficiency:** [[Mamba]]'s [[Hardware-Aware Scan]] did not use **tensor cores** (matrix-multiplication units). On an H100 GPU, BF16 matmul delivers **989 TFLOPS** vs. **67 TFLOPS** for FP32 scalar ops — a **~15× gap** that Mamba-1's scan algorithm could not exploit.

---

## Key Concepts

- **[[SSD]] (Structured State Space Duality):** The central contribution — a layer that is simultaneously a linear recurrence ([[State Space Model|SSM]]) and a form of linear [[Attention]], enabling both efficient inference and tensor-core training.
- **Scalar-times-identity $A_t$:** The key structural restriction over [[Mamba]]'s diagonal $A_t$; reduces the $N\times N$ matrix to a single scalar per timestep, enabling the attention-mode interpretation.
- **[[Diagonal recurrence]]:** The recurrence type $h_t = a_t \cdot h_{t-1} + B_t x_t$ used in [[SSD]], where $a_t \in \mathbb{R}$ is a scalar [[Exponential decay]].
- **[[Chunkwise recurrent]] algorithm:** [[SSD]] processes sequences in chunks, combining intra-chunk matmuls (quadratic/attention mode) with inter-chunk recurrence (linear mode) — analogous to [[FlashAttention]]'s tiling strategy.
- **Multi-head structure:** Analogous to multi-head [[Attention]], enabling larger per-head state sizes ($N=64$–$128$ vs. $N=16$ in [[Mamba]]).
- **Linear [[Attention]] connection:** The [[SSD]] quadratic form is equivalent to softmax-free [[Attention]] with a structured lower-triangular decay mask derived from $a_t$, connecting to [[RetNet]] and [[RWKV]].
- **[[KV cache]] analogy:** At inference, the SSM hidden state $h_t \in \mathbb{R}^{N \times P}$ plays the role of the [[KV cache]], but with **fixed, $O(1)$ memory** regardless of sequence length.
- **[[Channel-mixing]]:** Gating mechanism retained from [[Mamba]] in the overall block design.
- **Tensor-core utilization:** [[SSD]]'s matmul-based algorithm achieves **~989 TFLOPS** BF16 throughput on H100 vs. ~67 TFLOPS for [[Mamba]]'s scan — the primary systems motivation.

---

## The SSD Layer

### Core SSM equations (same as [[Mamba]])

$$h_t = A_t h_{t-1} + B_t x_t, \qquad y_t = C_t^\top h_t$$

- $x_t, y_t \in \mathbb{R}$ (scalars)
- $h_t \in \mathbb{R}^N$ (hidden state, size $N$)
- $(A, B, C)$ are **input-dependent** (selective) parameters

### The SSD Restriction

[[Mamba]] uses **diagonal** $A_t \in \mathbb{R}^{N \times N}$. The [[SSD]] layer restricts this further:

> **$A_t$ is a scalar × identity**: all diagonal elements are equal.

This means $A_t$ has shape $(T,)$ — a single scalar per timestep — giving a [[Diagonal recurrence]] of the form $h_t = a_t \cdot h_{t-1} + B_t x_t$.

This tiny change has outsized consequences: the [[SSD]] layer can be expressed **both** as:
- A **linear recurrence** ([[State Space Model|SSM]] mode) — efficient for inference; hidden state $h_t \in \mathbb{R}^{N \times P}$ serves as a fixed-size memory analogous to the [[KV cache]]
- A **quadratic attention-like form** ([[Attention]] mode) — efficient for training via matrix multiplications on tensor cores

This is the core of **State Space Duality ([[SSD]])**.

---

## Multi-Head Structure

To handle multi-dimensional inputs, [[Transformers Are SSMs|Mamba-2]] introduces a **multi-head** design analogous to multi-head [[Attention]]:

- Inputs $X, Y$ have shape $(T, H \cdot P)$ where $H$ = heads, $P$ = head dim
- Each head independently runs an [[SSD]] layer with its own $(A, B, C)$
- Larger state sizes (e.g., $N=64$ or $N=128$) are now tractable — vs. $N=16$ in [[Mamba]]

This is related to Multi-Query Attention in its grouped/shared-parameter philosophy.

---

## Mamba-2 Architecture

The overall [[Transformers Are SSMs|Mamba-2]] block differs from [[Mamba]] in:

| Feature | [[Mamba]] | [[Transformers Are SSMs|Mamba-2]] |
|---|---|---|
| A structure | Diagonal (N×N) | Scalar × Identity |
| State size N | Typically **16** | Typically **64–128** |
| Training algorithm | Parallel scan (no tensor cores) | [[SSD]] (tensor-core matmuls) |
| Training throughput | ~67 TFLOPS (scan) | ~989 TFLOPS (matmul, H100 BF16) |
| Head structure | No explicit heads | Multi-head |
| Normalization | — | Group norm per head |
| Inference memory | $O(N)$ fixed | $O(N \times P)$ fixed |

The block still uses a [[Channel-mixing]] gating mechanism and a local [[Diagonal recurrence]] for the conv layer.

---

## SSD vs. Attention

The [[SSD]] layer's quadratic form looks like:

$$Y = \text{masked\_softmax-free attention}(Q=C, K=B, V=X, \text{mask}=L)$$

where $L$ is a lower-triangular mask derived from the [[Exponential decay]] $a_t$. This is a form of **linear [[Attention]]** with a structured decay mask, connecting to models like [[RetNet]] and GLA (Gated Linear Attention).

Key contrasts with standard [[Transformer]] [[Attention]]:

| Property | [[Transformer]] [[Attention]] | [[SSD]] / [[Transformers Are SSMs|Mamba-2]] |
|---|---|---|
| Training complexity | $O(T^2)$ | $O(T^2)$ (quadratic mode) or $O(T)$ (recurrent) |
| Inference memory | $O(T)$ [[KV cache]] | $O(N \times P)$ fixed state |
| Decay mask | None (full attention) | Structured [[Exponential decay]] $a_t$ |
| Tensor-core friendly | Yes ([[FlashAttention]]) | Yes ([[SSD]] matmuls) |
| Selectivity | Via softmax | Via input-dependent $(A, B, C)$ |

---

## Key Insight

> The [[SSD]] model sits at the intersection of [[State Space Model|SSMs]] and linear [[Attention]] — enabling both efficient recurrent inference and GPU-optimized training via tensor cores, closing a **~15× hardware efficiency gap** vs. [[Mamba]]'s parallel scan.

---

## Why It Matters

- **Hardware efficiency:** [[SSD]]'s matmul-based training algorithm exploits tensor cores for **~989 TFLOPS** BF16 throughput on H100, vs. **~67 TFLOPS** for [[Mamba]]'s scan — a **~15× utilization improvement** and **2–8× end-to-end training speedup**.
- **Larger state capacity:** The multi-head design allows state sizes of $N=64$–$128$ (vs. $N=16$ in [[Mamba]]), giving the model **4–8× more memory capacity per layer** while remaining computationally tractable.
- **Theoretical unification:** By proving that [[SSD]] is simultaneously an [[State Space Model|SSM]] and a form of linear [[Attention]], the work provides a formal bridge between two previously disconnected paradigms, enabling cross-pollination of techniques (e.g., [[Sequence parallelism]], [[FlashAttention]]-style tiling) from [[Transformer]] systems research into recurrent models.

---

## Connections

- [[Mamba]] — predecessor selective [[State Space Model|SSM]]; [[Transformers Are SSMs|Mamba-2]] restricts its diagonal $A$ to scalar-times-identity
- [[Transformers Are SSMs|Mamba-2]] — the full model and architecture building on this [[SSD]] theory
- [[State Space Model|SSM]] — the broader family of state space models that [[SSD]] belongs to
- [[SSD]] — the specific algorithmic layer introduced here
- [[Attention]] — [[SSD]]'s quadratic form is equivalent to softmax-free linear attention with decay mask
- [[FlashAttention]] — systems inspiration for [[SSD]]'s chunkwise tiling algorithm; [[FlashAttention-2]] achieves analogous tensor-core utilization for standard [[Attention]]
- [[RetNet]] — uses fixed [[Exponential decay]] positional bias in a similar linear attention framework; [[SSD]] makes the decay input-dependent
- [[RWKV]] — another recurrent model with an attention-mode dual interpretation; [[SSD]] formalizes this duality more explicitly
- [[KV cache]] — [[SSD]]'s fixed-size hidden state $h_t$ is the recurrent analogue, with $O(1)$ memory vs. $O(T)$ for [[Transformer]]s
- [[Chunkwise recurrent]] — the [[SSD]] training algorithm; processes length-$C$ chunks with intra-chunk matmuls and inter-chunk recurrence
- [[Sequence parallelism]] — [[SSD]]'s matmul structure unlocks [[Transformer]]-style data and sequence parallelism for [[State Space Model|SSMs]]
- [[Jamba]] — hybrid [[Mamba]]/[[Transformer]] architecture that could benefit from [[SSD]]'s improved throughput

---

## Open Questions

1. **Optimal head/state-size tradeoffs:** [[Transformers Are SSMs|Mamba-2]] supports $N=64$–$128$, but the relationship between head count $H$, head dim $P$, and state size $N$ on downstream task quality is not fully characterized — how do these hyperparameters interact at scale (e.g., 7B+ parameters)?
2. **Beyond scalar-times-identity:** The SSD restriction ($A_t$ = scalar × identity) is sufficient for tensor-core efficiency, but are there intermediate structures (e.g., block-diagonal $A_t$) that recover more of [[Mamba]]'s expressivity while remaining matmul-friendly? The diagonal-to-scalar gap may leave modeling capacity on the table.
3. **Long-context scaling of the fixed state:** [[SSD]]'s $O(N \times P)$ inference memory is context-length-independent, but compression quality degrades for very long sequences. Can the chunkwise recurrent algorithm be adapted for hierarchical or multi-scale state updates (cf. [[speculative decoding]] for inference efficiency) to maintain quality beyond 100K+ tokens?

---

## Related Wiki Notes

- [[Mamba]] — predecessor selective [[State Space Model|SSM]]
- [[Transformers Are SSMs|Mamba-2]] — the full [[Transformers Are SSMs|Mamba-2]] model and training results
- [[State Space Model|SSM]] — broader state space model family
- [[SSD]] — the Structured State Space Duality algorithm
- [[Chunkwise recurrent]] — the [[SSD]] algorithm processes sequences in chunks
- [[Hardware-Aware Scan]] — [[Mamba]]'s approach replaced by [[SSD]] matmuls
- [[Diagonal recurrence]] — the recurrence type used in [[SSD]]
- [[Exponential decay]] — the scalar $a_t$ acts as a positional decay
- [[FlashAttention]] — systems inspiration for [[SSD]]
- [[Sequence parallelism]] — [[SSD]] unlocks [[Transformer]]-style parallelism for [[State Space Model|SSMs]]
- [[RWKV]] — another recurrent model with [[Attention]]-like duality
- [[RetNet]] — linear [[Attention]] with fixed [[Exponential decay]]; closely related to [[SSD]]
- [[KV cache]] — [[SSD]]'s fixed-size state as the recurrent analogue
- [[Jamba]] — hybrid architecture combining [[Mamba]] layers with [[Transformer]] blocks
