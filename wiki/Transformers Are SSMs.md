---
created: "2026-05-12"
title: "Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality"
authors: "Tri Dao, Albert Gu"
year: "2024"
arxiv: "2405.21060"
tags: [ssm, mamba, attention, transformers, state-space-models, linear-attention, structured-matrices, mamba-2, sequence-modeling, language-modeling]
tldr: "Proves SSMs and attention are dual forms of the same structured semiseparable matrix computation, yielding Mamba-2: 2-8x faster than Mamba-1 while matching or beating Transformers at language modeling."
citation_count: 1744
aliases: ["SSD", "Mamba-2", "State Space Duality"]
---

## TL;DR
SSMs (like Mamba) and variants of attention are not competing paradigms — they're two computational views of the same underlying structured matrix class: **semiseparable matrices**. This "Structured State Space Duality" (SSD) framework yields a new SSD algorithm that is **2-8× faster than Mamba-1's selective scan**, crosses over with FlashAttention-2 at sequence length 2K, and is **6× faster at 16K**. The resulting Mamba-2 architecture (2.7B params, 300B tokens) outperforms both Mamba-2.8B and Pythia-6.9B on the Pile.

## The Problem
Transformers scale **quadratically** in sequence length during training and maintain a **linear KV cache** during inference. SSMs (Mamba) solve this with linear training complexity and **constant state size** at inference, but suffer from two major issues:
1. **No matrix multiplication utilization**: Mamba's selective scan can't leverage GPU/TPU tensor cores, making it slower than its theoretical complexity implies.
2. **Ecosystem gap**: SSMs lack the rich tooling (tensor parallelism, sequence parallelism, variable-length batching) developed for Transformers, making large-scale training awkward.
3. **Theoretical isolation**: SSMs and attention were developed in parallel with no unifying framework, preventing cross-pollination of algorithmic ideas.

Concretely: Mamba's selective scan is ~2-8× slower than it needs to be, and the state size $N$ is constrained small because of the hardware-unaware implementation.

## Core Innovation
**Semiseparable matrices are the bridge.**

A 1-semiseparable matrix $M$ is one where every submatrix in the lower-triangular part has rank ≤ 1. The key insight: **the full output matrix of a scalar SSM is exactly a 1-semiseparable matrix**, and **linear attention's output matrix is also semiseparable**. Both are just different algorithms for multiplying by the same class of structured matrix.

Analogy: Think of it like FFT vs. naive DFT — they compute the same thing (DFT matrix multiplication) via different algorithms. SSD reveals that SSM recurrence and quadratic attention are both algorithms for "semiseparable matrix × vector" multiplication, and this insight unlocks a **block decomposition algorithm** that mixes both approaches optimally.

## Architecture / Method

### Semiseparable Matrices
An $N$-semiseparable matrix $M \in \mathbb{R}^{T \times T}$ satisfies: for all $i > j$, $\text{rank}(M_{i:T, 1:j}) \leq N$. The scalar SSM output can be written as:

$$Y = M X, \quad M_{ij} = \begin{cases} C_i^\top \left(\prod_{k=j+1}^{i} A_k\right) B_j & i \geq j \\ 0 & i < j \end{cases}$$

This is exactly a **1-semiseparable matrix** (scalar case) or **N-semiseparable** (full state size N).

### Selective SSM Recurrence (Mamba-style)
Time-varying parameters $(A_t, B_t, C_t)$:
$$h_t = A_t h_{t-1} + B_t x_t, \qquad y_t = C_t^\top h_t$$

This is inherently sequential — no parallelism without restructuring.

### Structured Masked Attention (SMA)
Linear attention with a structured mask matrix $L$ (lower-triangular, structured):
$$Y = (Q K^\top \odot L) V$$

When $L_{ij} = \prod_{k=j}^{i} a_k$ (the cumulative product of scalar gates), SMA is **exactly equivalent** to the scalar SSM. This is the duality.

### SSD Algorithm (Block Decomposition)
Partition the sequence into blocks of size $Q$. For a block matrix $M$ split into diagonal blocks $D$ (small, dense) and off-diagonal blocks (low-rank via semiseparability):

```
For each block b:
  1. Compute intra-block output using quadratic attention form: O(Q²N)
  2. Compute inter-block contributions by passing recurrent state between blocks: O(T/Q · N²)
  3. Optimal block size Q = N balances both terms → O(T · N · sqrt(T)) total, but in practice Q~64-256
```

The key win: **intra-block uses matrix multiplication (fast on GPU)**, inter-block uses recurrence (small state). This hybrid extracts the best of both worlds.

### Multi-Head SSM (Mamba-2 Architecture)
Importing the "heads" concept from MHA:
- **State dimension**: $(B, T, H, N)$ where $H$ = number of heads, $N$ = head state size
- **Head structure**: Mamba-1 is equivalent to "multi-value attention" (MVA) — many V heads, few Q/K heads. Mamba-2 uses **grouped-value attention (GVA)** structure
- **Parallel block**: All projections ($Q, K, V$, gates) computed in parallel at block start → enables tensor parallelism with half the synchronization points vs. Mamba-1

```
Mamba-2 block:
  z, x, B, C, Δ = split(linear(norm(input)))   # all projections parallel
  A = -exp(A_param)                              # scalar, per-head
  y = SSD(x, A, B, C, Δ)                        # core SSD layer
  output = linear(y * silu(z))                   # gating + project
```

## Key Results

### Language Modeling (Pile, standard evals)

| Model | Params | Tokens | LAMBADA | HellaSwag | PIQA | Arc-E | WinoGrande | Avg |
|-------|--------|--------|---------|-----------|------|-------|------------|-----|
| Mamba-1 | 2.8B | 300B | 69.2 | 66.5 | 75.2 | 65.5 | 67.0 | 68.7 |
| Pythia | 2.8B | 300B | 67.1 | 66.9 | 74.0 | 64.4 | 67.1 | 67.9 |
| Pythia | 6.9B | 300B | 73.0 | 72.2 | 76.6 | 67.8 | 70.1 | 71.9 |
| **Mamba-2** | **2.7B** | **300B** | **69.7** | **68.7** | **76.4** | **69.6** | **67.6** | **70.4** |

**Mamba-2 2.7B beats Pythia-6.9B** (2.6× more params) on average.

### Training Speed vs. Mamba-1

| Sequence Length | Speedup (SSD vs. Mamba selective scan) |
|----------------|----------------------------------------|
| 2K | ~2× |
| 8K | ~4× |
| 16K | ~8× |

### vs. FlashAttention-2

| Sequence Length | SSD vs. FA2 |
|----------------|-------------|
| 2K | ~1× (crossover point) |
| 8K | ~3× faster |
| 16K | ~6× faster |

### Scaling Laws
Mamba-2 **Pareto dominates** both Mamba-1 and Transformer++ in perplexity vs. wall-clock time (Chinchilla scaling law experiments).

### Multi-Query Associative Recall
Mamba-2 matches or exceeds Transformers on the difficult MQAR task (Arora et al. 2024), which specifically probes associative memory capacity.

## Why It Matters

- **Unified theory**: Proves SSMs and linear attention are not competing approaches but dual algorithms on the same mathematical object (semiseparable matrices). Any fast recurrent kernel attention *must* be an SSM.
- **Hardware efficiency**: SSD uses BLAS matrix multiplication for the intra-block computation, finally letting SSMs leverage GPU tensor cores. 2-8× speedup is real throughput, not FLOP counting.
- **Larger state sizes**: Mamba-1 was constrained to small $N$ due to hardware implementation. SSD supports **8× larger state sizes** with minimal slowdown, significantly increasing model expressivity.
- **Ecosystem compatibility**: Mamba-2's parallel block design enables tensor parallelism (TP), sequence parallelism, and variable-length batching — matching the infrastructure capabilities that made Transformers dominant at scale.
- **Scaling efficiency**: Mamba-2 2.7B matches a 6.9B Transformer (Pythia), suggesting ~2.5× parameter efficiency advantage, which compounds dramatically at scale.

## Connections to Other Work

### Builds On
- [[Mamba]] — Mamba-2 is a direct refinement of Mamba-1's selective SSM layer
- [[S4]] — foundational SSM work, diagonal A matrix structure
- [[Linear attention|Linear Attention]] — Katharopoulos et al. 2020, first showed attention-RNN duality; SSD significantly generalizes this
- [[FlashAttention-2]] — the hardware-efficient attention baseline SSD competes with and outperforms at long sequences

### Competes With
- [[LLaMA 2|Transformer++ / Llama]] — Mamba-2 matches quality with better scaling efficiency
- [[Pythia]] — direct comparison baseline on Pile
- [[RWKV]] — another linear-complexity sequence model with recurrent form
- [[RetNet]] — retention mechanism, also attempts SSM-attention unification
- [[Gated Linear Attention]] — related gated linear attention work in the SMA family

### Enables
- [[Jamba|Hybrid SSM-Attention]] — SSD framework makes mixing SSM and attention layers principled
- [[KV Cache Optimization|Long-Context Language Models]] — 6× speedup at 16K enables practical very long sequence training
- [[Sequence parallelism|Sequence Parallelism]] — directly described in the paper

## Limitations

1. **Scalar A constraint**: The SSD layer as described uses a **scalar (per-head) A matrix** rather than the full diagonal matrix of Mamba-1. This is a strict structural simplification that may reduce expressivity in some settings, traded for the duality and speed.
2. **Quadratic form still exists**: The intra-block computation is quadratic in block size $Q$; for very long sequences this is hidden inside blocks, but the algorithm still requires tuning block size as a hyperparameter.
3. **Not full softmax attention**: The duality connects SSMs to *linear* (kernel) attention, not full softmax attention with its normalization. The theoretical framework doesn't directly explain why softmax attention works so well.
4. **Scale validation**: Most experiments are "small to medium scale" (≤3B params, 300B tokens). Whether Mamba-2 continues to dominate Transformers at 70B+ params remains unvalidated.
5. **Task diversity**: Strong on standard LM benchmarks but the community has observed SSMs can lag on tasks requiring precise in-context lookup / retrieval at very long range.

## Open Questions

1. **Can the scalar-A restriction be lifted?** The SSD framework requires scalar (rank-1) state transitions for the clean semiseparable matrix correspondence. Can a richer structured matrix class (e.g., N-semiseparable with full rank-N transitions) be computed with similar hardware efficiency while maintaining the attention duality?

2. **What is the right head structure?** The paper introduces MVA, GVA, and MHA analogs for SSMs but doesn't definitively resolve which head structure maximizes quality-efficiency tradeoff. How do these interact with state size $N$ and model depth?

3. **Hybrid architectures via SSD**: Since SSD reveals SSMs and attention as dual, can we design layers that *dynamically* choose between the recurrent and quadratic form at inference time based on sequence length or content — a truly unified sequence model?

## Related Concepts
[[Semiseparable matrix|Semiseparable Matrices]]
[[Linear Attention]]
[[State Space Model|Structured State Space Models]]
[[Selective State Space Model|Selective State Space Models]]
[[Tensor Parallelism]]
[[Sequence Parallelism]]
[[Hardware-aware algorithms|Hardware-Aware Algorithm Design]]
[[LSTM|Recurrent Neural Networks]]
[[Semiseparable matrix|Semiseparable Matrix Decomposition]]
[[Chinchilla Scaling Laws]]
[[Multi-Head Attention]]
[[FlashAttention]]
