---
created: "2026-05-28"
title: "State Space Duality (Mamba-2) Part III – The Algorithm"
authors:
  - Tri Dao
  - Albert Gu
year: 2024
tags:
  - state space model
  - SSM
  - mamba
  - mamba-2
  - algorithm
  - hardware acceleration
  - tensor cores
  - chunkwise recurrent
  - deep learning
  - language model
  - SSD
  - block-matrix decomposition
tldr: >
  Deep dive into the SSD algorithm used in Mamba-2: a block-matrix decomposition of the
  SSM semiseparable matrix that splits computation into intra-chunk (matmul, tensor-core
  friendly) and inter-chunk (short scan) phases. Achieves up to 6× speedup over Mamba-1
  during training while supporting larger state sizes (N=64/128 vs. N=16).
aliases:
  - "SSD Algorithm"
  - "Mamba-2 Algorithm"
  - "State Space Duality Part 3"
  - "Chunkwise SSD"
wikilinks:
  - "[[Mamba]]"
  - "[[Transformers Are SSMs|Mamba-2]]"
  - "[[State Space Model|SSM]]"
  - "[[SSD]]"
  - "[[Chunkwise recurrent]]"
  - "[[Hardware-Aware Scan]]"
  - "[[FlashAttention]]"
  - "[[FlashAttention-2]]"
  - "[[Sequence parallelism]]"
  - "[[HBM]]"
  - "[[Memory hierarchy]]"
  - "[[Systolic array]]"
  - "[[Thread block]]"
  - "[[Diagonal recurrence]]"
  - "[[Exponential decay]]"
  - "[[RetNet]]"
  - "[[Attention]]"
---

# State Space Duality (Mamba-2) Part III – The Algorithm

**Source:** [https://tridao.me/blog/2024/mamba2-part3-algorithm/](https://tridao.me/blog/2024/mamba2-part3-algorithm/)
**Authors:** Tri Dao (Princeton), Albert Gu (CMU)
**Published:** May 31, 2024
**Paper:** [arXiv 2405.21060](https://arxiv.org/abs/2405.21060)

---

## TL;DR

The [[SSD]] (State Space Duality) algorithm in [[Transformers Are SSMs|Mamba-2]] rewrites the [[State Space Model|SSM]] sequence-mixing operation as a block-matrix decomposition of an $N$-semiseparable matrix. By splitting computation into tensor-core-friendly intra-chunk matmuls (~90% of FLOPs) and a short inter-chunk scan (on a sequence of length $T/Q$ instead of $T$), it achieves **2–6× training throughput improvement** over [[Mamba]] while unlocking state sizes of $N=64$–$128$ (vs. $N=16$ in [[Mamba]]) and enabling both sequence and tensor parallelism.

---

## Key Concepts

- **[[SSD]] (State Space Duality):** The core algorithm of [[Transformers Are SSMs|Mamba-2]]; recasts [[State Space Model|SSM]] computation as structured matrix multiplication over $N$-semiseparable matrices
- **$N$-semiseparable matrix:** The $(T \times T)$ sequence-mixing matrix $M$ whose block structure drives the four-step decomposition
- **Chunkwise decomposition:** Splitting the sequence into chunks of size $Q$; related to [[Chunkwise recurrent]] patterns in [[RetNet]] and Gated Linear Attention
- **Tensor core utilization:** Steps 1, 2, and 4 are pure matmuls; ~90% of total FLOPs run on tensor cores vs. 0% in [[Mamba]]
- **Short scan:** Step 3 scans only $T/Q$ chunk-boundary states rather than $T$ individual timesteps, dramatically reducing scan overhead
- **State size scaling:** [[Mamba]] capped at $N=16$ due to scan cost; [[SSD]] supports $N=64$–$128$ with negligible overhead
- **[[FlashAttention]]-style HBM efficiency:** Intra-chunk blocks fit in SRAM; cross-chunk state tensors are small ($N$ elements per boundary), minimizing [[HBM]] traffic
- **Log-space [[Exponential decay]]:** Numerically stable computation of the scalar decay $a_t$ for long sequences, preventing underflow
- **[[Sequence parallelism]]:** Enabled by [[SSD]]'s chunkwise structure; not possible with [[Mamba]]'s monolithic scan
- **[[Attention]] duality:** Intra-chunk computation is structurally analogous to masked [[Attention]], establishing the SSM–[[Attention]] duality of [[Transformers Are SSMs|Mamba-2]]

---

## Motivation: Tensor Cores vs. Scans

[[Mamba]] used a parallel associative scan ([[Hardware-Aware Scan]]) for training, which avoids $O(T^2)$ complexity but **cannot use tensor cores** (specialized matrix-multiply units). On modern GPUs:

| GPU | BF16 matmul TFLOPS | FP32 scalar TFLOPS | Ratio |
|---|---|---|---|
| A100 | 312 | 19 | 16× |
| H100 | 989 | 67 | ~15× |

[[Mamba]] was limited to state size $N=16$ because larger states made the scan prohibitively slow. **The [[SSD]] algorithm fixes this** by recasting the [[State Space Model|SSM]] computation primarily as matrix multiplications, routing ~90% of FLOPs through tensor cores.

---

## The SSD Algorithm: Block Matrix Decomposition

The [[SSD]] layer's sequence-mixing operation can be written as multiplication by an **$N$-semiseparable matrix** $M$ of shape $(T, T)$. The [[SSD]] algorithm decomposes this matrix into blocks of size $Q \times Q$:

### Four Steps

```
Step 1 (Intra-chunk, Orange):  Diagonal blocks
   Each diagonal block is a smaller semiseparable matrix.
   Compute using the quadratic (attention-like) form → pure matmul.

Step 2 (Green):  Chunk final states
   Compute the final SSM state at end of each chunk.
   Only T/Q distinct blocks → batched matmul.

Step 3 (Yellow):  State passing (short scan)
   The chunk final states form a short 1-semiseparable recurrence.
   Run a parallel or sequential scan on T/Q states.
   ← Only this step requires a scan; runs on a much shorter sequence.

Step 4 (Blue):  Contribution from initial states
   For each chunk, add the contribution from the true initial state
   computed in Step 3 → batched matmul.
```

### Chunking Interpretation

Equivalently: split the input sequence into **chunks of size Q**:
1. **Intra-chunk outputs** — local output assuming zero initial state (matmul)
2. **Chunk states** — final state of each chunk assuming zero initial (matmul)
3. **Pass states** — recurrence over all chunk final states (short scan on $T/Q$ elements)
4. **Output states** — contribution from true initial state per chunk (matmul)

Steps 1, 2, 4 use matmuls and run fully in parallel. Step 3 is a scan on a sequence of length $T/Q$ — fast in practice because $Q$ is typically 64–256, reducing scan length by 64–256×.

This is related to [[Chunkwise recurrent]] processing patterns, and the intra-chunk matmul is structurally equivalent to computing masked [[Attention]] within each chunk.

---

## Advantages Over Mamba-1

| Property | Mamba-1 | Mamba-2 (SSD) |
|---|---|---|
| State size N | 16 (limited by scan) | 64–128 |
| Tensor core usage | None (0%) | ~90% of FLOPs |
| Training throughput | Baseline | **~2–6× faster** |
| Sequence parallelism | No | Yes |
| Tensor parallelism | No | Yes |
| Scan sequence length | $T$ | $T/Q$ (64–256× shorter) |
| Dual [[Attention]] interpretation | No | Yes |

---

## Why It Matters

- **2–6× training throughput improvement** over [[Mamba]] on A100/H100 GPUs by routing ~90% of FLOPs through tensor cores (vs. 0% in [[Mamba]]-1's associative scan), directly translating to faster and cheaper model training at scale.
- **State size $N$ scales from 16 to 64–128** without prohibitive compute cost, giving [[Transformers Are SSMs|Mamba-2]] substantially higher representational capacity per layer and enabling it to match or exceed [[Transformer]] perplexity on language modeling benchmarks at equivalent parameter counts.
- **Unlocks sequence parallelism and tensor parallelism** for [[State Space Model|SSM]]-based models for the first time, enabling [[Transformers Are SSMs|Mamba-2]] to scale across multiple GPUs in the same way as [[Transformer]] training pipelines — a critical capability for training frontier-scale models.

---

## Connection to Prior Work

- **[[RetNet]] "chunkwise" mode:** a special case of [[SSD]] where the decay mask $L$ is a fixed geometric sequence (constant decay per position), rather than data-dependent decays; recovers [[RetNet]]'s recurrent chunk computation exactly
- **Gated Linear Attention (GLA):** another [[Chunkwise recurrent]] formulation, but derived from a different motivation; [[SSD]] subsumes GLA as a special case under the semiseparable matrix framework
- **[[FlashAttention]]:** [[SSD]] adopts the same HBM-minimization philosophy as [[FlashAttention]] and [[FlashAttention-2|FlashAttention-2]], tiling computation to fit in SRAM and avoiding large intermediate materialization in [[HBM]]
- **[[Jamba]]:** a hybrid [[State Space Model|SSM]]/[[Transformer]] architecture that interleaves [[Mamba]] and [[Attention]] layers — the [[SSD]] algorithm's [[Attention]] duality makes such hybrid designs more architecturally coherent
- SSD's derivation from block matrix decomposition makes parallelization more explicit than prior chunkwise methods

---

## Implementation Details

### HBM Efficiency
Like [[FlashAttention]], the [[SSD]] algorithm is designed to minimize reads/writes to [[HBM]]:
- Intra-chunk computation fits in [[Thread block|SRAM (shared memory)]]
- Cross-chunk state passing is small (size $N$ per chunk boundary; for $N=64$, this is 64 floats vs. $Q \times d$ for a full chunk)
- Leverages the [[Memory hierarchy]] of modern GPUs

### Numerical Stability
- The [[Exponential decay]] scalar $a_t$ can underflow for long sequences → log-space computation
- Discretization of continuous-time [[State Space Model|SSM]] parameters follows standard ZOH (zero-order hold)

### Chunk Size Selection
- $Q$ (chunk size) trades off intra-chunk matmul efficiency vs. scan frequency
- Typical values: $Q \in \{64, 128, 256\}$; larger $Q$ reduces scan overhead but increases SRAM pressure

---

## Code

A minimal Python implementation of [[SSD]] (Listing 1 from the paper):

```python
# From ssd_minimal.py in state-spaces/mamba
def segsum(x):
    """Stable segment sum for computing cumulative decays."""
    ...

def ssd_minimal_discrete(X, A, B, C, block_len):
    # X: (batch, seqlen, n_heads, d_head)
    # A: (batch, seqlen, n_heads)       — scalar decays
    # B: (batch, seqlen, n_heads, d_state)
    # C: (batch, seqlen, n_heads, d_state)
    # block_len: chunk size Q
    ...
    # Steps 1-4: chunked matmuls + short scan
    # Step 3 scan runs on T/Q elements, not T
```

---

## Open Questions

1. **Optimal chunk size $Q$ selection:** Is there a principled way to choose $Q$ dynamically based on sequence length, state size $N$, and hardware (A100 vs. H100 SRAM capacities), rather than fixing it as a hyperparameter? Could learned or adaptive chunking recover additional throughput?

2. **Data-dependent decay and numerical precision:** The log-space trick stabilizes scalar [[Exponential decay]] $a_t$, but as [[State Space Model|SSM]] state sizes grow toward $N=256+$ and sequence lengths reach 1M+ tokens, are there further numerical failure modes — particularly with matrix-valued (non-scalar) $A$ generalizations — and how should they be addressed?

3. **[[SSD]] in hybrid architectures:** Given that [[Jamba]] and similar hybrid models interleave [[State Space Model|SSM]] and [[Attention]] layers, can the [[SSD]] algorithm's block decomposition be co-designed with [[FlashAttention]] kernels to share SRAM tiling strategies and reduce end-to-end memory traffic across the full hybrid forward pass?

---

## Connections

- [[Mamba]] — predecessor model whose [[Hardware-Aware Scan]] approach [[SSD]] supersedes for training
- [[Transformers Are SSMs|Mamba-2]] — the full model built on the [[SSD]] algorithm; combines [[SSD]] layers with multi-head structure
- [[State Space Model|SSM]] — the mathematical framework underlying [[SSD]]; [[SSD]] is the efficient algorithm for computing [[State Space Model|SSM]] sequence mixing
- [[SSD]] — the State Space Duality framework connecting [[State Space Model|SSM]] and [[Attention]] computations
- [[Attention]] — [[SSD]]'s intra-chunk computation is structurally equivalent to masked [[Attention]]; establishes the SSM–[[Attention]] duality
- [[FlashAttention]] — systems design inspiration for HBM-efficient tiling; [[SSD]] applies analogous principles to [[State Space Model|SSM]] computation
- [[RetNet]] — [[SSD]] generalizes [[RetNet]]'s chunkwise mode by allowing data-dependent (non-constant) decay
- [[Chunkwise recurrent]] — the key computational pattern; [[SSD]] is the canonical hardware-aware realization
- [[KV cache]] — [[State Space Model|SSM]] recurrent state at inference time plays an analogous role to the [[KV cache]] in [[Transformer]]s; [[SSD]]'s fixed-size state ($N \times d$) makes this cache constant regardless of sequence length
- [[Jamba]] — hybrid [[State Space Model|SSM]]/[[Transformer]] architecture whose design is informed by the [[SSD]]–[[Attention]] duality
- [[Hardware-Aware Scan]] — [[Mamba]]-1's approach, replaced by [[SSD]] for training; still relevant for understanding the scan step (Step 3)
- [[HBM]] / [[Memory hierarchy]] — hardware considerations central to [[SSD]]'s design
- [[Systolic array]] / [[Thread block]] — GPU primitives leveraged by the matmul-heavy steps
- [[Sequence parallelism]] — now applicable to [[State Space Model|SSM]]s via [[SSD]]'s chunkwise structure
- [[Diagonal recurrence]] — the [[State Space Model|SSM]] recurrence form that [[SSD]] block-decomposes
- [[Exponential decay]] — role of scalar $a_t$ as decay; requires log-space handling for long sequences
