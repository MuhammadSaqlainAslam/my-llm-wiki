---
created: "2026-05-28"
title: "Gated Linear Attention (GLA)"
tags: [glossary, attention, linear-attention, ssm, efficient-transformers]
tldr: "A linear attention variant that adds a learnable, data-dependent decay gate to the hidden state, giving SSM-like selective memory while keeping O(1) recurrent inference and a hardware-efficient chunkwise-parallel training form."
aliases: [GLA]
---

## TL;DR
GLA fuses the best of linear attention and gated SSMs: it maintains a 2D key–value hidden state and multiplies it by a learnable, **input-dependent decay gate** at every step. This gives the model dynamic control over what to remember — closing much of the quality gap with softmax Transformers — while keeping ~O(1) per-token inference cost.

## Intuition
Standard linear attention rewrites `softmax(QKᵀ)V` as `Q(KᵀV)`, collapsing the sequence into a `d×d` hidden state `S_t = S_{t-1} + kₜᵀvₜ`. This is fast, but every past token contributes equally forever — old irrelevant tokens pollute new predictions, there is no forgetting. GLA fixes this by multiplying the accumulated state by a gating matrix `G_t = sigmoid(W_g · xₜ)` before adding the new association: `S_t = G_t ⊙ S_{t-1} + kₜᵀvₜ`. Because the gates are *data-dependent*, the model learns *what* to forget at each step — a 2D analogue of the LSTM forget gate operating on a full feature matrix, and closely related to Mamba's selective scan. In practice GLA uses a low-rank factored gate (rank ~16) to keep parameters manageable; a full `d×d` gate at d=2048 would cost ~4M extra parameters per layer just for the gate projection.

The key engineering win: although per-token gating breaks naive parallelism, GLA still trains efficiently via a **chunkwise-parallel** form. The sequence is split into chunks (~64–256 tokens); intra-chunk attention runs in parallel as a small materialized form, while inter-chunk hidden states are propagated recurrently with an accumulated gate product. This reaches roughly 2–3× the throughput of naive full attention at length 2K (and ~80% of FlashAttention throughput on A100s), scaling to 1B+ parameters on language modeling with little quality loss versus Transformers.

## Why It Matters
- **Truly O(1) decoding**: At generation time GLA runs as a pure RNN with a fixed-size hidden state (e.g. 256 key-dim × 256 value-dim) — no KV cache growing with context length, critical for long-document and streaming LLM deployment.
- **Competitive perplexity**: GLA at ~1.3B params matches or beats Mamba and RetNet at equivalent token budgets (e.g. ~300B tokens on The Pile), suggesting data-dependent decay is the key missing ingredient in earlier linear-attention designs.
- **Hardware-friendly**: The chunkwise CUDA kernel never materializes the full N×N attention matrix; memory is O(N·d) not O(N²), enabling 8K+ context without flash-attention tricks.

## Key Formula or Mechanism
Recurrent update at step t (hidden state `S ∈ R^{d_k × d_v}`):

```
S_t = diag(g_t) @ S_{t-1} + k_tᵀ @ v_t
o_t = q_t @ S_t

where  g_t = sigmoid(W_g · x_t)  ∈ (0,1)^{d_k}   # data-dependent decay, broadcast over d_v
```

Chunkwise-parallel form (chunk size L):
- **intra-chunk**: `O_chunk = tril(QKᵀ ⊙ Γ) V` — a masked, decay-weighted linear-attention block computed in parallel.
- **inter-chunk**: carry the hidden state `S` recurrently across chunks using the accumulated gate product `∏ g`.

## Where It Appears
- **GLA paper**: *"Gated Linear Attention Transformers with Hardware-Efficient Training"* — Yang et al., arXiv 2312.06635 (Dec 2023; ICML 2024).
- **[[HGRN2]]**: related gated recurrent design from the same lineage.
- **[[RetNet]]**: predecessor using a fixed exponential decay γ per head instead of learned, data-dependent gates.
- **[[Mamba]]**: parallel selective SSM whose data-dependent gating inspired GLA.
- **[[Transformers Are SSMs|Mamba-2]]**: converges on a related formulation (state-space duality), validating the design space.

## Related Concepts
[[Linear Attention]]
[[Mamba]]
[[RetNet]]
[[RWKV]]
[[State Space Model|State Space Model (SSM)]]
[[FlashAttention]]
