---
created: "2026-05-28"
title: "Gated DeltaNet"
tags: [glossary, ssm, linear-attention, memory, recurrent-models]
tldr: "A recurrent sequence model that combines DeltaNet's associative memory updates with a forget gate, enabling selective memory erasure and achieving competitive performance with Transformers on recall-intensive tasks."
---

## TL;DR
Gated DeltaNet extends DeltaNet by adding a data-dependent scalar gate that controls how much old memory to retain, yielding a hybrid between linear attention and gated RNNs like Mamba.

## Intuition
DeltaNet's core idea is the **delta rule**: when you want to store a new key-value association, first *subtract out* the old value at that key, then add the new one — like a targeted memory overwrite. The recurrence is `S_t = S_{t-1}(I - β_t k_t k_t^T) + β_t v_t k_t^T`, where the matrix `S` is the memory state. This is already more expressive than standard linear attention, which just accumulates without erasing.

Gated DeltaNet adds a **global forget gate** `α_t ∈ (0,1)` that scales the entire memory matrix before the delta update: `S_t = α_t · S_{t-1}(I - β_t k_t k_t^T) + β_t v_t k_t^T`. This one scalar per timestep lets the model flush stale context entirely — critical for tasks like multi-document QA or anything with clearly delimited episodes. In practice, models in the ~1.3B range match or exceed similarly-sized Transformers on recall benchmarks (MQAR, phonebook lookup) while remaining linear in sequence length at inference.

## Why It Matters
- **Recall-intensive tasks**: The forget gate solves the "memory pollution" problem in plain linear attention — old associations can be aggressively decayed, letting new ones dominate without interference.
- **Hardware-efficient inference**: Fixed-size matrix hidden state `S ∈ R^{d_k × d_v}` means O(1) memory per decoding step; no KV cache growth — critical for long-context or on-device inference.
- **Hybrid architecture synergy**: Gated DeltaNet layers can be interleaved with full attention (e.g., every 4th layer) to get associative memory + exact recall, matching Transformer quality at a fraction of the compute.

## Key Formula or Mechanism
# Gated DeltaNet recurrence
S_t = α_t · [S_{t-1} - β_t (S_{t-1} k_t) k_t^T] + β_t v_t k_t^T
y_t = S_t q_t

# where:
#   α_t  ∈ (0,1)  — scalar forget gate (data-dependent, learned)
#   β_t  ∈ (0,1)  — per-step learning rate (data-dependent)
#   k_t, q_t, v_t ∈ R^d — keys, queries, values
#   S_t  ∈ R^{d_k × d_v} — matrix-valued hidden state

The term `S_{t-1} k_t` retrieves the currently stored value at key `k_t`; subtracting it before writing is the delta rule — write the *residual*, not the raw value.

## Where It Appears
- **Yang et al. (2025)** — *"Gated Delta Networks: Improving Mamba2 with Delta Rule"* (primary paper)
- **DeltaNet (Schlag et al., 2021)** — foundational delta rule for fast weight / linear attention
- **Mamba / Mamba2 (Gu & Dao, 2023–24)** — gating design inspiration; Gated DeltaNet targets same efficiency class
- **HGRN2, RetNet, GLA** — related gated linear recurrences this work benchmarks against

## Related Concepts
[[DeltaNet]]
[[Linear Attention]]
[[State Space Model|State Space Models]]
[[Mamba]]
[[Associative Memory]]
