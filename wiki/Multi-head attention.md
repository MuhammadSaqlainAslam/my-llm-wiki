---
created: "2026-05-20"
title: "Multi-Head Attention (MHA)"
tags: [glossary, attention, transformer, architecture, foundational]
tldr: "The core operation in Transformers: run h independent attention heads in parallel, each with its own Q/K/V projections, then concatenate and linearly project. Each head can learn a different type of relationship (local syntax, long-range coreference, etc.)."
aliases: [MHA, multi-head attention, multi-headed attention]
---

## TL;DR

Multi-head attention runs $h$ attention computations in parallel, each on a lower-dimensional projection of the queries, keys, and values. The outputs are concatenated and linearly projected: $\text{MHA}(Q,K,V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O$ where $\text{head}_i = \text{Attention}(Q W_i^Q, K W_i^K, V W_i^V)$.

## Intuition

A single attention head computes one kind of relationship: "which tokens does each position attend to?" Different positions in a sentence need different kinds of attention — verbs might need to attend to their subjects (syntactic), pronouns to their antecedents (semantic), and local tokens to adjacent neighbors (local). Multi-head attention gives the model $h$ different attention patterns simultaneously.

The key efficiency: each head works in dimension $d/h$, so the total computation is the same as one head at dimension $d$. You get $h$ diverse relationship patterns for free.

After [[GQA]] and [[Multi-Query Attention]] (MQA) became standard, the full MHA is primarily used for smaller models or cases where KV cache cost isn't a concern.

## Why It Matters

- **It's the fundamental module of every Transformer.** All papers in this wiki build on or react to MHA.
- **Different heads genuinely specialize.** Empirical analysis shows head specialization for syntax, position, coreference, and more.
- **The KV cache grows as $h \times d/h \times T$ = $d \times T$ per layer.** This is the baseline cost that [[GQA]] and [[MQA]] reduce.

## Where It Appears in This Wiki

- [[Transformer]] — MHA is the core attention mechanism
- [[GQA]] — GQA is a cache-efficient generalization of MHA
- [[FlashAttention-2]] — introduces per-head parallelism across thread blocks

## Key Formula or Pseudocode

```
head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)
       = softmax(Q W_i^Q (K W_i^K)^T / √(d/h)) · (V W_i^V)

MHA(Q,K,V) = Concat(head_1, ..., head_h) W^O

Dimensions: W_i^Q, W_i^K, W_i^V ∈ R^{d × (d/h)},  W^O ∈ R^{d × d}
```

## Related Concepts

[[Transformer]] · [[GQA]] · [[Multi-Query Attention]] · [[Flash Attention]] · [[KV Cache]]
