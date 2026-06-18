---
created: "2026-05-20"
title: "Local Attention (Sliding Window Attention)"
tags: [glossary, attention, efficiency, hybrid, griffin, long-context]
tldr: "Attention restricted to a sliding window of the last W tokens. O(W·T) cost instead of O(T²). Used in Griffin, Mistral, LongFormer as the complement to a global recurrence: exact short-range lookup at fixed cost."
aliases: [sliding window attention, local attention, windowed attention, SWA]
---

## TL;DR

Local attention restricts each token to attending only to the last $W$ tokens (e.g., $W = 1024$). Instead of the full $T \times T$ attention matrix, you compute $T$ independent $W \times W$ attention blocks. Cost: $O(W \cdot T)$ — linear in sequence length. Exact recall within the window; tokens older than $W$ are invisible.

## Intuition

Full attention is a complete graph: every token talks to every other. Local attention is a neighborhood graph: each token only talks to its nearest $W$ neighbors in sequence. This is natural for language: most syntactic structure is local (within a sentence), most semantic structure is mediated through local chains.

The cost savings are significant: for $T = 32768$ and $W = 1024$, local attention costs $32 \times$ less than full attention.

Where local attention fails: long-range dependencies that skip many tokens (coreference across paragraphs, distant key facts). This is why hybrid models like [[Griffin]] pair local attention with a global recurrence (the [[RG-LRU]]): the recurrence handles long-range context compression, while local attention handles exact short-range lookup.

[[FlashAttention-2]] can compute local attention efficiently by using a block-sparse mask — only compute attention for blocks within the window, skip the rest.

## Why It Matters

- **It makes hybrid architectures tractable.** A few local attention layers in an otherwise-recurrent model ([[Griffin]], [[Nemotron-3]]) provide exact local recall at bounded cost.
- **It's the inference-efficient alternative to full attention for long contexts.** Mistral-7B uses local attention (4K window) with a sliding cache, enabling efficient 32K+ inference.
- **It determines the model's exact recall window.** Everything older than $W$ tokens must be compressed through the recurrence; nothing is retrieved exactly.

## Where It Appears in This Wiki

- [[Griffin]] — uses local attention (W=1024) in 1 of every 3 blocks, complementing the RG-LRU recurrence
- [[Nemotron-3]] — sparse attention layers serve a similar role: exact local recall within the attention layers

## Key Formula or Pseudocode

```
Standard attention:   A_ij = softmax(q_i k_j / √d)   for all i, j   → O(T²)
Local (window W):     A_ij = softmax(q_i k_j / √d)   if |i-j| ≤ W   → O(W·T)

Causal (left-context only):
  attend to tokens j ∈ [i-W, i]     (last W tokens only)
```

## Related Concepts

[[Griffin]] · [[Sliding window attention]] · [[Flash Attention]] · [[KV Cache]] · [[Transformer]] · [[Exponential decay]]
