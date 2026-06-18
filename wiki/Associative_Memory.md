---
created: "2026-06-03"
title: "Associative Memory"
tags: [glossary, memory, attention]
tldr: "A content-addressable storage mechanism where patterns are retrieved by similarity to a query, not by index — transformers implement this with ~64-dim keys matching against stored key-value pairs."
---

## TL;DR
Associative memory stores and retrieves information by pattern similarity, and modern transformers are mathematically equivalent to Hopfield networks doing exactly this at each attention layer.

## Intuition
Classic Hopfield networks store memories as energy minima — you give a noisy partial pattern and the network "snaps" to the nearest stored memory. The key insight (Ramsauer et al., 2020) is that the softmax attention operation in transformers is performing *exactly* this retrieval: queries are probes, keys are stored patterns, and the update rule is a single Hopfield energy minimization step. Modern continuous Hopfield networks can store exponentially many patterns (~2^(d/2) for dimension d), compared to the linear ~0.14N limit of classical binary Hopfield nets.

This reframing makes the KV cache literal: it's the memory bank. Each attention head maintains ~64-128 dimensional key vectors as addresses, and retrieval is a soft, differentiable nearest-neighbor lookup over all stored positions. The "memory capacity" of a layer scales with head dimension and sharpness of the softmax temperature.

## Why It Matters
- **KV Cache is explicit associative memory**: growing the context window from 4K → 128K tokens means storing 32× more key-value memories, directly explaining inference memory costs.
- **Attention heads specialize**: individual heads demonstrably learn to act as distinct associative stores (induction heads, copy heads), each retrieving different content types from the same memory bank.
- **MoE extends this**: routing in [[Mixture-of-Experts|MoE]] models can be viewed as associative lookup over expert "memories," selecting specialized computation by input similarity rather than position.

## Related Concepts
[[Attention]] [[KV Cache]] [[Transformer]] [[GQA]] [[FlashAttention]]
