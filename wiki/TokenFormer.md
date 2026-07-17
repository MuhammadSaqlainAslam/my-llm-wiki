---
created: "2026-07-16"
title: "TokenFormer: Rethinking Transformer Scaling with Tokenized Model Parameters"
authors: "Haiyang Wang, Yue Fan, Muhammad Ferjad Naeem, Yongqin Xian, Jan Eric Lenssen, Liwei Wang, Federico Tombari, Bernt Schiele"
year: "2024"
arxiv: "2410.23168"
tags: [architecture, scaling, transformer, efficiency]
tldr: "Replaces all linear projections in a Transformer with a token-parameter attention mechanism — treating model parameters themselves as tokens — enabling models to be incrementally scaled by adding new parameter tokens without retraining from scratch. ICLR 2025 Spotlight."
citation_count: 0
---

## TL;DR

TokenFormer replaces the fixed linear projections (Q/K/V matrices and FFN weights) in a standard [[Transformer]] with a token-parameter attention mechanism: a set of learnable "parameter tokens" that input tokens attend to, producing the equivalent of a matrix-vector product but via attention. The key benefit is incremental scalability: adding capacity means adding new parameter tokens without disturbing existing ones, so a smaller pretrained model can be grown into a larger one without retraining from scratch. ICLR 2025 Spotlight.

## The Idea

In a standard Transformer, the Q/K/V projections and FFN layers are fixed-size weight matrices — scaling the model up means training a larger matrix from scratch, with no principled way to grow an existing trained matrix.

TokenFormer parameterizes each projection as attention between input tokens and a set of learned "parameter tokens": output = Attn(X, P_key, P_value). This is mathematically equivalent to a linear projection when the parameter tokens are fixed, but because they're tokens rather than a fixed matrix, new ones can be appended to the existing set without changing the original parameters. Growing the model means adding more parameter tokens — the existing ones are unchanged, so the model inherits its prior capabilities immediately and only needs to train the newly added parameters.

## Why It Matters

- Addresses a real practical problem: the standard paradigm requires retraining from scratch whenever a larger model is needed, wasting all prior compute
- Token-parameter attention is a natural generalization — it collapses the distinction between "model weights" and "model inputs" into a unified attention framework
- Incrementally scaled models reported to match the performance of models trained at the larger size from scratch, at a fraction of the total compute cost
- ICLR 2025 Spotlight

## Limitations

- Token-parameter attention has somewhat higher inference cost than a standard linear projection of equivalent capacity, due to the attention overhead
- Validated at moderate scales (up to ~1.4B parameters) — whether the incremental-scaling advantage holds at frontier scale (70B+) is not yet established
- Requires architectural commitment from the start — an existing standard Transformer can't be converted to TokenFormer without retraining

## Related Concepts

[[Transformer]] · [[Scaling Laws]] · [[LoRA Low-Rank Adaptation of Large Language Models|LoRA]]
