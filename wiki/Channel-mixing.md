---
created: "2026-05-20"
title: "Channel-Mixing (RWKV)"
tags: [glossary, rwkv, ffn, architecture]
tldr: "The RWKV sublayer that handles within-position (channel-wise) transformation, analogous to a Transformer's FFN layer. Uses a gated activation with a time-shift for lightweight local context."
aliases: [channel mixing, RWKV channel-mixing, RWKV FFN]
---

## TL;DR

Every [[RWKV]] block has two sublayers: [[Time-mixing]] (cross-time interactions) and channel-mixing (within-position processing). Channel-mixing is a gated FFN-like layer: it mixes information across the model's hidden dimensions without any cross-time interaction. It uses the same [[Time shift]] trick as time-mixing — blending the current token's embedding with the previous token's — giving it a cheap one-step local context window.

## Intuition

If time-mixing is the "where to look in time" layer, channel-mixing is the "what to do with what you found" layer. Once the WKV mechanism has aggregated past context into a vector, channel-mixing applies a learned gated transformation to produce the final output for that position.

The channel-mixing formula:
```
r' = W_r' @ (μ' * x + (1 - μ') * x_prev)   # gating
k' = W_k' @ (μ' * x + (1 - μ') * x_prev)   # key
output = sigmoid(r') * (W_v' @ max(k', 0)²) # gated quadratic activation
```

The `max(k', 0)²` (squared ReLU) is a simple nonlinearity that avoids computing a full softmax or gelu. The gate `sigmoid(r')` mirrors the gating pattern from [[LSTM]].

## Why It Matters

- **It makes RWKV's block structure match Transformer's.** Time-mixing ≈ attention; channel-mixing ≈ FFN. The two-sublayer structure is familiar and easy to implement.
- **The time-shift in channel-mixing gives one-step local context for free.** A tiny fraction of the previous position's embedding bleeds into the current position's FFN — a cheap trick that smooths the representation.

## Where It Appears in This Wiki

- [[RWKV]] — channel-mixing is the second sublayer in every RWKV block

## Related Concepts

[[RWKV]] · [[Time-mixing]] · [[Time shift]] · [[Receptance]] · [[Transformer]]
