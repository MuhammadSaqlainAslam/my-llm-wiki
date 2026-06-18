---
created: "2026-05-20"
title: "Receptance (RWKV)"
tags: [glossary, rwkv, gating, rnn, architecture]
tldr: "RWKV's output gating vector r_t = sigmoid(W_r x̂_t) that controls how much of the WKV computation reaches the output. Analogous to LSTM's output gate — determines when the model 'speaks' vs. suppresses the context accumulation."
aliases: [RWKV receptance, receptance gate]
---

## TL;DR

In [[RWKV]], after computing the WKV (Weighted Key Value) accumulation, the output is gated by the receptance vector: $\text{output}_t = W_o \cdot (\sigma(r_t) \odot \text{WKV}_t)$. The receptance $r_t$ is a learned linear projection of the time-shifted input, passed through a sigmoid. It's the mechanism that decides how much of the historical context (in WKV) to pass forward vs. suppress.

## Intuition

Think of the WKV mechanism as a memory readout — it produces a weighted average of past values. Receptance is the gate that decides whether to "trust" that memory and output it, or to suppress it. A receptance near 1 means "yes, this context is relevant"; near 0 means "ignore the memory, output nothing."

This mirrors LSTM's output gate, which controls when the cell state contributes to the hidden state. The difference: RWKV's receptance gates the WKV accumulation directly, while LSTM's output gate gates the full cell state.

The RWKV paper names the four key components of its mechanism: **R**eceptance, **W**eights (decay), **K**ey, **V**alue — hence "RWKV."

## Why It Matters

- **It provides the model control over when to "activate" historical context.** Without the receptance gate, every position would output the same WKV accumulation, losing the ability to selectively respond.
- **It's what gives RWKV its gating expressivity** despite replacing dot-product attention with a fixed exponential decay.

## Where It Appears in This Wiki

- [[RWKV]] — receptance is one of the four core vectors in RWKV's time-mixing sublayer

## Related Concepts

[[RWKV]] · [[Time-mixing]] · [[Channel-mixing]] · [[Exponential decay]] · [[LSTM]]
