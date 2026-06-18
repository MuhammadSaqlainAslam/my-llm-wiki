---
created: "2026-05-20"
title: "Time Shift (RWKV)"
tags: [glossary, rwkv, rnn, local-context, architecture]
tldr: "RWKV's simple trick of blending each token's embedding with the previous token's embedding via a learned interpolation weight, giving every layer a one-step local context window at negligible cost."
aliases: [token shift, time-shift mixing, temporal shift]
---

## TL;DR

In [[RWKV]], before computing the key, value, and receptance vectors for any sublayer, each input vector $x_t$ is blended with the previous position's input $x_{t-1}$: $\hat{x}_t = \mu \odot x_t + (1 - \mu) \odot x_{t-1}$, where $\mu$ is a per-channel learned weight. This is the "time shift" — a free one-step local sliding window that costs nothing at inference (you already have $x_{t-1}$ from the previous step).

## Intuition

The WKV mechanism in [[Time-mixing]] handles long-range context via the exponential decay. But immediate local context (the previous token) is cheap to access — you just need to remember it. The time shift blends a fraction of the previous embedding into the current one, giving the model local bigram-like information without any attention computation.

Different layers use different $\mu$ vectors — so earlier layers might lean heavily on the previous token (more local), while later layers use primarily the current token (more global). This per-layer, per-channel $\mu$ is trained by gradient descent.

At inference, this costs zero extra computation: $x_{t-1}$ is already in the recurrent state.

## Why It Matters

- **It's a near-free local context window.** Each RWKV layer gets one-step local context without any attention mechanism.
- **It improves expressivity without breaking the RNN property.** The time shift uses only one past token, so it doesn't require the KV cache to grow.
- **It's applied in both sublayers.** Both [[Time-mixing]] and [[Channel-mixing]] use the time shift, giving the entire block local context.

## Where It Appears in This Wiki

- [[RWKV]] — the time shift is applied before computing keys, values, and receptances in every block

## Related Concepts

[[RWKV]] · [[Time-mixing]] · [[Channel-mixing]] · [[Receptance]] · [[Exponential decay]]
