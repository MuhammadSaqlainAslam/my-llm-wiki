---
created: "2026-05-20"
title: "Recurrent State"
tags: [glossary, rnn, retnet, rwkv, ssm, inference, memory]
tldr: "The fixed-size hidden state h_t maintained by RNN-style models (including RetNet, RWKV, Griffin, Mamba) that summarizes all past inputs. Unlike the KV cache (which grows with T), the recurrent state has constant size O(d²) or O(Nd) per layer regardless of sequence length."
aliases: [recurrent state, hidden state, recurrent memory, RNN state]
---

## TL;DR

The recurrent state is the compressed summary of all past inputs that an RNN-style model carries forward. At each step, it's updated using the current input and the previous state. In [[RetNet]], the state $S_t \in \mathbb{R}^{d \times d}$ is updated as $S_t = \gamma S_{t-1} + k_t^\top v_t$; in [[Mamba]], as $h_t = \bar{A} h_{t-1} + \bar{B} x_t$; in [[RWKV]], as running numerator and denominator vectors.

The crucial property: the recurrent state has **constant size** — $O(d^2)$ or $O(N \times d)$ per layer — regardless of sequence length $T$. Compare to the [[KV Cache]] in Transformers, which grows as $O(T \times d)$ per layer. At $T = 32768$, the KV cache is 32,768× larger than the recurrent state.

## Intuition

Think of the recurrent state as a lossy summary: it compresses everything the model has seen into a fixed-size matrix. The compression is lossy — not everything from the distant past can be perfectly recovered — but for most practical language tasks, the loss is acceptable. The reward is $O(1)$ per-step inference with constant memory.

For [[RetNet]]: the state is a $d \times d$ matrix per head, updated with a rank-1 outer product $k_t^\top v_t$ at each step. This is the "memory matrix" that accumulates a weighted sum of all past key-value associations. Reading from it costs $d^2$ FLOPs (one matrix-vector product).

## Why It Matters

- **It's the fundamental reason linear RNNs are memory-efficient at inference.** No growing cache, no paging, no eviction policy needed.
- **The $d \times d$ size is a design choice.** For $d = 256$ per head, it's 64K floats — tiny. For $d = 4096$ (per-layer aggregate), it's 16M floats — still much smaller than a 32K-token KV cache.
- **It enables streaming inference.** Process tokens one-by-one, maintaining only the recurrent state — no materialization of past tokens.

## Where It Appears in This Wiki

- [[RetNet]] — $S_t \in \mathbb{R}^{d \times d}$ per head; updated at each step
- [[RWKV]] — running numerator/denominator vectors
- [[Mamba]] — $h_t \in \mathbb{R}^{N}$ per channel
- [[Griffin]] — $h_t \in \mathbb{R}^{d}$ per RG-LRU layer

## Related Concepts

[[RetNet]] · [[RWKV]] · [[Mamba]] · [[Griffin]] · [[KV Cache]] · [[State Space Model]]
