---
created: "2026-05-20"
title: "Exponential Decay (in Linear RNNs)"
tags: [glossary, rnn, rwkv, retnet, linear-attention, recurrence]
tldr: "The mechanism by which linear RNNs (RWKV, RetNet, Griffin) discount older tokens: weight at distance d is γ^d or e^{-dw}. Older tokens contribute exponentially less. The decay rate is a key architectural hyperparameter controlling memory horizon."
aliases: [positional decay, exponential forgetting, decay factor]
---

## TL;DR

In linear RNN-style models ([[RWKV]], [[RetNet]], [[Griffin]]), the influence of token $k$ on output at position $n$ decays exponentially with distance: weight $\propto \gamma^{n-k}$ (RetNet) or $e^{-(n-1-k)w}$ (RWKV). Older tokens contribute exponentially less. This is the architectural choice that replaces the full softmax attention matrix with a computable recurrence — but at the cost of not being able to recall arbitrarily distant tokens.

## Intuition

Imagine you're trying to remember a conversation. Attention is perfect recall — every word you've heard stays equally accessible. Exponential decay is more like human memory — recent words are clear, words from an hour ago are faint, and last week's conversation is almost gone.

For most language tasks, this is fine: the most relevant context is usually recent. The exponential decay lets the model run in $O(1)$ per step (just update the running sum) without needing to store all past tokens. The tradeoff shows up on "needle in a haystack" tasks — finding one specific fact from 10,000 tokens ago — where exponential decay loses to full attention.

Key parameters:
- **RWKV**: per-channel decay $w \geq 0$, learned; weight $= e^{-w \cdot d}$
- **RetNet**: per-head $\gamma \in (0, 1)$, set as $\gamma_h = 1 - 2^{-5-h}$; weight $= \gamma^d$
- **Griffin RG-LRU**: per-step input-dependent $a_t \in (0, 1)$; weight varies by content

Making the decay input-dependent (Griffin, Mamba) is the key upgrade that closes the gap with attention on content-sensitive recall tasks.

## Why It Matters

- **It enables O(1) per-step inference.** With exponential decay, the recurrent state has bounded magnitude and doesn't grow with sequence length.
- **It's the fundamental tradeoff in linear RNNs.** Exact long-range recall requires keeping all past tokens (attention's KV cache); exponential decay trades recall precision for efficiency.
- **The decay rate controls the model's effective context window.** Small $w$ (RWKV) or large $\gamma$ (RetNet): long memory. Large $w$ or small $\gamma$: short memory.

## Where It Appears in This Wiki

- [[RWKV]] — per-channel decay $w$ in the WKV mechanism
- [[RetNet]] — per-head $\gamma$ in the retention mechanism
- [[Griffin]] — input-dependent $a_t$ in the RG-LRU

## Related Concepts

[[RWKV]] · [[RetNet]] · [[Griffin]] · [[Retention mechanism]] · [[Linear attention]] · [[Mamba]]
