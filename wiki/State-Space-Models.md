---
title: "State-Space Models"
authors: ""
year: ""
arxiv: ""
tags: [glossary, ssm, architecture, foundations]
tldr: "A sequence modeling formulation that processes inputs through a hidden state evolving via linear recurrence, enabling linear-time sequence processing as an alternative to attention"
citation_count: 0
---

# State-Space Models

## TL;DR

State-space models (SSMs) process sequences by maintaining a hidden state that evolves according to a linear recurrence relation, rather than computing pairwise attention between every pair of tokens. This gives linear-time complexity in sequence length, in contrast to the quadratic complexity of standard self-attention.

---

## Intuition

An SSM updates a hidden state $h_t$ at each step using a function of the previous hidden state and the current input — conceptually similar to an RNN, but parameterized and trained in a way that allows efficient parallel computation during training (unlike traditional RNNs, which are inherently sequential).

The core recurrence is:

$$h_t = \bar{A} h_{t-1} + \bar{B} x_t, \quad y_t = C h_t$$

where $\bar{A}$ and $\bar{B}$ are discrete-time parameters derived from continuous-time matrices via a discretization step (e.g., zero-order hold).

[[S4]] established the modern foundation for making this formulation both expressive and trainable at scale, by showing that the right initialization of $A$ (via the HiPPO polynomial projection framework) enables long-range memory and efficient computation via convolutions. [[Mamba]] introduced input-dependent ("selective") parameters — $B$, $C$, and $\Delta$ all depend on the current input — that let the model decide what to remember or forget per token, closing much of the quality gap with attention-based Transformers.

---

## Why It Matters

- **Linear-time sequence processing** vs. attention's quadratic cost — critical for very long contexts
- **O(1) state size at inference**: the recurrent state is constant-size regardless of sequence length, enabling fixed-memory decoding
- Forms the architectural basis for a growing family of hybrid models ([[Nemotron_3_Super|Nemotron 3 Super]], Zamba, Samba, Hymba) that interleave SSM layers with attention layers to balance efficiency and quality
- Active area of architecture research distinct from, but increasingly combined with, MoE scaling techniques like [[LatentMoE]]

---

## Related Concepts

*Core papers: [[S4]] · [[Mamba]] · [[Transformers Are SSMs]]*

*Other recurrent architectures: [[RWKV]] · [[RetNet]] · [[xLSTM]] · [[Griffin]]*

*Applied in: [[Nemotron-3]] · [[Nemotron_3_Super|Nemotron 3 Super]]*
