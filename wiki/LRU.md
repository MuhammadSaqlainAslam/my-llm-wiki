---
created: "2026-05-20"
title: "LRU (Linear Recurrent Unit)"
tags: [glossary, rnn, recurrence, efficiency, complex-valued]
tldr: "A complex-valued diagonal linear recurrence (Orvieto et al. 2023) that uses eigenvalues on the unit disk for stability. The direct predecessor of Griffin's RG-LRU; proved that simple linear recurrences can match LSTM on language tasks when properly initialized."
aliases: [Linear Recurrent Unit, LRU, Orvieto LRU]
---

## TL;DR

The LRU (Orvieto et al., 2023) showed that a very simple linear recurrence — a diagonal complex-valued state transition with eigenvalues constrained to the unit disk — can match or beat LSTM on sequence tasks. It established the foundation that [[Griffin]]'s [[RG-LRU]] builds on: the LRU removes complex values and adds input-gating, making it real-valued and more expressive.

## Intuition

The LRU update: $h_t = \lambda \odot h_{t-1} + \exp(i\theta) \odot (B x_t)$, where $\lambda$ and $\theta$ are parameters with $|\lambda| \leq 1$ (stability constraint via parameterization as $|\lambda| = e^{-e^\nu}, \nu \in \mathbb{R}$). The complex values allow the state to oscillate while decaying — capturing periodic patterns that real-valued recurrences can miss.

The key finding: with the right initialization (eigenvalues near the unit circle), a plain linear recurrence without any nonlinearity in the state transition can model long-range dependencies as well as LSTMs. Nonlinearity is still present (in the output mapping and in subsequent layers), but the recurrence itself is linear.

[[Griffin]] replaces complex $\lambda$ with real input-dependent $a_t = \sigma(W_a x_t)^2 \cdot \alpha$ — simpler to implement, avoids complex arithmetic, adds content-based gating.

## Why It Matters

- **It proved linear recurrences are competitive.** Before LRU, the assumption was that nonlinear state updates (LSTM-style) were necessary for good performance. LRU falsified this.
- **It gave [[Griffin]] a theoretical foundation.** RG-LRU is explicitly named as an extension of LRU.
- **It simplifies the architecture design space.** If a linear recurrence works, you don't need the overhead of full LSTM-style gating in the state transition.

## Where It Appears in This Wiki

- [[Griffin]] — RG-LRU is the real-valued, input-gated extension of LRU

## Related Concepts

[[Griffin]] · [[RG-LRU]] · [[Diagonal recurrence]] · [[RWKV]] · [[Mamba]]
