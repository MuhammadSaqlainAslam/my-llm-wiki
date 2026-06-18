---
created: "2026-05-20"
title: "RG-LRU (Real-Gated Linear Recurrent Unit)"
tags: [glossary, griffin, rnn, recurrence, gating, efficiency]
tldr: "Griffin's core recurrent layer: a diagonal linear recurrence with input-dependent decay, an output gate, and a norm-preserving update rule. Simpler than Mamba's selective SSM, more expressive than RWKV's fixed decay — the sweet spot for hardware-efficient gated recurrence."
aliases: [RG-LRU, Real-Gated Linear Recurrent Unit, recurrent gated LRU]
---

## TL;DR

The RG-LRU is [[Griffin]]'s replacement for attention. It's a diagonal linear recurrence:

$$h_t = a_t \odot h_{t-1} + \sqrt{1 - a_t^2} \odot (W_x x_t)$$

where $a_t = \sigma(W_a x_t)^2 \cdot \alpha$ is an input-dependent decay. Output: $y_t = h_t \odot \sigma(W_g x_t)$. Everything is elementwise — no matrix multiply in the recurrence — making it efficient on both GPUs and TPUs.

## Intuition

Three design choices make RG-LRU tick:

1. **Input-dependent decay** ($a_t$): Unlike [[RWKV]]'s fixed channel-wise decay or [[RetNet]]'s fixed per-head $\gamma$, RG-LRU's decay depends on the current input. If the model sees a surprising token, it can learn to "reset" the state ($a_t \to 0$); if the context should persist, $a_t \to 1$. This is what [[Griffin]] means by "gated": the gate controls the forgetting rate.

2. **Norm-preserving update** ($\sqrt{1-a_t^2}$ factor): If $a_t^2 + (1-a_t^2) = 1$ at each step and the input is also unit norm, then $\|h_t\|^2 \leq \|h_{t-1}\|^2$ — the state can't explode. This keeps gradients healthy without needing careful initialization.

3. **Output gate** ($\sigma(W_g x_t)$): After the recurrence produces $h_t$, an independent gate filters the output. This allows the model to "keep a secret" — maintain state without always broadcasting it.

Compared to [[Mamba]]: RG-LRU is simpler (no $B$, $C$, $\Delta$ per token — just one decay parameter per dimension), but less expressive. In practice [[Griffin]] (at 7B+) closes the gap with [[LLaMA 2]] using this simpler mechanism.

## Why It Matters

- **It's the minimum viable gated recurrence at scale.** Griffin-14B matches LLaMA-2 with 7× fewer training tokens, proving the simplicity doesn't hurt quality.
- **Diagonal operations are hardware-optimal.** The entire recurrence runs as elementwise multiplications — no matrix multiply in the temporal direction — maximizing throughput on tensor cores.
- **The norm-preserving update is a training stability win.** Models with RG-LRU train stably from random initialization without the HiPPO initialization that S4 requires.

## Where It Appears in This Wiki

- [[Griffin]] — RG-LRU is the temporal mixing layer in Griffin and Hawk
- [[Diagonal recurrence]] — RG-LRU is a specific diagonal linear recurrence with input-gated decay

## Key Formula or Pseudocode

```
# RG-LRU forward pass (one step)
a_t = sigmoid(W_a @ x_t)² * α      # input-dependent decay ∈ (0, 1)
h_t = a_t * h_{t-1} + sqrt(1 - a_t²) * (W_x @ x_t)  # gated state update
y_t = h_t * sigmoid(W_g @ x_t)     # output gate

α: fixed per-channel scalar initialized near 1 (learned initialization)
W_a, W_x, W_g: learned linear projections
```

## Related Concepts

[[Griffin]] · [[Diagonal recurrence]] · [[Mamba]] · [[RWKV]] · [[Exponential decay]] · [[Local attention]]
