---
created: "2026-05-20"
title: "Diagonal Recurrence"
tags: [glossary, griffin, rnn, linear-rnn, efficiency, hardware]
tldr: "A linear recurrence where the state transition matrix is diagonal: h_t = a_t ⊙ h_{t-1} + b_t, so each dimension of the state evolves independently. Enables full parallelization within the state dimension and efficient parallel scan computation."
aliases: [diagonal linear recurrence, diagonal RNN, diagonal state transition]
---

## TL;DR

A diagonal recurrence processes each hidden state dimension independently: $h_t^{(i)} = a_t^{(i)} \cdot h_{t-1}^{(i)} + b_t^{(i)}$. The state transition is elementwise (diagonal matrix), not a full $d \times d$ matrix multiply. This means: (1) each dimension can be run in parallel on GPU, (2) the parallel scan algorithm applies, enabling training in $O(L \log L)$, and (3) inference is $O(d)$ per step with no matrix multiply.

## Intuition

Compare to a full RNN (like the original LSTM): $h_t = W h_{t-1} + U x_t$. The $d \times d$ matrix $W$ couples all dimensions together — you must compute the full matrix multiply sequentially. A diagonal recurrence removes this coupling: dimension 1 only interacts with its own past, dimension 2 with its own past, etc.

This makes diagonal recurrences hardware-friendly: all $d$ dimensions run independently in parallel. The only sequential dependency is within one dimension across time — and that's where the parallel scan algorithm helps: instead of computing $h_1, h_2, \ldots, h_T$ sequentially, you can compute all $h_t$ in $O(\log T)$ parallel steps using prefix sums.

[[Griffin]]'s [[RG-LRU]] is a diagonal recurrence with an input-dependent diagonal: $a_t = f(x_t)$, so the decay rate changes per token. [[Mamba]]'s selective SSM is similarly a diagonal recurrence (diagonal $\bar{A}$) with input-dependent parameters.

## Why It Matters

- **It's what makes RG-LRU hardware-efficient.** Full matrix RNNs require $O(d^2)$ compute and are GPU-inefficient for sequential dependencies; diagonal is $O(d)$ and runs as fast as an elementwise op.
- **It enables the parallel scan training algorithm.** Any diagonal linear recurrence can be trained in $O(L \log L)$ or $O(L)$ using work-efficient parallel prefix scans.
- **It connects Griffin, Mamba, and RWKV architecturally.** All three use diagonal (or block-diagonal) recurrences; the distinctions are in how the diagonal values are parameterized.

## Where It Appears in This Wiki

- [[Griffin]] — the RG-LRU layer is a diagonal recurrence with input-dependent decay
- [[Mamba]] — uses diagonal $\bar{A}$ in the selective SSM
- [[RWKV]] — per-channel exponential decay is equivalent to a diagonal recurrence with fixed diagonal

## Related Concepts

[[Griffin]] · [[RG-LRU]] · [[Mamba]] · [[RWKV]] · [[Exponential decay]] · [[Hardware-Aware Scan]]
