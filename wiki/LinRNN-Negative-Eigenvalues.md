---
created: "2026-07-16"
title: "Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues"
authors: "Riccardo Grazzi, Julien Siems, Arber Zela, Jörg K. H. Franke, Frank Hutter, Massimiliano Pontil"
year: "2024"
arxiv: "2411.12537"
tags: [ssm, linear-rnn, architecture, theory, foundational]
tldr: "Proves that linear RNNs constrained to non-negative eigenvalues (Mamba, DeltaNet) cannot solve parity — a fundamental state-tracking limitation — and that allowing negative eigenvalues provably fixes it, with empirical validation up to 1.3B parameters. ICLR 2025 Oral."
citation_count: 91
---

## TL;DR

Proves a fundamental theoretical limitation of popular linear RNN architectures ([[Mamba]], [[DeltaNet]], [[RWKV]]) — they cannot solve the parity problem, a basic state-tracking task — because they constrain eigenvalues of the recurrence matrix to be non-negative. Allowing negative real eigenvalues provably removes this limitation, and the fix requires only a minimal change to the initialization/parameterization. ICLR 2025 Oral.

## The Problem

Modern linear RNNs like [[Mamba]], [[DeltaNet]], and [[RWKV]] achieve linear-time sequence processing by constraining their recurrence matrices' eigenvalues to a restricted non-negative range. This makes the recurrence stable and easy to train, but these architectures are provably incapable of solving the parity problem — determining whether the number of 1s in a binary sequence is odd or even — regardless of size or training procedure.

Parity is a canonical hard state-tracking task: solving it requires remembering a single bit of state across arbitrary context length, updating it at every step. The inability of standard linear RNNs to do this has implications for any task requiring precise long-horizon state tracking (code evaluation, modular arithmetic).

## The Idea

The impossibility result follows from the spectral structure of the recurrence matrix: a matrix with only non-negative eigenvalues can represent monotonically decaying or constant memory traces, but not the oscillatory dynamics parity-style state tracking requires. Extending the eigenvalue range to include negative real values (and, for non-diagonal recurrences like DeltaNet, non-triangular matrix structure) removes the restriction — a recurrence matrix with eigenvalue −1 oscillates between +1 and −1 at each step, exactly the behavior needed to track parity.

The paper proves this extends beyond diagonal architectures like Mamba to non-diagonal ones like DeltaNet, and further proves that LRNNs can learn any regular language when their state-transition matrices are products of identity-minus-outer-product matrices with eigenvalues in [−1, 1]. Critically, the fix requires no new architectural components — just a reparameterization allowing the existing eigenvalue-producing mechanism to output negative values. Models with this fix pretrain stably at 1.3B parameters with competitive language modeling performance and improved code/math task performance.

## Why It Matters

- A clean theoretical result — proves an impossibility rather than just observing empirically that something is hard
- Directly relevant to every linear RNN / SSM in this wiki that constrains eigenvalues to a bounded non-negative range
- The broader principle (oscillatory dynamics require non-positive-only spectra) is the same insight [[LinOSS]] arrives at independently via harmonic oscillators
- ICLR 2025 Oral

## Limitations

- The parity impossibility is proven for the idealized linear-recurrence setting — real architectures mixing linear RNN layers with nonlinear components (MLPs, attention) may partially circumvent the limitation in practice
- The fix can make training less stable without careful parameterization, though the paper shows this is manageable at scale
- Diagonal architectures (like Mamba) still can't solve harder state-tracking tasks (e.g. modular counting beyond 2) even with negative eigenvalues — the non-diagonal DeltaNet extension is needed for those

## Related Concepts

[[Mamba]] · [[DeltaNet]] · [[RWKV]] · [[LinOSS]] · [[Transformers Are SSMs]]
