---
created: "2026-07-16"
title: "Oscillatory State-Space Models"
authors: "T. Konstantin Rusch, Daniela Rus"
year: "2024"
arxiv: "2410.03943"
tags: [ssm, architecture, theory, foundational, oscillatory]
tldr: "LinOSS — an SSM built on discretized harmonic oscillators, with a universality proof showing it can approximate any continuous-time dynamical system. ICLR 2025 Oral. Competitive with Mamba on long-range benchmarks with provably richer dynamics than standard linear RNNs."
citation_count: 37
---

## TL;DR

LinOSS (Linear Oscillatory State-Space model) replaces the standard decaying-memory recurrence in SSMs with discretized harmonic oscillators — the mathematical model of a mass on a spring. This gives the recurrence complex eigenvalues by construction, letting LinOSS track state across arbitrary horizons in ways that [[Mamba]] and standard linear RNNs provably cannot (see [[LinRNN-Negative-Eigenvalues|Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues]]). A universality theorem proves LinOSS can approximate any continuous-time dynamical system. ICLR 2025 Oral.

## The Idea

A harmonic oscillator maintains two coupled state variables (position and velocity) that evolve as:

$$\ddot{x} + \omega^2 x = u(t)$$

Discretizing this with a stable numerical scheme gives a recurrence matrix with eigenvalues on (or near) the unit circle — complex numbers that oscillate indefinitely rather than decaying to zero. This is the minimal recurrence structure needed to represent persistent, non-decaying memory and oscillatory dynamics.

LinOSS parameterizes each SSM layer as a bank of harmonic oscillators with learnable frequencies, combined with learnable mixing weights. The discretization scheme is designed to guarantee stability while preserving the complex eigenvalue structure that gives the model its expressive power.

## Why It Matters

- Independently arrives at the same underlying fix identified theoretically in [[LinRNN-Negative-Eigenvalues|Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues]] — oscillatory (non-positive-only) spectra are needed for state tracking — but implements it via a physically-motivated harmonic oscillator structure rather than a direct reparameterization of an existing architecture
- Universality theorem: LinOSS can approximate any continuous-time dynamical system to arbitrary precision, a stronger theoretical guarantee than most competing linear RNN/SSM models offer
- Bio-inspired: harmonic oscillators appear throughout neuroscience, mechanics, and signal processing, drawing on a well-developed body of theory
- ICLR 2025 Oral

## Limitations

- Complex/oscillatory eigenvalues require careful discretization to avoid numerical instability during training, though the paper's scheme largely addresses this
- Evaluated primarily on long-range sequence benchmarks; performance at frontier LLM scale (multi-billion parameters, trillion-token training) is not yet established

## Related Concepts

[[S4]] · [[Mamba]] · [[RWKV]] · [[LinRNN-Negative-Eigenvalues|Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues]]
