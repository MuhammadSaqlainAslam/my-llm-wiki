---
title: "PonderNet"
aliases: ["PonderNet"]
year: 2021
tags: [adaptive-computation, halting, bayesian, inference, stub]
tldr: "Replaces the geometric halting approximation of Adaptive Computation Time with a proper probabilistic framework: the network emits a per-step 'halt' probability, the output is a mixture weighted by the probability of halting at each step, and the halting distribution is trained with a KL regularizer toward a geometric prior."
---

## TL;DR
ACT approximates a discrete halting decision with a soft geometric sum — PonderNet makes this exact by treating it as a proper probabilistic model. At each step `n`, the network outputs a candidate answer `ŷ_n` and a halting probability `λ_n`. The final prediction is `Σ p(N=n) · ŷ_n`, where `p(N=n)` is the probability of halting at step `n`. A KL regularizer penalizes divergence from a geometric(β) prior, replacing ACT's ad-hoc ponder cost.

## Intuition
ACT says "stop when you've accumulated enough halting probability." PonderNet says "the true answer is the expected output over all possible stopping points." This is cleaner theoretically and avoids the threshold-as-hyperparameter problem, replacing it with the geometric prior's β (a cleaner single knob controlling average depth).

## Why It Matters
- More principled probabilistic treatment of adaptive halting than [[Adaptive Computation Time|ACT]]
- Outperforms ACT on parity tasks and other structured reasoning benchmarks
- Part of the lineage leading to [[Fixed-Point Reasoners Stable and Adaptive Deep Looped Transformers|FPRM]]'s convergence-based halting

## See Also
[[Adaptive Computation Time]] · [[Universal Transformer]] · [[Deep Equilibrium Models]] · [[Fixed-Point Reasoners Stable and Adaptive Deep Looped Transformers]]
