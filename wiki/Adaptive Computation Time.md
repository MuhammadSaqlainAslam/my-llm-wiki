---
title: "Adaptive Computation Time"
aliases: ["ACT", "adaptive halting"]
year: 2016
tags: [recurrent, adaptive-computation, inference, halting, test-time-compute, stub]
tldr: "Trains a recurrent network to decide how many computation steps to spend on each input, halting early on easy inputs and running longer on hard ones — but the discrete halt is approximated by a soft geometric distribution, making it differentiable."
---

## TL;DR
Standard RNNs apply a fixed number of steps regardless of input complexity. ACT adds a *halting unit* — a scalar output from the cell that accumulates probability over steps. The network stops when the cumulative halt probability exceeds 1, with fractional credit given to the last partial step. A differentiable ponder cost (sum of halting probabilities) penalizes unnecessary compute.

## Intuition
Think of it as the network learning to raise its hand and say "I'm done thinking about this one." Easy inputs raise their hand after a couple steps; hard inputs keep going. The clever trick is making the discrete "stop" signal differentiable by spreading it over a soft probability distribution — the model emits a fraction of its "I'm done" budget at each step, and training can backprop through the total ponder time.

## Why It Matters
- First trainable mechanism for variable-depth computation in sequence models
- Inspired a line of work on adaptive inference: [[Universal Transformer]], [[PonderNet]], looped Transformers
- Foundational comparison point for [[Fixed-Point Reasoners Stable and Adaptive Deep Looped Transformers|FPRM]] (which replaces ACT with convergence detection)

## Limitations
- Ponder cost is a heuristic — tuning the regularization coefficient is finicky
- In practice, ACT often fails to allocate meaningfully more steps to harder inputs
- Discrete halting is still approximate; the soft weighting introduces a small approximation error

## See Also
[[Universal Transformer]] · [[Deep Equilibrium Models]] · [[Fixed-Point Reasoners Stable and Adaptive Deep Looped Transformers]] · [[Test-Time Compute Scaling]] · [[Looped World Models]]
