---
title: "Deep Equilibrium Models"
aliases: ["DEQ", "deep equilibrium"]
year: 2019
tags: [implicit-depth, fixed-point, equilibrium, transformer, inference, stub]
tldr: "Replace explicit layer stacking with a single layer solved to its fixed point — the forward pass finds z* = f(z*; x) via a root-finding solver (e.g. Broyden's method), and the backward pass differentiates through the fixed-point equation without storing intermediate activations."
---

## TL;DR
Instead of 48 distinct transformer blocks each mapping `z_i → z_{i+1}`, DEQ uses *one* block `f_θ` and finds `z*` such that `f_θ(z*; x) = z*`. This is solved iteratively (Newton/Broyden/Anderson acceleration), and because the fixed-point equation is available analytically, the gradient with respect to θ is a one-step solve — O(1) memory regardless of "effective depth."

## Intuition
Deep networks are expensive both in parameters (one set per layer) and in memory (activations stored for backprop). DEQ asks: what if we threw away the layer index entirely and just looked for the steady state? Once `z` stops changing, it has "solved" the network's forward pass. The backward pass is cheap because the implicit function theorem gives the gradient without unrolling the solver.

## Why It Matters
- O(1) memory for backprop vs. O(depth) for explicit deep networks
- Motivates the "fixed-point as convergence criterion" idea in [[Fixed-Point Reasoners Stable and Adaptive Deep Looped Transformers|FPRM]]
- Connects recurrent/looped architectures to the implicit-layer literature
- Extended to multiscale DEQ, DEQ-based MoE, and sequence models

## Limitations
- Convergence of the root-finding solver is not guaranteed for arbitrary learned functions
- Wall-clock training can be slow when the solver requires many iterations
- Less suitable for tasks where the "correct" answer is path-dependent (order matters)

## See Also
[[Adaptive Computation Time]] · [[Universal Transformer]] · [[Fixed-Point Reasoners Stable and Adaptive Deep Looped Transformers]] · [[Neural ODE]] · [[Looped World Models]]
