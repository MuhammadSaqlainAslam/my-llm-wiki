---
title: "Neural ODE"
aliases: ["Neural ODE", "Neural Ordinary Differential Equation", "NeuralODE"]
year: 2018
tags: [continuous-depth, differential-equations, implicit-depth, residual-networks, stub]
tldr: "Replace the discrete layer index in a residual network with a continuous time variable and parameterize the derivative dz/dt = f_θ(z, t) — the forward pass becomes an ODE solve, giving continuous-depth networks with O(1) memory backprop via the adjoint method."
---

## TL;DR
Residual networks compute `z_{l+1} = z_l + f_θ(z_l)` — a discrete Euler step. Neural ODEs generalize this to a continuous ODE `dz/dt = f_θ(z, t)` and solve it with an adaptive ODE solver (e.g. Dormand-Prince). The "depth" is now a continuous variable controlled by the solver's step size. Gradients flow via the adjoint method (solving a second ODE backward in time), so memory is constant regardless of how many solver steps are taken.

## Intuition
A residual network is just a discrete approximation of a continuous dynamical system. Neural ODEs say: why approximate? Just model the dynamics directly as an ODE and use a real numerical solver. You get adaptive depth for free (solvers take smaller steps near stiff regions) and O(1) memory for backprop as a bonus.

## Why It Matters
- Foundational connection between deep learning and dynamical systems theory
- Motivates the spectral stability constraints in [[Looped World Models|LoopWM]]
- Predecessor to [[Deep Equilibrium Models|DEQ]] (the fixed-point formulation)
- Latent ODEs extend this to time-series modeling with irregular observation times

## See Also
[[Deep Equilibrium Models]] · [[Looped World Models]] · [[Adaptive Computation Time]] · [[Recurrent State Space Model]] · [[S4]]
