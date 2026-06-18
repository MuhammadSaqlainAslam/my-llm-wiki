---
created: "2026-05-20"
title: "Discretization (SSMs)"
tags: [glossary, ssm, s4, mathematics, recurrence]
tldr: "Converting a continuous-time state space model (ODE x' = Ax + Bu) into a discrete recurrence (x_k = Ā x_{k-1} + B̄ u_k) suitable for processing tokenized sequences. The step size Δ is a learnable parameter that controls effective memory timescale."
aliases: [SSM discretization, ZOH discretization, bilinear transform, discrete SSM]
---

## TL;DR

State space models are naturally described as continuous-time ODEs, but sequences of text or audio tokens are discrete. Discretization is the conversion step. Given a step size $\Delta$, the standard Zero-Order Hold (ZOH) discretization converts the continuous matrices $(A, B)$ into discrete matrices $(\bar{A}, \bar{B})$ via matrix exponential. The resulting recurrence $x_k = \bar{A} x_{k-1} + \bar{B} u_k$ can then process token sequences.

## Intuition

Think of the continuous SSM as a pendulum with damping — a smooth dynamical system. You want to simulate it on a clock that ticks once per token. Discretization is choosing how to step the pendulum forward by one tick: take the exact solution of the ODE assuming the input $u$ is constant over each interval (ZOH rule).

The learnable step size $\Delta$ controls how much the state changes per token. A small $\Delta$ means slow dynamics — the model responds gently to each token, giving it long-range memory. A large $\Delta$ means fast dynamics — the model quickly forgets old tokens. In [[Mamba]], $\Delta$ becomes input-dependent ($\Delta_t = f(x_t)$), allowing the model to selectively control how much each token "resets" the state.

ZOH formula:
$$
\bar{A} = e^{\Delta A}, \quad \bar{B} = (A)^{-1}(e^{\Delta A} - I) B
$$

For the [[S4]] DPLR structure, $e^{\Delta A}$ is computable efficiently. For diagonal $A$ (Mamba), it reduces to elementwise exponentials.

## Why It Matters

- **It's the bridge between SSM theory and sequence models.** All the HiPPO theory is about continuous systems; discretization makes it usable for language modeling.
- **The step size Δ controls memory timescale.** This is a key design choice — and making it input-dependent (Mamba's selectivity) is what separates Mamba from S4.
- **Different discretization rules give different properties.** ZOH gives exact continuous-to-discrete correspondence; bilinear (Tustin) is sometimes used for stability; Euler is simplest but less accurate.

## Where It Appears in This Wiki

- [[S4]] — ZOH discretization converts the HiPPO ODE into the discrete recurrence used for training
- [[Mamba]] — $\Delta$ is made input-dependent; the selectivity mechanism is implemented through data-dependent discretization
- [[State Space Model]] — discretization is fundamental to all discrete SSMs

## Key Formula or Pseudocode

```
Continuous SSM:  x'(t) = Ax(t) + Bu(t),  y(t) = Cx(t)

ZOH Discretization with step Δ:
  Ā = exp(ΔA)              # matrix exponential
  B̄ = A⁻¹(exp(ΔA) - I)B   # may simplify to ΔB for small Δ

Discrete recurrence:
  x_k = Ā x_{k-1} + B̄ u_k
  y_k = C x_k

Mamba extension: Δ_t = softplus(W_Δ x_t + b_Δ)  # input-dependent
```

## Related Concepts

[[S4]] · [[State Space Model]] · [[HiPPO matrix]] · [[Mamba]] · [[Selective State Space Model]]
