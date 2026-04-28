---
title: "State Space Model (SSM)"
tags: [ssm, recurrence, math, linear-systems, mamba]
tldr: "A continuous-time linear dynamical system discretized for sequences. The dual recurrent/convolutional representation lets you train in parallel and infer in constant memory — the mathematical foundation of Mamba."
theme: efficiency
---

# State Space Model (SSM)

An SSM maps an input sequence $x(t)$ to output $y(t)$ through a latent state $h(t) \in \mathbb{R}^N$. The continuous-time equations:

$$h'(t) = \mathbf{A} h(t) + \mathbf{B} x(t)$$
$$y(t) = \mathbf{C} h(t)$$

This is a linear dynamical system — the same math as Kalman filters and control theory. For deep learning you discretize it. With step size $\Delta$ and zero-order hold:

$$\bar{\mathbf{A}} = \exp(\Delta \mathbf{A}), \quad \bar{\mathbf{B}} = (\Delta \mathbf{A})^{-1}(\exp(\Delta \mathbf{A}) - I) \cdot \Delta \mathbf{B}$$

The discrete recurrence is:

$$h_t = \bar{\mathbf{A}} h_{t-1} + \bar{\mathbf{B}} x_t, \quad y_t = \mathbf{C} h_t$$

The elegant duality: this exact recurrence can also be written as a **global convolution** $y = x * \bar{\mathbf{K}}$ where $\bar{\mathbf{K}} = (\mathbf{C}\bar{\mathbf{B}},\ \mathbf{C}\bar{\mathbf{A}}\bar{\mathbf{B}},\ \mathbf{C}\bar{\mathbf{A}}^2\bar{\mathbf{B}},\ \ldots)$ is the SSM's impulse response. This means you can **train with convolution** (all tokens visible in parallel, efficient on GPU) and **infer with recurrence** (O(1) per step, constant memory regardless of sequence length). The four parameters $(\Delta, \mathbf{A}, \mathbf{B}, \mathbf{C})$ fully define the model. Early SSMs (S4) kept these as fixed constants — [[Mamba]]'s breakthrough was making $\mathbf{B}$, $\mathbf{C}$, and $\Delta$ functions of the current input $x_t$, enabling selective rather than uniform compression of context.

## Where it appears

- **[[Mamba]]** — the entire architecture is built on this foundation; the SSM is the core sequence-mixing operation that replaces attention
- **[[Nemotron-3]]** — Mamba-2 SSM layers make up most of the layer stack; the SSM recurrent state is the thing that stays constant-size at 1M context

## Why it matters

- **It unifies training efficiency with inference efficiency.** Transformers are parallelizable at training time but sequential at inference (due to autoregressive KV cache growth). RNNs are sequential both ways. SSMs give you parallel training via convolution and constant-memory inference via recurrence — the best of both.
- **The $\mathbf{A}$ matrix controls long-range memory.** How eigenvalues of $\mathbf{A}$ are initialized and constrained determines how far back in the sequence the model can "remember." HiPPO initialization (used in S4/Mamba) is a principled way to set $\mathbf{A}$ so the state optimally reconstructs recent history.
- **Making parameters input-dependent is the key insight.** When $\mathbf{B}$, $\mathbf{C}$, $\Delta$ are fixed, the SSM processes every token identically — it can't selectively remember or forget. Input-dependence breaks the LTI constraint and gives the model content-aware memory, which is what closes the quality gap with attention on language tasks.

---

*Related: [[Mamba]] · [[Hardware-Aware Scan]] · [[Nemotron-3]]*
