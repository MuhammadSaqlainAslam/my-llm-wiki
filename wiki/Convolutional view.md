---
created: "2026-05-20"
title: "Convolutional View of SSMs"
tags: [glossary, ssm, s4, convolution, training, parallelism]
tldr: "State space models can be equivalently computed as a global convolution y = K̄ * u, where the convolutional kernel K̄ is derived from the SSM parameters. This view enables fully parallel training via FFT convolution — unlike the sequential recurrence view used at inference."
aliases: [SSM convolution, convolutional SSM, SSM convolutional kernel]
---

## TL;DR

An SSM with parameters $(\bar{A}, \bar{B}, C)$ and input $u_0, u_1, \ldots, u_{L-1}$ produces output $y_k = C \bar{A}^k \bar{B} u_0 + C \bar{A}^{k-1} \bar{B} u_1 + \ldots + C \bar{B} u_k$. This is a **convolution**: $y = \bar{K} * u$ where $\bar{K}_k = C \bar{A}^k \bar{B}$. Computing this convolution via FFT takes $O(L \log L)$ and is fully parallelizable across the sequence — the key to fast SSM training.

## Intuition

Think of the SSM as a linear filter applied to the input signal. A filter is exactly a convolution: each output is a weighted sum of past inputs, where the weights (the kernel $\bar{K}$) decay over time. The recurrent view computes these weights one step at a time; the convolutional view computes all outputs simultaneously.

In practice:
1. Compute the SSM kernel $\bar{K} = (\bar{C}\bar{B}, \bar{C}\bar{A}\bar{B}, \bar{C}\bar{A}^2\bar{B}, \ldots)$ — a vector of length $L$. For S4, this is where the [[Cauchy kernel]] computation happens.
2. Apply the convolution $y = \text{FFT}^{-1}(\text{FFT}(\bar{K}) \odot \text{FFT}(u))$ — $O(L \log L)$, fully parallel.

At inference, the recurrent view is cheaper: $x_k = \bar{A} x_{k-1} + \bar{B} u_k$ runs in $O(N)$ per step with constant memory. S4 (and Mamba) switches between the two views depending on the context: convolutional view for training (fast and parallel), recurrent view for inference ($O(1)$ memory per step).

## Why It Matters

- **It's why SSMs can train as fast as Transformers.** The convolutional view is $O(L \log L)$ and fully parallel, competitive with attention's $O(L^2)$ for moderate $L$.
- **It's the key to the SSM training/inference duality.** Same parameters, same math — two efficient computation strategies for two different settings.
- **It connects SSMs to signal processing.** The SSM kernel $\bar{K}$ is literally a digital filter impulse response — all the tools from DSP apply.

## Where It Appears in This Wiki

- [[S4]] — the convolutional view is one of S4's three equivalent representations; training uses this view
- [[State Space Model]] — the fundamental duality between convolutional and recurrent computation
- [[Mamba]] — uses the recurrent view at inference but the parallel scan (related to convolution) at training

## Key Formula or Pseudocode

```
SSM as convolution:
  K̄_k = C Ā^k B̄   for k = 0, ..., L-1        # SSM kernel
  y = K̄ * u                                      # output = convolution

Efficient computation:
  K̄ = compute_ssm_kernel(A, B, C, L)             # O(N log²N) via Cauchy
  Y = IFFT(FFT(K̄) ⊙ FFT(u))                     # O(L log L) via FFT

vs. recurrent view (inference):
  for k = 0..L: x_k = Ā x_{k-1} + B̄ u_k       # O(NL) sequential
```

## Related Concepts

[[S4]] · [[State Space Model]] · [[Cauchy kernel]] · [[Discretization]] · [[Mamba]] · [[Hardware-Aware Scan]]
