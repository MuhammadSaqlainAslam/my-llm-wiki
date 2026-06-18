---
created: "2026-05-12"
title: "Selective State Space Model (S6)"
tags: [ssm, mamba, selectivity, recurrence]
tldr: "A state-space model whose $\\mathbf{B}$, $\\mathbf{C}$, and $\\Delta$ parameters are functions of the current input $x_t$ — content-dependent rather than fixed. The core innovation behind [[Mamba]] that closed the quality gap between SSMs and attention."
---

# Selective State Space Model (S6)

An S4-style state-space model is **time-invariant**: the transition matrices $\mathbf{A}, \mathbf{B}, \mathbf{C}$ and step size $\Delta$ are the same for every position. That means the model processes every token with identical dynamics — it cannot decide at runtime to remember this token and forget that one. A *selective* SSM (S6) replaces $\mathbf{B}$, $\mathbf{C}$, $\Delta$ with $\text{Linear}(x_t)$, making them input-dependent. The selectivity gives the model content-aware memory: large $\Delta$ resets the state to attend to $x_t$; small $\Delta$ ignores $x_t$ and preserves history. This is the architectural breakthrough behind [[Mamba]].

## Where it appears

- [[Mamba]]
- [[Transformers Are SSMs|Mamba-2]]
- [[Transformers Are SSMs]]
- [[Nemotron-3]]

---

*Related: [[State Space Model]] · [[Hardware-Aware Scan]]*
