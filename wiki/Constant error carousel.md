---
created: "2026-05-12"
title: "Constant Error Carousel"
tags: [lstm, gradient, vanishing-gradient, history]
tldr: "The identity path through the LSTM cell state — when the forget gate is open, gradients pass through $c_t = c_{t-1}$ unchanged, avoiding the vanishing-gradient problem of vanilla RNNs. The 1997 idea that made deep recurrent learning possible."
---

# Constant Error Carousel

A vanilla RNN computes $h_t = \tanh(W h_{t-1} + U x_t)$. Backpropagating through time multiplies many Jacobians of $\tanh$, whose derivatives are bounded by 1, so gradients vanish exponentially with sequence length. Hochreiter & Schmidhuber's solution was the **constant error carousel**: a cell-state update $c_t = f_t c_{t-1} + i_t z_t$ where, with $f_t = 1$ and $i_t = 0$, the gradient $\partial c_t / \partial c_{t-1} = 1$ exactly. Errors flow through the cell state without decay or growth, regardless of sequence length. This is the architectural ancestor of every later identity-path mechanism — residual connections in [[Transformer]], skip connections in ResNets, the cell update of [[State Space Model]]s — and it's the reason the [[LSTM]] was the first sequence model that could learn long-range dependencies in practice.

## Where it appears

- [[LSTM]]
- [[xLSTM]]

---

*Related: [[Transformer]] · [[State Space Model]]*
