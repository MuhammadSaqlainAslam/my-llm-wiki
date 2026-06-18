---
created: "2026-05-12"
title: "mLSTM (matrix xLSTM cell)"
tags: [xlstm, lstm, matrix-memory, parallelizable]
tldr: "The xLSTM matrix-memory variant: a $d \times d$ matrix cell state updated by an outer-product covariance rule. Drops hidden-to-hidden recurrence to enable fully parallel training, like a linear attention with selective gates."
---

# mLSTM (matrix xLSTM cell)

The mLSTM cell maintains a matrix state $C_t \in \mathbb{R}^{d \times d}$ updated via the [[Covariance update rule]] $C_t = f_t C_{t-1} + i_t v_t k_t^\top$. Crucially, the gates $f_t, i_t$ depend only on the current input $x_t$, not on the previous hidden state — this breaks the LSTM's hidden-to-hidden recurrence and makes the cell **fully parallelizable in training time**, just like attention. Retrieval is $C_t q_t$, equivalent to one step of [[Linear attention]] with built-in [[Exponential gating]]. mLSTM provides the [[Matrix memory]] that gives [[xLSTM]] its associative-recall ability.

## Where it appears

- [[xLSTM]]

---

*Related: [[Matrix memory]] · [[Covariance update rule]] · [[sLSTM]] · [[Linear attention]]*
