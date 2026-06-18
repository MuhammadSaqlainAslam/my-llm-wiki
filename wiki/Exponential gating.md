---
created: "2026-05-12"
title: "Exponential Gating"
tags: [lstm, xlstm, gating, normalization]
tldr: "Replace the LSTM's sigmoid gates with $\\exp(\\cdot)$, paired with a running max-stabilizer and a normalizer. Lets a single token effectively reset the cell state — the missing ingredient that turns LSTM into a competitive language-model backbone."
---

# Exponential Gating

Sigmoid gates $f_t, i_t = \sigma(\cdot) \in (0, 1)$ are smooth and bounded, but that boundedness means a token can never *override* what's already in the cell — it can only slowly pull memory in its direction. Exponential gates $f_t, i_t = \exp(\cdot)$ are unbounded, so a single very-confident write can dominate the running normalizer and effectively flush the cell. To keep this numerically stable, [[xLSTM]] introduces a per-step running maximum $m_t = \max(\log f_t + m_{t-1}, \log i_t)$ and rescales the gates by $\exp(-m_t)$ before applying them. A separate normalizer $n_t$ tracks the integral of gate activity and divides it out at read time. Together these turn unbounded $\exp$ into a stable, learnable selectivity mechanism.

## Where it appears

- [[xLSTM]]

---

*Related: [[LSTM]] · [[Matrix memory]]*
