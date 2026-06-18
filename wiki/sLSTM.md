---
created: "2026-05-12"
title: "sLSTM (scalar xLSTM cell)"
tags: [xlstm, lstm, scalar-memory, gating]
tldr: "The xLSTM scalar-memory variant: keeps the LSTM's scalar cell but adds [[Exponential gating]] and \"memory mixing across cells within a head\". Not parallelizable across time, but cheap per step."
---

# sLSTM (scalar xLSTM cell)

The sLSTM cell replaces the LSTM's sigmoid gates with [[Exponential gating]] (with a max-stabilizer for numerical safety) while keeping a scalar cell state. The new "memory mixing" mechanism lets information flow between cells within the same head, which prior LSTM variants did not allow. The result is a cell with richer state evolution than LSTM but still fundamentally recurrent — sequential at training time, $O(1)$ per step at inference. sLSTM blocks alternate with mLSTM blocks in the [[xLSTM]] architecture; sLSTM contributes expressive sequential dynamics, mLSTM contributes parallelizable matrix memory.

## Where it appears

- [[xLSTM]]

---

*Related: [[LSTM]] · [[mLSTM]] · [[Exponential gating]]*
