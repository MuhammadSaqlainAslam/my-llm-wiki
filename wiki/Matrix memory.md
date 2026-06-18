---
created: "2026-05-12"
title: "Matrix Memory"
tags: [xlstm, memory, associative, hopfield]
tldr: "Replace the LSTM's scalar cell state with a $d \times d$ matrix $C_t$ updated by an outer product $C_t = f_t C_{t-1} + i_t v_t k_t^\top$. Turns the cell into a learned key-value associative store, recovering attention-like content-based recall in a fully recurrent model."
---

# Matrix Memory

Where the original [[LSTM]] stores a scalar (or vector of independent scalars) per cell, the [[xLSTM]] mLSTM variant stores a full $d \times d$ matrix $C_t$ updated via the **[[Covariance update rule]]** $C_t = f_t C_{t-1} + i_t v_t k_t^\top$. The outer product $v_t k_t^\top$ writes the key→value pair $(k_t, v_t)$ into the matrix; later, $C_t q_t$ retrieves the associated value for a query $q_t$ — the same operation as one step of [[Linear attention]]. The matrix memory gives a recurrent model the same content-based recall that attention has, at constant per-step cost. The cost: $d \times d$ memory per cell per layer, which can be substantial at $d = 4096$.

## Where it appears

- [[xLSTM]]

---

*Related: [[LSTM]] · [[Covariance update rule]] · [[Linear attention]]*
