---
created: "2026-05-12"
title: "Covariance Update Rule"
tags: [xlstm, memory, associative, kernel]
tldr: "$C_t = f_t C_{t-1} + i_t v_t k_t^\top$ — the outer-product update that turns a matrix memory into a learned key-value store. The same update that drives Hopfield networks and kernelized attention."
---

# Covariance Update Rule

The update writes a key/value pair $(k_t, v_t)$ into a matrix state $C_t$ via outer product — this is the same operation that builds covariance matrices in statistics, hence the name. Retrieval is $C_t q$ for a query $q$: the result is $\sum_{s \le t} f_{s+1:t} i_s (q^\top k_s) v_s$, i.e., a kernel-attention-like sum where the gates control what's in the cache. With exponential gates the memory can be selectively overwritten; with sigmoid gates it can only fade. The covariance update is what makes [[xLSTM]]'s mLSTM both parallelizable and expressive.

## Where it appears

- [[xLSTM]]

---

*Related: [[Matrix memory]] · [[LSTM]] · [[Linear attention]]*
