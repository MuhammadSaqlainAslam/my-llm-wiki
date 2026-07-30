---
created: "2026-05-12"
title: "Linear Attention"
tags: [attention, efficiency, kernel, ssm]
tldr: "Replace the softmax in attention with a kernel feature map so the computation reduces to outer products plus a running sum. $O(n)$ per token at the cost of expressive power — equivalent to a rank-1 SSM in the SSD framework."
---

# Linear Attention

Standard attention is $\text{softmax}(QK^\top)V$. The softmax couples positions and forces $O(n^2)$ compute. **Linear attention** approximates softmax by a kernel: $\text{softmax}(qk^\top) \approx \phi(q)\phi(k)^\top$ for some feature map $\phi$. With this swap you can re-associate: $\phi(Q)(\phi(K)^\top V)$. The bracketed term is a $d \times d$ matrix that you can update incrementally — $O(1)$ per token, no growing KV cache. The trade-off is expressivity: linear attention has trouble with sharp content-based recall, and in the [[SSD|Structured State Space Duality]] view it corresponds to a rank-1 mixing matrix. [[Mamba]]'s selectivity and [[xLSTM]]'s [[Matrix memory]] are essentially smarter linear-attention variants that get back some of the lost expressivity.

## Where it appears

- [[Transformers Are SSMs]]
- [[xLSTM]]
- [[Attention to Mamba]] — used as the intermediate stepping stone when distilling a Transformer teacher into a Mamba student

---

*Related: [[Transformer]] · [[KV Cache]]*
