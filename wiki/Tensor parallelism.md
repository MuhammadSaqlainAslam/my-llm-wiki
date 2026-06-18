---
created: "2026-05-12"
title: "Tensor Parallelism"
tags: [parallelism, training, gpu, transformer, mamba-2]
tldr: "Split each layer's weight matrices across GPUs on the same node so each GPU does a fraction of the matmul. The standard model-parallelism technique for training large Transformers; [[Transformers Are SSMs|Mamba-2]] is explicitly designed to be TP-friendly with half the synchronization points per block."
---

# Tensor Parallelism

When a model's weights don't fit on one GPU, you split them. Pipeline parallelism splits *layers* across devices; tensor parallelism splits *weights within a layer*. For a linear layer $y = xW$, you shard $W$ along its columns across $G$ GPUs, each computes $xW_g$, then an all-reduce concatenates the results. The cost is one all-reduce per linear layer, which is communication-heavy — modern hardware (NVLink, NVSwitch) is sized to make this affordable inside a single node. Transformer blocks have two big linear layers (Q/K/V projection and the FFN), each requiring a sync. [[Transformers Are SSMs|Mamba-2]] is engineered so its block needs **half** as many syncs as a Transformer block of comparable size — the SSD decomposition lets the SSM work happen inside one TP region without crossing it.

## Where it appears

- [[Transformers Are SSMs]]
- [[Nemotron-3]]

---

*Related: [[Sequence parallelism]] · [[Transformers Are SSMs|Mamba-2]] · [[Transformer]]*
