---
created: "2026-05-12"
title: "Multi-Head SSM"
tags: [ssm, mamba-2, attention, multi-head]
tldr: "Multiple parallel SSM heads operating on different projections of the input, by analogy with multi-head attention. The SSD framework shows that heads in attention map directly onto heads in SSMs once both are viewed as structured matrix mixers."
---

# Multi-Head SSM

In a multi-head Transformer, $H$ separate attention heads each compute their own $QK^\top$ pattern over different projections of $x$, and the outputs are concatenated. The SSD framework recognises that the same construction applies verbatim to a state-space model: split the channel dimension into $H$ groups, run an independent selective SSM on each, concatenate the outputs. This is what [[Transformers Are SSMs|Mamba-2]] does. Multi-head SSMs let different heads specialize on different mixing patterns just as multi-head attention does, and unlock the same kind of model parallelism at training time.

## Where it appears

- [[Transformers Are SSMs]]
- [[Transformers Are SSMs|Mamba-2]]

---

*Related: [[Transformer]] · [[State Space Model]]*
