---
created: "2026-05-12"
title: "Semiseparable Matrix"
tags: [math, ssm, ssd, attention]
tldr: "A matrix whose every off-diagonal block has rank at most $N$. The class of matrices computable by a state-space model with state size $N$ is exactly the class of $N$-semiseparable causal matrices — the central object of the SSD framework."
---

# Semiseparable Matrix

A matrix $M$ is $N$-semiseparable if every contiguous off-diagonal block — i.e., every sub-block that does not cross the diagonal — has rank at most $N$. For sequence models, the **mixing matrix** that maps inputs to outputs is exactly $T \times T$, and the question "how rich can long-range mixing be?" becomes "how high can the rank of off-diagonal blocks be?". A vanilla attention layer has unstructured off-diagonal blocks: rank can be as high as $T$. A linear attention layer is rank-1 off-diagonal everywhere — efficient but expressively weak. A selective SSM with state size $N$ sits in the middle: rank exactly $N$ off-diagonal, scalable, and equivalent to a structured form of attention. This is the central observation of [[Transformers Are SSMs]] — the structure of the mixing matrix is the architectural design choice, and selective SSMs are the rank-$N$ point in the design space.

## Where it appears

- [[Transformers Are SSMs]]
- [[Mamba]]
- [[Transformers Are SSMs|Mamba-2]]

---

*Related: [[Transformer]] · [[State Space Model]] · [[Linear attention]]*
