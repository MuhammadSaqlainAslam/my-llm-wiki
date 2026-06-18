---
created: "2026-05-12"
title: "Sequence Parallelism"
tags: [parallelism, training, long-context, attention, ssm]
tldr: "Split the sequence dimension across GPUs so each device holds only a slice of the activations. Originally designed for attention training at long context; the SSD paper shows how to apply it to SSMs by passing the recurrent state across devices."
---

# Sequence Parallelism

For very long sequences, even the activations of a single Transformer block may not fit on one GPU. Sequence parallelism shards the sequence dimension: GPU $g$ holds positions $[gL/G, (g+1)L/G)$. Attention requires each position to see every other, so the implementation chunks the QKV computation and uses ring-style all-to-all communication (the Ulysses / DeepSpeed-Ulysses pattern). For SSMs, the trick is different and arguably cleaner: each GPU runs the SSM recurrence on its slice, then **passes the final hidden state** as a boundary condition to the next GPU. The recurrent state is small ($O(N)$), so the sync is cheap. [[Transformers Are SSMs]] formalizes this construction.

## Where it appears

- [[Transformers Are SSMs]]
- [[Transformers Are SSMs|Mamba-2]]

---

*Related: [[Tensor parallelism]] · [[Mamba]]*
