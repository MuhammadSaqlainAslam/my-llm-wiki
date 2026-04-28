---
title: "KV Cache"
tags: [inference, attention, memory, bottleneck]
tldr: "Cached key-value tensors from past attention steps; eliminates redundant recomputation but grows linearly with sequence length — the root cause of most long-context engineering work."
theme: foundations
---

# KV Cache

When a [[Transformer]] generates token $t$, it needs attention scores against all previous tokens $1 \ldots t{-}1$. Recomputing queries, keys, and values from scratch at every step would cost $O(n^2)$ per generation step. The fix is simple: cache the key and value tensors from past steps and reuse them. At step $t$, you compute one new row of $K$ and $V$, append it to the cache, and run attention over the full cached sequence. Step cost drops to $O(n)$ in memory reads. The problem is that this cache grows by one KV row per token per attention layer — at 1M tokens with 32 layers, $d_{\text{model}} = 4096$, and FP16, the cache alone occupies tens of gigabytes, often exceeding the model weights. That single scaling property is the root cause of most long-context engineering work in this wiki: [[Mamba]] replaces the unbounded cache with a constant-size recurrent state; [[GQA]] shrinks the cache by sharing KV heads across groups of Q heads; FP8 quantization halves its footprint at inference time.

## Where it appears

- **[[Transformer]]** — the original paper doesn't use the term, but the KV cache is what makes autoregressive decoding tractable with attention
- **[[Mamba]]** — the constant-size SSM state is explicitly motivated as an alternative to the growing KV cache
- **[[Nemotron-3]]** — Nemotron-3 Super uses GQA (2 KV heads instead of 32) and FP8 KV cache quantization; the Mamba-dominant layer mix further reduces how many attention layers actually build a cache

## Why it matters

- **Memory is the bottleneck, not compute.** At inference time LLMs are memory-bandwidth-bound. The KV cache competes with model weights for GPU HBM; once it can't fit, you either reduce batch size or truncate context.
- **It's the direct cause of the O(n) memory growth problem.** Every architectural innovation aimed at long-context — sliding-window attention, linear attention, Mamba, GQA, MQA, FP8 KV — is fundamentally attacking the KV cache.
- **Its size determines what "long context" costs.** Serving a 1M-token context with standard MHA requires orders of magnitude more memory than a 4K context. Understanding the KV cache size formula ($2 \times L \times n \times d \times \text{bytes/element}$, where $L$ is layers and $n$ is sequence length) tells you exactly what you're paying for.

---

*Related: [[Transformer]] · [[GQA]] · [[Mamba]] · [[Nemotron-3]]*
