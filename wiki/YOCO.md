---
created: "2026-10-19"
title: "You Only Cache Once: Decoder-Decoder Architectures for Language Models (YOCO)"
authors: "Yutao Sun, Li Dong, Yi Zhu, Shaohan Huang, Wenhui Wang, Shuming Ma, Quanlu Zhang, Jianyong Wang, Furu Wei (Microsoft Research / Tsinghua)"
year: 2024
arxiv: "2405.05254"
citation_count: 172
tags: [architecture, kv-cache, long-context, inference, attention, retention, efficiency, memory]
tldr: "Stack a cross-decoder on a self-decoder: only the self-decoder writes the *global* KV cache once; every cross layer reuses it via cross-attention. KV memory drops ~L× (≈80× on 65B), prefill can early-exit before the heavy layers (512K: 180s → <6s; 1M: 71.8×), and a 3B YOCO matches strong Transformers while hitting near-perfect 1M needle retrieval."
aliases: [YOCO, You Only Cache Once, decoder-decoder, gRet, gated retention]
theme: efficiency
---

# YOCO (You Only Cache Once)

> Yutao Sun, Li Dong, et al. (Microsoft Research / Tsinghua), "You Only Cache Once: Decoder-Decoder Architectures for Language Models", 2024 (arXiv:2405.05254)

## TL;DR

A decoder-only Transformer keeps **L independent KV caches** (one per layer) — at 65B with [[GQA]] + 8-bit KV, **512K tokens ≈ 86 GB** (more than an H100). Prefill of 1M tokens can take minutes even with [[FlashAttention-2]]-class kernels.

YOCO is a **decoder–decoder**: bottom **self-decoder** with *cheap* self-attention (constant-ish cache) writes **one** global $(\hat K, \hat V)$; top **cross-decoder** layers only do **cross-attention onto that shared cache**. Behaviorally it is still an autoregressive LM. You cache once — hence the name.

Wins: **~L× less KV memory**, **prefill early-exit** after the self-decoder (skip half the stack on the prompt), plus clean **chunk parallel** training. 3B model trains to 1.6T tokens on par with StableLM-class Transformers and extends to **1M** context with near-perfect needles.

## The Problem / Motivation

Three classic layouts:

| Layout | AR generation | Long-context pain |
|---|---|---|
| Encoder-only | re-encode | no true AR cache |
| Encoder–decoder | decoder only | encoder params underused multi-turn |
| Decoder-only | KV cache | **memory + prefill** explode with $N$ |

KV complexity is $O(LND)$ ($N$ length, $L$ layers, $D$ dim). Prefill attention is $O(LN^2D)$. Serving long context is exactly this wiki’s *bytes and FLOPs per token* story.

## The Idea

Split the stack in half:

```
X⁰  ──►  Self-decoder (L/2 layers)
            ESA: gated retention OR sliding-window attention
            O(1)-ish local cache
         ──► X^{L/2}  ──► one global K̂, V̂
                              │
X^{L/2} ──►  Cross-decoder (L/2 layers)
            Attention(Q, K̂, V̂)  ← same cache every layer
            SwiGLU, RMSNorm, GQA-friendly
         ──► logits
```

- **Self-decoder:** efficient self-attention with **constant inference memory** (gated retention state $S_n$, or [[Sliding window attention]] of width $C$).
- **One global cache:** $\hat K = \mathrm{LN}(X^{L/2}) W_K$, $\hat V = \ldots W_V$ — projected **once**, reused by all cross layers.
- **Cross-decoder:** standard multi-head cross-attention + SwiGLU; causal; plays well with [[GQA]].

**KV complexity:** $O((N+CL)D)$ vs $O(LND)$ — about **$L\times$ smaller** global cache (self-decoder’s constant cache is noise at long $N$).

**Prefill early exit:** for the prompt, run only the self-decoder, build $\hat K,\hat V$, stop. Generation still uses the full stack. Attention prefill drops $O(LN^2D) \to O(LND)$ and skips half the layers entirely.

### Gated retention (default ESA)

[[Retention mechanism]] + data-dependent gate $\gamma = \mathrm{sigmoid}(XW_\gamma)^{1/\tau}$ (head-wise, Tensor-Core friendly):

$$
S_n = \gamma_n S_{n-1} + K_n^\top V_n, \quad \mathrm{gRet}(x_n) = Q_n S_n
$$

Parallel / recurrent / **chunkwise** forms are equivalent (RetNet lineage). Chunkwise is what training and prefill use; recurrent is decode. Sliding-window attention is the simpler ESA option.

## Key Results

**Quality**

| Setting | Result |
|---|---|
| YOCO-3B @ 1T / 1.6T tokens | comparable to OpenLLaMA-v2-3B, StableLM-3B-4E1T on ARC / BoolQ / HellaSwag / PIQA / … (e.g. 1.6T avg **0.636**) |
| Scaling 160M → 13B | loss curves match Llama-style Transformer; **YOCO-gRet > Transformer > YOCO-SWA** |
| Fine-grained PPL (160M) | YOCO-gRet **3.53** vs Transformer 3.56 (better AR-Hit recall) |
| Needle @ 1M | **near-perfect** (Fig. 4) |
| Multi-needle @ 128K | YOCO-3B-1M **0.98 / 0.98 / 0.84 / 0.56** (N=1,2,4,8) — competitive with 7B LWM-1M, above MiniCPM/ChatGLM3-128K |
| Long NLL | improves with context on books & repo code out to 1M |

**Systems (H100, vs strong Transformer w/ GQA + Flash-Decoding)**

| Metric | Result |
|---|---|
| KV memory (65B-class) | **~80×** reduction |
| 3B inference mem @ 32K / 1M | **~2×** / **>9×** lower (1M total **12.4 GB** vs **9.4×** that) |
| Prefill @ 32K / 512K / 1M | **2.87×** / **180 s → <6 s** / **71.8×** |
| Serving | 1 GB holds **128K** tokens (YOCO) vs **1.6K** at 65B Transformer+GQA |

Also: chunk parallelism for 1M training (all-gather KV **once**, not every layer).

## Comparison to Prior Work

- vs **decoder-only [[Transformer]]** — same AR interface; ~$L\times$ less global KV; much faster prefill.
- vs **encoder–decoder (T5)** — conceptually related, but YOCO looks like a decoder LM outside and uses the encoder side for *cache*, not a separate text encoder.
- vs **[[RetNet]] / [[RWKV]] / [[Mamba]]** — those drop softmax attention entirely; YOCO **keeps global exact attention** on one shared cache (gRet is only in the self-decoder).
- vs **[[GQA]] / [[MQA]]** — complementary: YOCO still uses GQA on the shared cache; the big win is *how many caches*, not how many KV heads.
- vs **[[KV Cache Optimization]] (evict/compress/offload)** — YOCO changes the *architecture* so the cache is single-layer; those methods shrink/evict a multi-layer cache after the fact. Can stack.
- vs **[[Native Sparse Attention|NSA]] / [[MiniMax-Sparse-Attention|MSA]]** — sparsity per query on a full multi-layer cache; YOCO reduces cache *rank in depth* first.
- vs **[[Griffin]] hybrids** — similar “cheap local + global story” vibe; YOCO’s global path is shared cross-attention, not per-layer recurrence.

## Limitations

- Self-decoder quality sets the ceiling for the shared $\hat K,\hat V$ — a weak ESA hurts every cross layer (SWA variant is worse than gRet in their scaling study).
- Cross layers cannot have *layer-specific* keys (that’s the point, but it is a real capacity change).
- Early-exit prefill assumes self-decoder output is enough for $\hat K,\hat V$; deep cross-only effects on the prompt are skipped (generation still uses them).
- gRet kernels / chunk parallel training are extra systems work (they ship Triton; not “stock PyTorch SDPA”).
- 3B/13B research scale in the paper; production 65B YOCO numbers are largely *complexity + profiling* extrapolations.

## Why It Matters

YOCO is the architectural answer to “the [[KV Cache]] is $L$ copies of the same context.” Pair it with the wiki’s kernel story ([[Flash Attention]] → [[FlashAttention-3]]) and cache taxonomy ([[KV Cache Optimization]], [[Paged attention]]): kernels make *each* cache layer cheap; YOCO makes there *fewer* cache layers. The decoder–decoder split also unlocks prefill early-exit and chunk-parallel long training — the same systems insight as “don’t move / don’t store what you can share.”

## Related Concepts

[[KV Cache]] · [[KV Cache Optimization]] · [[Paged attention]] · [[Prefill]] · [[Decode phase]] · [[GQA]] · [[Multi-Query Attention]] · [[Retention mechanism]] · [[Multi-scale retention]] · [[Sliding window attention]] · [[Flash Attention]] · [[FlashAttention-2]] · [[FlashAttention-3]] · [[Transformer]] · [[RetNet]] · [[Griffin]] · [[Mamba]] · [[Native Sparse Attention]] · [[Lost in the Middle]] · [[vLLM]] · [[GPU]]