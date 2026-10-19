---
created: "2026-10-19"
title: "MegaByte: Predicting Million-byte Sequences with Multiscale Transformers"
authors: "Lili Yu, Dániel Simig, Colin Flaherty, Armen Aghajanyan, Luke Zettlemoyer, Mike Lewis (Meta AI)"
year: 2023
arxiv: "2305.07185"
citation_count: 204
tags: [byte-level, tokenization, long-context, architecture, multiscale, efficiency, multimodal]
tldr: "Global Transformer over fixed-size byte patches + small local Transformer inside each patch. Sub-quadratic attention (optimal P gives O(N^{4/3})), fat per-patch FFNs (up to P× bigger at equal FLOPs), and parallel patch decoding (~P× fewer serial steps). Tokenization-free AR that is competitive with BPE models on long text and SOTA on ImageNet density — million-byte multimodal in one model."
aliases: [MegaByte, MEGABYTE, Multiscale Transformers]
theme: tokenization
---

# MegaByte

> Lili Yu, Dániel Simig, Colin Flaherty, Armen Aghajanyan, Luke Zettlemoyer, Mike Lewis (Meta AI), "MegaByte: Predicting Million-byte Sequences with Multiscale Transformers", 2023 (arXiv:2305.07185)

## The Problem / Motivation

Music, images, podcasts, code, books are **millions of bytes**. GPT-style decoders only chew a few thousand **tokens** — partly $O(T^2)$ attention, but mostly **per-position FFNs** (in GPT-3-scale models FFN is ~98% of FLOPs).

[[BPE_Tokenization|BPE]] is the usual escape hatch: fold bytes into a fixed vocabulary. That is a heuristic preprocessor — domain/modality brittle, lossy for some scripts, not truly end-to-end (the same critique [[Byte Latent Transformer|BLT]] sharpens later).

MegaByte’s claim: you can model **raw bytes** at million-length if you **split the work across scales**.

## The Idea

Most bytes are easy (finish “Mozart” after “M”). Spend a **small** model inside short patches; spend the **big** model only on patch-to-patch structure.

```
bytes  ──patch embed (concat)──►  K = T/P patches
                                      │
                          Global Transformer (big, causal, over patches)
                                      │
                    slice each patch → add prev-byte embeds
                                      │
                     Local Transformer (small, parallel over patches)
                                      │
                                   next-byte softmax
```

1. **Patch embedder** — embed each byte, concatenate $P$ embeddings into one patch vector (lossless layout).
2. **Global model** — large decoder-only Transformer on $K=T/P$ patches (pad + shift for AR).
3. **Local model** — small decoder that fills in the $P$ bytes of each patch conditioned on the global slice; **K copies in parallel** at train time.

Why local AR inside the patch? The joint over $P$ bytes has $256^P$ outcomes — only tractable if factored. Independent per-byte given global would smear mass (50% cat / 50% dog → “cag”, “dot”). Local AR keeps valid strings.

## Architecture / Method

### Efficiency (why this isn’t just “ViT on bytes”)

| Cost | Transformer | MegaByte |
|---|---|---|
| Attention | $O(T^2)$ | $O(T^2/P^2 + TP)$ → **$O(T^{4/3})$** at $P=T^{1/3}$ |
| FFN (dominant) | $2mT$ on $m$ params | $\approx 2T m_g / P$ if global ≫ local → **$P\times$ larger FFN** at same FLOPs |
| Decode serial depth | $O(P \cdot L)$ | $O(L_g + P \cdot L_l)$ → up to **~$P\times$** fewer serial layer-steps |

Also: **strided inference** (two offset passes, take first half of each patch) trades 2× compute for better end-of-patch quality; optional conv patch encoder and cross-patch local attention (Transformer-XL-like but fully differentiable).

### Multimodal

Same recipe on text, **ImageNet** (patch-scan of RGB bytes), and **raw audio files** (16-bit samples as bytes — no μ-law special case required).

## Key Results

| Task | Result |
|---|---|
| Compute-matched LM (80B bytes; PG-19, Stories, Books, arXiv, Code) | MegaByte **beats** Transformer and PerceiverAR on **all five** (e.g. Code **0.411** vs 0.575 / 0.546 bpb) |
| PG-19 scaled (400B bytes) | Word PPL **36.4** vs byte Transformer 69.4 / byte PerceiverAR 88.8 — **competitive with strong BPE models** (TransformerXL 36.3, Compressive 33.6) |
| ImageNet 64² | **3.40 bpb** = PerceiverAR SOTA at **~½ GPU hours** |
| ImageNet 256² / 640² (full 196K / **1.2M** bytes) | Beats compute-matched Transformer & PerceiverAR (640: **2.282** vs 2.345 / 2.847) |
| Audio from raw file bytes | Viable AR modeling without audio-specific discretizations |
| Speed | 1.5B MegaByte generates **~40% faster** than a 350M Transformer at better PPL (same train compute) |

Controlled setup: same FLOPs-per-byte and same training bytes (not “more GPUs win”).

## Comparison to Prior Work

- vs **[[Transformer]] on bytes** — same compute, much worse (bytes make $T$ huge). MegaByte is the structure that makes bytes affordable.
- vs **[[BPE_Tokenization|BPE]] / [[SentencePiece]] LMs** — MegaByte is competitive *without* a fixed vocab; patches are just reshaped bytes.
- vs **PerceiverAR** — latents + cross-attn long context; MegaByte is pure AR multiscale and wins on the long image grids.
- vs **[[Byte Latent Transformer|BLT]] (2024)** — BLT keeps the global/local split but **dynamic entropy patches**, hash n-grams, and two-way cross-attention; explicitly cites MegaByte’s *fixed* stride as wasteful (easy bytes still burn a global step).
- vs **MambaByte / linear byte models** — different axis (recurrence vs multiscale attention).

## Limitations

- Fixed stride $P$ is **not** content-adaptive (BLT’s main critique).
- Loss rises at the **end of patches** (local model weaker) → strided inference doubles cost.
- Patch boundaries are not translation-invariant (optional CNN encoder only partly helps).
- Two Transformers + reshape plumbing — more systems surface than a single decoder.
- “Competitive with BPE” on their long-text suite is strong but not a full LLM-scale chat/code model report.

## Why It Matters

MegaByte is the landmark **tokenization-free at million-byte scale** result and the parent of the modern byte-patch LLM stack ([[Byte Latent Transformer|BLT]], and the “patch = compute dial” idea in later hierarchical models). It also lands three claims this wiki cares about: (1) **FFNs, not attention**, dominate cost — so scale *width per patch*; (2) **decode is serial-depth bound** — patch parallelism is real speed; (3) one AR model can do text + image + audio **as files**, no tokenizer zoo.

## Related Concepts

[[Byte Latent Transformer]] · [[BPE_Tokenization]] · [[WordPiece_Tokenization]] · [[SentencePiece]] · [[minbpe]] · [[Transformer]] · [[Local attention]] · [[Sliding window attention]] · [[Long Range Arena]] · MambaByte · [[xLSTM]] · [[Flash Attention]] · [[Prefill]] · [[Decode phase]]