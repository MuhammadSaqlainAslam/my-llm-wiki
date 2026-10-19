---
created: "2026-10-19"
title: "Byte Latent Transformer: Patches Scale Better Than Tokens"
authors: "Artidoro Pagnoni, Ram Pasunuru, Pedro Rodriguez, John Nguyen, Benjamin Muller, Margaret Li, Chunting Zhou, Lili Yu, Jason Weston, Luke Zettlemoyer, Gargi Ghosh, Mike Lewis, Ari Holtzman, Srinivasan Iyer (Meta FAIR / UW / UChicago)"
year: 2024
arxiv: "2412.09871"
citation_count: 17
tags: [byte-level, tokenization, architecture, long-context, efficiency, robustness, scaling]
tldr: "Tokenizer-free LLM: entropy-based dynamic byte patches feed a big latent Transformer, with small local encoder/decoder and hash n-grams. First byte model to match Llama-3-class training-FLOP parity up to 8B / 4T bytes — up to 50% fewer inference FLOPs if you grow patches with model size. Better noise robustness and orthography than BPE."
aliases: [BLT, Byte Latent Transformer, entropy patching]
theme: efficiency
---

# Byte Latent Transformer (BLT)

> Artidoro Pagnoni, Ram Pasunuru, et al. (Meta FAIR), "Byte Latent Transformer: Patches Scale Better Than Tokens", 2024 (arXiv:2412.09871)

## TL;DR

Tokenization is the one non-differentiable step left in LLMs: a **static** BPE vocabulary decides how strings compress. That hurts noise, scripts, orthography, and multilingual equity — and every token gets the **same** global compute whether it is a trivial word ending or a hard new name.

BLT groups raw bytes into **dynamic patches** using **next-byte entropy** from a tiny byte LM (hard bytes start patches; easy runs get long patches). A heavy **latent global Transformer** runs on patches; light **local encoder/decoder** + **hash n-grams** keep byte fidelity. First FLOP-controlled byte scaling study to **8B params / 4T bytes**: **matches Llama 3** at train FLOPs, with up to **50% fewer inference FLOPs** when you co-scale patch size — a **new scaling axis** tokens do not have.

## The Problem / Motivation

| Tokenization pain | What breaks |
|---|---|
| Static vocab | Domain/modality mismatch; rare forms explode into junk tokens |
| Fixed compute per token | Easy suffixes and hard names cost the same in the global stack |
| No raw bytes | Weak orthography / phonology / noise robustness; multilingual inequity |

Naive bytes die on length: the cost is not only attention — **FFN on every byte** dominates (same punchline as [[MegaByte]]). Prior byte work mostly helps *small* models.

## The Idea

**Allocate global compute by information density.** Entropy patching builds patches of roughly uniform surprise. Predictable stretches → long patches → rare global steps. Hard bytes → short patches → more capacity where it matters. Local modules never throw the bytes away (unlike a frozen BPE id).

## Architecture / Method

```
bytes ──► Local Encoder (tiny, hash n-grams + cross-attn pool)
              │  dynamic patches
              ▼
        Latent Global Transformer  (big; block-causal on patches)
              │
              ▼
        Local Decoder (tiny, cross-attn expand → bytes)
```

### Entropy patching (the compute dial)

Train a **~100M** byte LM; for each byte compute $H(x_i)$. Start a new patch when

- **global:** $H(x_i) > \theta_g$, or
- **approx. monotonic:** $H(x_i) - H(x_{i-1}) > \theta_r$ (breaks “entropy drift” on repeats)

Threshold is calibrated to a target **average patch size** (the FLOP knob). Patching is **incremental** (prefix-consistent) so AR decode can decide boundaries without future bytes — BPE is *not* incremental.

Also compared: fixed stride (MegaByte), space patching, BPE-as-patches.

### Local encoder / decoder

- **Hash n-gram embeddings** (rolling poly-hash, n=3…8) added to byte embeddings — cheap multi-scale byte features.
- **Encoder cross-attn:** patch queries attend to their own bytes (Perceiver-style pooling into variable patches).
- **Global:** standard big Transformer on patch latents (block-causal).
- **Decoder cross-attn:** byte queries attend to the patch latent (roles swapped), then a small local Transformer.

Global runs **per patch**; locals run **per byte**. Average patch size is the primary cost control.

### Scaling knob tokens lack

At fixed inference FLOPs you can **grow the global model and the average patch size together** (longer patches → fewer global invocations). Token models only trade vocab size ↔ embedding cost; BLT gets a real second axis.

## Key Results

| Axis | Result |
|---|---|
| Train-FLOP parity | Matches **Llama 3** (BPE) through **8B / 4T bytes** |
| Inference FLOPs | Up to **50% fewer** for a small eval tradeoff, or parity at equal cost |
| Inference-FLOP-controlled scaling | BLT patch-size 6/8 curves **overtake** BPE Llama 2/3 trends |
| Long-tail / noise | Better than token models on noisy inputs; wins on **CUTE** (orthography), **G2P** phonology, **low-resource MT** |
| Character / code mix | Improvements on noisy data and subword-sensitive tasks |
| From Llama 3 | Can be trained/adapted into the byte regime (paper §6.2) |
| Scale study | First **FLOP-controlled** byte models to 8B / 4T (not just toy) |

Ablations: entropy > space > strided; hash n-grams ≫ bare bytes; cross-attn both ways matters; entropy model can shrink into a lookup if the window is small.

## Comparison to Prior Work

- vs **[[BPE_Tokenization]] / [[SentencePiece]] / Llama 3 tokenizer** — parity in quality without a fixed vocab; better robustness; dynamic compute.
- vs **[[MegaByte]]** — same global/local philosophy, but **fixed stride → entropy patches**, hash n-grams, richer cross-attn, and the first large FLOP-controlled scaling laws. MegaByte showed bytes can work; BLT shows bytes can **win the cost-quality Pareto**.
- vs **ByT5 / CANINE** (encoder byte models) — BLT is a decoder-only generative LLM at multi-billion scale.
- vs **MambaByte** — token-free via SSM instead of patch latents; complementary architecture choice.

## Limitations

- Entropy model + patching is extra pipeline (must be incremental for decode; quality of $\theta$ calibration matters).
- Structured/repetitive text can “entropy-drift” into huge patches (they reset context / use monotonic constraint — a heuristic, still).
- Local encoder/decoder are small but not free; hash n-grams are collision-prone (mitigated by table size).
- Released as research code (`facebookresearch/blt`); production stacks are still BPE-centric.
- Paper’s Llama-parity is **training-FLOP controlled** on their mix — not a drop-in chat product eval sweep.

## Why It Matters

Together with [[MegaByte]], BLT closes the “tokenizer-free is weaker” gap **at LLM scale** and reframes [[BPE_Tokenization]] as a *fixed* special case of *dynamic* patching. For this wiki’s efficiency thesis it adds a clean rule: **spend global FLOPs where entropy is high**, and when you need more quality at fixed serving cost, **grow patches and the latent model together**. If you care about long-tail text, multilingual fairness, or files that are not pre-tokenized English, start here.

## Related Concepts

[[MegaByte]] · [[BPE_Tokenization]] · [[WordPiece_Tokenization]] · [[SentencePiece]] · [[minbpe]] · MambaByte · [[Transformer]] · [[Local attention]] · [[Long Range Arena]] · [[RAG]] · [[Quantization]] · [[Prefill]] · [[Decode phase]] · [[LLaMA 2]] · [[The Llama 3 Herd of Models]]