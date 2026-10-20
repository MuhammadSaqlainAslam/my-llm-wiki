---
created: "2026-10-19"
title: "Large Language Diffusion Models (LLaDA)"
authors: "Shen Nie, Fengqi Zhu, Zebin You, Xiaolu Zhang, Jingyang Ou, Jun Hu, Jun Zhou, Yankai Lin, Ji-Rong Wen, Chongxuan Li (RUC / collaborators)"
year: 2025
arxiv: "2502.09992"
citation_count: 0
tags: [architecture, diffusion, transformers, generation, scaling, reasoning]
tldr: "Train a language **diffusion** Transformer from scratch (masking + unmasking) with the usual pretrain + SFT recipe. LLaDA 8B is competitive with LLaMA3-8B-class ARMs on ICL and math/code; after SFT it does multi-turn chat — and it fixes the reversal curse (beats GPT-4o on reversed poem completion). LLM skills are not AR-only."
aliases: [LLaDA, Large Language Diffusion Models, diffusion LLM]
theme: architecture
---

# LLaDA (Large Language Diffusion Models)

> Shen Nie, et al., "Large Language Diffusion Models", 2025 (arXiv:2502.09992)

## The Problem / Motivation

The default story: LLM capability **is** the autoregressive Transformer ([[Transformer]], GPT-style). That story has known cracks:

- **Reversal curse** — trained on “A is B,” models fail “B is A”
- Left-to-right commit is wrong for some structured / editing tasks
- Diffusion is SOTA in vision; language was left behind

Question: if you train a **diffusion language model** with the same data and compute discipline, do in-context learning, math, and chat **actually** require AR?

## The Idea

**Masked diffusion over tokens**: forward process randomly masks; reverse process predicts masked tokens with a shared Transformer. Optimize a **likelihood bound** (not teacher-forced next-token only). Pretrain + SFT as usual.

$$
\text{forward: corrupt with masks} \quad
\text{reverse: } p_\theta(x_{\mathrm{mask}} \mid x_{\mathrm{obs}}) \text{ in iterative unmasking steps}
$$

Multiple denoising steps can edit / fill non-left-to-right positions — that is the bet behind fixing reversal and flexible generation.

## Architecture / Method

- **Backbone:** standard Transformer encoder-style stack used to **predict masked tokens** (not a one-step AR softmax over next id only).
- **Training:** pretraining + supervised finetuning (same paradigm as ARMs).
- **Generation:** iterative unmasking (diffusion sampling); step count is a compute dial at inference.
- **ICL / chat:** prompted similarly to ARMs; SFT unlocks multi-turn instruction following in case studies.

## Key Results

| Axis | Result (paper) |
|---|---|
| Scalability | Competitive with **self-built ARM baselines** at scale |
| LLaDA **8B** | Competitive with **LLaMA3 8B** on in-context learning |
| Math / code / general | Strong across standard suites (vs their ARM controls) |
| After SFT | Convincing multi-turn dialogue case studies |
| **Reversal poem completion** | **Surpasses GPT-4o** (reversal curse mitigation) |
| Claim | Core LLM capabilities do not inherently require ARMs |

## Comparison to Prior Work

- vs **[[Transformer]] / ARMs (LLaMA, GPT)** — same-ish capacity story; different **generative process** (diffusion vs next-token).
- vs **[[GameNGen]] / [[DIAMOND]]** — diffusion in *world models / pixels*; LLaDA is diffusion for **text at LLM scale**.
- vs **BERT-style MLM** — LLaDA is **generative** diffusion with likelihood training + free-form sampling, not only encoding.
- vs **[[Byte Latent Transformer]] / [[MegaByte]]** — change *what* tokens are; LLaDA keeps tokens and changes *how* sequences are generated.

## Limitations

- Sampling is **multi-step** — latency/compute vs single-pass AR decode is a systems open question.
- “Competitive with LLaMA3 8B” is a point comparison; frontier ARMs keep moving.
- Reversal / editing wins do not mean every AR weakness disappears.
- Production serving stacks ([[Mooncake]], [[vLLM]]) are AR-centric; diffusion decode needs new kernels/schedulers.
- Case-study chat quality is not a full product eval.

## Why It Matters

This wiki’s architecture line usually ends at “attention vs SSM/linear recurrence.” LLaDA inserts a **third generative story**: *diffusion language models at 8B with standard pretrain+SFT*. It is the right note for the “is AR necessary?” debate and for readers coming from [[GameNGen]] / visual diffusion who ask why text stayed autoregressive.

## Related Concepts

[[Transformer]] · [[Attention]] · [[GameNGen]] · [[DIAMOND]] · [[BPE_Tokenization]] · [[MegaByte]] · [[BitNet b1.58]] · [[Supervised-Fine-Tuning]] · [[Qwen3 Technical Report]] · [[The Llama 3 Herd of Models]]