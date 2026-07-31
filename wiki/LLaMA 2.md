---
created: "2026-05-20"
title: "Llama 2: Open Foundation and Fine-Tuned Chat Models"
authors: "Hugo Touvron, Louis Martin, Kevin Stone et al. (Meta GenAI)"
year: 2023
arxiv: "2307.09288"
citation_count: 17760
tags: [language-model, rlhf, instruction-tuning, open-source, chat, safety, gqa, pretraining]
tldr: "Meta releases 7B–70B pretrained and RLHF-aligned chat models with full details on the alignment pipeline. Llama 2-Chat outperforms all public chat models and is competitive with closed models like ChatGPT on helpfulness. The technical report is one of the most detailed public accounts of the RLHF/RLAIF process."
aliases: [LLaMA 2, Llama 2, LLaMA-2, Llama-2-Chat]
theme: foundations
---

# LLaMA 2

> Hugo Touvron, Louis Martin, Kevin Stone et al. (Meta GenAI), "Llama 2: Open Foundation and Fine-Tuned Chat Models", arXiv:2307.09288 (2023)

## TL;DR

LLaMA 2 is the first fully open (weights + code + paper) large language model family that is:
1. Competitive with closed models (GPT-3.5, Claude-v1) on helpfulness
2. Thoroughly safety-aligned via a well-documented RLHF pipeline
3. Useful as a base for downstream finetuning (7B, 13B, 34B, 70B)

The technical paper is equally important as the models: it contains one of the most detailed public accounts of how to do RLHF at scale — from data collection through reward modeling through PPO to the "Ghost Attention" trick for multi-turn consistency. This made it a reference document for anyone building instruction-following LLMs.

---

## The Core Idea — Open RLHF at Scale

Before LLaMA 2, the production RLHF pipeline (reward model → PPO loop) was either unpublished (OpenAI) or only partially documented (Anthropic's Constitutional AI). Meta's contribution is unusual for a frontier lab: full disclosure of:

- Pretraining data composition and cleaning
- SFT (Supervised Fine-Tuning) data volume and quality bar
- Reward model architecture, training data, and inter-annotator agreement
- PPO reward shaping and KL penalty tuning
- Ghost Attention for multi-turn instruction following
- Safety red-teaming methodology and results

This transparency allows the community to reproduce and build on the work.

---

## Key Concepts

### Pretraining

- **2 trillion tokens** from publicly available sources (web, books, code — no proprietary data)
- Context length extended from LLaMA-1's 2K to **4K tokens**
- **[[GQA]] (Grouped Query Attention)** — the 34B and 70B models use GQA (8 KV heads shared across 64 query heads), dramatically reducing KV cache size during inference. The 7B and 13B models use multi-head attention.
- **[[RoPE]]** (Rotary Position Embedding) — used for position encoding, enabling length extrapolation
- Architecture: GPT-style decoder-only transformer (same as LLaMA-1), with pre-normalization (RMSNorm) and SwiGLU activations

### Fine-tuning

**Stage 1: SFT (Supervised Fine-Tuning)**
- Human-written instruction-response pairs (~27,500 pairs for initial SFT)
- Quality over quantity: fewer high-quality examples beat large low-quality datasets
- SFT data annotation focuses on responses that are accurate, helpful, and concise

**Stage 2: RLHF**

*Reward model training:*
- Human annotators compare two model responses and select the preferred one
- Two separate reward models trained: one for **helpfulness**, one for **safety**
- 1M+ human preference comparisons; inter-annotator agreement ~70%

*PPO loop:*
- Policy = finetuned LLM; reward = helpfulness RM minus KL penalty from reference model
- Safety reward model provides a "safety shield": heavily negative rewards for harmful outputs
- **RLHF context length**: 4096 tokens

*Ghost Attention:*
- Problem: after many turns of conversation, the model forgets instructions given at the start (e.g., "always respond in French")
- Solution: during training, append the system message to every turn in the conversation, then mask those tokens in the loss. The model learns to persist instructions without seeing them explicitly at inference time.

---

## Architecture / Method

| Model | Params | Attention heads | KV heads | Context | Architecture |
|---|---|---|---|---|---|
| Llama 2-7B | 7B | 32 | 32 (MHA) | 4K | Standard MHA |
| Llama 2-13B | 13B | 40 | 40 (MHA) | 4K | Standard MHA |
| Llama 2-34B | 34B | 64 | 8 ([[GQA]]) | 4K | GQA |
| Llama 2-70B | 70B | 64 | 8 ([[GQA]]) | 4K | GQA |

All models use:
- Pre-norm (RMSNorm instead of LayerNorm)
- SwiGLU activation (gated MLP)
- Rotary positional embeddings (RoPE)
- No bias terms in linear layers

---

## Key Results

**Pretraining:**
- Llama 2-70B outperforms GPT-3 (175B) on most benchmarks
- Competitive with PaLM-2-M and Falcon-40B on coding and reasoning

**Llama 2-Chat (after RLHF):**
- Human evaluation: preferred over ChatGPT on helpfulness in head-to-head comparisons (55% win rate)
- Safety: preferred over ChatGPT on safety in human evaluations
- GPT-4 evaluation: Llama 2-Chat-70B is competitive with ChatGPT across diverse tasks

**GQA ablation:**
- 70B with GQA (8 KV heads) achieves equivalent quality to 70B with full MHA
- Inference memory for KV cache: **8× smaller** vs. MHA (8 vs. 64 KV heads)

---

## Comparison to Prior Work

- vs. **LLaMA-1** — Llama 2 trains on 40% more data (2T vs. 1.4T tokens), doubles context length (4K vs. 2K), and adds the full RLHF pipeline. Llama 2-70B is strongly better than Llama-1-65B.
- vs. **GPT-3.5 / ChatGPT** — Llama 2-Chat is competitive in human evaluations of helpfulness. Safety is comparable (by Meta's methodology, slightly better on safety benchmarks).
- vs. **Falcon-40B** — Llama 2-70B outperforms Falcon-40B on most benchmarks despite being a similar parameter count; Falcon used a larger fraction of web data.
- vs. **[[Mixtral]] / Mistral** — Mistral-7B (Sep 2023, published after Llama 2) uses sliding window attention and GQA at 7B scale, matching Llama 2-13B. The Llama 2 architecture became the base for most subsequent 7B community models.

---

## Limitations

- **Context length is 4K.** By 2024 standards (Claude: 200K, GPT-4 Turbo: 128K), 4K is short. Llama 2 was extended by the community via RoPE scaling tricks.
- **No code specialization in the base model.** Code Llama was a separate release.
- **RLHF safety is imperfect.** The model can be jailbroken; adversarial prompts can elicit harmful content. Meta publishes the red-teaming results honestly.
- **Ghost Attention is a training trick, not a structural solution.** Long conversations can still drift from instructions.

---

## Why It Matters

- **It democratized RLHF.** Before Llama 2, the detailed RLHF recipe was a trade secret of OpenAI and Anthropic. Llama 2's paper gave the community a working blueprint.
- **GQA became the standard for large models.** Virtually every major open model after LLaMA 2 (Mistral, LLaMA 3, Gemma, Griffin) uses GQA. The paper's GQA ablation is the canonical justification.
- **The weights enabled the open-source ecosystem.** Llama 2 checkpoints became the base for thousands of community finetunes (Alpaca, Vicuna, WizardLM, Code Llama, etc.). Its architecture is now more standard than GPT-2 once was.
- **It set the benchmark for safety disclosure.** The red-teaming methodology and dual reward model (helpfulness + safety) are the template for subsequent safety-aligned models.

---

## Related Notes

[[Transformer]] · [[GQA]] · [[RLHF]] · [[GRPO]] · [[Griffin]] · [[EAGLE]] · [[Medusa]] · [[Speculative Decoding]] · [[KV Cache]] · [[Mixture-of-Experts]]
