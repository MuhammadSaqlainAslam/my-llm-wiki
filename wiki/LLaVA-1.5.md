---
created: "2026-06-21"
title: "Improved Baselines with Visual Instruction Tuning"
authors: "Liu, Li, Li, Lee"
year: "2023"
arxiv: "2310.03744"
tags: [vision-language, multimodal, foundations, instruction-tuning]
tldr: "LLaVA-1.5 — simple modifications (MLP connector + academic VQA training data) to the original LLaVA recipe establish state-of-the-art open vision-language baselines across 11 benchmarks, trained in ~1 day on a single 8-A100 node"
citation_count: 5410
---

## TL;DR

LLaVA-1.5 shows that the original LLaVA's fully-connected vision-language connector is already surprisingly powerful and data-efficient — two simple changes (swapping the linear projection for an MLP, and adding academic-task VQA data with simple response-formatting prompts) push it to state-of-the-art across 11 benchmarks, using only 1.2M training examples and roughly one day of training on a single 8-A100 node.

---

## The Problem

The original LLaVA excelled at open-ended visual conversation but performed poorly on academic VQA benchmarks that expect short, specific answers — it tended to produce overly verbose responses or default to "yes" on yes/no questions, since its training data lacked short-answer examples.

---

## The Idea

Two changes, both orthogonal to LLaVA's core architecture:

1. **MLP connector** — replace the single linear projection layer between the vision encoder and LLM with a two-layer MLP, improving cross-modal representation capacity at negligible parameter cost
2. **Academic VQA data + response formatting** — incorporate benchmark-oriented VQA datasets (VQAv2, GQA, OCR-VQA, TextCaps, RefCOCO) into training, paired with an explicit response-formatting instruction ("Answer the question using a single word or phrase") appended to prompts that expect short answers

Combined with a higher-resolution CLIP-ViT-L-336px vision encoder (vs. 224px in the original), these changes — without any fundamentally new architecture — produce LLaVA-1.5.

The training recipe uses Vicuna-13B or LLaMA 2-13B as the language backbone, with the MLP connector and language model fine-tuned while the vision encoder remains frozen.

---

## Why It Matters

- Became the reference baseline that most subsequent open vision-language models are compared against — essentially the ImageNet moment for open-source VLMs
- Demonstrates that careful data curation and small connector changes can matter as much as architectural novelty — a recurring theme also seen in [[Chinchilla_Scaling_Laws|Chinchilla]]'s findings about data vs. parameter scaling for language-only models
- Extremely accessible compute requirements (~1 day, single 8-A100 node) relative to the capability gained, lowering the barrier to vision-language research
- The response-formatting insight (explicitly instructing for short answers on VQA tasks) is a simple but broadly applicable technique for controlling output style

---

## Limitations

- Still single-image, not natively suited for video or multi-image reasoning without further extension
- Performance gains are from careful engineering rather than a new architectural paradigm — the ceiling may require more substantial changes (e.g. higher-resolution tiling, interleaved image-text, video)

---

## Related Concepts

[[MiniGPT-4]] · [[LLaMA 2]] · [[Transformer]] · [[Chinchilla_Scaling_Laws|Chinchilla]]
