---
created: "2026-06-24"
title: "Ministral 3"
authors: "Liu, Khandelwal, Subramanian, Jouault, et al. (Mistral AI)"
year: "2026"
arxiv: "2601.08584"
tags: [model-family, distillation, efficiency, fine-tuning-alignment, mistral]
tldr: "A family of 3B/8B/14B dense models derived from Mistral Small 3.1 via Cascade Distillation — iterative prune-distill-repeat — each released in base, instruct, and reasoning variants, all with vision support"
citation_count: 54
---

## TL;DR

Ministral 3 is a family of nine dense models (3B, 8B, 14B, each in base/instruct/reasoning variants) derived from a single 24B parent model, Mistral Small 3.1, via **Cascade Distillation** — an iterative prune-then-distill pipeline applied repeatedly to shrink the model in stages. Each child model trains on only 1–3 trillion tokens, far less than training Qwen3 or LLaMA 3 from scratch, while remaining competitive with same-size Gemma 3 and Qwen 3 models. All nine models include vision support and are released under Apache 2.0.

---

## The Idea

**Cascade Distillation** prunes the parent model down to a target size (via layer pruning, PCA-based hidden-dimension pruning, and feedforward-dimension pruning informed by SwiGLU activation importance), then continues training the pruned model with logit distillation from the original parent as teacher — first at short context, then extended via YaRN to 256K tokens. The newly distilled model is then itself pruned again to seed the next, smaller child, so the whole family is produced in one pass through the data rather than three separate training runs.

Post-training mirrors [[RLHF|InstructGPT / RLHF]]'s SFT-then-preference-optimization pattern, using Online [[Direct Preference Optimization Your Language Model is Secretly a Reward Model|DPO]] (ODPO) rather than offline DPO. Reasoning variants add a [[GRPO]] stage between SFT and ODPO, following [[DeepSeek-R1 Incentivizing Reasoning Capability in LLMs via Reinforcement Learning|DeepSeek-R1]]'s RL-for-reasoning approach.

---

## Key Empirical Findings

Three findings, independently confirming prior distillation literature, are likely the most reusable part of this paper beyond the models themselves:

- **Capacity gap**: a stronger teacher does not produce a better student during pretraining-stage distillation — counter-intuitively, the smaller Mistral Small 3.1 outperformed the larger Mistral Medium 3 as a pretraining teacher. Post-training, however, benefits from the stronger teacher.
- Distilling from a **post-trained** teacher (rather than a base/pretrained one) during pretraining produces a stronger student, with the largest effect on math and code.
- Distilling from a **preference-optimized** teacher beats distilling from one only tuned with SFT — and this advantage persists even after the student undergoes its own preference tuning.

---

## Why It Matters

- A direct complement to [[Chinchilla_Scaling_Laws|Chinchilla]] and [[Scaling_Laws|Scaling Laws]]: those papers address how to allocate compute when training from scratch; this paper addresses how to get a family of smaller models cheaply when a larger one already exists
- The capacity-gap finding is a useful caveat for anyone planning a distillation pipeline — bigger teacher is not automatically better during pretraining
- Concrete real-world use of Online DPO, giving [[Direct Preference Optimization Your Language Model is Secretly a Reward Model|DPO]] a citable downstream variant in this wiki

---

## Limitations

- All comparisons are against same-size open models (Gemma 3, Qwen 3) using Mistral's own evaluation harness — not independently verified
- 3B reasoning variant showed weaker gains from ODPO than 14B/8B, and was more sensitive to fine-tuning hyperparameters generally

---

## Related Concepts

*Distillation: [[Chinchilla_Scaling_Laws|Chinchilla]] · [[Scaling_Laws|Scaling Laws]]*

*Post-training: [[Direct Preference Optimization Your Language Model is Secretly a Reward Model|DPO]] · [[RLHF|InstructGPT / RLHF]] · [[GRPO]] · [[DeepSeek-R1 Incentivizing Reasoning Capability in LLMs via Reinforcement Learning|DeepSeek-R1]]*

*Efficiency: [[LoRA Low-Rank Adaptation of Large Language Models|LoRA]]*
