---
created: "2026-10-19"
title: "ORPO & KTO: Preference Optimization without a Reference Model (or without Pairs)"
authors: "ORPO: Jiwoo Hong, Noah Lee, James Thorne · KTO: Kawin Ethayarajh, Winnie Xu, Niklas Muennighoff, Dan Jurafsky, Douwe Kiela"
year: 2024
arxiv: "2403.07691"
technical_report: "KTO arXiv:2402.01306"
citation_count: 0
tags: [rlhf, post-training, alignment, training, reward, dpo, fine-tuning, preference]
tldr: "Two ways past DPO's reference model: **ORPO** folds preference into SFT with an odds-ratio penalty (no second phase) — strong UltraFeedback results on Phi-2/Llama-2/Mistral 7B. **KTO** maximizes prospect-theory utility from *binary* good/bad labels (no preference pairs) and matches preference methods 1B–30B. ICML 2024 (KTO); ORPO preprint + widely used."
aliases: [ORPO, KTO, Odds Ratio Preference Optimization, Kahneman-Tversky Optimization, reference-free preference optimization]
theme: training
---

# ORPO & KTO (preference optimization past DPO)

> **ORPO** — Hong, Lee, Thorne, "ORPO: Monolithic Preference Optimization without Reference Model" (arXiv:2403.07691)  
> **KTO** — Ethayarajh et al., "KTO: Model Alignment as Prospect Theoretic Optimization", ICML 2024 (arXiv:2402.01306)

## The Problem / Motivation

[[Direct Preference Optimization Your Language Model is Secretly a Reward Model|DPO]] needs **preference pairs** $(y_w, y_l)$ and a **reference model**. [[SimPO]] drops the reference via average logprob. Two other hard constraints stay:

1. **Two-phase recipes** — SFT then preference stage is heavy; SFT already carries most of the alignment if the loss is right (**ORPO**).
2. **Pairwise data** — many pipelines only have **binary** “this reply is good/bad,” not ranked pairs (**KTO**).

Also: *why* DPO ≻ plain CE is not fully “it’s a better likelihood” — it is a **human-aware loss (HALO)** shaped like human risk attitudes (KTO’s frame).

## ORPO — odds ratio *in* SFT

### Idea
One **monolithic** stage: SFT on chosen tokens **plus** a penalty on disfavored style using an **odds-ratio** contrast (log-odds of $y_w$ vs $y_l$). **No reference model**, **no separate preference phase**.

Minor penalty on the bad style is enough to steer preference-aligned SFT (their claim).

### Headline numbers (UltraFeedback)
- Phi-2 (2.7B), Llama-2 (7B), Mistral (7B) with ORPO **surpass** larger SOTA-style models (>7B / 13B class)
- Up to **12.20%** AlpacaEval 2.0 (their Fig. 1 scale), **66.19%** IFEval (loose), **7.32** MT-Bench
- Released Mistral-ORPO-α/β 7B checkpoints

## KTO — prospect theory utility, binary labels

### Idea
Align with **Kahneman–Tversky** utility (loss aversion, etc.). Maximize **utility of generations**, not log-likelihood of preference order. HALO family.

Learns from **desirable / undesirable** only — no $(y_w,y_l)$ pairs.

### Headline numbers
- **Matches or exceeds** preference-based methods at **1B–30B**
- Works with **binary** feedback (thumbs up/down style)
- Paper’s broader point: **no universally best HALO** — pick the inductive bias for the data you have

## How they sit next to SimPO / DPO

| Method | Reference model | Needs pairs? | Extra idea |
|---|---|---|---|
| [[RLHF\|PPO RLHF]] | reward model | rankings | online RL |
| **DPO** | yes | yes | BT on log-ratio |
| **[[SimPO]]** | no | yes | avg logprob + margin |
| **ORPO** | no | yes | odds-ratio **inside SFT** (monolithic) |
| **KTO** | often via HALO framing | **no** (binary) | prospect-theory utility |

## Comparison to Prior Work

- vs **DPO** — ORPO/KTO remove the **reference tax** (and ORPO merges stages; KTO drops pairs).
- vs **[[SimPO]]** — same post-DPO family; SimPO stays pairwise with length-avg reward + $\gamma$ margin.
- vs **[[RLHF]] / [[RLVR]]** — no online rollouts / RM; still offline labels. RLVR uses verifiable rewards when you have them.
- vs pure **SFT** ([[Supervised-Fine-Tuning]]) — ORPO says SFT + a small odds penalty replaces the second stage.

## Limitations

- ORPO UltraFeedback / AlpacaEval numbers are **setup-specific**; later baselines move.
- Binary labels (KTO) are cheaper but **noisier** than true pairs; bad labels still poison policy.
- Odds-ratio / prospect utilities are proxies — length, style, and sycophancy need data hygiene.
- Not a substitute for **online** RL when you have verifiable rewards and a large budget ([[DeepSeek-R1 Incentivizing Reasoning Capability in LLMs via Reinforcement Learning|R1]]-style).
- “No one best HALO” (KTO) means **ablate** on your distribution; don’t cargo-cult.

## Why It Matters

This wiki’s post-training map is **RLHF → DPO → simpler objectives**. ORPO and KTO are the two well-cited answers to “what if we cannot afford a reference or ranked pairs?” — with SimPO the clean pairwise no-ref option. Together they document the **HALO zoo** practical teams actually ship.

## Related Concepts

[[RLHF]] · [[Direct Preference Optimization Your Language Model is Secretly a Reward Model|DPO]] · [[SimPO]] · [[InstructGPT]] · [[Supervised-Fine-Tuning]] · [[RLVR]] · [[GRPO]] · [[Constitutional-AI]] · [[On-Policy Distillation]] · [[DeepSeek-R1 Incentivizing Reasoning Capability in LLMs via Reinforcement Learning|DeepSeek-R1]] · [[Qwen3 Technical Report]]