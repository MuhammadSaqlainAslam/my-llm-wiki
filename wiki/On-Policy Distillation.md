---
title: "On-Policy Distillation (OPD)"
tags: [post-training, distillation, rl, knowledge-distillation, deepseek]
aliases: [OPD, On-Policy Distillation, on-policy distillation]
tldr: "Train a single student model by having it generate its own rollouts (on-policy), then minimize reverse KL divergence against an ensemble of specialist teacher models. The student learns to match the best expert in each domain without keeping multiple deployed models. Used as the second stage of DeepSeek-V4's post-training pipeline, after independent specialist RL."
theme: efficiency
---

# On-Policy Distillation (OPD)

> Used in [[DeepSeek_V4]], 2026. Based on Lu and Lab, 2025.

## The Problem: You Have Great Specialists, Now What?

[[RLVR|Reinforcement learning]] fine-tuning works well when focused on a single domain — math, code, agent tasks, instruction following. Trying to run RL on all domains simultaneously causes **reward conflicts**: improvements on one task degrade another. The clean solution is to train separate specialists.

But you can't deploy six models. You need one.

**Naive merging** (model soups, weight averaging) works to some degree but loses the sharp peaks of each specialist — the rare but important behaviors that RL carved out. **Offline distillation** (train student on teacher outputs from a static dataset) is better, but the teacher outputs were generated for a fixed distribution that may not match what the student actually produces.

OPD solves both problems: the student generates its own prompts' completions and learns to match the teachers on *those*, not on a pre-collected static set.

---

## How It Works

### Setup

After stage 1 (specialist training), you have:
- A **base model** (post-SFT, pre-RL)
- $K$ **specialist teacher models** $\{T_1, \ldots, T_K\}$, each expert in one domain (math, code, agent, instruction following, …)
- The goal: one **student model** $S$ that matches all of them

### The On-Policy Loop

For each training batch:

1. **Student rollout:** $S$ generates completions for a batch of prompts (drawn from a mix of domain data)
2. **Teacher scoring:** each teacher $T_k$ scores the student's own outputs by computing $\log T_k(\text{student completion} \mid \text{prompt})$
3. **Reverse KL loss:** the student minimizes:

$$\mathcal{L}_\text{OPD} = \mathbb{E}_{x \sim S} \left[ \log S(x) - \log T_{\text{best}}(x) \right]$$

The reverse KL $\text{KL}(S \| T)$ is **mode-seeking**: $S$ concentrates on high-probability regions of $T$. This means the student learns the teachers' most reliable behaviors first, without spreading probability mass into low-confidence tail regions.

### Why On-Policy Matters

In **offline** distillation, the teacher labels a fixed dataset. The student is never asked about prompts it would naturally generate. Distribution shift accumulates: the student's errors compound because it was never trained on its own mistakes.

In **on-policy** distillation, every training example came from the student itself. There is no distribution shift between training and inference. The student sees exactly the inputs it will encounter at deployment.

---

## Tradeoffs

| Property | On-Policy Distillation | Offline Distillation | Naive Merging |
|---|---|---|---|
| Distribution shift | None | Accumulates | N/A |
| Teacher compute | Required at train time | Required offline only | None |
| Preserves specialist peaks | Yes (mode-seeking) | Partially | Averages them out |
| Tail coverage | Lower (reverse KL) | Higher (forward KL) | Blended |

The reverse KL's mode-seeking behavior is both a feature and a limitation: the student may underrepresent rare capabilities that live in the teachers' tails.

---

## Where It Appears

- **[[DeepSeek_V4]]** — second stage of the post-training pipeline; integrates $K$ domain specialists into one unified model via on-policy reverse KL distillation

---

## Related Concepts

- [[GRPO]] — the RL algorithm used in stage 1 (specialist training) before OPD
- [[RLVR]] — broader context for RL-based post-training; OPD is the consolidation step after RLVR-style specialist training
- [[Multi-Token Prediction]] — another post-training signal; complementary to OPD
- [[DeepSeek_V4]] — the model whose post-training pipeline introduced OPD at this scale
