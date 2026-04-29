---
title: "Group Relative Policy Optimization (GRPO)"
tags: [rl, post-training, optimization, policy-gradient, deepseek]
aliases: [GRPO, Group Relative Policy Optimization]
tldr: "A PPO variant that eliminates the value function by estimating advantages from a group of G rollouts for the same prompt. The reward of each rollout is normalized relative to the group mean and std — no critic network needed. Cheaper, more stable, and better suited to LLM fine-tuning than standard PPO."
theme: efficiency
---

# Group Relative Policy Optimization (GRPO)

> Used in [[DeepSeek_V4]] and earlier DeepSeek models, 2024–2026.

## Why Not Just Use PPO?

Proximal Policy Optimization (PPO) is the standard RL algorithm for LLM fine-tuning. It estimates the **advantage** of each action — how much better was this response than what the value function predicted? — using a separately-trained critic network. The critic has roughly the same size as the policy (i.e., another full LLM), which means:

- **2× memory** — you need both the policy and the critic loaded simultaneously
- **Critic training instability** — the critic lags the policy; early in training its estimates are wrong, which produces noisy gradient signal
- **Long rollouts are hard** — credit assignment over many tokens requires a good value function, which is hard to train

GRPO sidesteps the critic entirely.

---

## The Core Idea: Groups as Their Own Baseline

For each prompt $q$, generate $G$ independent rollouts $\{o_1, o_2, \ldots, o_G\}$ from the current policy. Score each with a reward function $r_i = R(q, o_i)$. Compute the advantage of rollout $i$ *relative to the group*:

$$\hat{A}_i = \frac{r_i - \text{mean}(\{r_1,\ldots,r_G\})}{\text{std}(\{r_1,\ldots,r_G\})}$$

This normalized advantage says: *was this response better or worse than the policy's average response to this prompt?* No critic needed — the group of responses is its own baseline.

The policy gradient loss is then:

$$\mathcal{L}_\text{GRPO} = -\mathbb{E} \left[ \sum_{t} \min\!\left( \frac{\pi_\theta(o_t|q,o_{<t})}{\pi_\text{ref}(o_t|q,o_{<t})} \hat{A}_i,\ \text{clip}(\cdot, 1\pm\epsilon) \hat{A}_i \right) - \beta \cdot \text{KL}(\pi_\theta \| \pi_\text{ref}) \right]$$

The clip and KL penalty are inherited from PPO — they prevent the policy from moving too far from the reference model in one update.

---

## Key Design Choices

| Parameter | Role |
|---|---|
| $G$ | Group size — more rollouts = more stable advantage estimate; typical $G = 8$–$16$ |
| $\epsilon$ | PPO clip threshold — limits how much the policy changes per step |
| $\beta$ | KL penalty weight — keeps the policy near the reference |
| $R(q, o)$ | Reward function — domain-specific (rule-based for math/code, model-based for instruction following) |

---

## Why It Works for LLMs

LLM responses are long sequences. Assigning credit to individual tokens is hard — most of the value is in the final answer. GRPO sidesteps this with **outcome-level rewards**: score the full response, then assign that score uniformly to all tokens in the response. Crude, but it works when the reward is reliable (e.g., checking a math answer against a verified solution).

The group normalization automatically handles reward scale — you don't need to carefully calibrate absolute reward magnitudes, only that the reward is *informative* (some responses to a prompt are better than others).

---

## GRPO in DeepSeek-V4's Post-Training

In [[DeepSeek_V4]]'s two-stage post-training:

1. **Stage 1 (Specialist Training):** Each domain expert is trained with SFT + GRPO. The reward model is domain-specific: exact match for math, test execution for code, rubric-based scoring for instruction following.
2. **Stage 2:** [[On-Policy Distillation]] unifies the specialists — GRPO is not used in stage 2.

---

## Where It Appears

- **[[DeepSeek_V4]]** — stage 1 specialist RL for math, code, agent, and instruction-following domains
- **[[RLVR]]** — the broader post-training paradigm that GRPO enables; GRPO is the RL algorithm underlying RLVR-style training

---

## Related Concepts

- [[RLVR]] — RL with verifiable rewards; GRPO is the optimizer, RLVR is the paradigm
- [[On-Policy Distillation]] — stage 2 of DeepSeek-V4 post-training; consolidates GRPO-trained specialists
- [[Multi-Token Prediction]] — complementary training signal during pre-training
- [[DeepSeek_V4]] — the model whose post-training pipeline uses GRPO at scale
