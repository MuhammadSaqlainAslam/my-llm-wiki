---
created: "2026-05-20"
title: "RLHF (Reinforcement Learning from Human Feedback)"
authors: "Ouyang et al."
tags: [glossary, training, alignment, fine-tuning, llama2, reward-model]
arxiv: "2203.02155"
citation_count: 22920
tldr: "A fine-tuning pipeline that trains a reward model from human preference comparisons, then uses PPO to optimize the language model's policy toward higher rewards. The standard method for making LLMs helpful, harmless, and honest — used in InstructGPT, Claude, ChatGPT, and LLaMA 2-Chat."
aliases: [RLHF, reinforcement learning from human feedback, RLHF fine-tuning, "Reinforcement-Learning-from-Human-Feedback"]
---

## TL;DR

RLHF trains a language model to produce outputs that humans prefer, without needing explicit labels. Human annotators compare pairs of model outputs and select the better one. A **reward model** is trained to predict these preferences. Then the language model is fine-tuned using **PPO** (Proximal Policy Optimization) to maximize the reward model's scores — while a KL penalty keeps it from drifting too far from the original model.

## Intuition

Three stages:

**Stage 1: Supervised Fine-Tuning (SFT).** Start with a pretrained LLM. Fine-tune it on high-quality human-written demonstrations of desired behavior (instruction following, helpful answers). This sets the starting point for the RL loop.

**Stage 2: Reward Model (RM) Training.** For each prompt, show the SFT model to generate multiple responses. Human annotators rank the responses (or pick the better of two). Train a regression model (typically a smaller LLM with a linear head) to predict the human preference score.

**Stage 3: PPO Fine-Tuning.** Use the reward model as a reward signal and run PPO on the SFT model. The model generates responses, the reward model scores them, PPO updates the model parameters to increase expected reward. The KL penalty: $\text{total reward} = r_\text{RM} - \beta \log(\pi_\text{new}(y|x) / \pi_\text{ref}(y|x))$ prevents the model from generating gibberish that fools the reward model.

[[LLaMA 2]] uses two separate reward models — one for **helpfulness**, one for **safety** — and runs a modified PPO loop that jointly optimizes both.

## Why It Matters

- **It's the standard for making LLMs safe and helpful.** All major commercial assistants (ChatGPT, Claude, Gemini, Llama-2-Chat) use some form of RLHF.
- **The reward model is the bottleneck.** If the RM is biased or wrong, the resulting model learns to exploit the bias rather than genuinely improve.
- **[[GRPO]] and [[RLVR]] are successors** — GRPO removes the value function to reduce complexity; RLVR uses verifiable task rewards instead of a learned RM.

## Where It Appears in This Wiki

- [[LLaMA 2]] — detailed public account of the RLHF pipeline: data collection, dual RM (helpfulness + safety), PPO loop, Ghost Attention
- [[GRPO]] — DeepSeek's simplification of PPO for language model training
- [[FLAN]] — the earlier, purely-supervised instruction-tuning stage that RLHF pipelines like InstructGPT build a reward model and PPO on top of
- [[Self-Instruct]] — shows the supervised instruction-tuning half of the pipeline can be bootstrapped from the model itself instead of human annotators, though it produces no preference/reward signal to replace the RLHF stage

## Key Formula or Pseudocode

```
Reward: r(x, y) = r_RM(x, y) - β · KL(π_new(·|x) || π_ref(·|x))

PPO update: maximize E[r(x, y)] subject to KL constraint
  π_new = argmax E_π[r(x,y)] s.t. KL(π||π_ref) ≤ δ

Reward model loss: L_RM = -log(σ(r_θ(x, y_w) - r_θ(x, y_l)))
  y_w = preferred response, y_l = rejected response
```

## Related Concepts

[[LLaMA 2]] · [[GRPO]] · [[RLVR]] · [[Ghost Attention]] · [[Transformer]]
