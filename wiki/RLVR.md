---
title: "RLVR — Reinforcement Learning with Verifiable Rewards"
tags: [post-training, rl, reasoning, nemotron, reward]
tldr: "Restrict RL rewards to ground-truth verifiable signals (correct answer, passing test suite) rather than a learned reward model. No reward hacking possible. Nemotron-3 runs 21 environments simultaneously with async GRPO."
theme: synthesis
---

# RLVR

Standard RLHF trains a **reward model** — a neural network that scores outputs based on human preference data — then optimizes the policy to maximize that score. The problem: the reward model is a neural network. It can be exploited. The policy finds patterns in the reward model's distribution that score high without corresponding to genuinely better responses. This is **reward hacking**, and it degrades real-world quality even as the reward model score climbs. RLVR restricts rewards to **verifiable, ground-truth signals** that cannot be gamed:

- **Math:** the final numeric answer either matches the reference or it doesn't
- **Code:** the submitted program either passes the test suite or it doesn't
- **Instruction following:** constraints are either satisfied (countable, checkable) or not
- **Formal logic:** the proof either verifies or it doesn't

There is no learned reward model to exploit. The policy must solve the actual problem. [[Nemotron-3]] trains **21 such environments simultaneously** in a single RL run using **Async GRPO** (Group Relative Policy Optimization with asynchronous rollout generation): 256 prompts per step, 16 rollout responses per prompt, effective batch 4096, max generation length 64K tokens. [[Multi-Token Prediction]] heads provide [[Speculative Decoding]] during rollout, making 64K-token generations fast enough to be practical. The 21 environments include: competitive math (AMC/AIME-level), competitive coding, software engineering (SWE-bench), instruction following, search, chat, agentic tool use, long context, economics, formal logic, and multiple-choice question answering. All environments train simultaneously — unlike prior NVIDIA models that staged separate RL runs per capability.

## Where it appears

- **[[Nemotron-3]]** — Stage 2 of the post-training pipeline; the core capability-building RL stage before SWE-RL and RLHF

## Why it matters

- **It eliminates reward hacking by construction.** A verifiable reward can't be exploited — the answer is right or wrong according to an external oracle. The policy improvement is real and generalization is genuine. This is why RLVR models consistently outperform RLHF-only models on held-out reasoning tasks even when RLHF scores are higher.
- **Simultaneous multi-environment training is crucial.** Training environments sequentially causes **catastrophic forgetting** — the capabilities from early stages degrade when you train later stages. Running all 21 simultaneously with a shared policy means each gradient step balances all environments, improving stability and generalization.
- **It defines the frontier of post-training.** RLVR (or close variants like DeepSeek's GRPO, OpenAI's RLVR for o1) is what separates models that "feel smart in chat" from models that actually solve hard problems. The limiting factor is not the RL algorithm — it's the quality and breadth of verifiable environments.

---

*Related: [[Nemotron-3]] · [[Multi-Token Prediction]] · [[Speculative Decoding]]*
