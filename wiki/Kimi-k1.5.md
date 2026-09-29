---
title: "Kimi k1.5: Scaling Reinforcement Learning with LLMs"
authors: "Kimi Team (Moonshot AI)"
year: "2025"
arxiv: "2501.12599"
tags: [reasoning, reinforcement-learning, rlvr, multimodal, frontier-models]
tldr: "Multi-modal LLM trained with RL using two key ingredients — long-context scaling (RL context window to 128k) and improved policy optimization — with no MCTS, value functions, or process reward models. Long-CoT matches o1 (77.5 AIME, 96.2 MATH500); long2short distillation gives SOTA short-CoT (60.8 AIME, 94.6 MATH500, 47.3 LiveCodeBench), far ahead of GPT-4o and Claude Sonnet 3.5."
citation_count: 0
created: "2026-09-29"
---

## TL;DR
Kimi k1.5 treats RL as a new axis for scaling LLM intelligence beyond the limits of pretraining data. Its two key ingredients are long-context scaling (extending the RL context window to 128k tokens so the model can learn from much longer reasoning trajectories) and improved policy optimization, giving a simple, effective RL framework that avoids Monte Carlo tree search, value functions, and process reward models.

Two headline results:
- **Long-CoT:** 77.5 on AIME, 96.2 on MATH500, 94th percentile on Codeforces, 74.9 on MathVista — matching OpenAI's o1.
- **Short-CoT (via long2short methods that use long-CoT techniques to improve short-CoT models):** 60.8 on AIME, 94.6 on MATH500, 47.3 on LiveCodeBench — outperforming GPT-4o and Claude Sonnet 3.5 by a large margin.

Technical report from Moonshot AI (makers of [[Kimi-K2|Kimi K2]]).

## The Problem
Pretraining with next-token prediction is limited by the amount of available high-quality data. Once the internet's useful text has been consumed, pretraining scaling yields diminishing returns. RL offers a different axis: the model can generate its own training signal by exploring with rewards. But prior published work had not produced competitive results with this approach.

## The Idea
Two ingredients enable effective RL scaling. **Long-context scaling** extends the RL context window to 128k tokens, allowing the model to generate and learn from far longer reasoning trajectories. **Improved policy optimization** yields a simpler, more stable pipeline than MCTS-style approaches that nonetheless reaches strong results at scale.

Multi-modal RL training is also included — the model learns from both text and image inputs with verifiable rewards, making k1.5 one of the first frontier models to demonstrate RL reasoning improvements in the multi-modal setting. The report also covers multi-modal data recipes and infrastructure optimization.

## Why It Matters
- One of the first detailed public accounts of how to scale RL training for LLM reasoning with long-context trajectories — released the same week as [[DeepSeek-R1 Incentivizing Reasoning Capability in LLMs via Reinforcement Learning|DeepSeek-R1]] and an influence on subsequent RLVR work.
- The "RL as a new scaling axis" framing became a dominant narrative of early-to-mid 2025 LLM research, alongside DeepSeek-R1 and [[s1-Simple-Test-Time-Scaling|s1]].
- Establishes Moonshot AI as a serious player in frontier reasoning-model development, leading to [[Kimi-K2|Kimi K2]].

## Limitations
- A technical report, not a full recipe; model weights were not released with k1.5.
- The multi-modal RL improvements are demonstrated but less thoroughly ablated than the text-only results.
- The headline "beats GPT-4o and Claude Sonnet 3.5" comparison is for the short-CoT (long2short) model; the long-CoT model is the one that matches o1.

## Related Concepts
[[DeepSeek-R1 Incentivizing Reasoning Capability in LLMs via Reinforcement Learning|DeepSeek-R1]] · [[GRPO]] · [[RLVR]] · [[Multi-Environment_RLVR_Training|Multi-Environment RLVR Training]] · [[Kimi-K2|Kimi K2]] · [[s1-Simple-Test-Time-Scaling|s1]] · [[VinePPO]] · [[Absolute-Zero|Absolute Zero]]
