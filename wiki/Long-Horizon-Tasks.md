---
title: "On Training Large Language Models for Long-Horizon Tasks: An Empirical Study of Horizon Length"
authors: "Sunghwan Kim, Junhee Cho, Beong-woo Kwak, Taeyoon Kwon, Liang Wang, Nan Yang, Xingxing Zhang, Furu Wei, Jinyoung Yeo"
year: "2026"
arxiv: "2605.02572"
tags: [reasoning, reinforcement-learning, agentic, training, long-context]
tldr: "Systematic empirical study showing that increasing task horizon length alone is a training bottleneck for LLM agents — horizon reduction stabilizes RL training and yields horizon generalization to longer horizons at inference time. ICML 2026."
citation_count: 0
created: "2026-09-29"
---

## TL;DR
Constructs controlled tasks where agents face identical decision rules and reasoning structure but differ only in the length of action sequence required for success. Finds that increasing horizon length alone causes severe training instability, driven by exploration difficulties and credit assignment challenges. The key remedy is **horizon reduction** — e.g., macro actions that compose several atomic actions into one step — which stabilizes RL and improves performance on long-horizon tasks. Models trained under reduced horizons also generalize better to longer-horizon variants at inference time (**horizon generalization**). Accepted to ICML 2026.

## The Problem
LLM agents are increasingly deployed on long-horizon tasks requiring extended sequences of environment interactions. Prior work has focused on system-level optimizations or algorithmic improvements, but the role of horizon length in shaping training dynamics remains poorly understood. Is a longer-horizon task simply a harder task, or does it introduce qualitatively different training challenges?

## The Idea
Isolate horizon length as an independent variable by constructing controlled tasks — **Sudoku** (fill a 9×9 grid cell by cell) and **Rush Hour** (sliding-block puzzle) — where reasoning complexity is held constant but the number of actions needed varies. Train with RL under atomic actions (long horizon) versus macro actions (horizon-reduced equivalent: several cell fills per step in Sudoku, multi-cell moves in Rush Hour) and measure training stability and performance separately. The primary model trained is Qwen3-1.7B; findings are further checked on a larger model and validated on **WebShop**, a web-interaction benchmark with natural-language observations.

Key finding: horizon length is not just a proxy for task difficulty — it is an independent training bottleneck. A task that is learnable at short horizon becomes unlearnable at long horizon because of exploration difficulty and sparse credit signals, not increased reasoning complexity.

## Why It Matters
- Gives a principled empirical basis for why long-horizon agentic RL is hard — not just "tasks are complex" but specifically how horizon length degrades exploration and credit assignment.
- Horizon generalization has practical implications: train on structured short-horizon curricula, deploy on longer tasks.
- Connects to [[Multi-Environment_RLVR_Training|Multi-Environment RLVR Training]] (multi-environment agentic RL at scale) and [[VinePPO]] (credit-assignment focus).
- ICML 2026 acceptance provides an independent peer-review signal.

## Limitations
- The controlled studies use two puzzle domains (Sudoku, Rush Hour) with WebShop as external validation — generalization to more diverse agentic tasks (coding, scientific reasoning) is not directly demonstrated.
- Horizon reduction via macro actions requires domain knowledge about what constitutes a meaningful macro action — may not be straightforward in open-ended domains.

## Related Concepts
[[Multi-Environment_RLVR_Training|Multi-Environment RLVR Training]] · [[VinePPO]] · [[DeepSeek-R1 Incentivizing Reasoning Capability in LLMs via Reinforcement Learning|DeepSeek-R1]] · [[EAPO]] · [[Kimi-k1.5|Kimi k1.5]] · [[RLVR]]
