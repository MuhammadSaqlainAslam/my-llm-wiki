---
created: "2026-06-29"
title: "Training Agents Inside of Scalable World Models"
authors: "Hafner, Yan, Lillicrap"
year: "2025"
arxiv: "2509.24527"
tags: [world-models, reinforcement-learning, transformer, deepmind]
tldr: "Dreamer 4 is the first agent to obtain diamonds in Minecraft purely from offline data with zero environment interaction, using a block-causal transformer world model trained via a novel 'shortcut forcing' objective, with 100x less data than OpenAI's VPT"
citation_count: 118
---

## TL;DR

Dreamer 4 trains a 2B-parameter agent to solve the long-horizon "obtain diamonds" challenge in Minecraft — requiring over 20,000 sequential mouse/keyboard actions from raw pixels — purely from a fixed offline dataset, with zero environment interaction during training. It is the first agent to do this, substantially outperforming OpenAI's VPT offline agent despite using 100x less data, and runs the world model in real time on a single GPU.

---

## The Idea

Replaces [[DreamerV3]]'s recurrent state-space model with a block-causal Transformer trained via a novel "shortcut forcing" objective, enabling the world model to be trained almost entirely on large amounts of unlabeled gameplay video (2,541 hours from OpenAI's VPT Minecraft dataset), with only a small subset paired with actual action labels. The policy is then trained via reinforcement learning entirely inside this learned world model — the agent never interacts with the real Minecraft environment during training, only during final evaluation.

---

## Why It Matters

- The direct successor to [[DreamerV3]] from the same lead author (Hafner), extending the Dreamer lineage from recurrent to Transformer-based world modeling
- Strong validation of the "learn general knowledge from abundant unlabeled video, action-condition with little labeled data" pattern also central to [[V-JEPA 2]] — two independent lines of research (DeepMind's Dreamer lineage, Meta's JEPA lineage) converging on a similar data-efficiency strategy via different architectures
- Outperformed even general vision-language model finetuning baselines (reported against Gemma 3) on this specific behavioral-cloning-style task, a notable result for a purpose-built world-model approach vs a general VLM

---

## Limitations

- Demonstrated primarily on Minecraft — a rich but still bounded and well-instrumented game environment relative to open-ended real-world robotics
- The "shortcut forcing" objective and block-causal Transformer are specific architectural choices whose generality to substantially different domains (e.g. continuous robot control) is not directly established in this paper

---

## Related Concepts

*Lineage: [[World Models]] · [[DreamerV3]] · [[V-JEPA 2]]*

*Landscape: [[World-Model-Landscape|The World Model Landscape (2019-2026)]]*
