---
created: "2026-06-29"
title: "Mastering Diverse Domains through World Models"
authors: "Hafner, Pasukonis, Ba, Lillicrap"
year: "2023"
arxiv: "2301.04104"
tags: [world-models, reinforcement-learning, foundations]
tldr: "A single algorithm and configuration that outperforms specialized RL methods across 150+ diverse tasks by learning a world model and imagining future scenarios; first to collect diamonds in Minecraft from scratch without human data"
citation_count: 1206
---

## TL;DR

DreamerV3 is a general reinforcement learning algorithm that learns a model of its environment and improves behavior by imagining future scenarios within that model, using a single fixed configuration across over 150 diverse tasks spanning continuous and discrete actions, visual and proprioceptive inputs, and dense and sparse rewards. It is the first algorithm to collect diamonds in Minecraft from raw pixels without human demonstration data — a long-standing challenge requiring over 20,000 sequential actions.

---

## The Idea

Builds directly on the V-M-C lineage from [[World Models]], replacing the original's VAE and LSTM-MDN with a Recurrent State-Space Model (RSSM) and applying a set of robustness techniques — normalization, balancing, and transformations of inputs and targets — that allow the same hyperparameters to work unmodified across domains as different as robot locomotion, Atari, and Minecraft. The agent learns entirely by imagining trajectories inside its learned world model and training an actor-critic pair on those imagined rollouts, rather than learning directly from real environment interaction.

---

## Why It Matters

- The third generation of the Dreamer lineage (DreamerV1 → DreamerV2 → DreamerV3), each step removing more domain-specific tuning from [[World Models]]'s original architecture
- Demonstrates that a model-based, "imagination" approach can match or beat specialized model-free methods across a genuinely broad task spectrum, not just the narrow domains earlier world models were validated on
- Published in *Nature* as well as arXiv — an unusually strong independent validation signal for an RL paper
- Appears in the BVP World Model Landscape (2023, Google DeepMind) — see [[World-Model-Landscape|The World Model Landscape (2019-2026)]]

---

## Limitations

- Even with 100 million environment steps, DreamerV3 only occasionally solves the Minecraft diamond task rather than reliably succeeding every episode
- Still requires substantial compute relative to simpler model-free baselines on tasks where those baselines already work well

---

## Related Concepts

*Lineage: [[World Models]] · [[IRIS]] · [[DIAMOND]]*

*Landscape: [[World-Model-Landscape|The World Model Landscape (2019-2026)]]*

*Credit assignment: [[VinePPO]]*
