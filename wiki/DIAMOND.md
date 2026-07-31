---
created: "2026-06-29"
title: "Diffusion for World Modeling: Visual Details Matter in Atari"
authors: "Alonso, Jelley, Micheli, Kanervisto, Storkey, Pearce, Fleuret"
year: "2024"
arxiv: "2405.12399"
tags: [world-models, diffusion, reinforcement-learning]
tldr: "DIAMOND (DIffusion As a Model Of eNvironment Dreams) trains an RL agent entirely inside a diffusion-based world model, achieving a new best 1.46 human-normalized score on Atari 100k for agents trained purely in imagination, and scales to a playable Counter-Strike: Global Offensive world model"
citation_count: 284
---

## TL;DR

DIAMOND argues that compressing observations into discrete latent tokens (as in [[IRIS]] or Dreamer-style world models) can discard visual detail that matters for reinforcement learning. It instead trains a diffusion model to predict future frames directly, and trains an RL agent entirely inside this diffusion-generated world. DIAMOND reaches a mean human-normalized score of 1.46 on Atari 100k — a new best among agents trained entirely within a world model — and its diffusion world model alone, trained on 87 hours of static gameplay, functions as a playable neural game engine for Counter-Strike: Global Offensive.

---

## The Idea

Rather than compressing each frame into a small set of discrete tokens, DIAMOND trains a diffusion model to directly denoise and predict the next frame conditioned on past frames and actions. The paper identifies specific design choices needed to make diffusion stable and efficient enough for world modeling over long horizons (few denoising steps, careful noise scheduling) — without these, diffusion models are too slow or unstable to substitute for a real RL environment during training.

The Atari-trained model uses only 4.4M parameters; the CS:GO variant, trained on a fixed dataset of human gameplay rather than RL-collected rollouts, is scaled up to 381M parameters (including a 51M-parameter upsampling stage) to handle the higher visual fidelity of a full 3D game.

---

## Why It Matters

- Direct architectural contrast with [[IRIS]] and Dreamer-style discrete/recurrent latents — demonstrates that preserving more visual detail via diffusion, rather than compressing it away, can directly improve RL performance
- NeurIPS 2024 Spotlight — a meaningful independent quality signal beyond arXiv alone
- The CS:GO demonstration shows a world model trained purely on fixed human gameplay data (no RL interaction at all) can function as a standalone, playable simulator — a different use case from training an RL policy

---

## Limitations

- The 381M parameter figure (appearing in some industry summaries) refers specifically to the CS:GO variant; the core Atari world model is dramatically smaller at 4.4M parameters — important to distinguish when citing model size
- Diffusion sampling, even with few steps, adds computational overhead per imagined step relative to a single forward pass through a recurrent or token-based world model

---

## Related Concepts

*Lineage: [[World Models]] · [[IRIS]] · [[DreamerV3]]*

*Landscape: [[World-Model-Landscape|The World Model Landscape (2019-2026)]]*
