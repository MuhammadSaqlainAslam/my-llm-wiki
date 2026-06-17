---
title: "DIAMOND"
aliases: ["DIAMOND world model", "Diffusion as a World Model"]
year: 2024
tags: [world-models, diffusion, reinforcement-learning, video-prediction, stub]
tldr: "A world model that uses a diffusion process to generate the next observation frame — treating environment simulation as conditional image generation, which produces significantly sharper and more temporally consistent predictions than autoregressive token-based world models."
---

## TL;DR
Instead of predicting the next frame as a sequence of discrete tokens (IRIS-style) or a compact latent vector (Dreamer-style), DIAMOND generates the full next observation frame via score-based diffusion, conditioned on the action and recent history. This gives high-fidelity visual predictions at the cost of more inference steps per rollout.

## Intuition
Diffusion models are the best image generators we have. If you want to "imagine" what the next game frame looks like, why not use the best visual prior available? DIAMOND plugs diffusion into the world model slot: the model's "dream" is literally a sequence of generated images, which a policy can then train on.

## Why It Matters
- Demonstrates that diffusion can replace the predictive head in world models
- Achieves state-of-the-art visual fidelity in imagined rollouts
- Shows the modular nature of world model architectures: the generative backbone is swappable
- Comparison baseline for [[Looped World Models|LoopWM]]'s parameter efficiency claims

## See Also
[[DreamerV3]] · [[IRIS]] · [[Looped World Models]] · [[Recurrent State Space Model]]
