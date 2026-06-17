---
title: "DreamerV3"
aliases: ["Dreamer", "DreamerV3"]
year: 2023
tags: [world-models, reinforcement-learning, model-based-rl, latent-space, stub]
tldr: "A single world model that learns purely from pixels/observations, imagines millions of virtual rollouts in a compact latent space, and trains an actor-critic entirely inside that imagination — mastering Atari, continuous control, and Minecraft without any task-specific tuning."
---

## TL;DR
DreamerV3 encodes environment observations into a compact latent state using a recurrent state-space model (RSSM), predicts how that latent state evolves under actions, and trains a policy entirely on *imagined* trajectories inside the latent world model. No environment interaction is needed during policy improvement — the agent learns from dreams.

## Intuition
Instead of running thousands of real game episodes, train a "mental model" of the world, then let the agent practice inside that model at 1000× the speed of reality. The world model is a learned simulator; the policy is optimized inside the simulator. Return to the real environment only to collect a little more data to keep the mental model accurate.

## Why It Matters
- First model-based method to achieve strong results across diverse domains (Atari, DMC, Minecraft) with a single architecture and hyperparameter set
- Demonstrated that latent-space imagination can replace environment interaction for large parts of training
- Baseline / competitor for [[Looped World Models]] and IRIS
- Inspires parameter-efficient successors like [[Looped World Models|LoopWM]]

## See Also
[[Looped World Models]] · [[IRIS]] · [[DIAMOND]] · [[Recurrent State Space Model]] · [[Adaptive Computation Time]]
