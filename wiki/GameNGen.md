---
created: "2026-06-29"
title: "Diffusion Models Are Real-Time Game Engines"
authors: "Valevski, Leviathan, Arar, Fruchter"
year: "2024"
arxiv: "2408.14837"
tags: [world-models, diffusion, gaming, google]
tldr: "GameNGen is the first game engine powered entirely by a neural model — a diffusion model trained on recorded RL-agent gameplay simulates DOOM in real time at 20fps, with human raters barely better than chance at distinguishing it from the real game"
citation_count: 238
---

## TL;DR

GameNGen demonstrates that a diffusion model can function as a complete, real-time game engine with no traditional game-engine code at all. Trained in two phases — an RL agent first learns to play DOOM and its play sessions are recorded, then a diffusion model learns to predict the next frame conditioned on past frames and actions — the resulting system runs DOOM interactively at over 20fps on a single TPU, with next-frame prediction quality (PSNR 29.4) comparable to lossy JPEG compression. Human raters were only slightly better than random chance at telling real gameplay clips apart from GameNGen's simulated ones, even after 5 minutes of continuous generation.

---

## The Idea

Unlike [[DIAMOND]], which trains an RL agent to act inside a diffusion world model, GameNGen inverts the relationship for its second phase: the RL agent's role is purely to generate training data (diverse, skillful gameplay trajectories) which is then used to train a diffusion model — built on an augmented Stable Diffusion 1.4 — to predict subsequent frames. The diffusion model itself becomes the playable artifact, not the policy.

---

## Why It Matters

- The first demonstration that a complex, established commercial game (DOOM) can be "re-implemented" entirely as a neural network, with no underlying game logic code at all
- A direct precedent for [[Odyssey-2]]-style and similar "neural game engine" products appearing later in [[World-Model-Landscape|The World Model Landscape (2019-2026)]]'s Gaming/Interactive category
- Demonstrates long-horizon stability (multi-minute play sessions without degradation) — a known failure mode (drift, inconsistency) for many video-generation-based world models, explicitly flagged as an open research gap in BVP's broader industry analysis

---

## Limitations

- Trained and demonstrated specifically on DOOM — a relatively simple, low-resolution 1993 game by modern standards, chosen partly for tractability; generalization to visually richer modern games is not demonstrated
- The paper's open-source distinction in industry summaries refers to the published methodology and demo, not necessarily a full public release of trained model weights — worth confirming directly before citing this as a freely reproducible artifact

---

## Related Concepts

*Lineage: [[World Models]] · [[DIAMOND]]*

*Forward ref (not yet created): [[Odyssey-2]]*

*Landscape: [[World-Model-Landscape|The World Model Landscape (2019-2026)]]*
