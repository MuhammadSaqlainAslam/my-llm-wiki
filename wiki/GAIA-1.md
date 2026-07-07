---
created: "2026-06-29"
title: "GAIA-1: A Generative World Model for Autonomous Driving"
authors: "Hu, Russell, Yeo, Murez, Fedoseev, Kendall, Shotton, Corrado"
year: "2023"
arxiv: "2309.17080"
tags: [world-models, autonomous-driving, generative-models]
tldr: "Wayve's generative world model for driving — tokenizes video, text, and action into a sequence-modeling problem to generate realistic future driving scenarios with fine-grained control, the first world model in this wiki applied to autonomous driving rather than games or robotics"
citation_count: 597
---

## TL;DR

GAIA-1 ("Generative AI for Autonomy") casts world modeling for autonomous driving as an unsupervised sequence-modeling problem: video, text, and action inputs are each mapped to discrete tokens, and the model predicts the next token in the combined sequence. The resulting model generates realistic driving videos with fine-grained, controllable ego-vehicle behavior, and exhibits emergent understanding of 3D geometry and the causal relationships between road users' decisions.

---

## The Idea

Following the same next-token-prediction philosophy as [[IRIS]] (discrete tokens, an autoregressive model), but applied to the much higher-stakes domain of real-world driving footage. Trained on extensive real-world driving data from British cities, GAIA-1 learns to predict plausible future video continuations conditioned on text descriptions and action inputs, functioning as both a generative simulator and a potential planning tool for autonomous vehicles.

---

## Why It Matters

- The first world model in this wiki applied outside games/robotics — autonomous driving is a third major application cluster in the field, alongside the Robotics and Gaming/Interactive clusters [[World Models]], [[IRIS]], and [[DIAMOND]] represent
- Demonstrated genuinely emergent properties (understanding of 3D geometry, causal reasoning between road agents) not explicitly trained for — a recurring theme across larger world models as they scale
- Wayve later scaled the underlying approach to 9 billion parameters in an internal technical report — the BVP landscape figure's listed 6.5B reflects an earlier scale point in this same model's development, not the final size
- A direct predecessor to GAIA-2 (2025), the next entry in this wiki's planned World Models coverage

---

## Limitations

- Self-reported emergent capabilities — no independent benchmark suite comparable to Atari 100k exists for this domain at the time of publication
- Generated driving scenarios are evaluated largely qualitatively; the paper does not establish a standardized quantitative benchmark the way Atari 100k does for game-based world models

---

## Related Concepts

*Lineage: [[World Models]] · [[IRIS]] · [[GAIA-2]]*

*Landscape: [[World-Model-Landscape|The World Model Landscape (2019-2026)]]*
