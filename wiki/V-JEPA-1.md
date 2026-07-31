---
created: "2026-06-29"
title: "Revisiting Feature Prediction for Learning Visual Representations from Video"
authors: "Bardes, Garrido, Ponce, Chen, Rabbat, LeCun, Assran, Ballas"
year: "2024"
arxiv: "2404.08471"
tags: [world-models, jepa, self-supervised-learning, video, meta]
tldr: "V-JEPA — video representations learned purely by predicting masked spatio-temporal regions in latent space, with no pretrained image encoders, text, negative examples, or pixel reconstruction; ViT-H/16 reaches 81.9% on Kinetics-400"
citation_count: 412
---

## TL;DR

V-JEPA trains video representations using only a feature-prediction objective: predict masked spatio-temporal regions of a video in a learned latent space, without pretrained image encoders, text supervision, negative examples, or pixel-level reconstruction. The largest model (ViT-H/16), trained purely on video, reaches 81.9% on Kinetics-400, 72.2% on Something-Something-v2, and 77.9% on ImageNet1K using only a frozen backbone.

---

## The Idea

Based on Yann LeCun's Joint-Embedding Predictive Architecture (JEPA) concept: rather than predicting raw pixels (which wastes capacity on irrelevant visual detail — the same critique [[DIAMOND]]'s authors raise about discrete-token world models, just resolved in the opposite direction: predict in latent space instead of preserving pixel detail), V-JEPA predicts the *representation* of masked video regions. A context encoder processes masked video, a target encoder processes the full video, and a predictor learns to map one to the other in representation space.

---

## Why It Matters

- A fundamentally different design philosophy from [[World Models]]'s VAE-based pixel reconstruction and [[IRIS]]/[[DIAMOND]]'s pixel-or-token prediction — predicting in abstract representation space rather than observation space
- Outperforms pixel-reconstruction approaches under a frozen-backbone evaluation protocol, and is more label-efficient as labeled data decreases
- Direct foundation for [[V-JEPA 2]], which extends this same objective to internet-scale video and adds action-conditioned robot planning

---

## Limitations

- Evaluated on representation-quality benchmarks (Kinetics-400, Something-Something-v2, ImageNet1K) rather than direct control/planning tasks — V-JEPA 1 alone does not demonstrate robot action or planning capability
- Trained on 2 million videos, considerably smaller scale than V-JEPA 2's 1M+ hours

---

## Related Concepts

*Lineage: [[World Models]] · [[V-JEPA 2]] · [[IRIS]] · [[DIAMOND]]*

*Landscape: [[World-Model-Landscape|The World Model Landscape (2019-2026)]]*
