---
created: "2026-06-29"
title: "V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning"
authors: "Assran, Bardes, Fan, Garrido, Howes, et al. (Meta)"
year: "2025"
arxiv: "2506.09985"
tags: [world-models, jepa, robotics, self-supervised-learning, meta]
tldr: "Scales V-JEPA to 1M+ hours of internet video and 1B parameters, then adds action-conditioned planning (V-JEPA 2-AC) using under 62 hours of unlabeled robot video — enabling zero-shot pick-and-place on real robot arms in new environments"
citation_count: 557
---

## TL;DR

V-JEPA 2 scales [[V-JEPA 1]]'s feature-prediction objective to internet-scale data (1M+ hours of video, up to 1B encoder parameters), achieving state-of-the-art human action anticipation. After pretraining, the video encoder is frozen and a new action-conditioned predictor (V-JEPA 2-AC) is trained on under 62 hours of unlabeled robot manipulation video, enabling zero-shot prehensile manipulation — grasping and pick-and-place of novel objects in new environments — without any task-specific training or reward signal.

---

## The Idea

Two-stage training, directly testing the "world knowledge vs. action knowledge" distinction discussed in BVP's industry analysis (see [[World-Model-Landscape|The World Model Landscape (2019-2026)]]): first, action-free pretraining on internet video learns general physical/visual understanding; second, a small amount of robot-specific interaction data teaches the model to condition its predictions on actions, without needing to relearn physics from scratch.

---

## Why It Matters

- Directly demonstrates the core claim explored in BVP's "world knowledge vs. action knowledge" framing — that broad physical understanding can transfer from internet video, while only the much smaller embodiment-specific action mapping needs robot-specific data
- 80% zero-shot pick-and-place success across different labs, with no task-specific training — a strong result for sample efficiency in robotics, an area where data is far scarcer than for language or vision-only tasks
- At inference, planning with V-JEPA 2-AC takes roughly 16 seconds per action — a real practical bottleneck for real-time robot control noted as an open challenge in the broader BVP industry analysis

---

## Limitations

- ~16 second per-action planning latency is far from real-time control requirements
- Action-conditioned planning is demonstrated on relatively constrained manipulation tasks (grasp, pick-and-place); broader embodiment generalization remains open

---

## Related Concepts

*Lineage: [[World Models]] · [[V-JEPA 1]] · [[Dreamer 4]]*

*Landscape: [[World-Model-Landscape|The World Model Landscape (2019-2026)]]*
