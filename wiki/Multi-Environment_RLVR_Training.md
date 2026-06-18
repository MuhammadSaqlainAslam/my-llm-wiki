---
title: "Multi-Environment RLVR Training"
authors: "NVIDIA"
year: "2025"
arxiv: ""
technical_report: "https://arxiv.org/pdf/2512.20856"
source_type: "technical_report"
tags: [reinforcement-learning, rlvr, post-training, agentic, nvidia]
tldr: "Three-stage RL post-training pipeline combining multi-environment RLVR, SWE-RL for software engineering, and RLHF with a principle-following GenRM, using asynchronous GRPO"
citation_count: 0
---

# Multi-Environment RLVR Training

## TL;DR

A three-stage reinforcement learning post-training recipe used in [[Nemotron_3_Super|Nemotron 3 Super]] and [[Nemotron_3_Ultra|Nemotron 3 Ultra]]:

1. **[[RLVR]]** across 21 diverse environments and 37 datasets spanning math, code, STEM, safety, and agentic tasks
2. **SWE-RL** for end-to-end software engineering using OpenHands containers
3. **[[RLHF|RLHF]]** with a principle-following Generative Reward Model (GenRM)

Training uses asynchronous [[GRPO]] that decouples training from inference, with [[Multi-Token Prediction]] heads accelerating rollout generation.

---

## The Problem

**Single-environment RL overfits to one task shape.** A model trained only on math RL doesn't necessarily transfer that reasoning discipline to agentic tool-use or software engineering tasks. Earlier [[RLVR]] work (e.g. [[DeepSeek-R1 Incentivizing Reasoning Capability in LLMs via Reinforcement Learning|DeepSeek-R1]]) demonstrated RL could dramatically improve reasoning, but the gains were concentrated in math and code and didn't generalize as broadly.

**Synchronous RL creates a compute bottleneck.** Standard synchronous PPO/GRPO stalls training while waiting for the model to generate rollouts (inference is the bottleneck). This is especially painful when rollouts are long (64K tokens at reasoning tasks).

---

## The Idea

### Stage 1: RLVR Across 21 Environments

All 21 environments are trained **simultaneously in a single RL run** — not sequentially. This is the key system-level choice:

- Sequential per-capability RL causes capability regression: each stage overwrites some gains from earlier stages
- Simultaneous training is more stable and produces better cross-task generalization

**Environments include:** competitive math, competitive coding, software engineering, instruction following, search, chat, agentic tool use, long context, economics, formal logic, MCQ, and more (21 configurations across 37 datasets).

**Scale:** 1.2 million environment rollouts generated across the 21-environment stage.

**Implementation:** Async [[GRPO]] — training and inference run on separate GPU devices with in-flight weight updates:
- 256 prompts per step
- 16 responses per prompt (effective batch: 4096)
- Max generation length: 64K tokens
- [[Multi-Token Prediction]] heads used as free speculative drafts during rollout — critical for throughput when sampling thousands of 64K-token traces

**PivotRL:** A sample-efficiency technique that reuses offline SFT traces and focuses RL training on "pivot" turns — turns where the model has high uncertainty or where the conversation branches. More efficient use of rollout budget, faster convergence on hard problems.

### Stage 2: SWE-RL

Dedicated RL stage for software engineering:

- **Environments:** SWE-bench tasks (real GitHub issues requiring multi-file code edits)
- **Reward:** Whether the submitted patch passes the test suite
- **Scaffold:** OpenHands sandboxed containers for code execution

Trained after Stage 1 RLVR to specialize on interactive coding without disrupting general capabilities. The primary driver of improvements on SWE-bench vs. earlier NVIDIA models (Nemotron 3 Super achieves 60.47% on SWE-bench with OpenHands).

### Stage 3: RLHF with GenRM

Standard RLHF stage but with a twist — the reward model is a **Generative Reward Model (GenRM)**. Instead of outputting a scalar preference score, the GenRM reasons about which response better follows stated principles before giving a verdict.

This is more expensive than a standard RM but produces better-calibrated rewards, especially for nuanced instruction-following and safety cases where the right answer depends on context that a simple scalar reward can't capture.

---

## Why It Matters

**Simultaneous multi-environment training beats sequential.** Prior practice was to train math RL, then code RL, then instruction-following RL — each stage risked regressing the previous. Nemotron 3's single simultaneous run is a systems-level win that produces more stable convergence.

**Async GRPO is a distinct systems choice.** Standard synchronous PPO/GRPO stalls training on inference. Decoupling to separate devices with in-flight weight updates means the training pipeline is never idle — a significant throughput improvement at the scale of 64K-token reasoning traces.

**RLVR generalizing across task types.** A common criticism of earlier RLVR work was that it was math-specific. This pipeline demonstrates RLVR scaling across 21 heterogeneous environments including agentic tasks, long-context, and safety — without a dedicated task-specific recipe for each.

**MTP as inference accelerator during training.** Using [[Multi-Token Prediction]] heads to draft tokens during RL rollout generation reduces the wall-clock time of the inference-heavy rollout generation step — a practical feedback loop between the architecture choice and training efficiency.

---

## Related Concepts

*RL methods: [[RLVR]] · [[GRPO]] · [[RLHF]]*

*Related work: [[DeepSeek-R1 Incentivizing Reasoning Capability in LLMs via Reinforcement Learning|DeepSeek-R1]] · [[Direct Preference Optimization Your Language Model is Secretly a Reward Model|DPO]]*

*Nemotron 3 family: [[Nemotron-3]] (whitepaper) · [[Nemotron_3_Super|Nemotron 3 Super]] · [[Nemotron_3_Ultra|Nemotron 3 Ultra]]*

*Co-used with: [[Multi-Token Prediction]] · [[LatentMoE]]*
