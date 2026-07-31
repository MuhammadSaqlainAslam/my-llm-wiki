---
created: "2026-07-30"
title: "EAPO: Experience Augmented Policy Optimization for LLM Reasoning"
authors: "Jinda Lu, Kexin Huang, Junkang Wu, Shuo Yang, Jinghan Li, Chiyu Ma, Shaohang Wei, Xiang Wang, Guoyin Wang, Jingren Zhou"
year: 2026
tags: [rl, post-training, reasoning, rlvr, policy-optimization]
aliases: [EAPO, Experience Augmented Policy Optimization, Experience-Augmented Policy Optimization]
tldr: "Inject a prior RL-optimized policy's actions at critical decision points during rollout — rather than replaying fixed trajectories — and correct the bias with importance sampling. Consistently beats state-of-the-art RLVR baselines on Qwen-2.5-Math-7B and Qwen3-8B across five reasoning benchmarks."
theme: synthesis
arxiv: "2606.30420"
citation_count: 1
---

# EAPO: Experience Augmented Policy Optimization for LLM Reasoning

## TL;DR

Standard [[RLVR]] trains on-policy from scratch every run, wasting all the reasoning experience accumulated in earlier checkpoints. Naively replaying old trajectories fails once the policy has moved on — the actions no longer match what the current policy would choose (policy mismatch). EAPO instead treats prior experience as an **action-level prior**: at select decision points during rollout it injects the experience-policy's suggested action, then corrects the resulting bias with importance sampling. The result is a way to reuse reasoning experience without inheriting the instability of stale trajectories.

## The Problem

RLVR works, but every improvement in reasoning ability normally requires fresh, expensive on-policy rollouts — past experience (trajectories from earlier or related policies) is discarded once the policy moves past them. Reusing that experience as **fixed trajectories** runs into **policy mismatch**: the actions a fixed trajectory took no longer match what today's policy would actually do, so gradients computed on it are biased.

## The Idea

Don't reuse experience as fixed sequences of actions — reuse it as a policy that can be *queried* for a suggested action at any state, and only *sometimes* followed. During rollout the current policy generates as normal, but at selected critical decision points the experience policy's action is substituted in. Because this happens at the action level rather than by replaying whole trajectories, the current policy still explores and adapts on its own — it borrows guidance, not a fixed path. To keep the training objective unbiased, EAPO reweights the affected steps with an adapted importance-sampling correction, so the gradient still reflects the current policy's true objective rather than the experience policy's.

## Key Results

- Base models: Qwen-2.5-Math-7B and Qwen3-8B.
- Evaluated across five reasoning benchmarks.
- Consistently improves reasoning performance over state-of-the-art RLVR methods (both from-scratch on-policy RLVR and fixed-trajectory replay baselines).

## Why It Matters

This amortizes RL training cost across runs: rather than throwing away everything a policy learned once it's superseded by a newer checkpoint, later training runs can draw on it as a soft, adaptive prior. That's directly useful for iterative reasoning-model development, where checkpoints are trained frequently and past rollouts would otherwise go to waste.

## Limitations

- Requires maintaining and querying a separate "experience" policy alongside the policy being trained — added infrastructure and compute versus plain RLVR.
- Evaluated at 7B–8B scale; unclear how the gains scale to frontier-size reasoning models.

## Related Concepts

[[RLVR]] · [[GRPO]] · [[Multi-Environment_RLVR_Training|Multi-Environment RLVR Training]] · [[On-Policy Distillation]]
