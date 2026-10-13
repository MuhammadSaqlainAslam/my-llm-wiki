---
created: "2026-10-13"
title: "Stabilizing Recurrent Dynamics for Test-Time Scalable Latent Reasoning in Looped Language Models"
authors: "Xiao-Wen Yang, Ziyu Han, Xi-Hua Zhang, Wen-Da Wei, Jie-Jing Shao, Lan-Zhe Guo, Yu-Feng Li"
year: "2026"
arxiv: "2605.26733"
venue: "ICML 2026"
tags: [looped-transformer, recurrent-depth, latent-reasoning, test-time-compute, stability, regularization]
citation_count: 21
tldr: "Looped language models promise test-time scaling by iterating more, but performance often peaks at some loop depth and then collapses. STARS trains the loop toward asymptotically stable fixed points with a cheap Jacobian Spectral Radius Regularization plus random loop sampling, giving reliable scaling on arithmetic and much milder degradation (and a higher peak) on math reasoning. ICML 2026."
aliases: ["STARS", "STAbility-driven Recurrent Scaling", "JSRR"]
---

# Stabilizing Recurrent Dynamics for Test-Time Scalable Latent Reasoning in Looped Language Models

> Xiao-Wen Yang, Ziyu Han, Xi-Hua Zhang, Wen-Da Wei, Jie-Jing Shao, Lan-Zhe Guo, Yu-Feng Li, "Stabilizing Recurrent Dynamics for Test-Time Scalable Latent Reasoning in Looped Language Models", May 2026 (arXiv:2605.26733), ICML 2026

## The Problem / Motivation

The pitch of [[Recurrent-Depth-Reasoning|Huginn]]-style looped language models (LoopLMs) is a free test-time dial: run the recurrent block more times on hard problems. In practice that dial is unreliable. The paper observes that performance often **peaks at a certain iteration depth and then collapses with further recurrence**, so "just loop more" can make answers worse. A depth dial that bends the wrong way past its sweet spot is hard to use.

## The Idea

Treat the loop as a dynamical system and ask why it misbehaves. Latent-dynamics analysis shows an inherent **trade-off between stability and effectiveness** in existing architectures and strategies: setups that stay stable (e.g. Post-Norm) fail to do deep reasoning, while setups that reason well (e.g. Pre-Norm) let hidden states blow up as iterations accumulate. Framing reasoning as **uncertainty reduction**, the authors argue the sweet spot is for the latent state to **converge toward a stable fixed point while staying effective**, so extra iterations refine the answer instead of destroying it.

## Method

**STARS (STAbility-driven Recurrent Scaling)** is a training framework that constrains latent states to approach asymptotically stable fixed points.

- **Jacobian Spectral Radius Regularization (JSRR).** By Lyapunov linearization, a fixed point is asymptotically stable when the spectral radius of the loop's Jacobian is below 1, so the loss penalizes that radius. Computing eigenvalues exactly is prohibitively expensive, so it is estimated cheaply with single-step power iteration using Jacobian-vector products.
- **Random loop sampling.** Train with a randomly sampled number of loop iterations, so the model works at many depths rather than one.
- **Objective.** A mix of the usual supervised fine-tuning loss and the JSRR term, weighted by λ.

## Key Results

| Setting | Result |
|---|---|
| Arithmetic (multi-digit addition) | Reliable test-time scaling; reported 100% accuracy across recurrent iterations with stable latent trajectories |
| Math reasoning | Substantially mitigates degradation as recurrence depth grows, and improves peak performance |
| Ouro-1.4B fine-tuned on NuminaMath-1.5 (reported) | GSM8K at 4 steps 81.96% vs 80.06% for plain SFT; at 8 steps 74.45% vs 60.05% |

**Ablations the paper reports:** a Prelude/Coda split as in Huginn does not change the stability tendencies, plain L2 regularization gives negligible benefit, and random loop sampling alone helps retain performance but does not fundamentally fix stability.

## How It Fits the Wiki

It extends the Looped Transformers & Recurrent Depth cluster from "does looping help?" to "when does looping break?":

- [[Recurrent-Depth-Reasoning|Huginn]] introduced the inference-time loop-count dial; STARS is a training fix for that dial's failure mode.
- [[LOTUS]] stabilizes latent reasoning with supervision on the latents; STARS stabilizes it with a *dynamical-systems* constraint and needs no gold reasoning trace.
- [[SMELT]] loops under matched compute at a fixed depth; STARS is about what happens when the depth is varied at test time.
- It echoes [[Illusion-of-Thinking|The Illusion of Thinking]] (reasoning collapses past a threshold) and [[Deep-Thinking-Tokens|Think Deep, Not Just Long]] (more compute is not automatically more useful), but traces the collapse to *latent instability* rather than to reasoning difficulty. See also [[Test-Time-Compute]].

## Limitations

- Evaluation centers on mathematical reasoning; generalization to other complex reasoning domains is an open question (the authors' own caveat).
- Performance still does not always improve monotonically with more steps, so fully reliable and predictable test-time scaling remains a challenge.
- Demonstrated at small scale (about 1.4B parameters) on fine-tuned models, not pretrained from scratch at frontier scale.

## Related Concepts

[[Recurrent-Depth-Reasoning|Huginn]] · [[LOTUS]] · [[SMELT]] · [[Test-Time-Compute]] · [[Inference-Time-Scaling]] · [[Deep-Thinking-Tokens|Think Deep, Not Just Long]] · [[Illusion-of-Thinking|Illusion of Thinking]]
