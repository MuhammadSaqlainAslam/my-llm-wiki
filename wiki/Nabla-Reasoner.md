---
created: "2026-10-13"
title: "∇-Reasoner: LLM Reasoning via Test-Time Gradient Descent in Latent Space"
authors: "Peihao Wang, Ruisi Cai, Zhen Wang, Hongyuan Mei, Qiang Liu, Pan Li, Zhangyang Wang"
year: "2026"
arxiv: "2603.04948"
venue: "ICLR 2026"
tags: [reasoning, test-time-compute, inference, gradient-descent, reward-model, kl-regularized-rl]
citation_count: 6
tldr: "Replaces discrete test-time search (best-of-N, trial-and-error prompting) with first-order optimization: during decoding, run gradient descent on the token logits against a reward model plus the LLM's own likelihood, then accept or reject the refined token by rejection sampling. Proves this is dual to KL-regularized RL policy alignment, and reports over 20% accuracy improvement on a hard math benchmark with ~10-40% fewer model calls than strong baselines. ICLR 2026."
aliases: ["Nabla-Reasoner", "DTO", "Differentiable Textual Optimization"]
---

# ∇-Reasoner: LLM Reasoning via Test-Time Gradient Descent in Latent Space

> Peihao Wang, Ruisi Cai, Zhen Wang, Hongyuan Mei, Qiang Liu, Pan Li, Zhangyang Wang, "∇-Reasoner: LLM Reasoning via Test-Time Gradient Descent in Latent Space", March 2026 (arXiv:2603.04948), ICLR 2026

## The Problem / Motivation

[[Test-Time-Compute]] scaling is how most current reasoning gains are bought, but the standard ways of spending that compute are *zeroth-order*: sample many completions and pick one ([[Self-Consistency]], best-of-N), run a tree or beam search, or keep re-prompting and hoping the model self-corrects. Each of these treats the model as a black box that you can only query. They are expensive (every extra candidate is a full generation) and they only explore what the base policy already happens to sample.

## The Idea

If you have a reward model and white-box access to the LLM, you don't have to *guess* a better answer — you can follow a gradient toward one. ∇-Reasoner moves the optimization from "search over discrete samples" to "first-order optimization over the logits", right inside the decoding loop. The authors frame it as a paradigm shift from zeroth-order search to first-order optimization at test time.

## Method

The core component is **Differentiable Textual Optimization (DTO)**:

- The pre-softmax logits of the generated tokens are treated as continuous variables (made differentiable through the discrete vocabulary with a Gumbel-softmax straight-through estimator).
- DTO takes gradient steps on them against a combined objective: **reward** from a reward model, plus the **LLM's own log-likelihood**. The likelihood term acts as a KL-style anchor to the base model, which is what keeps the optimization from reward-hacking its way into fluent nonsense.
- It plugs into decoding as an iterative loop: generate a continuation, refine the logits with DTO, resample the next token from the refined logits, and — via **rejection sampling** — only keep the change if the resulting full response scores a higher reward than the original rollout. Otherwise revert and move on.
- Three **acceleration** tricks make it affordable: caching gradients when the one-hot token assignment hasn't changed, reusing rollouts (sharing generated tokens and [[KV Cache]] across consecutive decoding steps), and skipping DTO for high-confidence tokens or tokens whose gradient is negligible.

**Theory.** The paper shows that doing gradient descent in sample space to maximize reward at inference time is dual to aligning a policy with KL-regularized reinforcement learning. In other words, the same objective that [[RLHF]] optimizes into the *weights* during training is here optimized into the *sample* at inference — test-time and training-time alignment are two ends of one connection. (Compare [[Direct Preference Optimization Your Language Model is Secretly a Reward Model|DPO]], which also exploits the KL-regularized-RL closed form, but on the training side.)

## Key Results

| Aspect | Result |
|---|---|
| Accuracy | Over 20% improvement on a challenging math reasoning benchmark (per the abstract) |
| Cost | ~10–40% fewer model calls than strong baselines such as Best-of-N / self-consistency |
| Benchmarks | MATH-500, AMC, AIME24, AIME25 |
| Models | Qwen-2.5-7B / -Instruct, Llama-3.1-8B-Instruct, with the Skywork-V2 family as reward model |

## How It Fits the Wiki

This is a third distinct way of spending test-time compute, and it sits well next to the notes already here:

- **More tokens** — [[s1-Simple-Test-Time-Scaling|s1]]'s budget forcing lengthens the chain of thought.
- **More samples** — [[Self-Consistency]] and best-of-N vote over many generations; ∇-Reasoner's baselines are of this kind.
- **More latent computation** — [[Recurrent-Depth-Reasoning|Huginn]] and [[LOTUS]] iterate a looped block in hidden space. ∇-Reasoner also works in continuous latent space at test time, but optimizes the *output logits against a reward* rather than iterating the network.

It also connects the Reasoning & RL cluster ([[RLHF]], [[KL-Divergence]], [[RLVR]]) to inference-time methods: it makes explicit that KL-regularized RL has an inference-time dual.

## Limitations

- Performance is bounded by the base model and the reward model, particularly under a small compute budget.
- Requires the base model and reward model to share the same vocabulary.
- Needs white-box access and backward passes through the models, so it does not apply to API-only models, and fitting it into an efficient serving stack needs careful system co-design (the authors say as much).
- Depends on a good reward model for the task; for domains without a reliable verifier or reward model the gradient signal is only as good as that proxy.

## Related Concepts

[[Test-Time-Compute]] · [[Inference-Time-Scaling]] · [[Self-Consistency]] · [[RLHF]] · [[KL-Divergence]] · [[Direct Preference Optimization Your Language Model is Secretly a Reward Model|DPO]] · [[s1-Simple-Test-Time-Scaling|s1]] · [[Recurrent-Depth-Reasoning|Huginn]] · [[LOTUS]] · [[KV Cache]]
