---
created: "2026-09-01"
title: "Revisiting Reinforcement Learning with Verifiable Rewards from a Contrastive Perspective"
authors: "Feng Zhang, et al."
year: 2026
arxiv: "2605.12969"
tags: [rlvr, grpo, reinforcement-learning, contrastive-learning, reasoning, post-training]
citation_count: 1
tldr: "Shows GRPO is secretly a discriminative method — it maximizes the score gap between verified-positive and verified-negative rollouts — then exposes two problems this reveals (misaligned surrogate scores, insensitive credit assignment) and fixes both with ConSPO: length-normalized log-probabilities as scores, contrasted via a group-wise InfoNCE objective with a curriculum-scheduled margin."
aliases: ["ConSPO", "Contrastive Sequence-level Policy Optimization"]
---

# Revisiting Reinforcement Learning with Verifiable Rewards from a Contrastive Perspective

> Feng Zhang, et al., "Revisiting Reinforcement Learning with Verifiable Rewards from a Contrastive Perspective", May 2026 (arXiv:2605.12969)

## The Problem / Motivation

[[GRPO]] is the workhorse algorithm for [[RLVR]] post-training — sample a group of rollouts per question, score each against a verifier (correct/incorrect, or a scalar verifiable reward), and update the policy toward the higher-scoring rollouts within the group, using the group's own mean as a baseline instead of a learned value function. It works, but this paper points out that GRPO's actual mechanics are usually described in policy-gradient language ("increase the probability of high-advantage rollouts") when there's a cleaner and more revealing way to see what it's really doing.

## The Idea

**GRPO is already a discriminative (contrastive) method in disguise.** The paper shows GRPO admits an exactly equivalent discriminative reformulation: its update is really maximizing the expected *score gap* between verified-positive rollouts and verified-negative rollouts drawn from the same group — structurally the same shape as a contrastive loss, not fundamentally different from a policy-gradient loss described differently. This reframing isn't just a change of vocabulary — it exposes two concrete problems that were hard to see in the standard framing:

1. **Likelihood-misaligned surrogate scores** — GRPO optimizes clipped importance-sampling ratio scores, not the actual sequence likelihoods that govern how the model generates at inference time. Training and inference are working with different notions of "how likely is this sequence."
2. **Score-insensitive credit assignment** — the rollout-level credit GRPO assigns doesn't reflect how close the current score gap between positives and negatives actually is; a positive that's already easily separated from its negatives gets pushed just as hard as one that's barely distinguishable from them.

**ConSPO** (Contrastive Sequence-level Policy Optimization) fixes both: replace the clipped-ratio score with **length-normalized sequence log-probabilities** (directly aligning the training score with what actually governs generation), and replace the credit-assignment mechanism with a **group-wise InfoNCE-style contrastive objective** — each verified-positive rollout is contrasted against negative distractors sampled for the same question, with a curriculum-scheduled margin that keeps separation pressure meaningful as training progresses.

## Architecture / Method

| | GRPO | ConSPO |
|---|---|---|
| Rollout score | clipped importance-sampling ratio | length-normalized sequence log-probability |
| Objective shape | policy-gradient advantage weighting | group-wise InfoNCE contrastive loss |
| Credit assignment | uniform within advantage sign, insensitive to current separation | strengthened when positive is close to negatives, smoothly attenuated once well-separated |
| Margin | implicit / fixed via clipping | explicit, curriculum-scheduled (widens as training progresses) |

The credit-assignment mechanism is the key behavioral difference: because ConSPO's loss is a softmax over positive-vs-negative scores (InfoNCE-style), the gradient on a positive rollout automatically depends on how well-separated it already is from its negative distractors — a positive still close to its negatives gets a strong gradient, one that's already clearly separated gets a weak one. GRPO's advantage-weighted update has no equivalent mechanism; it doesn't "know" how separated a rollout already is.

ConSPO is explicitly distinguished from **DisCO** (Li et al., 2025), a prior discriminative reinterpretation of GRPO under binary rewards: DisCO explores alternative scoring functions but does not address the score-insensitive credit-assignment problem that ConSPO's InfoNCE-style objective and curriculum margin specifically target.

## Key Results

- Evaluated on seven mathematical reasoning benchmarks (avg@32 on AIME, HMMT, AMC; pass@1 on the rest, including OlympiadBench) with DeepSeek-R1-Distill-Qwen-1.5B.
- ConSPO outperforms strong baselines (including GRPO) across these benchmarks.
- Training-dynamics analysis shows ConSPO achieves **consistently higher rewards than GRPO during training**, not just at convergence — the improved credit assignment appears to translate into faster, more stable learning, not only a better final score.
- Generalizes to larger reasoning models and to a different dataset (DAPO-Math-17k), and to Qwen3-4B-Base.
- Ablations confirm both pieces matter: removing the contrastive objective (replacing it with a linear positive-negative score difference) and removing likelihood-aligned scores (reverting to clipped importance-sampling ratios) each independently hurt performance, as does removing the curriculum-scheduled margin in favor of a fixed one.

## Comparison to Prior Work

- vs. **[[GRPO]]** — ConSPO is presented as a strict methodological upgrade: same group-relative structure (sample a group per question, compare within-group), but with a training score that matches inference-time likelihood, and a contrastive objective that's sensitive to how separated positives and negatives currently are, which GRPO's advantage weighting is not.
- vs. **DisCO** — both reinterpret GRPO discriminatively, but DisCO doesn't fix the credit-assignment problem ConSPO targets with its InfoNCE-style objective and curriculum margin.
- vs. **[[On the Direction of RLVR Updates|On the Direction of RLVR Updates for LLM Reasoning]]** (already in this wiki) — that paper analyzes and exploits the *direction* GRPO-style updates push the policy; ConSPO instead changes *what quantity is being optimized* (contrastive score gap with aligned likelihoods) — complementary angles on improving RLVR post-training rather than competing claims.
- vs. **[[EAPO]] and [[VinePPO]]** — both of those address credit assignment in RL-for-LLM-reasoning from a value-function/advantage-estimation angle; ConSPO addresses the same underlying problem (poor credit assignment) but via a contrastive reformulation of the scoring function itself, avoiding the need for a value function at all — consistent with GRPO's original motivation for dropping the value network in the first place.

## Limitations

- Code was not yet released at publication time ("upon acceptance"), so independent reproduction wasn't possible as of the paper's release.
- Evaluated primarily on mathematical reasoning benchmarks with verifiable (correct/incorrect) rewards — the paper's claims about likelihood-alignment and credit assignment are demonstrated in this narrow, clean-verifier setting; whether the same gains hold under noisier or partial-credit verifiers (the setting several other RLVR papers explicitly worry about) isn't established here.
- The discriminative reformulation cleanly explains GRPO's behavior in the binary/verifiable-reward regime; extending the same contrastive framing to non-verifiable, continuous reward signals (e.g., RLHF-style learned reward models) is not addressed.

## Why It Matters

This paper does something valuable independent of ConSPO's specific empirical gains: it gives a genuinely different, discriminative *lens* for understanding what GRPO — the dominant RLVR algorithm covered throughout this wiki's Reasoning & RL theme — is actually doing. Seeing GRPO as "maximize the score gap between verified positives and negatives" rather than "policy-gradient with a group-relative baseline" makes it obvious that the *scoring function* and the *credit-assignment mechanism* are both separately improvable design choices, not fixed consequences of the RL formulation. That reframing is likely to be reused by future RLVR work the same way this wiki already tracks multiple independent lines of GRPO refinement ([[EAPO]], [[VinePPO]], [[On the Direction of RLVR Updates|RLVR update-direction analysis]]) — ConSPO adds a fourth angle: treat post-training as contrastive learning, not policy-gradient learning.

## Related Concepts

[[GRPO]] · [[RLVR]] · [[Proximal-Policy-Optimization|PPO]] · [[EAPO]] · [[VinePPO]] · [[On the Direction of RLVR Updates|On the Direction of RLVR Updates for LLM Reasoning]]
