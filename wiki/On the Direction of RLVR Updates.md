---
created: "2026-08-02"
title: "On the Direction of RLVR Updates for LLM Reasoning: Identification and Exploitation"
authors: "Kexin Huang, Haoming Meng, Junkang Wu, Jinda Lu, Chiyu Ma, Ziqian Chen, Xue Wang, Bolin Ding, Jiancan Wu, Xiang Wang, Xiangnan He, Guoyin Wang, Jingren Zhou"
year: 2026
tags: [rl, rlvr, reasoning, policy-optimization, interpretability]
aliases: [Direction of RLVR Updates]
tldr: "RLVR-induced weight changes are sparse, but prior work only looked at their magnitude. The signed, token-level log-probability shift Δlog p between base and RLVR model — the direction of change — is a better lens: it pinpoints the reasoning-critical tokens more precisely, enabling training-free test-time extrapolation and a training-time reweighting method that both improve reasoning accuracy."
theme: synthesis
arxiv: "2603.22117"
citation_count: 0
---

# On the Direction of RLVR Updates for LLM Reasoning: Identification and Exploitation

## TL;DR

[[RLVR]] changes only a sparse subset of a model's weights/behavior — this much was already known. But prior analyses of that sparsity looked only at the *magnitude* of change (divergence, entropy). This paper argues the *direction* of change matters more: the signed, token-level log-probability difference $\Delta\log p$ between the base model and the RLVR-trained model. That signed quantity identifies the reasoning-critical tokens more precisely than magnitude-based metrics — and once you can identify them, you can exploit them, both at test time (no retraining) and during training.

## The Problem

RLVR training only meaningfully changes the policy's behavior on a small fraction of tokens — most of the model's output distribution is barely touched. Prior work quantified this sparsity using magnitude-based metrics like KL divergence or entropy change. But magnitude alone conflates two very different things: a token whose probability went *up* after RLVR (the model now trusts this token more) and a token whose probability went *down* (the model now avoids it) can show the same magnitude of change while meaning opposite things for reasoning quality.

## The Idea

Track the **signed** token-level log-probability difference:

$$\Delta\log p = \log p_{\text{RLVR}}(\text{token}) - \log p_{\text{base}}(\text{token})$$

Through statistical analysis and token-replacement interventions, the paper shows $\Delta\log p$ identifies the sparse, reasoning-critical tokens more effectively than magnitude-only metrics — because it distinguishes tokens the policy learned to *prefer* from tokens it learned to *avoid*, rather than just flagging "something changed here."

This identification unlocks two practical applications:

1. **Test-time extrapolation** — amplify the policy's behavior *along the learned $\Delta\log p$ direction*, without any further training, to improve reasoning accuracy. Since the direction of a useful update is already known from training, you can push further along it at inference time for free.
2. **Training-time reweighting** — focus the training signal on low-probability tokens (i.e. tokens with higher $\Delta\log p$), which improves reasoning performance across models and benchmarks by concentrating learning where RLVR's signal is actually informative.

## Key Results

- $\Delta\log p$ more effectively isolates reasoning-critical, sparse updates than magnitude-based metrics (KL divergence, entropy) in direct comparison.
- Test-time extrapolation along the $\Delta\log p$ direction improves reasoning accuracy with no additional training.
- Training-time reweighting toward high-$\Delta\log p$ tokens improves reasoning performance across multiple models and benchmarks.

## Why It Matters

This reframes how to think about what RLVR actually does to a model: not just "a few things changed" but "a few things changed in a specific, exploitable direction." That distinction turns an analysis result into two concrete levers — one that needs zero extra training (test-time extrapolation) and one that improves the training process itself (reweighting) — both grounded in the same underlying signal.

## Limitations

- Test-time extrapolation amplifies an existing learned direction; it doesn't discover new capabilities beyond what RLVR training already encoded.
- Identifying the useful $\Delta\log p$ direction still requires having already run RLVR once to compute the base-vs-RLVR log-probability difference.

## Related Concepts

[[RLVR]] · [[EAPO]] · [[Sparrow]] · [[GRPO]]
