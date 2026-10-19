---
created: "2026-10-19"
title: "SimPO: Simple Preference Optimization with a Reference-Free Reward"
authors: "Yu Meng, Mengzhou Xia, Danqi Chen (Princeton / NLP group)"
year: 2024
arxiv: "2405.14734"
citation_count: 0
tags: [rlhf, post-training, alignment, training, reward, dpo, fine-tuning]
tldr: "Replace DPO's reference-model reward with average log-prob of the sequence (length-normalized) and add a target margin in the Bradley-Terry loss. No reference model — cheaper and more stable. Beats DPO by up to 6.4 pts AlpacaEval 2 / 7.5 pts Arena-Hard; Gemma-2-9B-it tops Chatbot Arena <10B. NeurIPS 2024."
aliases: [SimPO, Simple Preference Optimization, reference-free DPO]
theme: training
---

# SimPO

> Yu Meng, Mengzhou Xia, Danqi Chen, "SimPO: Simple Preference Optimization with a Reference-Free Reward", NeurIPS 2024 (arXiv:2405.14734)

## The Problem / Motivation

[[Direct Preference Optimization Your Language Model is Secretly a Reward Model|DPO]] made RLHF-style preference tuning stable by rewriting the reward as a log-ratio against a **frozen reference** model:

$$
r_{\mathrm{DPO}}(x,y) = \beta \log \frac{\pi_\theta(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}
$$

That reference is a tax:

- **Memory + compute** — keep a full copy (or PEFT twin) every step  
- **Objective mismatch** — the reward is a *ratio* vs $\pi_{\mathrm{ref}}$, while generation is from $\pi_\theta$’s own logprobs  
- **Length** — sequence logprobs grow with $y$; long wins can dominate without normalization  

## The Idea

**Implicit reward = average log-probability of the response** under the policy (no reference):

$$
r_{\mathrm{SimPO}}(x,y) = \frac{\beta}{|y|} \sum_{t} \log \pi_\theta(y_t \mid x, y_{<t})
$$

Then Bradley-Terry preference loss **plus a target margin** $\gamma$ so winners beat losers by a designed gap (not just a sign):

$$
\mathcal{L} = -\log \sigma\!\left(r_{\mathrm{SimPO}}(x,y_w) - r_{\mathrm{SimPO}}(x,y_l) - \gamma\right)
$$

Length normalization ($|y|$) stops long replies from winning by verbosity; the margin sharpens preference separation.

## How It Works

1. Sample $(x, y_w, y_l)$ preference pairs (same data as DPO).
2. Compute average logprob under $\pi_\theta$ for winner and loser (**no** $\pi_{\mathrm{ref}}$ forward).
3. Apply margin + logistic loss; update $\pi_\theta$.

Hyperparameters: $\beta$ (scale), $\gamma$ (target margin). Much simpler bookkeeping than DPO + reference.

## Key Results

| Benchmark | Result vs DPO-style baselines |
|---|---|
| AlpacaEval 2 | up to **+6.4** points |
| Arena-Hard | up to **+7.5** points |
| Length | **without** inflating response length substantially |
| Gemma-2-9B-it + SimPO | **72.4%** length-controlled AlpacaEval 2, **59.1%** Arena-Hard |
| Chatbot Arena | **1st** among &lt;10B models with real user votes (paper) |

Studied across Mistral / Llama 3 / Gemma 2, base and instruction-tuned setups.

## Comparison to Prior Work

- vs **[[Direct Preference Optimization Your Language Model is Secretly a Reward Model|DPO]]** — drop $\pi_{\mathrm{ref}}$; use length-avg logprob + explicit margin. Simpler and stronger in their sweeps.
- vs **[[RLHF|PPO RLHF]]** — no reward model, no online rollouts; still offline pairs.
- vs **KTO / ORPO / IPO** — same “simplify preference opt” family; SimPO’s bet is *average logprob reward* + *margin*, not KL to a reference (see paper §2 for KL discussion).
- vs reward-model ranking — no separate RM; the policy’s logprob *is* the score.

## Limitations

- Offline pairs quality still dominates (garbage preferences → garbage policy).
- Average logprob is a proxy reward — can favor generic “always likely” style without good data / margin.
- Margin $\gamma$ and $\beta$ need tuning per model family.
- Not a full replacement for online RL ([[RLVR]], PPO) when you have verifiable rewards and huge exploration budgets.
- Arena “#1 &lt;10B” is a snapshot; later models move the board.

## Why It Matters

This wiki’s training thread is **how you turn base LMs into assistants** ([[RLHF]] → DPO → …). SimPO is the cleanest widely used **reference-free** step after DPO: less memory, better length behavior, and strong chat benchmarks. Pair it with [[Self-Instruct]], on-policy distillation notes, and RLVR when you want the full post-training map.

## Related Concepts

[[RLHF]] · [[Direct Preference Optimization Your Language Model is Secretly a Reward Model|DPO]] · [[InstructGPT]] · [[Constitutional-AI]] · [[Supervised-Fine-Tuning]] · [[Parameter-Efficient-Fine-Tuning]] · [[RLVR]] · [[GRPO]] · [[On-Policy Distillation]] · [[Self-Instruct]] · [[Qwen3 Technical Report]] · [[DeepSeek-R1 Incentivizing Reasoning Capability in LLMs via Reinforcement Learning|DeepSeek-R1]]