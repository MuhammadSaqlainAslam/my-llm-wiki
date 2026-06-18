---
created: "2026-06-10"
title: "Direct Preference Optimization: Your Language Model is Secretly a Reward Model"
aliases: ["DPO", "Direct-Preference-Optimization"]
authors: Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, Chelsea Finn
year: 2023
arxiv: 2305.18290
tags: [alignment, rlhf, fine-tuning, preference-learning, foundational, inference]
citation_count: 5500
tldr: DPO replaces the unstable two-stage RLHF pipeline (reward model + PPO) with a single binary cross-entropy loss, matching or beating PPO-based RLHF on sentiment, summarization, and dialogue with models up to 6B parameters.
---

## The Problem

Getting a language model to behave the way you want after pretraining is hard. The dominant approach is **RLHF**: collect human preference labels (response A is better than response B), train a reward model to score outputs, then use reinforcement learning (typically PPO) to nudge the LM toward high-scoring outputs while not drifting too far from the original model via a KL-divergence penalty.

This pipeline works, but it's a mess in practice. You're training *at least three separate models*: the SFT base model, the reward model, and the policy itself — and the policy must sample from itself during training. PPO is notoriously finicky: it introduces dozens of hyperparameters, requires careful tuning of learning rates, clipping coefficients, and KL penalties, and can be numerically unstable. The whole thing is expensive and brittle. Small mistakes in the reward model or the RL loop can cause reward hacking, mode collapse, or training divergence.

The deeper question the DPO paper asks is: do we actually *need* an explicit reward model and RL at all? It turns out the answer is no — the optimal policy under the RLHF objective can be expressed analytically in terms of the policy itself, meaning the reward model is implicit in the policy and you can train directly on preferences.

## The Idea

**The language model policy and the reward model are two sides of the same coin** — given the constrained RLHF objective (maximize reward subject to a KL penalty from a reference policy), the optimal policy has a closed-form expression in terms of the reward. This means you can invert the relationship: express the reward in terms of the policy, and then plug that expression directly into the preference loss. The reward model disappears; you're left with a loss purely over policies.

The result is a simple binary cross-entropy objective over preference pairs — preferred vs. dispreferred completions — with no RL loop, no reward model to train separately, and no sampling during fine-tuning.

## How It Works

**Step 1: The standard RLHF objective.** RLHF solves:

$$\max_{\pi} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi(y|x)} [r(x, y)] - \beta \, \mathbb{D}_{\text{KL}}[\pi(y|x) \| \pi_{\text{ref}}(y|x)]$$

The KL term keeps the fine-tuned policy from wandering too far from the SFT reference policy $\pi_{\text{ref}}$.

**Step 2: The closed-form optimal policy.** This constrained optimization has a known analytical solution:

$$\pi^*(y|x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y|x) \exp\!\left(\frac{1}{\beta} r(x, y)\right)$$

where $Z(x)$ is a partition function normalizing over all responses. This is a Boltzmann/softmax reweighting of the reference policy by the reward.

**Step 3: Invert to get reward in terms of policy.** Rearranging the equation above:

$$r(x, y) = \beta \log \frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \log Z(x)$$

The partition function $Z(x)$ cancels when you take the *difference* of rewards between two responses (as in the Bradley-Terry preference model). So the human preference probability becomes:

$$p^*(y_w \succ y_l | x) = \sigma\!\left(\beta \log \frac{\pi^*(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi^*(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)$$

**Step 4: The DPO loss.** Substitute the parameterized policy $\pi_\theta$ for $\pi^*$ and maximize log-likelihood over preference pairs $(x, y_w, y_l)$:

$$\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}}\!\left[\log \sigma\!\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)\right]$$

Intuitively: **increase the log-probability of preferred responses relative to the reference policy, and decrease it for dispreferred ones** — but weight each example by how surprised the current model is, preventing degenerate solutions where the model collapses to a trivial answer.

The key trick is that the per-example gradient weight $\hat{u}(x, y_w, y_l)$ acts as an importance weight: examples where the model already confidently prefers the right answer contribute less, while "hard" examples where the model currently gets it wrong contribute more. This is what distinguishes DPO from a naive log-ratio objective that causes model degeneration.

**In practice:** You need (1) the frozen SFT reference model $\pi_{\text{ref}}$, (2) the trainable policy $\pi_\theta$ initialized from $\pi_{\text{ref}}$, and (3) a dataset of preference pairs. Two forward passes per batch (one through each model), one cross-entropy loss, done.

## Key Results

- **Sentiment control:** DPO exceeds PPO-based RLHF in controlling output sentiment on the IMDb dataset, achieving higher reward at lower KL divergence from the reference — a better Pareto frontier.
- **Summarization (TL;DR):** DPO matches or improves over PPO on human-judged summary quality, with GPT-4 evaluations showing DPO summaries preferred over SFT baselines.
- **Single-turn dialogue (Anthropic HH):** DPO matches PPO on win rate against reference completions while being dramatically simpler to train.
- **Scale:** Results hold with models up to **6B parameters**.
- **Simplicity:** DPO eliminates the need for a separate reward model, RL training loop, online sampling, or significant hyperparameter search — implementation is roughly as complex as standard supervised fine-tuning.

## Limitations

- **Offline only:** DPO trains on a fixed dataset of preference pairs; it does not do online exploration. RLHF with PPO can in principle generate new rollouts and keep improving, whereas DPO is bounded by the quality of the offline preference data.
- **Data distribution matters:** If the preference dataset was generated by a very different model than $\pi_{\text{ref}}$, the implicit reward extrapolation may be unreliable.
- **No explicit reward signal:** You cannot query the implicit reward model at inference time for reward-guided search or rejection sampling without reconstructing the log-ratio. Some downstream uses of RLHF (e.g., best-of-N sampling with a standalone reward model) require extra steps.
- **Hyperparameter $\beta$:** While simpler than PPO, $\beta$ still needs tuning — it controls the KL penalty strength and affects how aggressively the policy diverges from the reference.
- **Theoretical gap:** The derivation assumes the Bradley-Terry preference model. Whether this holds for complex real-world preferences (multi-dimensional quality, safety) is not guaranteed.

## Why It Matters

DPO is arguably the most practically impactful alignment paper since the original InstructGPT/PPO-RLHF work. It democratized preference fine-tuning: instead of needing engineering infrastructure for RL training loops, reward models, and online sampling, any lab with a GPU cluster and a preference dataset could fine-tune aligned models with essentially the same code as supervised fine-tuning.

It directly enabled a wave of open-source aligned models (Zephyr, OpenHermes, many Llama fine-tunes) that were trained with DPO variants. It also spawned follow-on algorithmic work — IPO, KTO, SimPO, ORPO — all trying to address DPO's offline limitation or improve its stability. The insight that **the policy implicitly encodes a reward model** reframed how the community thinks about the relationship between supervised and RL-based alignment.

## See Also

[[RLHF|Reinforcement Learning from Human Feedback]] · [[Proximal Policy Optimization]] · [[InstructGPT]] · [[Bradley-Terry Model]] · [[KL Divergence]] · [[Supervised Fine-Tuning]] · [[Constitutional AI]] · [[Transformer|Attention Is All You Need]]
