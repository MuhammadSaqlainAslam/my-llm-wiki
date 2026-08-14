---
created: "2026-06-10"
title: "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning"
authors: DeepSeek-AI
year: 2025
arxiv: 2501.12948
tags: [reinforcement-learning, reasoning, chain-of-thought, foundational, inference, fine-tuning]
aliases: ["DeepSeek-R1"]
citation_count: 5517
tldr: DeepSeek-R1 trains LLMs to reason using pure RL (GRPO) without human-annotated reasoning traces, achieving performance surpassing supervised baselines on AIME and other STEM benchmarks with emergent self-reflection behaviors.
---

## The Problem

Building capable reasoning models has historically required enormous amounts of human-annotated chain-of-thought demonstrations. You need experts to write out step-by-step solutions, verify them, and curate them into training sets — a process that is expensive, slow, and fundamentally bottlenecked by human cognitive bandwidth. Models trained this way learn to mimic human reasoning patterns, which caps their ceiling: if your best human annotator would solve a math problem a certain way, the model learns that way, and no other.

The deeper problem is that human reasoning traces introduce cognitive biases. People tend to use familiar shortcuts, avoid unconventional solution paths, and rarely discover genuinely novel strategies. When a model is forced to imitate these trajectories via supervised fine-tuning (SFT), it inherits those same biases. More importantly, it cannot explore reasoning strategies that humans wouldn't have written down in the first place.

Prior RL-based approaches (like RLHF with PPO) still relied heavily on reward models that were themselves trained on human preferences, and those reward models are fragile — susceptible to reward hacking at scale, expensive to retrain, and tricky to keep well-calibrated. The field needed a way to let models discover good reasoning strategies on their own, with only a correctness signal as feedback.

## The Idea

**Core insight:** If you give a large language model a question and a binary signal for whether its final answer is correct, it will — through pure trial and error — discover that thinking longer, checking its work, and trying alternative approaches leads to more correct answers. No human ever has to write a single reasoning trace.

This is the aha moment of DeepSeek-R1: reasoning capability is not something you have to teach by example. It can *emerge* from RL alone, the same way AlphaGo discovered superhuman Go strategies without being shown human games. The model invents self-reflection, verification loops, and strategy-switching all on its own because those behaviors correlate with getting rewarded.

## How It Works

**Step 1: Start with a capable base model.** They begin with DeepSeek-V3-Base, a strong pretrained LLM. No SFT warmup — they go straight to RL, deliberately skipping the phase that would imprint human reasoning patterns.

**Step 2: Use GRPO instead of PPO.** Group Relative Policy Optimization (GRPO) is a simplified RL algorithm that sidesteps the need for a separate critic network. For each question, sample a *group* of G outputs (they use G=16). The advantage for each output is computed relative to the group's mean reward, normalized by standard deviation:

$$A_i = \frac{r_i - \text{mean}(\{r_1, \ldots, r_G\})}{\text{std}(\{r_1, \ldots, r_G\})}$$

This is elegant: you don't need an external value function. The group itself provides the baseline. The policy is then updated with a PPO-style clipped objective plus a KL penalty toward a reference model, keeping the policy from drifting too far.

**Step 3: Rule-based rewards only.** No neural reward models. For math, check if the boxed final answer matches the ground truth. For code, run against test cases. Format rewards encourage the model to wrap its thinking in `<think>...</think>` tags. That's it. Binary, unambiguous, unhackable.

**Step 4: Let the model grow its thinking budget.** Max response length starts at 32,768 tokens and expands to 65,536 tokens at step 8,200 of training. The model naturally learns to use more tokens — average response length grows from a few hundred tokens to over 20,000 tokens across training as it discovers that longer deliberation pays off.

**Step 5: Multi-stage pipeline for DeepSeek-R1.** The pure-RL version (DeepSeek-R1-Zero) has great reasoning but poor readability — it mixes languages and is hard to read. DeepSeek-R1 fixes this with a multi-stage pipeline: (1) cold-start SFT on a small set of curated long-CoT examples, (2) RL training for reasoning tasks, (3) rejection sampling + SFT on non-reasoning tasks to preserve general capabilities, (4) final RL pass to align with human preferences. The result is a model with the reasoning power of R1-Zero but readable, well-behaved outputs.

**Distillation:** The emergent reasoning patterns can be distilled into smaller models by fine-tuning them on R1's outputs, producing capable small models that punch well above their weight class.

## Key Results

- **AIME 2024:** DeepSeek-R1-Zero achieves ~71% pass@1 and ~86% majority vote (cons@16), exceeding the average human participant score, starting from zero human reasoning demonstrations.
- **DeepSeek-R1** matches or surpasses OpenAI o1 on a range of math, coding, and STEM benchmarks at the time of release.
- Response length grows organically from ~hundreds of tokens to **>20,000 tokens** average by end of training — entirely self-directed, not forced.
- Distilled smaller models outperform the original instruction-tuned versions of the same size, demonstrating that the emergent reasoning patterns are transferable.
- The jump at training step 8,200 (when context length expanded) produced a **sharp discontinuous improvement** in both AIME accuracy and response length — a visible "aha moment" in the training curves.

## Limitations

- **DeepSeek-R1-Zero** suffers from language mixing (combining English and Chinese mid-thought) and poor readability — a known failure mode of unconstrained RL on multilingual base models.
- The rule-based reward system only works cleanly for **verifiable tasks** (math, code). Open-ended tasks like writing, summarization, or subjective QA don't have ground-truth answers to check against, so the reward signal degrades.
- RL training at this scale is **computationally expensive** — the infrastructure requirements are substantial even with GRPO's efficiency gains over PPO.
- The multi-stage pipeline for DeepSeek-R1 reintroduces some human curation (the cold-start SFT data), partially walking back the "pure RL" story.
- Reward hacking on format rewards is a known risk — the model could learn to emit `<think>` tags without actually reasoning meaningfully inside them.

## Why It Matters

DeepSeek-R1 is one of the clearest existence proofs that **test-time compute scaling via RL is real and powerful**. It validates the hypothesis — long discussed but rarely demonstrated cleanly — that LLMs can bootstrap reasoning capabilities without human demonstrations, simply by exploring a large space of outputs and learning which ones lead to correct answers.

This matters for the broader ecosystem in several ways. First, it removes the human annotation bottleneck for reasoning: you can keep improving the model by throwing more RL compute at it rather than hiring more annotators. Second, it shows that behaviors like self-reflection and multi-step verification are not things you have to carefully engineer — they emerge naturally when the reward structure favors correctness. Third, the distillation result means the community can benefit from these emergent capabilities at smaller, more accessible model sizes.

The paper also helps explain *why* longer chain-of-thought works: it's not just about following a template, it's about the model having the freedom to explore, backtrack, and verify — behaviors that RL selects for because they genuinely help get the right answer.

## See Also

[[Chain-of-Thought Prompting Elicits Reasoning in Large Language Models|Chain-of-Thought Prompting]] · [[Proximal Policy Optimization]] · [[RLHF]] · [[DeepSeek-V3 Technical Report|DeepSeek-V3]] · [[RLHF|Reinforcement Learning from Human Feedback]] · [[Scaling Laws]] · [[Test-Time Compute]]
