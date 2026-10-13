---
created: "2026-10-13"
title: "Toward Skill-Native LLMs: Skill Entropy for Benchmarking and Training Long-Horizon Reasoning"
authors: "Yinghui He, Ling Yang, Jiarui Liu, Yongjin Yang, Lechen Zhang, Yingcheng Wu, Zhenfei Yin, Mengdi Wang, Sanjeev Arora"
year: "2026"
arxiv: "2608.05139"
tags: [reasoning, benchmarks-evaluation, reinforcement-learning, rlvr, long-horizon, skills]
citation_count: 0
tldr: "Introduces Skill Entropy, a directed measure of how hard it is to switch from one reasoning skill to another inside a single chain, and builds Skill²-Bench (558 skills, 9 domains) on it. Frontier models lose accuracy as skill entropy rises; turning the measure into an RL reward (Skill-Entropy RL) lifts Qwen3-4B-Instruct from 34.4% to 68.4% and Qwen3-1.7B from 14.6% to 40.1% on the benchmark."
aliases: ["Skill²-Bench", "Skill-Entropy RL", "Skill Entropy"]
---

# Toward Skill-Native LLMs: Skill Entropy for Benchmarking and Training Long-Horizon Reasoning

> Yinghui He, Ling Yang, Jiarui Liu, Yongjin Yang, Lechen Zhang, Yingcheng Wu, Zhenfei Yin, Mengdi Wang, Sanjeev Arora, "Toward Skill-Native LLMs: Skill Entropy for Benchmarking and Training Long-Horizon Reasoning", August 2026 (arXiv:2608.05139)

## The Problem / Motivation

Long-horizon reasoning in recent LLMs means switching between *different* skills inside one chain: do a math derivation, then use the result to plan a schedule, then write something up. The paper calls these **cross-skill long-horizon tasks** — multi-step tasks whose steps require different skills and depend on earlier outputs. Most benchmarks measure skills one at a time, so there is no principled way to ask "how good is this model at *switching*?" — and no way to train it on exactly that.

This is a different axis from the one in [[Long-Horizon-Tasks]]. That paper holds reasoning difficulty fixed and varies *how many actions* a task needs; this one holds length roughly fixed and varies *how different the consecutive skills are*.

## The Idea

Define a number for the difficulty of a skill switch, then use it twice: as a **benchmark scale** and as a **training signal**.

**Skill Entropy.** A directed, pairwise quantity measuring how hard it is to go from skill *a* to skill *b*. Roughly, it compares the model's accuracy on the two skills individually against its accuracy when they are chained: if the pair does much worse than the skills alone, the switch is hard (value above 1); if not, it is easy (at or below 1). A task-level score aggregates over the switches in the task.

**Skill²-Bench.** Cross-skill long-horizon tasks built over **558 skills across 9 verifiable and open-ended domains** (including math, coding, science, planning, logic and creative writing). Each task carries a task-level skill-entropy score and is grouped into three difficulty levels. The held-out test set has 300 tasks, with a separate pool of about 9K tasks for training.

**Skill-Entropy RL.** An RL framework in which the model predicts, at each step, both the **answer and the skill it used** to produce it. The reward combines step-level correctness with a skill-entropy reward measuring how well the predicted skill sequence aligns with the gold skill sequence. This is an [[RLVR]]-style setup (verifiable step correctness) with an added structured, auxiliary reward on the *form* of the reasoning.

## Key Results

| Aspect | Result |
|---|---|
| Evaluation | 8 frontier and 4 open-source models evaluated on Skill²-Bench |
| Skill-switching gap | Accuracy falls nearly monotonically as task-level skill entropy rises; the same skill does 4–13% worse inside a cross-skill task than as a single-skill question |
| Qwen3-4B-Instruct | 34.4% → 68.4% on Skill²-Bench with Skill-Entropy RL |
| Qwen3-1.7B | 14.6% → 40.1% on Skill²-Bench with Skill-Entropy RL |
| Generality | The same pipeline applies to off-the-shelf training data such as OpenR1-Math, suggesting skill entropy is a reusable training signal |

## How It Fits the Wiki

It sits where the **Reasoning & RL** and **Benchmarks & Evaluation** themes meet:

- Like [[Long-Horizon-Tasks]], it attributes agentic/long-chain failure to something *structural* about the task rather than raw difficulty — and, like it, converts the diagnosis into a training recipe.
- Like [[TRACE]], it looks *inside* the trajectory instead of scoring only the final answer, though TRACE evaluates tool-use trajectories and this paper evaluates skill sequences.
- It extends the [[RLVR]] line ([[Absolute-Zero|Absolute Zero]], [[Kimi-k1.5|Kimi k1.5]], [[Multi-Environment_RLVR_Training|Multi-Environment RLVR Training]]) with a *reward shaped on reasoning structure* rather than only on the final answer.
- It complements [[Illusion-of-Thinking|The Illusion of Thinking]], which also builds a controllable complexity axis to probe where reasoning models break.

## Limitations

- The headline gains (34.4→68.4, 14.6→40.1) are measured on Skill²-Bench, the benchmark the authors built, with training tasks from the same construction. Whether the gains transfer to *other* reasoning benchmarks is the thing to check; the OpenR1-Math result indicates the pipeline is reusable, not that it improves other benchmarks.
- Skill entropy is computed from accuracies measured with a reference model, so the entropy values (and hence the difficulty levels) are not model-independent, and the setup needs skill annotations or skill-labelled data.
- The skill taxonomy (558 skills, 9 domains) is a modeling choice; results depend on how skills are carved up.

## Related Concepts

[[Long-Horizon-Tasks]] · [[TRACE]] · [[RLVR]] · [[Absolute-Zero|Absolute Zero]] · [[Kimi-k1.5|Kimi k1.5]] · [[Multi-Environment_RLVR_Training|Multi-Environment RLVR Training]] · [[Illusion-of-Thinking|Illusion of Thinking]] · [[LLM Benchmarks]] · [[Qwen3 Technical Report|Qwen3]]
