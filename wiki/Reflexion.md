---
created: "2026-08-14"
title: "Reflexion: Language Agents with Verbal Reinforcement Learning"
authors: "Noah Shinn, Federico Cassano, Edward Berman, Ashwin Gopinath, Karthik Narasimhan, Shunyu Yao"
year: 2023
arxiv: "2303.11366"
tags: [agents, reinforcement-learning, self-reflection, reasoning, code-generation]
citation_count: 0
tldr: "An LLM agent that improves within a task not by updating weights, but by writing itself a natural-language note about what went wrong and rereading that note on the next attempt. On AlfWorld this 'verbal RL' loop lifts success from 75% to 97% over 12 trials; on HumanEval it pushes GPT-4 from 80% to 91% pass@1."
aliases: ["Reflexion", "verbal reinforcement learning"]
---

# Reflexion: Language Agents with Verbal Reinforcement Learning

> Noah Shinn, Federico Cassano, Edward Berman, Ashwin Gopinath, Karthik Narasimhan, Shunyu Yao, "Reflexion: Language Agents with Verbal Reinforcement Learning", NeurIPS 2023 (arXiv:2303.11366)

## TL;DR

Standard reinforcement learning improves a policy by computing a gradient from a reward signal and updating millions or billions of weights. That's expensive, and for an LLM agent it's often not even possible — you don't get to fine-tune GPT-4 between every episode.

Reflexion asks a much cheaper question: what if the "policy update" is just a sentence? After a failed attempt, the agent looks at what happened, writes itself a short natural-language diagnosis of *why* it failed, and stores that diagnosis in a memory buffer. On the next attempt, that memory is fed back into the prompt as context. No gradients, no fine-tuning — the agent's behavior changes purely because its context changed. Across coding, reasoning, and decision-making benchmarks, this "verbal reinforcement learning" loop produces large gains: AlfWorld success rate goes from 75% (ReAct alone) to 97% over 12 trials, and Reflexion pushes GPT-4's HumanEval pass@1 from a baseline of 80% to 91%.

## The Problem / Motivation

Language agents that act in an environment (write code, browse the web, navigate a household simulator) fail constantly on their first try. The natural fix is reinforcement learning: try, get a reward, update the policy to do better next time. But policy-gradient RL over an LLM requires collecting a lot of trajectories, defining a reward model, and running an expensive fine-tuning loop — infrastructure most people don't have, and even when you do, it's slow to iterate.

Humans don't need any of that to improve within a single afternoon of practice. You attempt a task, notice you made a specific mistake, say *"I made an off-by-one error in the loop bound — next time double-check the boundary conditions"* to yourself, and do better on the retry. That correction never touched your synapses' weights in any deliberate, gradient-like way — it's just a fact you now remember. Reflexion asks whether an LLM agent can improve the same way: through remembered, self-generated verbal feedback rather than a weight update.

## The Idea

Replace the gradient update with a **natural-language lesson, stored in an episodic memory buffer and re-injected as context on the next attempt.**

Concretely: an agent tries a coding task and fails a unit test with an off-by-one error. Instead of computing a loss and backpropagating, Reflexion has the agent look at the failing trajectory and the test output, and generate a self-reflection like:

> "I made an off-by-one error in the loop bound — next time double-check whether the range should be inclusive or exclusive of the last index."

This sentence gets appended to the agent's memory. On the next attempt, that sentence is part of the prompt. The agent doesn't need to "learn" anything at the weight level — the correction is sitting right there in its context, and a capable LLM conditions on it just like it would condition on any other instruction.

## Architecture / Method

Reflexion has three components:

- **Actor** — an LLM that generates text and actions given the current state and its own memory. In interactive environments the Actor is literally a [[ReAct Synergizing Reasoning and Acting in Language Models|ReAct]]-style agent (interleaving reasoning traces with actions).
- **Evaluator** — scores the Actor's trajectory. This can be as simple as a heuristic (did the unit tests pass? did the agent hallucinate the same action repeatedly?) or an exact-match check against a known answer (used for HotpotQA). Reflexion does not require a learned reward model.
- **Self-Reflection model** — given the trajectory and the Evaluator's (often just binary) signal, generates a specific, actionable natural-language critique of what went wrong and how to do better.

The loop, run for several trials per task:

```
        ┌─────────────────────────────────────────────┐
        │                                               │
        ▼                                               │
   ┌─────────┐   trajectory   ┌────────────┐   score   ┌──────────────────┐
   │  Actor  │ ─────────────▶ │ Evaluator  │ ────────▶ │ Self-Reflection   │
   │ (ReAct- │                │ (heuristic │           │ model             │
   │  style) │                │  / exact   │           │ "what went wrong, │
   └─────────┘                │  match)    │           │  what to do next" │
        ▲                     └────────────┘           └────────┬──────────┘
        │                                                        │
        │              append reflection text                   │
        └──────────────── to episodic memory ────────────────────┘
                        (included in next trial's prompt)
```

Critically, **no parameters change at any point.** The Actor, Evaluator, and Self-Reflection model can even all be calls to the same frozen LLM with different prompts. The only thing that persists and accumulates across trials is the text in the memory buffer.

## Key Results

- **AlfWorld** (embodied household decision-making, 134 tasks): ReAct + Reflexion completes 130/134 tasks (97%) after 12 trials, versus 75% for ReAct alone — a 22-point absolute improvement. Hallucination rate drops from 32% to 3% over the trials, while ReAct alone plateaus around a 22% hallucination rate with no further improvement after trial 6–7.
- **HotpotQA** (multi-hop QA, exact-match reward): Reflexion reaches 51% success versus a base ReAct agent's ~34% — a 17-point absolute improvement, using a memory of just the last 3 reflections.
- **HumanEval** (Python code generation): the paper's headline claim is Reflexion reaching **91% pass@1**, versus **80%** for the prior GPT-4 state of the art — at the time, the best reported result on the benchmark, achieved with zero additional training.
- **Negative result, reported honestly**: on WebShop (an e-commerce navigation benchmark), Reflexion does *not* meaningfully outperform plain ReAct — the authors attribute this to WebShop's narrower, less diverse action/observation space giving the agent less useful signal to reflect on.

## Comparison to Prior Work

- vs. **[[ReAct Synergizing Reasoning and Acting in Language Models|ReAct]]** — Reflexion doesn't replace ReAct, it wraps it: the Actor *is* a ReAct-style reason-then-act loop. Reflexion's contribution is the Evaluator + Self-Reflection + memory loop layered on top, which lets the same ReAct agent keep improving across multiple attempts at the same task instead of only reasoning within a single attempt.
- vs. **traditional RL fine-tuning** — no trajectory dataset collection, no reward model training, no gradient updates. The entire "training" loop runs at inference time, in-context, and can adapt to a brand-new task in a handful of trials rather than requiring a training run.

## Limitations

- Needs *some* signal of success or failure (the Evaluator). This is easy for coding (unit tests) and QA (exact match), but much harder to define for open-ended tasks with no ground truth.
- The memory buffer is finite and shares the context window with everything else — as reflections accumulate, older ones may need to be dropped or summarized.
- The quality of the "lesson learned" depends entirely on the base LLM's ability to accurately diagnose its own failure; a model that misattributes the cause of failure will write itself a useless or actively misleading reflection.
- Gains are not universal — WebShop showed the approach can fall flat when the environment doesn't offer enough distinguishing signal between successive attempts.

## Why It Matters

Reflexion popularized the idea that an LLM agent can "learn" purely through accumulated natural-language self-critique, with no gradient step anywhere in the loop — a framing now generally called **verbal reinforcement learning**. It's a strikingly cheap way to get RL-like iterative improvement (try, fail, reflect, retry) without any of RL's usual infrastructure, and it directly influenced later self-refinement, self-critique, and agentic coding/reasoning systems that treat memory, not weights, as the thing that gets "trained" between attempts.

## Related Concepts

[[ReAct Synergizing Reasoning and Acting in Language Models|ReAct]] · [[Generative-Agents|Generative Agents]] · [[Self-RAG]] · [[Chain-of-Thought Prompting Elicits Reasoning in Large Language Models|Chain of Thought]]
