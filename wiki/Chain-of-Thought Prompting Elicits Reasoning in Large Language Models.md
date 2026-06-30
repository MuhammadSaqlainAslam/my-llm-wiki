---
created: "2026-06-10"
title: Chain-of-Thought Prompting Elicits Reasoning in Large Language Models
aliases: ["Chain-of-Thought-Prompting"]
authors: Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed H. Chi, Quoc V. Le, Denny Zhou
year: 2022
arxiv: 2201.11903
tags: [foundational, attention, inference, benchmarks]
citation_count: 19555
tldr: Chain-of-thought prompting — adding just 8 step-by-step reasoning exemplars to a prompt — lets PaLM 540B achieve 57% on GSM8K math word problems, surpassing fine-tuned GPT-3 with a verifier at 55%.
---

## The Problem

Language models get bigger and bigger, and most things improve with scale — but reasoning tasks stubbornly don't. Arithmetic word problems, commonsense reasoning, and symbolic manipulation all remained largely unsolved by raw scaling. A model like GPT-3 would confidently output a wrong number for a multi-step math problem, having seemingly no mechanism to "think through" intermediate steps. Standard few-shot prompting (just show input→output pairs as examples) just teaches the model to guess the format of the answer, not to reason to it.

The two existing alternatives were both painful. You could train or fine-tune a model on large datasets of annotated rationales — but creating those rationales is expensive and locks you into one task per checkpoint. Or you could use purely symbolic/neuro-symbolic systems, but those don't generalize and require carefully engineered formal languages. Neither approach was satisfying.

The deeper frustration: scaling to 100B+ parameters helped with knowledge retrieval and language fluency, but performance on benchmarks like GSM8K (grade-school math) essentially plateaued. More compute wasn't the answer on its own — the model needed a way to allocate computation across intermediate steps, not just emit an answer token.

## The Idea

The core insight: **if you show a large language model examples of problems solved with explicit intermediate reasoning steps, it will learn to generate those steps itself — and get the right answer.**

Think about how you solve a multi-step math problem. You don't just stare at it and produce the answer; you work through it: "She started with 10 apples, gave away 3, so she has 7, then bought 5 more, so 12." Chain-of-thought prompting simply encodes this natural human strategy into the few-shot exemplars. Instead of showing the model `(question, answer)` pairs, you show it `(question, reasoning steps, answer)` triples. The model picks up the pattern and generates its own reasoning chain at test time before committing to a final answer.

The beautiful part: no fine-tuning, no new parameters, no new training data. Just a smarter prompt.

## How It Works

**Step 1 — Build chain-of-thought exemplars.** Manually write 8 examples where each example includes the question, a natural-language step-by-step solution, and the final answer. For a math problem about Roger and tennis balls, you'd write: "Roger started with 5 balls. 2 cans of 3 tennis balls each is 6 tennis balls. 5 + 6 = 11. The answer is 11." These don't require prompt engineering heroics — robustness experiments show the method works across different annotators and phrasings.

**Step 2 — Prompt with triples at test time.** At inference, prepend those 8 `⟨input, chain of thought, output⟩` exemplars to the new test question. The model now generates a chain-of-thought reasoning sequence before emitting its final answer. You parse the final answer out of whatever it produces.

**Step 3 — Scale matters.** This is crucial: chain-of-thought prompting is essentially useless on small models (≤ ~8B parameters) and only starts to shine at ~100B parameters. The reasoning ability appears to be an *emergent* property — it's latent in large models and the prompting simply unlocks it. Smaller models tend to produce fluent-sounding but logically incorrect chains of thought.

Formally, standard prompting conditions on `p(output | input, exemplars)`, while chain-of-thought prompting conditions on `p(output | chain_of_thought, input, exemplars)`, where the chain of thought is generated autoregressively by the model itself as part of the output sequence.

The method generalizes beyond math: the same principle applies to commonsense reasoning (StrategyQA, sports understanding), symbolic tasks (last-letter concatenation, coin flipping), and date arithmetic — all using the same idea of showing worked-out reasoning in the prompt.

## Key Results

- **GSM8K (math word problems):** PaLM 540B with chain-of-thought prompting hits **57% solve rate**, vs. 33% for standard prompting with the same model. This beats fine-tuned GPT-3 175B with a verifier (55%), previously the state of the art — and PaLM isn't even fine-tuned.
- **SVAMP:** Chain-of-thought prompting with PaLM 540B reaches 79%, compared to 69% for the standard prompting baseline.
- **AQuA (algebraic word problems):** PaLM 540B chain-of-thought achieves 35% vs. 18% standard.
- **Commonsense reasoning (CSQA, StrategyQA):** Consistent gains across the board with chain-of-thought, though margins vary by task.
- **Symbolic reasoning:** Near-perfect in-distribution performance, and critically, chain-of-thought prompting enables **out-of-distribution generalization** (e.g., trained on 4-character strings, generalizes to 10-character strings) that standard prompting completely fails at.
- The gains are almost entirely concentrated in models with **≥100B parameters** — below that, chain-of-thought prompting offers little to no benefit.

## Limitations

- **Requires a very large model.** The method only works reliably at ~100B+ parameters (PaLM, GPT-3 175B, etc.). It's not a technique you can apply to smaller, cheaper, or on-device models and expect gains.
- **No guarantee of correct reasoning.** The chain of thought can sound plausible and still be logically wrong. The interpretability benefit is real but incomplete — you can see *where* it went wrong, but it may still go wrong.
- **Manually crafted exemplars.** Someone still has to write those 8 chain-of-thought examples per task domain. Compared to fine-tuning on thousands of examples this is cheap, but it's not zero effort, and quality matters.
- **Evaluation is brittle at the margins.** Parsing the final answer out of a free-form generated chain is heuristic-based and can fail for unusual output formats.
- **Only tested on tasks solvable via language.** The authors explicitly scope to tasks that humans can solve through natural language reasoning — this doesn't extend to tasks requiring external tools, grounding in perception, or very precise numerical computation.

## Why It Matters

Chain-of-thought prompting is one of the most practically impactful ideas in the modern LLM era. It shifted the field's understanding of what prompting could accomplish: not just pattern-matching to output format, but actually eliciting structured multi-step cognition from a frozen model. It made it clear that reasoning ability was latent in large models all along — the key was giving the model "permission" and structure to think out loud.

This work directly seeded an enormous downstream research tree: zero-shot chain-of-thought ("Let's think step by step"), self-consistency decoding (sample many chains, take the majority answer), least-to-most prompting, tree-of-thought, and program-of-thought all build directly on this foundation. It also quietly established that **emergent capabilities** in large models can be unlocked by inference-time interventions, not just training-time ones — a conceptual shift with massive implications for how we think about deploying and eliciting capabilities from LLMs.

In practical terms, chain-of-thought prompting is now the default approach whenever you need an LLM to do anything non-trivial — it's baked into system prompts, agents, and reasoning frameworks across the industry.

## See Also

[[Transformer]] · [[Transformer|Attention Is All You Need]] · [[Few-Shot Learners]] · [[Self-Consistency Improves Chain of Thought Reasoning]] · [[Large Language Models are Zero-Shot Reasoners]] · [[Tree of Thoughts]] · [[GPT-3]] · [[PaLM]]
