---
created: "2026-08-14"
title: "Self-Instruct: Aligning Language Models with Self-Generated Instructions"
authors: "Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A. Smith, Daniel Khashabi, Hannaneh Hajishirzi"
year: 2022
arxiv: "2212.10560"
tags: [instruction-tuning, alignment, self-supervised, data-generation, bootstrapping]
citation_count: 3557
tldr: "Bootstraps an instruction-tuning dataset from a language model's own generations, starting from just 175 human-written seed tasks. Generates ~52K instructions and ~82K instances, filters them with heuristics (ROUGE-L novelty check, validity checks), and fine-tunes on the result — a 33% absolute improvement over vanilla GPT-3 on Super-NaturalInstructions, closing to within 5% of InstructGPT-001 despite using almost no human annotation."
aliases: ["Self-Instruct"]
---

# Self-Instruct: Aligning Language Models with Self-Generated Instructions

> Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A. Smith, Daniel Khashabi, Hannaneh Hajishirzi, "Self-Instruct: Aligning Language Models with Self-Generated Instructions" (arXiv:2212.10560)

## TL;DR

[[InstructGPT]] showed that instruction-following comes from fine-tuning on (instruction, output) pairs plus human feedback — but that data pipeline is expensive: OpenAI paid annotators to write and rate tens of thousands of examples. What if a capable base model could generate its own instruction-tuning data instead of a human writing it by hand?

Self-Instruct does exactly that. Starting from a small seed pool of 175 human-written tasks, it repeatedly prompts a vanilla GPT-3 to generate new instructions in the style of the seeds, generates matching inputs/outputs for those instructions, filters out low-quality or redundant ones with cheap heuristics, and adds the survivors back into the pool to bootstrap further generations. The result: over 52K instructions and 82K instances, generated almost entirely by the model itself, that — when used to fine-tune the same base GPT-3 — produce a 33% absolute improvement on Super-NaturalInstructions and performance within 5% of InstructGPT-001 on human-written test instructions, for a total data-generation cost of about $600.

## The Problem / Motivation

Instruction-tuned models need large, diverse sets of (instruction, output) pairs to learn to follow arbitrary natural-language commands. Human-written instruction data is the bottleneck: it's expensive to collect, and human annotators tend to write instructions that are narrower and less diverse than what a model needs to generalize well. InstructGPT's pipeline required OpenAI's own paid labelers and users' submitted prompts — a resource most researchers and labs simply don't have access to.

## The Idea

Use the language model itself as the instruction-data generator. Give it a small, diverse seed set of human-written tasks as in-context examples, and have it produce *new* tasks in the same spirit — new instructions, new inputs, new outputs — via nothing more than clever prompting and few-shot generation. Filter out the bad generations with cheap automatic heuristics (not a human reviewer), and repeat: feed some of the model's own accepted outputs back in as additional few-shot examples so the pool of style diversity keeps growing.

Concretely, one seed task looks like:

```
Instruction: "Given a dialogue, classify whether the user is satisfied with the service. 
              Output 'Satisfied' or 'Unsatisfied'."
Instance:    input  = "Agent: Is there anything else I can help you with?
                        Customer: No, that's all. Thank you for your help!"
             output = "Satisfied"
```

From seeds like this, the pipeline generates entirely new instructions — e.g. "Given a tweet, classify whether it contains political content or not" — that were never written by a human, along with plausible example inputs and outputs for them.

## Architecture / Method

1. **Seed pool.** 175 hand-written tasks (25 classification, 150 non-classification), each with an instruction plus one example instance. Every task also carries an `is_classification` flag, since classification tasks (fixed output label space) and open-ended tasks need different generation strategies downstream.
2. **Instruction generation.** At each step, sample 8 task instructions as in-context examples for the LM — 6 from the human-written seed pool, 2 from tasks the model generated in earlier steps — and prompt the LM to produce a new instruction in the same style. Mixing in model-generated examples pushes the pool toward greater diversity over time rather than collapsing back to the seed distribution.
3. **Classification-type identification.** Before generating an instance, the LM is prompted few-shot (12 classification + 19 non-classification seed examples) to decide whether the new instruction is a classification task or not, since that determines the instance-generation strategy in step 4.
4. **Instance generation.** For non-classification tasks, use an **input-first** approach: the LM invents a plausible input for the instruction, then generates the corresponding output. Classification tasks use the reverse ordering to avoid the LM collapsing onto one dominant label.
5. **Filtering.** Cheap automatic heuristics reject bad generations before they enter the pool: a **ROUGE-L similarity threshold (< 0.7)** against every existing instruction in the pool (rejects near-duplicates), forbidden-keyword checks, and basic input/output validity checks (non-empty, not just a copy of the instruction, etc.).
6. **Iterate and accumulate.** Repeat steps 2–5, growing the pool with each round, until reaching the target scale.
7. **Fine-tune.** Supervised fine-tune the same vanilla GPT-3 on the final filtered dataset — this fine-tuned model is referred to as GPT3<sub>SELF-INST</sub>.

## Key Results

| Metric | Result |
|---|---|
| Generated instructions | 52K+ |
| Generated instances (input/output pairs) | 82K+ |
| Tasks judged to describe a valid task | 92% |
| Tasks with all fields fully valid | 54% |
| Improvement over vanilla GPT-3 on Super-NaturalInstructions | **+33% absolute** |
| Gap to InstructGPT-001 on 252 expert-written user-oriented instructions | **~5% absolute** |
| Total cost (generation + fine-tuning) | ~$600 + $338 |

Diversity analysis (parsing instructions with a Berkeley Neural Parser to extract root verb + direct object) found roughly half the generated instructions follow a clean verb-noun pattern (e.g. "classify the sentiment"), while the other half have more complex phrasing or are framed as questions — evidence the generation process isn't just repeating a narrow template.

## Comparison to Prior Work

- vs. **[[InstructGPT]] / [[RLHF]]** — InstructGPT's data pipeline requires paid human annotators to write instructions, write demonstrations, and rank outputs for reward-model training; Self-Instruct requires 175 human-written seed tasks total and generates everything else automatically. Self-Instruct doesn't incorporate a human-preference signal at all — it produces supervised fine-tuning data only, not a reward model — so it closes most, but not all, of the gap to a fully RLHF'd model.
- vs. hand-curated instruction datasets (e.g. Super-NaturalInstructions itself) — Self-Instruct trades some per-example quality control for near-zero marginal cost per additional example, and the ROUGE-L deduplication filter became a standard cheap technique that later synthetic-data pipelines (Alpaca, and later variants like SeDi-Instruct) reused directly.

## Limitations

- Quality control is entirely heuristic (ROUGE-L overlap, keyword filters, basic validity checks) rather than human-reviewed — only 54% of generated tasks are fully valid across every field, meaning a meaningful fraction of the pipeline's raw output is noise that relies on scale to wash out.
- The method inherits and can amplify the base model's own biases and blind spots, since the model is generating instructions, inputs, *and* outputs — there's no independent human signal correcting systematic errors.
- No human-preference/reward signal is captured at all, unlike RLHF — Self-Instruct only produces supervised instruction-following data, not preference data for a reward model.
- Later work found that a smaller, carefully curated instruction set can outperform Self-Instruct-scale but noisier data — bigger is not strictly better once quality variance is high.

## Why It Matters

Self-Instruct was the paper that proved instruction-tuning data itself doesn't require a human in the loop for every example — a capable base model can bootstrap its own training set from a small seed. This directly inspired Stanford's Alpaca (which used Self-Instruct-style generation against a stronger teacher model to fine-tune LLaMA cheaply) and the wider wave of open-source instruction-tuned models that followed, fundamentally changing the cost structure of building an instruction-following LLM from "hire annotators" to "run generation against an existing model."

## Related Concepts

[[InstructGPT]] · [[RLHF]] · [[In-Context Learning]] · [[Toolformer]]
