---
created: "2026-08-14"
title: "Finetuned Language Models Are Zero-Shot Learners"
authors: "Jason Wei, Maarten Bosma, Vincent Y. Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M. Dai, Quoc V. Le"
year: 2021
arxiv: "2109.01652"
tags: [instruction-tuning, zero-shot, generalization, finetuning, multi-task]
citation_count: 5340
tldr: "Finetune a large pretrained LM on 60+ NLP datasets rewritten as natural-language instructions, grouped into task clusters. Held-out-cluster evaluation shows the model generalizes to unseen task types, not just unseen datasets — FLAN zero-shot beats 175B GPT-3 zero-shot on 20 of 25 datasets, and even beats GPT-3 few-shot on several."
aliases: ["FLAN", "Finetuned Language Net"]
---

# Finetuned Language Models Are Zero-Shot Learners

> Jason Wei, Maarten Bosma, Vincent Y. Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M. Dai, Quoc V. Le, "Finetuned Language Models Are Zero-Shot Learners" (arXiv:2109.01652)

## TL;DR

GPT-3-style models are strong *few-shot* learners: give them a handful of examples in the prompt and they generalize well. But give the same model a plain zero-shot instruction with no examples, and performance drops sharply — the model was never trained to map an instruction directly onto the right behavior, only to imitate patterns from web text.

FLAN's fix: take a large pretrained LM and finetune it on a huge mixture of existing NLP datasets, each rewritten into natural-language instruction form ("Is the sentiment of this review positive or negative?" instead of a bare classification setup). Do this across many different *kinds* of tasks at once, and the model doesn't just memorize those tasks — it learns the general skill of following an instruction, which transfers to task types it never saw during finetuning.

The headline result: FLAN's zero-shot performance surpasses zero-shot 175B GPT-3 on 20 of 25 datasets evaluated, and even beats GPT-3's *few-shot* performance by a large margin on ANLI, RTE, BoolQ, AI2-ARC, OpenbookQA, and StoryCloze — despite FLAN using no in-context examples at all.

## The Problem / Motivation

Large LMs pretrained purely on next-token prediction are excellent at continuing a pattern once you show them a few examples (in-context / few-shot learning). But zero-shot prompting — just describing the task in plain language with no examples — performs much worse. The model has no trained-in notion of "an instruction" as a distinct kind of input; it only knows how to continue text that resembles its training distribution, and a bare instruction with no examples often doesn't look like anything common in that distribution.

## The Idea

Explicitly train the model to bridge that gap. Take a large collection of existing, already-labeled NLP datasets, and for each one write several different natural-language instruction templates that describe the task the way a person would ask for it — e.g., a sentiment dataset gets rephrased as "Is the sentiment of this review positive or negative?" (and several other phrasings, including some with the input/output relationship reversed for diversity). Group the datasets into task clusters by task type — natural language inference, sentiment analysis, closed-book QA, translation, summarization, and so on — and finetune the model on a mixture spanning many clusters at once.

The key evaluation trick is **held-out-cluster generalization**, not held-out-dataset generalization. To test whether the model learned the *skill* of instruction-following rather than memorizing task-specific patterns, evaluate zero-shot on datasets from a cluster whose entire task type was excluded from finetuning — not just a specific held-out dataset, but every dataset belonging to that whole cluster. Concretely: to test natural language inference (NLI) performance, remove **every** NLI-style dataset (and anything from a cluster containing NLI-like tasks) from the finetuning mixture, finetune on everything else, and only then test zero-shot on an NLI dataset like RTE. If the model performs well despite never having seen a single NLI example, it learned something more general than a specific dataset's quirks.

## Architecture / Method

1. **Base model.** A 137B-parameter decoder-only LM (the same pretrained LaMDA-PT model family), pretrained on web documents, dialog data, code, and Wikipedia (2.49T BPE tokens, 32k vocabulary, ~10% non-English).
2. **Dataset aggregation.** Gather 62 text datasets from TensorFlow Datasets, spanning both language understanding and generation tasks.
3. **Cluster grouping.** Organize the 62 datasets into 12 task clusters by task type: Natural Language Inference, Reading Comprehension, Closed-Book QA, Translation, Commonsense Reasoning, Coreference Resolution, Sentiment Analysis, Paraphrase Detection, Struct-to-Text, Summarization, plus two hybrid clusters (Reading Comprehension + Commonsense, Paraphrase + NLI).
4. **Instruction templates.** For each dataset, manually write up to 10 unique natural-language instruction templates describing the task — for NLI, instead of GPT-3's awkward sentence-completion framing, FLAN asks directly: *"Does [premise] mean that [hypothesis]?"* — with some templates reversing the input/output direction for extra diversity.
5. **Instruction tuning.** Finetune the base model on a mixture of all instruction-rephrased examples, sampling proportionally across datasets (capped at 30k examples per dataset, mixing rate maximum 3k) — a small compute cost, under 2% of pretraining's gradient steps.
6. **Held-out-cluster evaluation.** For each cluster being evaluated, exclude every dataset belonging to that cluster (and clusters containing similar task types) from finetuning entirely, then evaluate zero-shot on that cluster's datasets. Repeat with a different cluster held out each time to cover all 12.

```
Raw NLI example (premise, hypothesis, label)
        │
        ▼
Instruction template #1: "Does [premise] mean that [hypothesis]?"
Instruction template #2: "[premise] Based on that information, is [hypothesis] true?"
Instruction template #3: "Given [premise], is it guaranteed true that [hypothesis]?"
        │
        ▼
Finetune on templates from clusters A, B, C, ... (NLI cluster excluded)
        │
        ▼
Zero-shot evaluation on an NLI dataset (e.g. RTE) — never seen during finetuning
```

## Key Results

- FLAN zero-shot **surpasses zero-shot 175B GPT-3 on 20 of 25 evaluated datasets**.
- FLAN zero-shot **outperforms GPT-3's few-shot performance by a large margin** on ANLI, RTE, BoolQ, AI2-ARC, OpenbookQA, and StoryCloze — despite using no in-context examples.
- **Scaling ablation:** instruction tuning substantially helps held-out-task performance at 100B+ parameter scale, but at 8B parameters and below, instruction tuning actually *hurts* held-out performance — likely because the model's limited capacity gets consumed learning many surface-level task formats rather than a transferable instruction-following skill.
- Performance on held-out clusters improves as more distinct task clusters are included in the finetuning mixture — task diversity, not just data volume, drives generalization.

## Comparison to Prior Work

- vs. **GPT-3** — GPT-3 relies purely on scale and in-context few-shot examples; it has no training stage specifically targeting instruction-following, so zero-shot performance lags behind few-shot. FLAN adds one lightweight finetuning stage that closes (and often reverses) that gap without needing any in-context examples at inference time.
- vs. **[[InstructGPT]] / [[RLHF]]** — InstructGPT builds directly on the instruction-tuning idea FLAN introduces, then adds a second stage: training a reward model on human preference comparisons and using RL (PPO) to further align outputs with what humans actually prefer. FLAN is the earlier, simpler, purely-supervised instruction-tuning step; InstructGPT is instruction tuning *plus* preference-based RL on top.
- vs. **[[Self-Instruct]]** — Self-Instruct addresses the bottleneck FLAN doesn't: FLAN depends on humans hand-writing instruction templates for a large collection of already-labeled datasets, which doesn't scale indefinitely. Self-Instruct instead bootstraps new instructions and examples from the model itself, removing the human-authored-template dependency.

## Limitations

- Depends on a large existing collection of labeled NLP datasets, each manually rewritten into instruction templates by humans — doesn't scale to generating novel instructions from scratch (see [[Self-Instruct]] for a bootstrapped alternative).
- The benefit is scale-dependent: instruction tuning helps at 68B–137B parameters but can actively hurt smaller (≤8B) models on held-out tasks.
- Purely supervised instruction tuning, with no preference modeling or RL — outputs can still be technically instruction-following but misaligned with more nuanced human preferences (the gap [[RLHF]] later targets).
- Possible train/test contamination: some evaluation examples may overlap with pretraining data, though the authors' post-hoc analysis found no evidence this substantially affected results.

## Why It Matters

FLAN established **instruction tuning** as a distinct, effective, and now-standard training stage for large language models — finetune on a diverse mixture of tasks phrased as natural-language instructions to unlock strong zero-shot generalization to unseen task types. This directly set up the recipe that [[InstructGPT]] and the broader RLHF pipeline build on: instruction tuning first to get a model that reliably follows instructions, then preference-based RL on top to align its outputs with what humans actually want.

## Related Concepts

[[Self-Instruct]] · [[InstructGPT]] · [[RLHF]] · [[In-Context Learning]]
