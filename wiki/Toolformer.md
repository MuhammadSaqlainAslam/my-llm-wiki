---
created: "2026-06-10"
title: "Toolformer: Language Models Can Teach Themselves to Use Tools"
authors: "Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Luke Zettlemoyer, Nicola Cancedda, Thomas Scialom"
year: 2023
arxiv: "2302.04761"
tags: [tool-use, api-calls, language-models, agents, self-supervised]
citation_count: 5309
tldr: "Fine-tunes language models to autonomously decide when and how to call external APIs (calculator, search, calendar, translator, Q&A) mid-generation, using a self-supervised bootstrapping procedure that requires only a handful of demonstrations per tool — no human annotation of when to call."
aliases: ["Toolformer"]
---

# Toolformer: Language Models Can Teach Themselves to Use Tools

> Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Luke Zettlemoyer, Nicola Cancedda, Thomas Scialom, "Toolformer: Language Models Can Teach Themselves to Use Tools" (arXiv:2302.04761)

## TL;DR

Large LMs are strong at open-ended reasoning but embarrassingly weak at things a calculator or lookup table solves trivially — arithmetic, exact facts, up-to-date information. The obvious fix is to let the model call external tools. The hard part is teaching it *when* to call a tool, *what arguments* to pass, and *how* to weave the result back into the text — without expensive human-labeled examples of tool use for every position in a corpus.

Toolformer solves this with a self-supervised bootstrapping trick: use the LM itself (via a handful of in-context examples per tool) to generate candidate API calls at many positions in a text corpus, actually execute those calls, and keep only the calls that measurably reduce the model's loss on the following text. The resulting API-call-annotated corpus is used to fine-tune the model, so it learns to insert API calls exactly where they help — not because a human annotated them, but because they were empirically useful.

## The Idea

1. **Sample candidate call positions.** For each tool (calculator, Q&A system, two search engines, a translator, a calendar), prompt the base LM with a few hand-written demonstrations of that tool's API-call format. The LM proposes candidate insertion points and call arguments throughout a large unlabeled text corpus.
2. **Execute and filter.** Actually run each candidate API call, get its result, and splice it into the text. Compare the model's loss on the text *following* the call with the call+result present vs. absent (and vs. a "wrong" result). Keep only calls where the *correct* result meaningfully lowers the loss more than no call or an incorrect one — this is the self-supervised filtering signal, no human judgment needed.
3. **Fine-tune on the filtered corpus.** The surviving examples — text interleaved with the API calls that provably helped predict what comes next — become fine-tuning data. The model learns a general policy for *when* an API call is worth making, in whatever new context it later encounters.
4. **Inference.** At generation time, the fine-tuned model can emit an API-call token sequence mid-generation; the call is executed, the result is inserted, and generation continues — normal decoding, augmented with tool calls exactly where the model has learned they help.

## Key Results

- Substantially improved **zero-shot** performance across a range of downstream tasks (arithmetic, question answering, multilingual QA, temporal/factual lookup) versus the same base model without tool access.
- Often **competitive with much larger models** that don't use tools — the smaller tool-augmented model closes much of the gap by offloading exactly the sub-tasks it's bad at.
- Core language modeling ability is preserved — fine-tuning to use tools does **not** degrade general perplexity/fluency, because the filtering step only keeps calls that help, so the model isn't forced to call tools indiscriminately.

## Why It Matters

Toolformer's central contribution isn't "LMs can call APIs" (that was already being explored) — it's the **self-supervised loss-based filter** for deciding which tool calls are worth learning from, removing the need for humans to annotate exactly where and how each API should be invoked. This bootstrapping recipe — generate candidates with the model itself, execute them, keep only what measurably helps — became a template for later tool-use and agent training approaches that need to scale past hand-labeled demonstrations.

## Limitations

- Tools are fixed and hand-specified in advance (a small set of APIs with hand-written few-shot demonstrations) — the model doesn't discover new tools or interfaces on its own.
- Each API call is treated independently; there's no support for multi-step tool-use plans or chaining tool outputs into further tool calls within the paper's method.
- The filtering procedure requires actually executing candidate calls at data-generation time, which has a real compute/API cost proportional to corpus size and number of tools.

## Related Concepts

[[ReAct Synergizing Reasoning and Acting in Language Models|ReAct]] · [[In-Context Learning]] · [[RLHF]] · [[InstructGPT]]
