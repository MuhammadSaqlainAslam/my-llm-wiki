---
created: "2026-08-14"
title: "Gorilla: Large Language Model Connected with Massive APIs"
authors: "Shishir G. Patil, Tianjun Zhang, Xin Wang, Joseph E. Gonzalez"
year: 2023
arxiv: "2305.15334"
tags: [tool-use, api-calls, retrieval-augmented, finetuning, hallucination-reduction]
citation_count: 1567
tldr: "Finetunes LLaMA-7B specifically to write correct calls to real ML-model APIs (HuggingFace, TorchHub, TensorHub), using a Retriever-Aware Training scheme so the model actually uses retrieved API docs at inference instead of ignoring them. Beats GPT-4 and Claude on both accuracy and hallucination rate for API-call generation, and adapts automatically when an API's docs change."
aliases: ["Gorilla", "Gorilla LLM"]
---

# Gorilla: Large Language Model Connected with Massive APIs

> Shishir G. Patil, Tianjun Zhang, Xin Wang, Joseph E. Gonzalez, "Gorilla: Large Language Model Connected with Massive APIs" (arXiv:2305.15334, NeurIPS 2024)

## TL;DR

Ask GPT-4 "I want to classify images of different kinds of dogs" and it will happily write you a `pipeline(...)` call — for a model that doesn't exist. Not a wrong choice, an *invented* one: a plausible-looking model name that was never real. This is the sharp edge of hallucination once you leave open-ended text generation and enter a domain with exact, closed answers — real APIs have exact names, exact required arguments, and thousands of near-identical siblings to confuse a model that's only ever seen a handful of examples of each during pretraining.

Gorilla's fix is to stop treating API-calling as something a general-purpose LLM should just know, and instead finetune a model specifically for it — on a large dataset of real API documentation, and critically, trained to actually *use* a retrieved document when one is handed to it at inference time. The result: a 7B LLaMA-based model that beats GPT-4 and Claude at writing correct API calls, while also inventing far fewer nonexistent ones.

## The Problem / Motivation

HuggingFace, TorchHub, and TensorHub together expose well over a thousand ML-model APIs, each documented in natural language with its own exact function name and required argument names/types. A user's request ("classify dog breeds from an image") maps to one correct API call out of this large, constantly-changing set — get the model name slightly wrong, or invent an argument that doesn't exist, and the call simply fails.

Even GPT-4, tested directly on this task, frequently **hallucinates**: it produces syntactically clean, confident-looking API calls that reference models or arguments which do not exist. This is a different failure mode from being merely *wrong* — recommending Stripe's API to check a bank balance is an incorrect choice; inventing a Stripe API to classify images is a hallucination, because that endpoint was never real. General-purpose LLMs, trained broadly rather than on this narrow, exact-match domain, are unusually prone to the latter.

## The Idea

Two ingredients, combined:

1. **APIBench** — a large finetuning dataset of (natural-language instruction, ground-truth API call) pairs, built by exhaustively collecting real API documentation from HuggingFace, TorchHub, and TensorHub, then generating synthetic instructions per API (self-instruct style) so the model sees many phrasings of "how would a user ask for this API."
2. **Retriever-Aware Training (RAT)** — rather than only finetuning zero-shot (no docs shown), Gorilla is *also* trained with the correct API documentation included in the prompt, so it learns to actually condition on retrieved text rather than ignore it. This matters because the paper found something counterintuitive: naively bolting a retriever onto an LLM at inference time (without training for it) can *hurt* accuracy, since the model was never taught to trust or parse retrieved context correctly.

At inference, a retrieval system (e.g. BM25 or a GPT-based retriever) fetches the most relevant API doc snippets for the user's request and prepends them to the prompt; the RAT-trained Gorilla model then reliably grounds its output in whatever documentation it's handed — including *updated* documentation for APIs that changed after Gorilla was trained, without any retraining.

**Concrete example** (from the paper): given "I want to classify images of different kinds of dogs," GPT-4 invents a plausible but nonexistent model name; Claude picks a real API but from the wrong library entirely; Gorilla correctly identifies the task and returns a fully-qualified, real HuggingFace `pipeline(...)` call.

## Architecture / Method

```
 user instruction
       │
       ▼
 ┌─────────────┐        (optional) relevant API docs
 │  Retriever  │───────────────────┐
 │ (BM25 / IR) │                   │
 └─────────────┘                   ▼
                          ┌──────────────────────┐
 instruction  ──────────▶ │ Gorilla (LLaMA-7B,   │ ──▶  correct, fully-
                          │  Retriever-Aware      │      qualified API call
                          │  finetuned)           │
                          └──────────────────────┘
```

1. **Build APIBench.** Exhaustively collect API documentation: every TorchHub API (94), every TensorHub API (696), and the most-downloaded HuggingFace models per task category (~925) — roughly 1,645 unique APIs after filtering and dedup. Generate ~10 synthetic (instruction, API call) pairs per API via self-instruct, yielding ~16,450 instruction-API pairs.
2. **Finetune LLaMA-7B** on this dataset in two regimes: zero-shot (no docs in the prompt) and retriever-aware (the correct doc chunk included in training prompts some of the time), so the model learns both to recall common APIs from memory and to defer to retrieved documentation when it's present and relevant.
3. **At inference**, an information-retrieval system fetches the most relevant API doc(s) for the user's instruction and prepends them to the prompt before Gorilla generates its call — this is what lets Gorilla track API updates (renamed arguments, new versions) without retraining, something a purely memorized zero-shot model cannot do.
4. **Evaluate with AST sub-tree matching**, not exact string match: the generated function call is parsed into an abstract syntax tree and checked for whether it matches the ground-truth API's structure and required arguments as a sub-tree. This lets functionally-correct calls with superficially different formatting or extra optional arguments still count as correct, while cleanly separating two distinct failure types — **inaccuracy** (a valid, real API used incorrectly) from **hallucination** (an API that doesn't exist at all).

## Key Results

- Gorilla **outperforms GPT-4, GPT-3.5, and Claude on both accuracy and hallucination rate** for API-call generation across all three sources (HuggingFace, TorchHub, TensorHub) in the paper's AST-matching evaluation.
- **Retrieval alone does not reliably help** — the paper shows that giving GPT-4/Claude retrieved docs via naive prompting can *hurt* their accuracy, since they weren't trained to prioritize retrieved context over their own priors. Gorilla's Retriever-Aware Training is what makes retrieval reliably beneficial rather than a coin flip.
- Gorilla generalizes to **test-time document changes**: because it was trained to condition on retrieved docs, showing it an updated API specification at inference (e.g., a renamed argument) lets it adapt correctly without any retraining.
- 3-shot in-context examples improve GPT-family models' *syntactic* correctness (and can match Gorilla on the smaller TorchHub subset), but Gorilla's advantage in accuracy and low hallucination holds broadly, especially at HuggingFace's larger, noisier API scale.

## Comparison to Prior Work

- vs. **[[ToolLLM]]** — ToolLLM targets a much broader and more heterogeneous space (16,000+ real-world REST APIs across 49 categories, with multi-tool, multi-step planning via DFSDT). ToolLLM's own evaluation used APIBench (Gorilla's benchmark) as an out-of-distribution test and found ToolLLaMA generalizes to it zero-shot — Gorilla is narrower in scope (single-call, ML-model APIs) but was the benchmark others measured against.
- vs. **[[Toolformer]]** — Toolformer self-supervises a small, fixed set of simple tools (calculator, search, calendar) by filtering on loss reduction; Gorilla targets a much larger set (1,645+) of complex APIs with exact-match syntax requirements, using supervised finetuning on curated documentation plus retrieval, rather than self-supervised bootstrapping.
- vs. plain **[[RAG]]**-style retrieval-augmentation — Gorilla's central finding is that retrieval augmentation by itself is not sufficient (and can even hurt); the model must be *trained* to use retrieved context (RAT), not just have it appended at inference.

## Limitations

- Scope is specifically ML-model APIs from HuggingFace, TorchHub, and TensorHub — not general-purpose REST APIs (see [[ToolLLM]] for that broader setting).
- Best performance depends on a working retrieval system at inference time; without it (pure zero-shot), Gorilla still helps but the adaptability to changing APIs is lost.
- APIBench is a snapshot of real API documentation at one point in time, so its instructions and ground truths need periodic refreshing as the underlying libraries evolve.
- AST sub-tree matching evaluates structural/functional correctness, not runtime behavior — an API call can pass evaluation and still fail if executed against a live, changed service.

## Why It Matters

Gorilla was one of the first papers to demonstrate that a small, specifically-finetuned open model can beat much larger general-purpose LLMs at a narrow but practically essential task — writing exact, executable API calls — and that doing so well requires *training* a model to use retrieval, not merely giving it retrieved text. That distinction (retrieval-augmentation helps only when the model is trained to trust it) directly informs how modern function-calling and tool-use systems are built and evaluated.

## Related Concepts

[[ToolLLM]] · [[Toolformer]] · [[RAG]] · [[In-Context Learning]]
