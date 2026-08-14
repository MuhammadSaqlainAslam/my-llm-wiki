---
created: "2026-08-14"
title: "Lost in the Middle: How Language Models Use Long Contexts"
authors: "Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, Percy Liang"
year: 2023
arxiv: "2307.03172"
tags: [long-context, retrieval, positional-bias, evaluation, empirical-analysis]
citation_count: 0
tldr: "Language models don't use long contexts uniformly — accuracy is highest when the relevant information sits at the very start or very end of the input, and drops sharply (sometimes below a no-context baseline) when it's buried in the middle, even for models explicitly built for long context. A purely diagnostic paper that became one of the most-cited practical constraints on RAG system design."
aliases: ["Lost in the Middle"]
---

# Lost in the Middle: How Language Models Use Long Contexts

> Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, Percy Liang, "Lost in the Middle: How Language Models Use Long Contexts" (arXiv:2307.03172)

## TL;DR

By 2023, language models advertised context windows of 16K, 32K, even 100K tokens. The implicit promise: stuff in everything relevant and the model will find what it needs, wherever it sits. This paper tests that promise directly and finds it's false in an interesting, consistent way.

Take a question the model can answer, put the one document containing the answer inside a list of N irrelevant distractor documents, and measure accuracy as you slide that document's position from first to last. The result is a U-shaped curve: accuracy is highest when the answer is at position 1 or position N, and it sags — sometimes dropping below the accuracy of giving the model *no documents at all* — when the answer sits in the middle of the list. This holds across GPT-3.5-Turbo, Claude, and open models like MPT and LongChat, and it holds even for models specifically extended and marketed for long-context use. Bigger context windows don't fix it; models with longer windows just have a longer, deeper middle to lose things in.

## The Problem / Motivation

Long-context capability is usually benchmarked with "needle in a haystack" tests: can the model retrieve one specific fact planted somewhere in a huge context? Those tests report a single retrieval-accuracy number, or maybe accuracy averaged over positions — but averaging hides exactly the effect this paper is looking for. The real question for anyone building a RAG system, a long-document QA system, or a multi-document summarizer isn't "can the model ever find the right passage" — it's "does *where* the right passage sits in the prompt change whether the model uses it correctly." If it does, that's a design constraint on every system that concatenates retrieved chunks into a context window, not just a curiosity about model internals.

## The Idea / Method

Two controlled experiments, designed to isolate positional effects from everything else:

**Multi-document question answering.** Take a question with a known, verified answer contained in exactly one "gold" document. Assemble a context of that gold document plus $k-1$ distractor documents (topically related but not answer-bearing, so the model can't just skim for topic mismatch). Vary two things independently: the position of the gold document within the list (1st, in the middle, or last), and the total number of documents $k$ (i.e., the total context length). Everything else — the question, the gold document's content, the distractor pool — stays fixed. Only position moves.

**Key-value retrieval (synthetic control).** To check whether the effect is about *position* per se, rather than something about natural-language document semantics (e.g., documents near the edges happening to be easier to compare against the question), the authors also build a synthetic task: a long JSON object of random key-value pairs, where the model must retrieve the value for one specified key. There's no semantic content to exploit here at all — just a pure "find the item at position $p$ out of $k$" test. The same U-shape appears, confirming it's a positional effect, not a content effect.

**Concrete example.** Say a context holds 20 documents and the gold document — the one containing the actual answer to "What year was the Eiffel Tower completed?" — is placed at position 1, position 10 (middle), or position 20 (last). The model reliably finds and uses the fact at position 1 or 20. At position 10, accuracy drops substantially — the model may fail to even mention the correct year, despite the fact being sitting right there in its context.

## Key Results

The signature finding is the U-shaped accuracy-vs-position curve:

```
accuracy
   ^
   |*                                          *
   |  *                                      *
   |    *                                  *
   |      *                              *
   |        *  *  *  *  *  *  *  *  *  *
   +----------------------------------------------> position of
   1        5       10 (middle)      15       20     relevant doc
```

- Accuracy is consistently **highest when the relevant document is at the very beginning or the very end** of the context.
- Accuracy **degrades substantially when the relevant document is in the middle** — in several settings, performance in the worst middle positions falls **below the accuracy of giving the model no documents at all** (i.e., forcing it to rely only on parametric/closed-book knowledge). Concatenating more context can actively hurt.
- The degradation **gets worse as the number of documents (total context length) grows** — more distractors means a deeper, more damaging middle.
- This holds for **models explicitly extended and marketed for long context** (not just base models with short native windows) — extending the context window does not by itself fix positional robustness.
- The same U-shape appears in the **synthetic key-value retrieval task**, which has zero natural-language semantics to exploit, confirming the effect is structurally about position in the input sequence, not about document content or topical relevance patterns.

## Comparison to Prior Work

- vs. **"needle in a haystack" benchmarks** — those typically report a single aggregate retrieval-accuracy number (or an average over positions), which averages away exactly the signal this paper isolates. A model can score well on an aggregate needle-in-a-haystack metric while still being unreliable whenever the needle happens to land in the middle third of the context.
- vs. **marketing claims of "N-token context windows"** — a stated context window length says nothing about whether the model uses that whole window *uniformly*. This paper shows window length and effective, position-robust utilization are different properties, and the gap between them doesn't close just by training longer-context models.

## Limitations

- The core experiments test specific task types — multi-document QA and synthetic key-value retrieval — that isolate the positional effect cleanly but may not capture every way long-context degradation manifests in more open-ended tasks (e.g., long-form summarization, multi-hop reasoning across several buried facts at once).
- The paper is purely diagnostic: it does not propose an architectural or training fix for the U-shaped bias, only documents it and its implications for evaluation protocol design.
- Findings are specific to the models evaluated at the time (GPT-3.5-Turbo, Claude, MPT, LongChat); the underlying architectural or positional-encoding causes are analyzed but not definitively isolated to a single root cause.

## Why It Matters

This became one of the most frequently cited *practical* findings in applied LLM work, because it directly shapes how RAG and long-context systems should be engineered: how many chunks to retrieve, and critically, *where to place the most relevant chunk* in the assembled prompt (favor the start or the end; don't bury the best evidence in the middle of a long context stack). It's a standard reference point whenever someone argues that a bigger context window alone solves a long-context problem — this paper is the counterevidence that context length and effective context utilization are separate axes, and that evaluation protocols for long-context models need to test position robustness explicitly, not just maximum retrievable distance.

## Related Concepts

[[RAG]] · [[RAPTOR]] · [[Self-RAG]] · [[KV Cache Optimization]] · [[In-Context Learning]]
