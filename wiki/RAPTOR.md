---
created: "2026-08-14"
title: "RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval"
authors: "Parth Sarthi, Salman Abdullah, Aditi Tuli, Shubh Khanna, Anna Goldie, Christopher D. Manning"
year: 2024
arxiv: "2401.18059"
tags: [retrieval, rag, hierarchical-summarization, long-document-qa, clustering]
citation_count: 0
tldr: "Flat-chunk RAG only ever retrieves local snippets, so it fails on questions that need a whole-document view. RAPTOR recursively clusters and summarizes chunks into a tree of increasingly abstract summaries, then retrieves from every level at once — a narrow question pulls a leaf chunk, a thematic question pulls a root-level summary. Coupled with GPT-4, it improves the best prior QuALITY benchmark accuracy by 20 absolute points. ICLR 2024."
aliases: ["RAPTOR"]
---

# RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval

> Parth Sarthi, Salman Abdullah, Aditi Tuli, Shubh Khanna, Anna Goldie, Christopher D. Manning, "RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval", ICLR 2024 (arXiv:2401.18059)

## TL;DR

Standard [[RAG]] chunks a document, embeds the chunks, and at query time retrieves the top-k most similar chunks. This works fine for narrow fact-lookup ("what year was X founded") but breaks down for questions that require synthesizing information spread across an entire document or book ("what is the overall arc of this novel's plot") — no single chunk, or even the top handful of chunks, contains that answer. Retrieving *more* chunks doesn't fix it either, since [[Lost-in-the-Middle|Lost in the Middle]] showed models struggle to use information buried in a long, undifferentiated context.

RAPTOR's fix: don't just index the raw chunks — recursively **cluster and summarize** them into a tree. Leaves are the original chunks; each level up is an LLM-generated summary of a cluster of nodes from the level below, continuing until a single root summary remains. At query time, RAPTOR retrieves from *every level of the tree at once*, so a narrow factual query naturally matches a specific leaf chunk while a broad thematic query naturally matches a high-level summary node. Coupled with GPT-4, RAPTOR retrieval improves the best prior accuracy on the QuALITY long-document QA benchmark by **20 absolute points** — state of the art at publication.

## The Problem / Motivation

Retrieval-augmented generation retrieves fixed-size, contiguous text chunks and hands them to the LLM as context. This is a good match for questions whose answer lives in one localized place in the source text. It is a poor match for questions that require connecting information *across* many chunks or reasoning about the document as a whole — e.g., summarizing a whole book, identifying a novel's overarching theme, or tracing how a character's motivations evolve across chapters. No amount of top-k tuning fixes this: the information needed simply isn't compressed into any single retrievable unit in a flat index.

## The Idea

Build a tree, bottom-up, over the document, then retrieve from the whole tree at once instead of just the leaves.

1. Split the document into chunks (leaves).
2. Cluster semantically similar chunks together.
3. Ask an LLM to **summarize each cluster** — that summary becomes a new node one level up.
4. Recurse: cluster the summary nodes, summarize again, repeat until a single root node (a summary of the whole document) remains.
5. At query time, search across nodes from **all levels simultaneously** rather than only the leaves.

Concrete example: ask *"what did character X say to character Y in chapter 3"* and the query embedding matches a specific leaf chunk almost verbatim. Ask *"what is the overall theme of this novel"* and the query embedding instead matches a root-level (or near-root) summary node, which already condenses the whole book's arc — something no single leaf chunk could ever contain.

```
                    [ Root summary ]
                    /              \
          [ Summary A ]        [ Summary B ]        ← level 2 (summaries of summaries)
          /     |    \          /    |    \
      [S1]   [S2]   [S3]     [S4]  [S5]  [S6]        ← level 1 (cluster summaries)
       |       |      |        |     |     |
    chunks  chunks chunks   chunks chunks chunks     ← level 0 (leaves = raw text chunks)

  query: "chapter 3 dialogue"  →  matches a level-0 leaf
  query: "novel's overall theme" → matches the root or a level-2 summary
```

## Architecture / Method

1. **Chunking + embedding.** Split the source document into small chunks (~100 tokens) and embed each with SBERT — these become the tree's leaf nodes.
2. **Dimensionality reduction (UMAP).** Raw embeddings are high-dimensional, where distance metrics used by clustering algorithms behave poorly. UMAP reduces dimensionality first, with its "number of neighbors" parameter tuned to separate global structure (broad themes) from local structure (fine detail) — effectively producing a hierarchy of cluster granularities for free.
3. **Soft clustering (GMM).** Gaussian Mixture Models cluster the reduced embeddings. Clustering is *soft*: a chunk can belong to more than one cluster, which matters because real text passages often relate to multiple themes at once. The number of clusters is chosen automatically via the Bayesian Information Criterion (which penalizes overly complex models while rewarding fit), and cluster parameters (means, covariances, mixture weights) are fit with Expectation-Maximization. If a cluster's combined text would exceed the summarizer's context limit, it's recursively re-clustered into smaller sub-clusters first.
4. **Summarization.** Each cluster's member texts are concatenated and summarized by an LLM (gpt-3.5-turbo in the paper) into a single new node, placed one level up the tree.
5. **Recurse.** Re-embed and re-cluster the newly created summary nodes, summarize again, and continue until clustering collapses to a single root node — a summary of the entire document.
6. **Retrieval — two strategies compared:**
   - **Tree traversal:** start at the root, retrieve the top-k children by cosine similarity to the query, descend into their children, repeat for a fixed depth. Gives explicit control over breadth (k) vs. specificity (depth) but requires committing to a traversal path level by level.
   - **Collapsed tree:** flatten *every* node from *every* level into one undifferentiated pool, and retrieve the top-k by cosine similarity to the query directly, until a token budget is filled — no notion of "level" at retrieval time at all. The paper finds collapsed tree **outperforms** tree traversal, since it lets retrieval settle on whatever granularity level actually matches the query, rather than being forced through a fixed layer-by-layer path.

## Key Results

- Coupling **GPT-4 with RAPTOR retrieval improves the best previously reported accuracy on the QuALITY benchmark by 20 absolute points** — a new state of the art at publication. QuALITY is a long-document QA benchmark specifically designed to require multi-step, whole-passage reasoning rather than single-sentence fact lookup.
- RAPTOR outperforms flat-chunk baselines (BM25, dense passage retrieval) across multiple backbone language models, with the largest gains concentrated on questions that require synthesizing information across multiple parts of a document — exactly the failure mode flat retrieval can't address.
- The **collapsed tree** retrieval strategy consistently beats **tree traversal**, validating the "retrieve at whatever level matches the query" design over a fixed top-down path.

## Comparison to Prior Work

- vs. **[[RAG]]** (Lewis et al.) — RAG's index is flat: chunks in, chunks out, no hierarchy. RAPTOR changes *what's in the index* — a multi-level tree of abstractions — while remaining compatible with the same retrieve-then-generate pipeline.
- vs. **[[Self-RAG]]** — orthogonal concerns. Self-RAG controls *whether and when* to retrieve, and critiques retrieved passages for quality; RAPTOR controls *what structure the retrieval corpus has*. The two could in principle compose.
- vs. **[[Lost-in-the-Middle|Lost in the Middle]]**'s findings — that paper shows models struggle to use information buried in the middle of a long, flat context. RAPTOR sidesteps this by pre-condensing information into summary nodes *before* retrieval, so the model isn't handed a pile of raw chunks and asked to find the needle itself — the needle is already distilled into a short summary.
- vs. **[[RETRO]]** — RETRO retrieves from a massive external corpus of text chunks at every few tokens during generation, optimized for injecting broad world knowledge cheaply at pretraining scale. RAPTOR instead builds a bespoke hierarchical index over a *specific* document/corpus for downstream QA, optimized for multi-granularity reasoning over that document rather than open-domain knowledge injection.

## Limitations

- Building the tree costs LLM summarization calls proportional to the number of clusters at every level — a real upfront indexing cost that scales with document size and tree depth, unlike flat chunking which needs no LLM calls to index.
- Summary nodes necessarily lose fine-grained detail present in the leaves; a summary is a lossy compression, so retrieval that lands on a high-level node may miss the exact phrasing or numeric detail a leaf chunk would have contained.
- Cluster quality depends on UMAP/GMM hyperparameters (number of neighbors, BIC-selected cluster count) — poorly tuned clustering produces incoherent groupings and, downstream, incoherent summaries.
- The benefit is concentrated on documents with genuine hierarchical/thematic structure (long-form narrative, technical reports, books); short or unstructured documents gain little from building a multi-level tree over them.

## Why It Matters

RAPTOR reframed RAG's design space: instead of treating retrieval quality purely as a scoring/embedding problem, it treats the **structure of the retrieval index itself** as a first-class design choice. Multi-scale, hierarchical retrieval — rather than a flat pool of equally-sized chunks — became a standard tool for long-document and whole-corpus QA, directly targeting the failure mode that flat chunking and long-context stuffing (per [[Lost-in-the-Middle|Lost in the Middle]]) both struggle with.

## Related Concepts

[[RAG]] · [[Self-RAG]] · [[Lost-in-the-Middle|Lost in the Middle]] · [[RETRO]]
