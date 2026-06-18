---
created: "2026-06-03"
title: "WordPiece Tokenization"
tags: [glossary, tokenization]
tldr: "A subword tokenization algorithm that greedily merges character pairs to maximize likelihood of the training corpus, producing vocabularies typically of 30k tokens."
---

## TL;DR
WordPiece splits words into subword units by iteratively merging character pairs that maximize language model likelihood, balancing vocabulary size (~30k) against coverage of rare words.

## Intuition
Instead of tokenizing by whole words (huge vocabulary, can't handle "unhappiness" if unseen) or characters (tiny vocabulary, sequences become absurdly long), WordPiece finds a middle ground. It starts with individual characters and repeatedly merges the pair whose merger most increases the likelihood of the training corpus under a unigram language model. The result: common words like "running" stay whole, while rare words decompose into pieces like "un" + "##happen" + "##iness" — the `##` prefix signals a non-leading subword chunk.

BERT uses a 30,522-token WordPiece vocabulary. A typical English sentence of ~10 words becomes ~15–20 tokens. The key insight vs. BPE (used by GPT): WordPiece picks merges by likelihood ratio rather than raw frequency, making it slightly more principled statistically. In practice the outputs are very similar, but WordPiece tends to produce more linguistically meaningful splits.

## Why It Matters
- **Vocabulary efficiency**: 30k tokens cover ~99%+ of English text while keeping embedding tables small enough to fit in memory — crucial since embedding layers can be 30k × 768 = 23M parameters just for BERT-base.
- **Sequence length budget**: Every token consumes one slot in the context window; poor tokenization of numbers or code (e.g., each digit separate) wastes [[KV Cache]] memory and inflates compute quadratically via [[Attention]].
- **Generalization to rare/foreign words**: Subword decomposition means the model sees "sub-morpheme" patterns during training, enabling nonzero representations for OOV words at inference — no UNK token required for in-distribution text.

## Related Concepts
[[Transformer]], [[Attention]], [[KV Cache]], [[RoPE]], [[Speculative Decoding]]
