---
created: "2026-05-28"
title: "BPE Tokenization"
tags: [glossary, tokenization, nlp, preprocessing]
tldr: "A bottom-up subword tokenization algorithm that iteratively merges the most frequent byte/character pairs until a target vocabulary size is reached."
---

## TL;DR
BPE builds a vocabulary by repeatedly merging the most frequent adjacent token pairs in a corpus, landing between character-level (too granular) and word-level (too sparse) representations.

## Intuition
Start with every character as its own token. Count all adjacent pairs, find the most frequent one (say `t` + `h` → `th`), merge it everywhere, repeat. After ~50,000 merges you have a vocabulary that naturally carves up frequent words into single tokens (`the`, `running`) while handling rare words via subword pieces (`un` + `believ` + `able`). GPT-2 uses 50,257 tokens; GPT-4 uses ~100k. The algorithm never throws away information — any string is still representable, just as more pieces.

Common strings in the training corpus get compressed to single tokens (saving context length), while rare strings like `supercalifragilistic` fragment gracefully. A key insight: the merge order is the vocabulary — you must store the ~50k merge rules and apply them greedily at inference time in the same order they were learned. This is why tokenization is **deterministic but corpus-dependent**: tokenize on a different dataset and you get a different vocabulary.

## Why It Matters
- **Context efficiency**: "ChatGPT" costs 1 token; naive char-level would cost 7, cutting effective context window ~7×.
- **Vocabulary coverage without OOV**: No unknown-token problem — worst case any Unicode character falls back to its UTF-8 bytes, which BPE initializes from.
- **Tokenization artifacts cause real bugs**: `" 1"` and `"1"` are different tokens; `"Paris"` ≠ `" Paris"`. Models learn arithmetic badly partly because `"9"+"9"` ≠ `"99"` in token space.

## Key Formula or Mechanism
```python
# Core BPE training loop (pseudocode)
vocab = all_unique_characters(corpus)
for i in range(num_merges):          # e.g. 50,000 iterations
    pairs = count_adjacent_pairs(corpus)
    best = max(pairs, key=pairs.get)  # e.g. ('e', 's') → 50,000 occurrences
    corpus = merge(corpus, best)      # replace all ('e','s') → 'es'
    vocab.add(best)                   # add 'es' to vocabulary
# Store merge rules in order — this IS the tokenizer
```

## Where It Appears
- **Sennrich et al. 2016** — *Neural Machine Translation of Rare Words with Subword Units*: original NLP application of BPE
- **Radford et al. 2019** — GPT-2 paper: introduced byte-level BPE (starts from bytes, not chars — true zero OOV)
- **Brown et al. 2020** — GPT-3: same byte-level BPE, 50,257 vocab
- **Kudo & Richardson 2018** — SentencePiece: alternative BPE implementation used by LLaMA/Mistral

## Related Concepts
[[WordPiece Tokenization]]
[[SentencePiece]]
[[BPE_Tokenization|Vocabulary Size Tradeoffs]]
[[BPE_Tokenization|Tokenization Artifacts]]
[[BPE_Tokenization|Byte-Level Encoding]]
