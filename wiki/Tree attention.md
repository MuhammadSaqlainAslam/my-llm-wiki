---
created: "2026-05-20"
title: "Tree Attention"
tags: [glossary, medusa, speculative-decoding, inference, tree, verification]
tldr: "A modified attention mask that represents a tree of candidate token continuations as a flat batch, allowing the LLM to verify all candidate paths in a single forward pass. Used by Medusa and EAGLE to verify speculative token trees efficiently."
aliases: [tree attention, tree-based attention, tree attention mask]
---

## TL;DR

Tree attention packs a tree of $N$ candidate token sequences into a single forward pass of the LLM by constructing a custom attention mask. Each node in the tree corresponds to one position in the packed input; the mask ensures each node can only attend to its ancestors in the tree (not its siblings or cousins). The LLM processes all nodes simultaneously and outputs a probability distribution at each node — allowing simultaneous verification of all paths.

## Intuition

Speculative decoding needs to verify candidate token sequences. Naive approach: run the LLM once per candidate sequence — $N$ forward passes. Expensive.

Better approach: pack all candidates into one batched input. Standard batching would let token B at depth 2 attend to both its actual parent A1 and the sibling A2's path — wrong, because B should only attend to A1. Tree attention adds a custom attention mask that enforces tree structure: node B can only attend to nodes on the path from the root to B.

```
Tree structure:
[root] → [A1] → [B1] → [C1]
                → [B2] → [C2]
         → [A2] → [B3]

Packed as one sequence: [root, A1, B1, B2, A2, B3, C1, C2]
Tree attention mask: each token attends only to its ancestors
```

The mask is computed once per step based on the tree structure. The LLM treats it like a standard forward pass with a custom causal mask.

## Why It Matters

- **It reduces N candidate verifications to 1 forward pass.** For a tree of 80 nodes, tree attention is ~80× more efficient than sequential verification.
- **It's a general primitive for speculative decoding.** Any method that generates multiple candidate continuations (Medusa, EAGLE, lookahead decoding) can use tree attention for verification.
- **The tree size is the main knob for speedup vs. memory.** Larger trees catch more candidates but use more KV cache for the verification pass.

## Where It Appears in This Wiki

- [[Medusa]] — Medusa heads generate the tree; tree attention verifies all paths
- [[EAGLE]] — EAGLE uses a similar tree-based verification for its feature-level draft

## Key Formula or Pseudocode

```
# Tree structure (Medusa with K=3, s=3 per head)
# Candidates: 3^3 = 27 paths → tree with 1+3+9=13 nodes (with pruning)
# Pack as: [token_t, draft_1a, draft_1b, draft_1c, draft_2a, ...]
# Mask: M[i,j] = 1 iff node j is ancestor of node i

attention_mask = build_tree_mask(tree_structure)  # boolean matrix
output = llm_forward(tokens, attention_mask=attention_mask)
# output[i] → distribution at position i (conditioned only on its path)
```

## Related Concepts

[[Medusa]] · [[Medusa heads]] · [[EAGLE]] · [[Speculative Decoding]] · [[Draft model]]
