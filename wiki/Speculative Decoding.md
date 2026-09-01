---
created: "2026-04-28"
title: "Speculative Decoding"
tags: [inference, throughput, efficiency, draft-model]
year: 2025
tldr: "A fast draft model proposes K tokens; the large model verifies all K in one parallel pass. Expected accepted tokens per verifier call = K × acceptance_rate. With MTP heads as drafts, Nemotron-3 achieves ~97% acceptance and ~2× throughput on long generations."
theme: efficiency
citation_count: 1926
arxiv: "2211.17192"
cited_by_top: ["EAGLE-3", "QuantSpec", "SpecAttn", "Mamba Drafters", "TreeSpec", "ParallelSpec", "SpecFusion", "DraftAlign", "LongSpec", "SpecMamba"]
cited_by_details:
  - title: "EAGLE-3"
    year: 2025
    citations: 400
    theme: "inference"
    arxiv: "2503.01840"
  - title: "QuantSpec"
    year: 2025
    citations: 200
    theme: "inference"
    arxiv: "2502.10424"
  - title: "SpecAttn"
    year: 2026
    citations: 180
    theme: "inference"
    arxiv: "2602.07223"
  - title: "Mamba Drafters"
    year: 2025
    citations: 150
    theme: "inference"
    arxiv: "2506.01206"
  - title: "TreeSpec"
    year: 2025
    citations: 130
    theme: "inference"
  - title: "LongSpec"
    year: 2025
    citations: 70
    theme: "inference"
  - title: "SpecFusion"
    year: 2025
    citations: 90
    theme: "inference"
  - title: "DraftAlign"
    year: 2025
    citations: 80
    theme: "inference"
  - title: "ParallelSpec"
    year: 2025
    citations: 110
    theme: "inference"
  - title: "SpecMamba"
    year: 2025
    citations: 60
    theme: "hardware"
    arxiv: "2509.19873"
---

# Speculative Decoding

Autoregressive LLM generation is **memory-bandwidth-bound**: at each step the full model is loaded from GPU HBM to produce exactly one token, then loaded again for the next. You're paying the full memory read cost regardless of batch size. Speculative decoding amortizes this cost across multiple tokens. A fast **draft model** generates $K$ candidate tokens sequentially — fast because it's small, or because it shares weights with the main model (as in [[Multi-Token Prediction]]). The large **verifier model** then processes all $K$ candidates in a **single parallel forward pass**: because all $K$ tokens are known in advance, attention can run in parallel over them (just like training mode). The verifier accepts or rejects each position: it accepts candidate $k$ if its probability distribution agrees with the draft's choice, rejects the first position where it doesn't, replaces it with its own sample, and restarts. Expected tokens per verifier call = $K / (1 + r)$ where $r$ is the rejection rate. With $K = 2$ and 97% acceptance rate (as in [[Nemotron-3]]): $2 / (1 + 0.03) \approx 1.94$ tokens per verifier pass vs. 1.0 without speculative decoding — roughly 2× throughput on long sequences. The critical requirement: the draft model must be fast enough that generating $K$ drafts plus the verifier pass costs less wall-clock time than $K$ verifier passes. In Nemotron-3 the [[Multi-Token Prediction]] auxiliary heads are the draft model — zero extra memory cost, no separate deployment artifact, the heads exist for free as a training byproduct.

## Origins

The technique was independently named and formalized in two concurrent papers: [[Speculative-Decoding-Leviathan|Speculative Decoding (Leviathan 2023)]] (Google, ICML 2023 Oral) and [[Speculative-Sampling-Chen|Speculative Sampling (Chen 2023)]] (DeepMind, arXiv 2023). Both prove that the accept/reject sampling scheme preserves the target model's exact output distribution; Leviathan et al. is the more commonly cited canonical reference, while Chen et al. demonstrated the technique at larger scale (Chinchilla 70B).

## Where it appears

- **[[Nemotron-3]]** — MTP heads serve as the draft model; used both during [[RLVR]] rollout generation (critical for sampling 64K-token traces efficiently) and at final inference deployment
- **[[Multi-Token Prediction]]** — MTP is the specific mechanism that provides the draft tokens
- **[[Speculative Chain-of-Thought]]** — raises the unit of speculation from tokens to whole reasoning chains: a small model drafts a full chain-of-thought, the target model verifies and corrects only the wrong parts

## Why it matters

- **It converts parallelism at training time into speed at inference time.** Attention is inherently parallelizable when all tokens are known. Speculative decoding exploits this: the verifier pass processes $K$ tokens at nearly the cost of 1, because the matrix multiplications over $K$ known token positions run in parallel.
- **It's particularly valuable for reasoning models.** Long chain-of-thought generations (8K–64K tokens) are where LLM inference is most expensive. A 2× throughput multiplier on a 64K-token generation is a much bigger absolute saving than on a 128-token reply. [[RLVR]] training over 21 environments with 64K-token rollouts is only feasible because MTP speculative decoding cuts rollout time in half.
- **The acceptance rate determines everything.** A draft model that's wrong 50% of the time gives you $2/(1+0.5) = 1.33$ tokens per pass — barely worth the overhead. A draft at 97% acceptance gives you 1.94 — nearly 2×. The acceptance rate reflects how aligned the draft distribution is with the verifier's, which is why shared-weight MTP heads (same model, just an auxiliary head) achieve such high rates.

---

*Related: [[Speculative-Decoding-Leviathan|Speculative Decoding (Leviathan 2023)]] · [[Speculative-Sampling-Chen|Speculative Sampling (Chen 2023)]] · [[Medusa]] · [[EAGLE]] · [[Multi-Token Prediction]] · [[Nemotron-3]] · [[RLVR]] · [[KV Cache]]*
