---
created: "2026-05-28"
title: "state-spaces/mamba – Official Mamba / Mamba-2 / Mamba-3 Repository"
authors:
  - Albert Gu
  - Tri Dao
year: 2024
tags:
  - state space model
  - SSM
  - mamba
  - deep learning
  - language model
  - hardware acceleration
  - github
  - repository
aliases:
  - mamba-ssm repo
  - state-spaces/mamba
  - Mamba official implementation
tldr: >
  The official GitHub repository for the Mamba family of selective state space models,
  covering Mamba-1 (selective SSM with hardware-aware scan), Mamba-2 (SSD: structured
  state space duality with tensor-core matmuls), and Mamba-3 (MIMO, inference-first SSM).
  Provides pip-installable CUDA kernels, pretrained checkpoints up to 2.8B parameters,
  and lm-evaluation-harness integration.
wikilinks:
  - "[[Mamba]]"
  - "[[Transformers Are SSMs|Mamba-2]]"
  - "[[State Space Model|SSM]]"
  - "[[SSD]]"
  - "[[FlashAttention]]"
  - "[[Hardware-Aware Scan]]"
  - "[[Chunkwise recurrent]]"
  - "[[Diagonal recurrence]]"
  - "[[Sequence parallelism]]"
  - "[[Memory hierarchy]]"
  - "[[HBM]]"
  - "[[Inference optimization]]"
  - "[[KV cache]]"
  - "[[Attention]]"
---

# state-spaces/mamba – Official Mamba Repository

**URL:** [https://github.com/state-spaces/mamba](https://github.com/state-spaces/mamba)
**Authors:** Albert Gu (CMU), Tri Dao (Princeton)
**Papers:**
- Mamba-1: [arXiv 2312.00752](https://arxiv.org/abs/2312.00752)
- Mamba-2: [arXiv 2405.21060](https://arxiv.org/abs/2405.21060)
- Mamba-3: [arXiv 2603.15569](https://arxiv.org/abs/2603.15569)

---

## TL;DR

The `state-spaces/mamba` GitHub repository is the canonical implementation of three generations of selective [[State Space Model|state space models]]. [[Mamba]] (v1) introduced input-dependent parameters with a hardware-aware parallel scan, achieving [[Transformer]]-competitive perplexity with **5× faster inference** at 2K sequence length. [[Transformers Are SSMs|Mamba-2]] restructured the recurrence into the [[SSD]] (Structured State Space Duality) framework, enabling tensor-core matmuls and **2–8× training throughput improvement** over Mamba-1. Mamba-3 (2025) adds MIMO (multiple-input multiple-output) SSM structure for improved long-range recall with an inference-first design philosophy. The repo ships pip-installable CUDA kernels, pretrained checkpoints (130M–2.8B parameters trained on 300B tokens), and direct integration with `lm-evaluation-harness`.

---

## Repository Overview

This is the **official implementation** of the Mamba family of models, including three generations of selective state space models. All models are available as `pip install mamba-ssm`.

---

## Key Concepts

- **[[Mamba]]** — Selective [[State Space Model|SSM]] (S6) with input-dependent $(A, B, C)$ parameters; breaks the time-invariance constraint of classical SSMs to enable content-based reasoning
- **[[State Space Model|Selective State Space Model]]** — Continuous-time system discretized per token; Mamba-1 uses state size $N=16$, Mamba-2 expands to $N=64$–$128$
- **[[SSD]] (Structured State Space Duality)** — [[Transformers Are SSMs|Mamba-2]]'s core algorithm; scalar-times-identity $A_t$ constraint enables rewriting the recurrence as a masked matrix multiplication, unlocking tensor-core acceleration
- **Hardware-Aware Scan** — Fused CUDA kernel that keeps the scan state in SRAM rather than [[HBM]], eliminating the memory-bandwidth bottleneck; inspired by [[FlashAttention]]'s IO-aware design
- **Chunkwise recurrent computation** — Splits sequences into chunks (e.g., 256 tokens); each chunk is computed with a matmul (parallel, fast) and chunk boundaries are connected by a short recurrent scan
- **MIMO SSM (Mamba-3)** — Multiple-input multiple-output extension; rank-$r$ input/output projections instead of rank-1, controlled by `mimo_rank` parameter
- **[[KV cache]] analogy** — Mamba's fixed-size recurrent state (e.g., $B \times D \times N$ floats) replaces the linearly growing [[KV cache]] of [[Transformer]]s; constant memory per token at inference
- **Recurrent inference** — At generation time, all three Mamba variants run as pure RNNs with **O(1) memory and O(1) compute per step**, vs. [[Attention]]'s O(L) memory and O(L) compute
- **[[Sequence parallelism]]** — [[Transformers Are SSMs|Mamba-2]]'s chunkwise formulation naturally enables distributed sequence parallelism across devices via the SSD structure
- **Diagonal recurrence** — Scalar $a_t$ (vs. full matrix $A_t$ in general SSMs) is the key structural constraint that makes [[SSD]] tractable while preserving expressivity

---

## Mamba-1

- **Core:** Selective SSM (S6) with input-dependent $(A, B, C)$ parameters
- **Algorithm:** Hardware-aware parallel associative scan (see [[Hardware-Aware Scan]])
- **State size:** $N = 16$ (limited by scan efficiency)
- **Training:** Scan-based, no tensor cores
- **Inference speed:** **5× faster** than [[Transformer]] of equal size at sequence length 2K; gap grows with sequence length
- **Memory:** Constant recurrent state replaces growing [[KV cache]]; e.g., 2.8B model state is ~100MB regardless of context length

```python
from mamba_ssm import Mamba
model = Mamba(d_model=16, d_state=16, d_conv=4, expand=2).cuda()
```

---

## Mamba-2 (SSD)

- **Core:** [[SSD]] layer — scalar-times-identity $A_t$, multi-head structure
- **Algorithm:** Chunkwise matmul + short scan (see [[Chunkwise recurrent]])
- **State size:** $N = 64$–$128$ (tensor-core friendly; **4–8× larger** than Mamba-1)
- **Training:** **2–8× faster** than Mamba-1 (2× at small batch; up to 8× at large batch on A100)
- **Perplexity:** Matches or slightly exceeds Mamba-1 at equal parameter count; both competitive with [[Transformer]]++ on Pile

Key implementation files:
- `mamba_ssm/modules/mamba2.py` — full Mamba-2 block
- `mamba_ssm/modules/mamba2_simple.py` — simplified version
- `mamba_ssm/modules/ssd_minimal.py` — minimal SSD (Listing 1 from paper)

```python
from mamba_ssm import Mamba2
model = Mamba2(d_model=16, d_state=64, d_conv=4, expand=2).cuda()
```

---

## Mamba-3

- **Core:** MIMO (multiple-input multiple-output) SSM for improved expressivity
- **Focus:** Inference-first design, improved long-range recall
- **Parameters:** `is_mimo=True`, `mimo_rank`, `chunk_size`

```python
from mamba_ssm import Mamba3
model = Mamba3(d_model=768, d_state=128, headdim=64, is_mimo=True,
               mimo_rank=4, chunk_size=16, dtype=torch.bfloat16).cuda()
```

---

## Pretrained Models (HuggingFace: `state-spaces`)

| Model | Params | Tokens | Notes |
|---|---|---|---|
| `mamba-130m` .. `mamba-2.8b` | 130M–2.8B | 300B | Mamba-1, Pile |
| `mamba2-130m` .. `mamba2-2.7b` | 130M–2.7B | 300B | Mamba-2 / [[SSD]] |
| `transformerpp-2.7b` | 2.7B | 300B | [[Transformer]]++ baseline |
| `mamba2attn-2.7b` | 2.7B | 300B | [[Transformers Are SSMs|Mamba-2]] + [[Attention]] hybrid |
| `mamba-2.8b-slimpj` | 2.8B | 600B | SlimPajama dataset |

Model dimensions follow GPT-3 conventions (24–64 layers, 768–2560 hidden dim).

---

## Hardware Requirements

- Linux, NVIDIA GPU (CUDA 11.6+), PyTorch 1.12+
- AMD GPU support available (see repo docs)
- Optional: `causal-conv1d` for efficient Conv1d in Mamba block

---

## Evaluation

Zero-shot evaluation via `lm-evaluation-harness`:

```sh
lm_eval --model mamba_ssm \
        --model_args pretrained=state-spaces/mamba2-2.7b \
        --tasks lambada_openai,hellaswag,piqa,arc_easy,arc_challenge,winogrande,openbookqa \
        --device cuda --batch_size 256
```

---

## Why It Matters

- **Inference efficiency at scale:** [[Mamba]] models generate tokens with **O(1) memory per step** (fixed recurrent state) vs. [[Transformer]]'s O(L) [[KV cache]]; at 16K context, a 2.8B Mamba-2 uses ~200MB of state memory vs. ~14GB KV cache for an equivalent [[Transformer]]
- **Training throughput breakthrough:** [[Transformers Are SSMs|Mamba-2]]'s [[SSD]] algorithm achieves **2–8× higher training throughput** than Mamba-1 on A100 GPUs by replacing CUDA scan kernels with BLAS-level tensor-core matmuls, enabling larger state sizes ($N=128$ vs $N=16$) at the same cost
- **Ecosystem and reproducibility:** With 9,000+ GitHub stars, pip-installable kernels, and direct `lm-evaluation-harness` integration, this repo has become the reference benchmark for alternative architectures like [[RWKV]], [[xLSTM]], [[RetNet]], [[Griffin]], and [[Jamba]], enabling fair apples-to-apples comparison at the 130M–2.8B scale

---

## Connections

- [[Mamba]] — Mamba-1 foundational paper; introduces selectivity mechanism and hardware-aware scan
- [[Transformers Are SSMs|Mamba-2]] — Mamba-2 / [[SSD]] paper; theoretical duality between SSMs and [[Attention]], enabling tensor-core training
- [[State Space Model|SSM]] — Broader family of state space models (S4, S5, H3) that Mamba extends with selectivity
- [[SSD]] — Structured State Space Duality; the algorithmic core of Mamba-2
- [[FlashAttention]] — Direct inspiration for IO-aware kernel design; both avoid materializing full intermediate tensors in [[HBM]]
- [[Attention]] — The mechanism Mamba's recurrent form replaces at inference; [[Transformers Are SSMs|Mamba-2]] paper formalizes their mathematical duality
- [[KV cache]] — Transformer inference bottleneck that Mamba eliminates by using fixed-size recurrent state
- [[Jamba]] — Hybrid architecture interleaving Mamba-2 layers with [[Attention]] layers (as in `mamba2attn-2.7b`)
- [[RWKV]] — Concurrent linear RNN work; different parameterization, similar inference efficiency goals
- [[RetNet]] — Another linear recurrence model with chunkwise training; predates SSD's tensor-core formulation
- [[Griffin]] — DeepMind's hybrid RNN; uses real-gated linear recurrences, comparable to Mamba-1 scale
- [[xLSTM]] — Extended LSTM with matrix memory; competing approach to large-state recurrent models
- [[speculative decoding]] — Orthogonal inference acceleration technique; Mamba's O(1) step cost makes it attractive as a draft model

---

## Open Questions

1. **MIMO expressivity vs. efficiency tradeoff:** Mamba-3's MIMO structure increases expressivity via rank-$r$ projections, but the optimal `mimo_rank` relative to `d_state` for different task types (recall-intensive vs. language modeling) remains poorly characterized — does higher rank always help, or does it overfit on shorter contexts?

2. **Hybrid architecture design principles:** The `mamba2attn-2.7b` checkpoint mixes [[Transformers Are SSMs|Mamba-2]] and [[Attention]] layers, and [[Jamba]] explores similar hybrids; there is no principled theory yet for *which* layers should be attention (1 in 8? 1 in 4?) and whether the optimal ratio changes with model scale, context length, or task distribution.

3. **Long-context recall ceiling:** Fixed-size recurrent state fundamentally limits exact retrieval from long contexts (e.g., "needle in a haystack" tasks), unlike [[Attention]] with full [[KV cache]]; whether Mamba-3's MIMO design or future architectural changes (e.g., learned state expansion) can close this gap without sacrificing O(1) inference memory remains an open research direction.

---

## Related Wiki Notes

- [[Mamba]] — the foundational selective SSM paper
- [[Transformers Are SSMs|Mamba-2]] — SSD: Structured State Space Duality
- [[State Space Model|SSM]] — state space models background
- [[SSD]] — the core Mamba-2 algorithm
- [[Hardware-Aware Scan]] — Mamba-1's training algorithm
- [[Chunkwise recurrent]] — Mamba-2's SSD training algorithm
- [[Diagonal recurrence]] — scalar-$a_t$ recurrence in SSD
- [[FlashAttention]] — systems inspiration (IO-aware design)
- [[Sequence parallelism]] — enabled for SSMs by SSD
- [[Attention]] — mechanism replaced by Mamba's recurrent form
- [[KV cache]] — Transformer inference bottleneck Mamba avoids
- [[Jamba]] — production hybrid Mamba-2 + Attention model
- [[RWKV]], [[xLSTM]], [[RetNet]], [[Griffin]] — concurrent linear recurrence alternatives
- [[speculative decoding]] — complementary inference optimization
- [[HBM]] / [[Memory hierarchy]] — GPU memory design principles
