---
created: "2026-06-24"
title: "MAI-Thinking-1: Building a Hill-Climbing Machine"
authors: "The Microsoft AI Team"
year: "2026"
arxiv: ""
technical_report: "https://microsoft.ai/pdf/mai-thinking-1.pdf"
source_type: "technical_report"
tags: [model-family, moe, reasoning, reinforcement-learning, microsoft, scaling, safety, infrastructure]
tldr: "Microsoft AI's first from-scratch reasoning model — 35B active / 1T total parameter MoE trained on 30T tokens of exclusively licensed/public data with no distillation, achieving 52.8% SWE-Bench Pro and 97.0% AIME 2025, competitive with Sonnet 4.6"
citation_count: 0
---

## TL;DR

MAI-Thinking-1 is Microsoft AI's first reasoning model trained entirely from scratch — no distillation from any third-party model. A 35B active / 1T total parameter MoE, pretrained on 30T tokens of exclusively licensed and public data, then taken through a three-stage RL "climb" (STEM, agentic coding, helpfulness/safety) before being consolidated into a single model. Reaches 52.8% on SWE-Bench Pro, 97.0% on AIME 2025, and is described as competitive with Sonnet 4.6 across a wide benchmark range.

---

## Architecture

- Decoder-only Transformer with interleaved local/global attention (5:1 ratio, following Gemma 3's design) and alternating dense/MoE feed-forward blocks
- Adopts [[LatentMoE]] (compressing tokens before all-to-all expert dispatch), citing NVIDIA's design directly
- 8 of 512 experts activated per token
- GQA with 8 KV heads, RoPE on local attention layers, no position encoding on global attention layers
- o200k_base tokenizer (200,019 vocab)
- FP8 mixed precision (E4M3 forward, E5M2 data-gradient), fully deterministic training (bitwise-reproducible runs)
- Trained on 8,192 GB200 GPUs using YOLO, Microsoft's in-house training framework

---

## Training Cluster

Microsoft frames **goodput** — the ratio of ideal training duration to actual wall-clock duration — as the primary production KPI, rather than peak MFU alone. Overhead is tracked in two layers:

- **Visible failures:** crashloops, node failures, InfiniBand/NVLink link flaps, OOM errors, pod terminations, checkpoint stalls, manual requeues
- **Silent efficiency losses:** MFU degradation, recomputation, long startup paths, slow process scheduling, checkpoint-induced stalls, degraded network or memory behavior, fabric conditions that reduce throughput without crashing

Determinism is treated as a first-class infrastructure property: the cluster must eliminate silent data corruption, keep communication topology stable, and preserve floating-point reduction order across checkpoint/restart boundaries — not just for reproducibility, but because silent correctness failures can degrade model quality in ways that only surface in downstream evaluations.

**MAI-Base-1 pre-training run results (8K GB200 GPUs):**

| Overhead category | Hours | Share of overhead |
|---|---|---|
| MFU drop | 18 h | 35% (largest remaining) |
| Non-stepping time | 14 h | 27% |
| Recomputation | 6.5 h | 15% |
| Total overhead | 51 h | — |
| **Goodput** | **90.0%** | — |

The RL climb (MAI-Thinking-1) ran on 4.6K GB300s — a homogeneous accelerator generation chosen to reduce experimental variance. Inference runs on MAIA-200 hardware, which delivers **40%+ higher token generation throughput per watt** vs a GB200-based deployment under the same rack power budget.

*(Full hardware and cluster specifications are in Appendix L of the technical report, not covered here.)*

---

## Training Recipe

**Pre-training:** 30T tokens, no synthetic or LM-generated data anywhere in the corpus. Explicit exclusion of huggingface.co and similar ML-repository domains to avoid benchmark contamination.

**Mid-training:** Two-stage context extension (64K then 256K tokens) with re-weighted, STEM/code-biased subsets of the same corpus.

**RL climb:** Starts from a checkpoint with zero prior exposure to reasoning traces. Uses a modified [[GRPO]] objective with adaptive entropy control and an outer ratio clip for stability. Three domain specialists are trained in parallel:

1. **STEM / competitive code** — math olympiad + competitive programming
2. **Agentic coding / tool use** — multi-step tool-calling, SWE-Bench-style tasks
3. **Helpfulness / safety** — instruction following and alignment

The three specialists are then consolidated into a single model via SFT, followed by a final lightweight RL stage to restore generality.

---

## Safety

Red-teaming ran in parallel with model development across 15 engagements (early, mid, and late stages), covering 2,170+ goal-based adversarial scenarios across 25 policy categories. Each scenario ran 5–10 conversational turns to allow escalation past first-turn refusals.

**Six attack patterns identified as the durable adversarial surface** (these recurred independently across red teamers and model checkpoints — the patterns, not the individual prompts, are treated as what needs covering):

1. Multi-turn escalation under a benign pretext
2. Fictional or novelistic framing
3. Credentialed-persona pretexts (claiming researcher, medical, or authority status)
4. Gradual recursion / formatting drift — repeated requests to expand, reformat, or operationalize a previously hedged answer
5. In-context age-indicator bypass
6. Authoritative-document fabrication

**Mitigation effectiveness** (pre- vs. post-mitigation in the final candidate):

| Attack class | ASR reduction |
|---|---|
| Jailbreaks (overall) | −44% |
| Hate & fairness | −43% |
| Child safety | −30% |
| Mental health attacks | −20% |
| **Aggregate** | **−22%** |

**Jailbreak ASR comparison** (Figure 21; lower = stronger safety; third-party results include provider-side filtering):

| Technique type | MAI-Thinking-1 | GPT-5.4 | Claude Opus 4.6 | Claude Sonnet 4.6 |
|---|---|---|---|---|
| Foundational | 4.4% | 7.0% | 3.0% | 5.7% |
| Compositional | 17.6% | 13.9% | 17.4% | 15.0% |
| Adaptive | 26.8% | 32.3% | 25.1% | 26.4% |

*Foundational* = single-step transforms (wrappers, templates). *Compositional* = multi-transform rewrites, PyRIT, PAP-style, non-English variants. *Adaptive* = multi-turn / search-based (TAP, multi-turn attacks).

**Two named vulnerabilities from independent red-teaming (AIRT + third-party vendors):**

- **TAP (Tree of Attacks with Pruning)** — surfaced as a robustness gap. Fixed via a closed-loop adversarial data pipeline: broad generation of realistic harmful scenarios → diverse attack-transformation templates → TAP-style adaptive refinement against the current model → output fed back as targeted remediation data. Resulted in a "large reduction" in TAP susceptibility, bringing MAI-Thinking-1 to parity with SOTA models on the same attack vectors.

- **Low-resource language framing** — content reliably refused in English was elicited in Yoruba, Telugu, Amharic, Burmese, Khmer, and Malay. Fixed by expanding safety training data with multilingual adversarial seeds and re-targeting high-yield English attack patterns into the affected languages. Closed a "significant portion" of the English/non-English gap; multilingual robustness in the long tail remains an ongoing investment area.

*These are documented applications of standard red-teaming methodology (TAP and multilingual safety gaps are known issues in the literature), not novel techniques. This section is included for the unusual transparency of publishing concrete ASR numbers and specific vulnerability names — most vendor reports omit both.*

---

## Why It Matters

- A rare, unusually transparent technical report — documents a failure mode ("data-mixture rank non-invariance": a data mixture that wins at small scale can lose at large scale) that most vendor reports omit
- Explicit, deliberate choice to train without any distillation data, positioned as a design principle ("capabilities should be learned, not inherited") rather than a limitation
- Direct architectural lineage from papers already in this wiki: Gemma 3's local/global attention interleaving and NVIDIA's [[LatentMoE]]
- Reports BPB comparisons against [[DeepSeek-V3 Technical Report|DeepSeek-V3]], Kimi-K2, and Gemma 4 31B base models — useful cross-reference for benchmark comparisons elsewhere in the wiki
- First major reasoning model to publicly report SWE-Bench Pro (52.8%) alongside AIME 2025 (97.0%), offering a joint math+coding capability snapshot

---

## Limitations

- Self-reported; no independent evaluator benchmarks cited in the report
- "Competitive with Sonnet 4.6" is a vendor characterization — worth verifying against the full report's comparison table before citing specific numbers in other notes
- No arXiv submission — technical report only, same caveat as [[Nemotron_3_Super|Nemotron 3 Super]], [[Nemotron_3_Ultra|Nemotron 3 Ultra]], and [[DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence|DeepSeek-V4]]

---

## Related Concepts

*Architecture: [[LatentMoE]] · [[Mixture-of-Experts]] · [[GRPO]]*

*Comparable models: [[Nemotron_3_Super|Nemotron 3 Super]] · [[Nemotron_3_Ultra|Nemotron 3 Ultra]] · [[DeepSeek-V3 Technical Report|DeepSeek-V3]]*

*Training: [[Multi-Environment RLVR Training]] · [[Chinchilla_Scaling_Laws|Chinchilla]]*
