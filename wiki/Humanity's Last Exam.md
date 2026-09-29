---
created: "2026-06-03"
title: "Humanity's Last Exam: A Comprehensive Evaluation of Expert-Level AI Reasoning"
authors: "Long Phan, Alice Gatti, Ziwen Han, Nathaniel Li, et al. (Center for AI Safety, Scale AI)"
year: "2025"
arxiv: "2501.14249"
tags: [benchmarks, reasoning, evaluation, knowledge, expert-level, multi-domain]
tldr: "A 2,500-question multi-modal benchmark written by domain experts across dozens of subjects, designed as the final closed-ended academic benchmark now that LLMs exceed 90% on MMLU; frontier models scored below 10% at release. Published in Nature."
citation_count: 376
---

## TL;DR
Humanity's Last Exam (HLE) is a closed-ended, multi-modal benchmark of 2,500 expert-level questions across dozens of academic subjects, crowdsourced from hundreds of PhD-level contributors worldwide. Frontier models including GPT-4o and Claude 3.5 Sonnet score below 10% on it, making it the hardest publicly released benchmark to date. It is designed to resist saturation for years and serve as a meaningful ceiling test for reasoning AI.

## The Problem
Most existing benchmarks are saturating fast. MMLU, which once seemed ambitious, is now routinely cleared at >85% by frontier models. Even harder successors like GPQA (graduate-level science), MATH, and AIME are being approached or cleared. The LLM evaluation community lacks a benchmark that:
1. Is genuinely hard enough that no current system is close to solving it
2. Spans all serious academic disciplines (not just STEM)
3. Has verifiable, closed-form answers that resist gaming
4. Cannot be easily contaminated because the questions are entirely novel

HLE fills this gap by crowdsourcing questions that expert question-writers personally consider extremely difficult, across the full breadth of human knowledge.

## What It Measures
**Dataset structure:**
- **2,500 questions** in the current release, finalized after quality filtering from a much larger submission pool (the first arXiv version reported ~3,000)
- **Dozens of subject areas**: mathematics, physics, chemistry, biology, law, history, economics, linguistics, medicine, computer science, philosophy, and more
- **Two formats**: majority are text-only; ~30% include images (multimodal subset)
- **Answer types**: exact short answers (numbers, expressions, names) and multiple-choice — all machine-gradable
- **Scoring**: Pass@1 accuracy (exact match or equivalent); no partial credit

**Construction process:**
- Open call to academics, PhD students, and domain experts worldwide; hundreds of contributors
- Each question vetted for: (a) having a definitive correct answer, (b) being unsolvable by a simple web search, (c) being beyond what a generalist expert could quickly answer
- Questions designed so that even knowing the topic area, solving them requires deep specialized reasoning

**Domains breakdown (approximate):**
- STEM fields (~60%): math-heavy problems, chemistry synthesis, physics derivations, biology mechanisms
- Humanities/social sciences (~25%): history, law, linguistics, philosophy
- Mixed/applied (~15%): medicine, engineering, economics

## Key Results

| Model | Score (%) | Notes |
|-------|-----------|-------|
| o1 (OpenAI) | ~9.1% | Best performer at release |
| o1-preview | ~7.0% | Earlier checkpoint |
| Claude 3.5 Sonnet | ~4.3% | Top Anthropic model tested |
| GPT-4o | ~3.3% | Standard GPT-4 class |
| Gemini 1.5 Pro | ~3.0% | Google flagship |
| Grok-2 | ~3.0% | xAI model |
| Random baseline (MC) | ~1–2% | Depends on question format |

*All scores are rough figures from the paper at time of publication; exact numbers vary slightly by evaluation setup.*

Human performance (domain experts on their own questions): effectively ~100% by construction — questions were submitted with verified answers.

## Why It Matters for LLM Evaluation

- **Genuine headroom**: Best models score <10%, meaning there is an enormous gap to close before saturation — unlike MMLU (~90%), MATH (~80%), or even GPQA (~50% for o1). This benchmark will remain meaningful for years.
- **Contamination-resistant by design**: All questions are newly authored and have never appeared on the internet prior to release; there is no training data overlap by construction.
- **Breadth is unique**: Unlike GPQA (only biology/chemistry/physics) or competition math benchmarks, HLE covers dozens of disciplines including law, history, and linguistics — stress-testing whether models have genuine cross-domain expertise, not just STEM fluency.
- **Verifiable answers at scale**: Every question has a definitive, short, machine-checkable answer, avoiding the inter-annotator disagreement and LLM-judge noise that plagues open-ended evaluations.
- **Reveals calibration failures**: Models are not just wrong — they are confidently wrong. Reported confidence scores from models on incorrect answers are very high, indicating poor uncertainty quantification at the frontier, which is critical for safety-relevant deployments.

## Limitations

- **Static dataset**: Once released publicly, questions can enter training corpora; future models trained after release may have contamination. The authors acknowledge HLE will need refreshing over time.
- **English-centric**: Submissions were predominantly in English; performance on non-English expert knowledge is untested.
- **No process/chain-of-thought scoring**: Only final answer correctness is measured; models that reason correctly but format the answer slightly differently may be penalized.
- **Contributor variability**: Questions come from hundreds of different experts with varying standards of difficulty calibration — some subjects may be systematically harder or easier than others, creating uneven difficulty distribution across domains.
- **Multimodal subset not uniformly evaluated**: Not all models tested on the image-containing questions, making cross-model comparison on the full set incomplete.
- **No adversarial robustness testing**: Questions are not probed for sensitivity to rephrasing; a model that memorizes question patterns rather than reasoning could in principle exploit surface features.
- **Potential expert blind spots**: Questions vetted primarily by the submitting experts themselves; systematic coverage gaps in less-represented academic fields (e.g., non-Western history, minority languages) are likely.
- **Rolling revisions:** the benchmark team released **HLE-Rolling**, a dynamic fork of the benchmark with public update logs and community-contributed questions, so the question set keeps changing after the original release (see lastexam.ai).

## Publication and Later Revisions

- The paper was later published in **Nature** (DOI 10.1038/s41586-025-09962-4, linked from the arXiv page), and the arXiv record has gone through 11 versions as of July 2026 — the question set and counts have been revised since the first release, so older secondary sources quoting "~3,000 questions" or "100+ subjects" reflect the earliest version.
- Later reasoning-model results on HLE are far above the single-digit scores of the original release, so the "<10%" figures in the table below are launch-time numbers only. See [[LLM Benchmarks]] and [[Opus 4.8 Benchmarks]] for more recent numbers.
- Benchmark quality has been questioned: an independent review (FutureHouse, July 2025) reported that a sizable share of text-only chemistry and biology answers may be incorrect. Treat small score differences on HLE with caution.
- Reasoning-limit studies such as [[Illusion-of-Thinking|The Illusion of Thinking]] and agent-trajectory evaluation such as [[TRACE]] are complementary views: HLE scores final answers on hard questions, while those papers probe *how* and *where* reasoning breaks.

## Related Concepts
[[LLM Benchmarks]] · [[LLM evaluation]] · [[MMLU]] · [[GPQA]] · [[MATH Benchmark]] · [[RLHF]] · [[RLVR]] · [[Transformer]] · [[Speculative Decoding]] · [[Illusion-of-Thinking|Illusion of Thinking]] · [[TRACE]]
