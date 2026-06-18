---
created: "2026-06-03"
title: "Data Quality Standards"
tags: [meta, quality, standards, benchmarks]
tldr: "Standards for data accuracy in this wiki — source hierarchy, verification requirements, known discrepancies, and update procedures for benchmark scores."
---

## TL;DR
This wiki aims to be an authoritative AI research resource. Every quantitative claim — especially benchmark scores — should carry a source, a date, and an honest note about whether it was independently verified.

## Source Hierarchy

1. **Official system cards** ← highest trust
   Vendor-published with detailed methodology.
   Example: Anthropic system cards.

2. **Official announcements** ← high trust
   Vendor blog posts with benchmark tables.
   Risk: methodology details sometimes omitted.

3. **Independent evaluators** ← high trust
   Examples: Artificial Analysis (GDPval-AA), LMSYS Chatbot Arena, HLE leaderboard (agi.safe.ai), SWE-bench leaderboard (swebench.com).

4. **Third-party analysis** ← medium trust
   News articles and blog posts citing sources above.

5. **Internal comparison images / community reports** ← low trust
   Score tables of unknown provenance. Use only when no better source exists; always flag as unverified.

## Required for Every Score Table

- Source URL or document name
- Date of measurement (Elo scores shift over time!)
- Whether vendor self-reported or independently verified
- Known caveats (harness differences, variant differences, etc.)

## Known Score Discrepancies (June 2026)

### Opus 4.6 MRCR 1M

| Source | Score | Variant |
|--------|-------|---------|
| Internal comparison image | **92.9** MMR | Unknown MRCR variant |
| Anthropic official system card | **76%** MMR | MRCR v2, 8-needle, 1M tokens |

**Resolution:** Cite 76% when sourcing from the official system card. The discrepancy likely reflects different MRCR versions (v1 vs v2) or different needle counts. Both may be accurate for their respective variants.

### Opus 4.6 GDPval-AA

| Source | Score |
|--------|-------|
| Internal comparison image | 1,619 Elo |
| Anthropic official system card | 1,606 Elo |

**Resolution:** Cite 1,606 from the official system card. 13-point gap is likely evaluation date or Elo pool composition.

### Opus 4.6 HLE with tools

| Source | Score | Date |
|--------|-------|------|
| Internal comparison image | 53.1% | — |
| Anthropic system card | 53.0% | Updated Feb 23 2026 |

**Resolution:** Use 53.0% — updated after a cheating-detection pipeline improvement.

## Score Freshness

Elo-based benchmarks (GDPval-AA, LMSYS) change daily as more models are evaluated. Always record the date when noting an Elo score. A score from two months ago may be materially stale.

Static benchmarks (SWE-bench, HLE) are more stable but can be updated when evaluation harnesses change — note the harness version where known.

## Update Procedure

1. Find the official system card for the new model.
2. Cross-check with at least one independent evaluator.
3. Note any discrepancies with a brief explanation.
4. Add `Last verified:` date to the note header.
5. Update `citation_count` via `agent.py update-citations`.

```bash
# Find and add new papers around a topic
python3 agent.py topic "latest LLM benchmark official 2026"

# Refresh citation counts
python3 agent.py update-citations
```

## Benchmark Harness Caveats

Different harnesses for the same benchmark produce non-comparable scores. Always check:

- **Terminal-Bench 2.1:** GPT-5.5 uses Codex CLI; other models use Terminus-2. Not directly comparable.
- **OSWorld:** Harness updated between Opus 4.6 and 4.8 evaluations. Part of the gain is methodology.
- **SWE-bench Verified vs Pro:** Pro has harder multi-file tasks and less contamination risk. Not interchangeable.
- **MRCR v1 vs v2:** v2 applies a cheating-detection fix; scores are not comparable across versions.

## Related Notes

[[LLM Benchmarks]] · [[Opus 4.8 Benchmarks]] · [[SWE-bench]] · [[Humanity's Last Exam]] · [[OSWorld]] · [[LLM evaluation]]
