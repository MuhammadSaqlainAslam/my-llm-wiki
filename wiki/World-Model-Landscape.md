---
created: "2026-06-29"
title: "The World Model Landscape (2019-2026)"
authors: "Bessemer Venture Partners (Nagda, Ma, Goldberg)"
year: "2026"
arxiv: ""
technical_report: "https://www.bvp.com/atlas/can-world-models-unlock-general-purpose-robotics"
source_type: "industry_report"
tags: [world-models, reference, landscape, robotics, industry-analysis]
tldr: "A timeline of 24 frontier world models released 2019-2026, organized by organization, parameter count, and application category (Robotics, Gaming/Interactive, Autonomous Driving) — sourced from BVP's research, spot-checked against arXiv/Nature for accuracy"
citation_count: 0
---

## TL;DR

A timeline of frontier world models released between 2019 and 2026, compiled by Bessemer Venture Partners from published papers and technical reports. Organizes 24 models across three application categories — Robotics, Gaming/Interactive, and Autonomous Driving — showing a clear trend toward larger parameter counts and more organizations entering the space over time.

> **Source note:** This table reproduces a landscape figure from a venture capital industry report, not a peer-reviewed survey. Two entries (DreamerV3, Cosmos) were independently spot-checked against arXiv/Nature and confirmed accurate. The remaining entries are presented as reported by BVP and have not each been individually re-verified — treat parameter counts and dates as industry-research-grade, not citation-grade, until cross-checked against each model's own paper when adding that model's own wiki note.

---

## The Landscape

### 2019
| Model | Org | Size | Category | Open Source |
|-------|-----|------|----------|-------------|
| PlaNet | Google | ~2M | Robotics | ✅ |
| DreamerV1 | Google | ~10M | Robotics | ✅ |

### 2020
| Model | Org | Size | Category | Open Source |
|-------|-----|------|----------|-------------|
| DreamerV2 | Google | ~20M | Robotics | ✅ |

### 2022
| Model | Org | Size | Category | Open Source |
|-------|-----|------|----------|-------------|
| DayDreamer | Berkeley | ~20M | Robotics | ✅ |
| [[IRIS]] | University of Geneva | ~15M | Gaming/Interactive | ✅ |

### 2023
| Model | Org | Size | Category | Open Source |
|-------|-----|------|----------|-------------|
| [[DreamerV3]] | Google DeepMind | — | Robotics | — |
| UniSim | Google DeepMind | — | Robotics | — |
| RT-2 | Google DeepMind | — | Robotics | — |
| GAIA-1 | Wayve | 6.5B | Autonomous Driving | — |

### 2024
| Model | Org | Size | Category | Open Source |
|-------|-----|------|----------|-------------|
| V-JEPA 1 | Meta | 632M | Robotics | — |
| [[DIAMOND]] | INRIA | 381M | Gaming/Interactive | — |
| Genie 1 | Google DeepMind | — | Robotics | — |
| GameNGen | Google | ~860M | Gaming/Interactive | ✅ |
| Cosmos | NVIDIA | 14B | Robotics | ✅ |
| Oasis | Decart | 500M | Gaming/Interactive | — |
| Genie 2 | Google DeepMind | — | Robotics | — |
| GameGen-O | Tencent | — | Gaming/Interactive | — |

### 2025
| Model | Org | Size | Category | Open Source |
|-------|-----|------|----------|-------------|
| V-JEPA 2 | Meta | ~1-2B | Robotics | — |
| GAIA-2 | Wayve | 8.4B | Autonomous Driving | — |
| Dreamer 4 | Google DeepMind | — | Robotics | — |
| MUSE | Microsoft | 1.6B | Gaming/Interactive | — |
| Odyssey-2 | Odyssey | — | Gaming/Interactive | — |
| Marble | World Labs | — | Robotics | — |

### 2026
| Model | Org | Size | Category | Open Source |
|-------|-----|------|----------|-------------|
| Lucy 2 | Decart | — | Robotics | — |
| Odyssey-2 Pro | Odyssey | — | Gaming/Interactive | — |

---

## Verified Entries (cross-checked beyond BVP's report)

- **DreamerV3** — confirmed via arXiv 2301.04104 ("Mastering Diverse Domains through World Models," Hafner et al.) and independently published in *Nature* as "Mastering diverse control tasks through world models." Matches BVP's listed year (2023) and organization (Google DeepMind) exactly.
- **Cosmos** — confirmed via arXiv 2501.03575 ("Cosmos World Foundation Model Platform for Physical AI," NVIDIA) and NVIDIA's official open-source release. Organization, year, and open-source status match BVP's listing; the 14B figure corresponds to the Cosmos-1.0-Diffusion-14B-Text2World checkpoint.

---

## Why It Matters

- Shows the field's parameter-count trajectory clearly: ~2-20M (2019-2022) → hundreds of millions (2024) → multi-billion (2025-2026), a scaling pattern similar to the LLM field's own trajectory roughly 3-4 years earlier
- Three distinct application clusters emerge: Robotics-focused (DeepMind's Dreamer/Genie line, Meta's V-JEPA line, NVIDIA Cosmos), Gaming/Interactive (IRIS, DIAMOND, GameNGen, Odyssey), and Autonomous Driving (Wayve's GAIA line) — each with different scaling and open-source norms
- Open-source releases cluster heavily in the smaller, earlier models (2019-2022, plus GameNGen and Cosmos) — most 2025-2026 multi-billion-parameter models are closed, mirroring the LLM field's own open/closed split at comparable scale

---

## Related Concepts

*Foundational: [[World Models]]*

*Models in the table (notes not yet in wiki): [[DreamerV3]] · [[IRIS]] · [[DIAMOND]]*

*Note: DreamerV3, IRIS, and DIAMOND links will resolve once those individual notes are added and verified — same as the forward references in the [[World Models]] note.*
