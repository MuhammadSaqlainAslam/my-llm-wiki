# HHRI-AI Research Wiki

> AI research knowledge base — papers, concepts & intuitions
> Designed by Artificial Intelligence Research Center, Hon Hai Research Institute

🏛️ **Institute:** https://hhri.foxconn.com/en
🌐 **Live Demo:** https://MuhammadSaqlainAslam.github.io/my-llm-wiki

[![Wiki Notes](https://img.shields.io/badge/dynamic/json?url=https://MuhammadSaqlainAslam.github.io/my-llm-wiki/notes.json&query=$.length&label=wiki+notes&color=7c3aed&style=flat)](https://MuhammadSaqlainAslam.github.io/my-llm-wiki)
![GitHub last commit](https://img.shields.io/github/last-commit/MuhammadSaqlainAslam/my-llm-wiki?color=7c3aed&style=flat)
![GitHub repo size](https://img.shields.io/github/repo-size/MuhammadSaqlainAslam/my-llm-wiki?color=0d9488&style=flat)
![License](https://img.shields.io/badge/license-MIT-green?style=flat)
![Live Demo](https://img.shields.io/badge/live-demo-brightgreen?style=flat&logo=github)

---

## Overview

This is a personal AI/ML research knowledge base built from academic papers, concept glossaries, and cross-linked notes. Every paper is distilled into a structured Obsidian note with YAML frontmatter (title, authors, year, arXiv ID, citation count, tags, TL;DR), a full intuition-first explanation, and `[[wikilinks]]` connecting it to every related concept. The result is a single navigable graph where clicking any node — whether a paper or a sub-concept — reveals exactly how it fits into the broader ecosystem.

The writing philosophy is borrowed from Andrej Karpathy: lead with the concrete problem and why it matters, introduce the math only after the intuition is clear, and always give real numbers. Every note leads with intuition, backs claims with concrete numbers, and skips filler — the same standard used in the best ML writing.

What makes this wiki distinct is a full citation intelligence layer on top of the content. Every core paper carries its real citation count from Semantic Scholar, a `cited_by_details` list of the ten most-cited downstream works, and arXiv IDs throughout. The live web demo exposes all of this through a Citation Explorer, an interactive D3 knowledge graph sized by citation count, and a full-text search interface — all from a single static site deployed automatically via GitHub Actions to GitHub Pages.

> Citation counts sourced from [Semantic Scholar](https://semanticscholar.org). Last verified: June 2026. Auto-updated monthly via `python3 agent.py update-citations`.

The most referenced note in the wiki is "Attention Is All You Need" with 46 backlinks — reflecting its foundational role across all modern LLM research.

---

## 🌐 Live Web Demo Features

- Full-text search, multi-tag filtering, and sort controls across all notes
- Citation Explorer — master-detail panel with ranked citation bars and top-10 citing papers
- Interactive D3.js knowledge graph sized and colored by citation count and research theme
- Theme Navigation — notes grouped by research area
- Three appearance modes: System / Dark / Light with live theme switching
- Reading progress tracker with per-note status badges (Unread / Reading / Done)
- Timeline view — papers arranged chronologically 2017–2026
- Knowledge graph node color coding by research theme with color and size legends

### Browse & Search

- Full-text search across all notes by title, tag, or keyword
- Multi-tag AND filtering — click multiple tags to narrow results simultaneously
- Note cards showing title, year badge, citation count badge (🔖 N citations), tags, and 3-line TL;DR
- Sort by title A–Z, year (newest first), or citation count descending
- Side panel opens the full note — rendered Markdown with KaTeX math, live `[[wikilink]]` resolution, and clickable cross-references

### Citation Explorer

A master-detail two-panel layout — click the **Citation Explorer** tab to open it.

**Left panel** — ranked list of every paper with `citation_count > 0`, sorted descending:
- Each row: `#rank · title · year · proportional citation bar · count`
- Bar width proportional to citations (`citation_count / 100,000`), minimum 4 px so even small counts are visible
- Bar color by impact tier:
  - 🟣 Purple → 10,000+ citations
  - 🩵 Teal → 1,000–9,999 citations
  - 🔵 Blue → 100–999 citations
  - ⬜ Gray → under 100 citations
- Filter box at top to search the ranked list in real time

**Right panel — default state** (nothing clicked): summary dashboard showing:
- Total papers in wiki
- Total citations tracked
- Most cited paper
- Fastest growing paper (Mamba: 7,268 citations since 2023)
- Citation cluster pills: Foundations · Efficiency · Inference · Hardware

**Right panel — paper selected**:
- Title, authors, year, and arXiv badge with external link
- Large citation count number + full-width color bar
- **What this paper introduced** — the note's TL;DR
- **Top 10 papers that cite this work** — ranked table with: title (clickable), year, ~citations, theme badge, arXiv ↗ link
  - Clicking a citing paper's title: loads its detail view if it exists in the wiki, otherwise opens `arxiv.org/abs/[id]` in a new tab
- **Related papers in this wiki** — all resolved `[[wikilinks]]` from the note shown as clickable pills; clicking one switches to the All tab and opens that note's detail panel
- **← All Papers** back button returns to the summary dashboard

### Knowledge Graph

- Interactive D3.js v7 force-directed graph (`graph.html`)
- Every node = one wiki note, sized logarithmically by `citation_count`; nodes without citation data sized by incoming link count
- Every edge = a `[[wikilink]]` connection between notes
- Color-coded by theme tag (foundations, efficiency, hardware, inference-optimization, scaling, synthesis, meta)
- Click any node → opens full note detail in the side panel with two action buttons:
  - **📄 Read on arXiv** — opens the paper on arxiv.org (shown when arXiv ID is available)
  - **🔗 Open in wiki** — opens the full note in the browse view
- Hover a node → highlights direct neighbors, dims everything else; tooltip shows title, year, citations, TL;DR, and arXiv ID if available
- Filter by tag — dims non-matching nodes
- Legend: color = theme, node size = citation count (where available, else link count)
- Zoom, pan, drag nodes freely

### Knowledge Graph Enhancements

- Node color coding by research theme:
  - Purple → SSM / Mamba / sequence modeling papers
  - Teal → concept glossary stubs
  - Amber → inference optimization papers
  - Blue → hardware & systems papers
  - Green → foundational papers
  - Red → benchmarks & evaluation
- Color legend displayed in the top-left corner of the graph
- Node size legend displayed in the bottom-left:
  - Small = under 100 citations (sized by incoming link count)
  - Medium = 100–1,000 citations
  - Large = 1,000–10,000 citations
  - Extra large = 10,000+ citations (e.g. Attention Is All You Need)
- Theme colors update live when appearance mode is switched — no page reload

### Theme Navigation

Notes grouped by theme in the **By Theme** tab:

| Theme | Contents |
|---|---|
| Foundations | Self-attention, RLHF, open models |
| State Space Models | SSMs, linear RNNs, LSTM variants |
| Mixture-of-Experts | Sparse routing, load balancing |
| Inference Optimization | Speculative decoding, KV cache, draft models |
| Hardware & Systems | GPUs, TPUs, FPGAs, ASICs, LPUs, IO-aware kernels |
| Attention & KV | Attention variants, KV cache techniques |
| Training & RL | RLHF, RLVR, GRPO, optimizers |
| Benchmarks | Frontier model comparisons |
| Concepts & Glossary | 100+ sub-concept stub notes |

### Appearance Modes

- Three-way toggle in the nav bar: **System** / **Dark** / **Light**
- System: automatically follows your OS preference (`prefers-color-scheme`)
- Dark: deep background (`#0d1117`), glowing colored nodes, high-contrast text
- Light: clean white background, rich color nodes, dark labels
- Preference saved between sessions via `localStorage` key `appearance`
- All components respond: cards, graph nodes, citation bars, side panel, legends

### Reading Progress Tracker

- Track your reading status per note directly in the browser — no account needed
- Three states per note: **Unread** / **Reading** / **Done**
- Click the status badge on any note card to cycle through states
- Progress bar at the top of the browse view: "X of Y notes done · M reading"
- Filter the grid by reading status: **All** / **Unread** / **Reading** / **Done**
- Progress filter applies across All, By Theme, and Timeline tabs
- State persisted in `localStorage` — survives page reloads and browser restarts

### Timeline View

- Horizontal scrollable timeline spanning **2017 to 2026**
- Papers arranged in columns by publication year — one column per year
- Each card shows: title, up to 3 tags, and theme color stripe
- Click any card to open the full note detail panel
- Color-coded by research theme — same scheme as the knowledge graph
- Concept stubs and reference notes without a year in a dedicated **Reference** column
- Reveals the field's evolution at a glance:
  - 2017 — Transformer (Attention Is All You Need)
  - 2022 — FlashAttention, S4 Structured State Spaces
  - 2023 — Mamba, RWKV, RetNet, LLaMA 2, FlashAttention-2
  - 2024 — xLSTM, Griffin, Mamba-2, Medusa, EAGLE
  - 2025–2026 — Speculative Decoding, KV Cache Optimization, DeepSeek-V4, Hardware Acceleration

### Math Rendering

- KaTeX renders LaTeX inline (`$O(n^2)$`) and display (`$$\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$`)
- Math is protected during Markdown preprocessing so the renderer never mangles LaTeX delimiters
- Supports `$...$`, `$$...$$`, `\(...\)`, `\[...\]` delimiters

### Full-text Search

- Indexes complete note body content — not just title and tags
- Results ranked by relevance: title match → tag match → tldr → body content
- Press `/` anywhere on page to focus search bar
- Result count shown: "X notes found"

**Ctrl+F style match navigation:**
- Opening a note from search auto-scrolls to the first match instantly
- Match navigation bar appears at top of note: 🔍 "search term"  1 of 6  ‹  ›
- ‹ › buttons navigate between every occurrence
- Current match = bright purple highlight
- All other matches = light purple highlight
- Keyboard shortcuts: Enter or ↓ → next match · Shift+Enter or ↑ → previous match
- Navigation bar hides automatically when note is closed or search is cleared
- Case insensitive — finds "Attention", "attention", "ATTENTION" equally
- Single-match notes disable the › button automatically

**Example workflow:**
1. Type `"exponential gating"` in search bar
2. xLSTM note appears in results (term is in the body, not the title)
3. Click xLSTM → note opens, auto-scrolls to first match, shows "1 of 3 ‹ ›"
4. Press › to jump through all 3 occurrences
5. Clear search → highlights disappear

### Search Modes

Three search scope modes are available via the mode selector above the result bar:

| Mode | Button | Scope |
|------|--------|-------|
| **Papers only** | 📄 | Academic papers — filters by arXiv ID, citation count, and year; searches title and tags only |
| **All notes** | 💡 | Default — all notes including concept stubs; searches title, tags, and TL;DR |
| **Full text** | 🔍 | All notes with complete body text search (slowest but most thorough) |

**Enter-key card highlighting:**
- Press **Enter** while typing to freeze the current search results as a visual overlay:
  - Matching cards glow purple with a soft border shadow
  - Non-matching cards dim to 20% opacity
  - Text inside matching cards is highlighted with `<mark>` spans
  - Status bar shows `"X matching cards highlighted  Esc to clear"`
- Press **Escape** to clear highlighting and return to normal filtering
- Typing while in highlight mode auto-clears the highlight (returns to live filtering)
- Press `/` anywhere on the page to jump focus to the search bar

### Paper Comparison

- Click **⊕ Compare** on any note card to select it for comparison
- Select 2 or 3 papers — a comparison tray slides up from the bottom of the page showing selected titles as pills
- Click **"Compare (X)"** to open a full side-by-side table view
- Toggle between **Summary** (quick 5-row) and **Technical** (10-row default) views
- **Technical mode** shows 10 rows extracted from each note:
  - Core Innovation, Architecture Type, Sequence Complexity, Key Formula / Mechanism,
    Best Benchmark Result, Hardware Requirement, Limitations, vs Transformer,
    Year & Citations, Reading Status
- **Key differences:** auto-detected — a banner highlights which rows actually differ between papers
- Row labels color-coded: **purple** when papers differ (key differences), gray when they match
- Column top borders color-coded: first paper = purple, second = teal, third = amber
- Alternating row backgrounds for readability
- Maximum 3 papers at once; selecting a 4th bumps the oldest
- Press **Escape** or click **✕ Close** to dismiss

### Backlinks

- Every note tracks which other notes reference it — the inverse of the `[[wikilink]]` graph
- **"Referenced by X notes"** section appears at the bottom of every open note detail panel (only shown when X > 0)
- Clicking any backlink pill opens that note in the same panel
- First 8 backlinks shown; click **"+X more"** to expand the full list
- **← X** backlink count badge on every note card
- Example: "Attention Is All You Need" shows ← 46, meaning 46 other notes reference it — the most central paper in the wiki

---

## 📚 Wiki Contents

[![Wiki Notes](https://img.shields.io/badge/dynamic/json?url=https://MuhammadSaqlainAslam.github.io/my-llm-wiki/notes.json&query=$.length&label=wiki+notes&color=7c3aed&style=flat)](https://MuhammadSaqlainAslam.github.io/my-llm-wiki) — 30+ full paper articles + growing concept glossary (badge updates automatically with each new paper).

### Papers

#### Foundations

| Paper | Year | ~Citations | arXiv |
|-------|------|-----------|-------|
| Attention Is All You Need | 2017 | ~178,574 | [arXiv:1706.03762](https://arxiv.org/abs/1706.03762) |
| LLaMA 2 | 2023 | ~17,078 | [arXiv:2307.09288](https://arxiv.org/abs/2307.09288) |

#### Efficient Sequence Modeling

| Paper | Year | ~Citations | arXiv |
|-------|------|-----------|-------|
| S4: Structured State Spaces | 2022 | ~3,636 | [arXiv:2111.00396](https://arxiv.org/abs/2111.00396) |
| RWKV | 2023 | ~1,063 | [arXiv:2305.13048](https://arxiv.org/abs/2305.13048) |
| RetNet | 2023 | ~670 | [arXiv:2307.08621](https://arxiv.org/abs/2307.08621) |
| Mamba | 2023 | ~7,268 | [arXiv:2312.00752](https://arxiv.org/abs/2312.00752) |
| Griffin | 2024 | ~239 | [arXiv:2402.19427](https://arxiv.org/abs/2402.19427) |
| Transformers are SSMs (Mamba-2) | 2024 | ~1,526 | [arXiv:2405.21060](https://arxiv.org/abs/2405.21060) |
| xLSTM | 2024 | ~595 | [arXiv:2405.04517](https://arxiv.org/abs/2405.04517) |

#### Scaling / Parameter Efficiency

| Paper | Year | ~Citations | arXiv |
|-------|------|-----------|-------|
| Switch Transformers / Mixtral | 2022 | — | — |

#### Hardware & Systems

| Paper | Year | ~Citations | arXiv |
|-------|------|-----------|-------|
| FlashAttention | 2022 | ~4,404 | [arXiv:2205.14135](https://arxiv.org/abs/2205.14135) |
| FlashAttention-2 | 2023 | ~2,769 | [arXiv:2307.08691](https://arxiv.org/abs/2307.08691) |
| Hardware Acceleration for Neural Networks | 2025 | ~3 | [arXiv:2512.23914](https://arxiv.org/abs/2512.23914) |

#### Inference Optimization

| Paper | Year | ~Citations | arXiv |
|-------|------|-----------|-------|
| Medusa | 2024 | ~728 | [arXiv:2401.10774](https://arxiv.org/abs/2401.10774) |
| EAGLE | 2024 | ~474 | [arXiv:2401.15077](https://arxiv.org/abs/2401.15077) |
| Speculative Decoding | 2025 | ~6 | [arXiv:2601.11580](https://arxiv.org/abs/2601.11580) |
| KV Cache Optimization | 2026 | ~1 | [arXiv:2603.20397](https://arxiv.org/abs/2603.20397) |

#### Reasoning & RL

| Paper | Year | ~Citations | arXiv |
|-------|------|-----------|-------|
| Chain-of-Thought Prompting | 2022 | ~18,000 | [arXiv:2201.11903](https://arxiv.org/abs/2201.11903) |
| ReAct: Reasoning + Acting | 2022 | ~4,500 | [arXiv:2210.03629](https://arxiv.org/abs/2210.03629) |
| DeepSeek-R1 | 2025 | ~3,000 | [arXiv:2501.12948](https://arxiv.org/abs/2501.12948) |

#### Fine-tuning & Alignment

| Paper | Year | ~Citations | arXiv |
|-------|------|-----------|-------|
| LoRA | 2021 | ~22,000 | [arXiv:2106.09685](https://arxiv.org/abs/2106.09685) |
| Direct Preference Optimization (DPO) | 2023 | ~8,000 | [arXiv:2305.18290](https://arxiv.org/abs/2305.18290) |

#### Frontier Model Families

| Paper | Year | ~Citations | arXiv |
|-------|------|-----------|-------|
| The Llama 3 Herd of Models | 2024 | ~5,000 | [arXiv:2407.21783](https://arxiv.org/abs/2407.21783) |
| DeepSeek-V3 | 2024 | ~2,500 | [arXiv:2412.19437](https://arxiv.org/abs/2412.19437) |
| Qwen3 Technical Report | 2025 | ~800 | [arXiv:2505.09388](https://arxiv.org/abs/2505.09388) |

#### Modern Synthesis

| Paper | Year | ~Citations | arXiv |
|-------|------|-----------|-------|
| Nemotron-3 | 2025 | — | — |
| DeepSeek-V4 | 2026 | — | — |
| LLM Benchmarks | 2025 | — | — |

### Concept Glossary

Over 100 concept stub notes cover every sub-component referenced across the paper articles, organised into six clusters: **Attention & Transformers** (multi-head attention, GQA, MQA, RoPE, XPOS, online softmax, tiling, recomputation, IO-awareness, warp, occupancy, thread block, linear attention, local attention, sliding window attention, attention sinks, CSA, HCA); **State Space Models** (State Space Model, HiPPO, HiPPO matrix, Cauchy kernel, discretization, convolutional view, Selective SSM, semiseparable matrix, multi-head SSM, retention mechanism, chunkwise recurrent, RG-LRU, LRU, exponential gating, matrix memory, covariance update rule, sLSTM, mLSTM, Long Range Arena); **Inference Optimization** (KV Cache, paged attention, KV offloading, prefill, decode phase, cache eviction, cache compression, hybrid memory, eviction policy, H2O eviction, draft model, verification step, lookahead decoding, tree attention, Medusa heads, Medusa-1, Medusa-2, lossless speedup, feature-level drafting, one-step token advance, multi-token prediction); **Hardware & Systems** (GPU, TPU, NPU, FPGA, ASIC, LPU, systolic array, tensor cores, HBM, SRAM, memory hierarchy, kernel fusion, operator fusion, hardware-aware scan, NVFP4, pallas kernel, in-memory computing, near-memory computing, neuromorphic computing); **Training & Optimization** (RLHF, RLVR, GRPO, ghost attention, load balancing loss, on-policy distillation, Muon optimizer, tensor parallelism, sequence parallelism); **Architecture Components** (LatentMoE, Manifold-Constrained Hyper-Connections, selective state spaces, LLM evaluation, LLM benchmarks). Every concept links back to the paper where it originates, and every `[[wikilink]]` in a paper note resolves to a concept page — 0 orphan links.

### Citation Intelligence

The wiki includes a full citation intelligence layer: `wiki/Citation Map.md` tracks total Semantic Scholar citation counts for the six core papers plus their top-10 highest-impact downstream works. Every core paper's YAML frontmatter carries `citation_count` (integer), `arxiv` (arXiv ID string), `cited_by_top` (name list), and `cited_by_details` (list of dicts: title, year, citations, theme, arxiv) — all emitted by `build.py` into `notes.json` and consumed by the Citation Explorer in the web demo. The `wiki/🗺️ Dashboard.md` Obsidian dashboard includes a live **Citation Leaderboard** Dataview query. Papers with full `cited_by_details`: Attention Is All You Need, Mamba, xLSTM, Speculative Decoding, KV Cache Optimization, Hardware Acceleration for Neural Networks.

---

## Repository Structure

```
my-wiki/
├── wiki/                          # Markdown notes — grows automatically (Obsidian vault)
│   ├── Home.md                    # Entry point with concept map + reading paths
│   ├── 000 Index.md               # Full index: themes, glossary, narrative, citation table
│   ├── Citation Map.md            # Citation scores + top-10 citing papers per core paper
│   ├── 🗺️ Dashboard.md           # Live Dataview queries (Obsidian only)
│   ├── Mamba.md                   # Paper notes (20 total)
│   ├── Transformer.md
│   ├── ...
│   ├── KV Cache.md                # Concept stub notes (100+ total)
│   ├── HBM.md
│   └── ...
│
│   💡 Grows automatically — run `python3 agent.py topic "..."` to add papers.
│      Note count updates live on the badge at the top of this README.
│
├── docs/                          # Static web demo (auto-deployed)
│   ├── index.html                 # Browse, search, Citation Explorer
│   ├── graph.html                 # D3 knowledge graph
│   └── notes.json                 # All notes as structured JSON
├── raw/                           # Source PDF papers
├── build.py                       # wiki/*.md → docs/notes.json + HTML
├── build_wiki.py                  # PDF → wiki note (Claude API)
├── agent.py                       # Agentic ingestion system
├── agent_log.json                 # Log of all agent runs
├── setup_cron.sh                  # Optional cron scheduler
├── cron.log                       # Cron output log (if scheduled)
├── .github/workflows/deploy.yml   # Auto-deploy to GitHub Pages on push
└── README.md
```

---

## Workflow

```
PDF papers in raw/
      ↓
build_wiki.py  —  Claude API extracts & summarises
      ↓
wiki/*.md  —  structured notes with YAML frontmatter
      ↓
build.py  —  converts Markdown + math + wikilinks → JSON + HTML
      ↓
docs/notes.json  +  docs/index.html  +  docs/graph.html
      ↓
GitHub Actions  —  auto-deploys on every push to main
      ↓
https://MuhammadSaqlainAslam.github.io/my-llm-wiki
```

`build.py` handles: YAML frontmatter parsing, math protection (swap LaTeX for placeholders before Markdown rendering → restore raw LaTeX for KaTeX), Obsidian callout → HTML div conversion, `![[embed]]` → clickable link conversion, Dataview/Mermaid block stubbing, and emission of `id`, `title`, `authors`, `tags`, `year`, `tldr`, `aliases`, `links`, `citation_count`, `arxiv`, `cited_by_details`, and rendered `html` per note.

---

## Agentic Ingestion

The wiki has an autonomous research agent (`agent.py`) that automatically finds, evaluates, and adds new content from multiple sources using Claude as the decision-making brain.

### Six Modes

**1. Topic search — add content about a specific topic:**
```bash
python3 agent.py topic "DeepSeek R2 architecture"
python3 agent.py topic "LLM inference optimization 2025"
python3 agent.py topic "Mamba 2 SSM improvements"
```

**2. Citation tracking — find high-impact papers citing existing wiki papers:**
```bash
python3 agent.py citations
```
Queries Semantic Scholar API for papers with 100+ citations that cite existing wiki papers. Automatically downloads any high-impact papers not yet in the wiki.

**3. Daily monitor — broad sweep of new content:**
```bash
python3 agent.py daily
```
Searches arXiv (cs.LG, cs.CL, cs.AI), checks GitHub repos for updates, fetches blog posts. Run this once a week for a broad update.

**4. Update citation counts — refresh from Semantic Scholar:**
```bash
python3 agent.py update-citations
```
Queries the Semantic Scholar batch API for every note that has an arXiv ID in its frontmatter, updates `citation_count` in the YAML, rebuilds the web demo, and commits + pushes automatically. This is a deterministic API call (no LLM), so it's fast and cheap. Run monthly to keep counts current.

**5. Conference papers sweep:**
```bash
python3 agent.py conference
```
Sweeps recent arXiv submissions for papers accepted at top conferences (NeurIPS, ICLR, ICML, ACL, EMNLP, COLING) matching existing wiki themes, using each paper's arXiv comment field to detect a conference-acceptance marker. Report-only — generates `conference_candidates.txt` for manual review, verified via `verify_arxiv_paper()` before any candidate is listed. Never auto-adds, never auto-commits. Run weekly alongside the daily sweep.

**6. Audit citation data integrity:**
```bash
python3 agent.py audit-citations
```
Scans every `cited_by_details` block across the wiki and verifies each arXiv ID actually resolves to the claimed paper title using the live arXiv API. Report-only — flags issues for manual review, never auto-edits. Catches corrupted citation data from the Semantic Scholar pipeline (wrong or fabricated IDs) that may have slipped through before the write-time `verify_arxiv_paper()` guard was added. Run monthly alongside `update-citations`.

> Known informal-name false positives are allowlisted in `agent.py` (`CONFIRMED_CITING_PAPERS`) after manual verification. New flags each month are genuinely worth reviewing — the allowlist only covers entries already checked by hand.

### Suggested Schedule

| Command | Frequency | Purpose |
|---------|-----------|---------|
| `python3 agent.py update-citations` | Monthly | Keep citation counts current |
| `python3 agent.py audit-citations` | Monthly | Verify citation data integrity |
| `python3 agent.py citations` | Monthly | Find new high-impact citing papers |
| `python3 agent.py daily` | Weekly | Broad new-paper sweep |
| `python3 agent.py conference` | Weekly | Find top conference papers for review |
| `python3 agent.py topic "..."` | On demand | Add specific topic papers |

### Sources the Agent Monitors

| Source | What it fetches | API |
|--------|----------------|-----|
| arXiv | New papers by keyword/topic | Free |
| Semantic Scholar | Citation tracking | Free |
| GitHub | READMEs, notebooks, releases | Free |
| Blogs | Karpathy, Lilian Weng, HuggingFace, Distill | Web fetch |
| YouTube | Lecture transcripts (yt-dlp) | Free |

### Why Manual Over Automated

The agent is intentionally triggered manually rather than running on a cron schedule:
- Full control over what enters the knowledge base
- No unnecessary server load on shared infrastructure
- Every update is intentional and visible
- API credits used only when needed

### Agent Log

Every agent run is logged to `agent_log.json`:

```bash
# See what was added in the last run
cat agent_log.json

# See cron output if scheduled
cat cron.log
```

### Relevance Filter

Claude evaluates every candidate paper or article against these wiki themes before adding:
transformer, attention, SSM, Mamba, LSTM, speculative decoding, KV cache, MoE, FlashAttention, hardware acceleration, LLM inference, scaling, RLHF, tokenization, language models

Only content scoring 7/10+ relevance is added. Everything else is logged as skipped.

### For Staying Current (use the agent)

```bash
# Weekly: broad new paper sweep
python3 agent.py daily

# On demand: specific topic
python3 agent.py topic "your research interest"

# Monthly: citation updates
python3 agent.py citations
```

---

## How to Add a New Paper

1. Drop the PDF into `raw/`
2. `python build_wiki.py` — Claude API extracts and writes `wiki/<Title>.md`
3. `python build.py` — regenerates `docs/notes.json`
4. `git add . && git commit -m "Add <Paper>" && git push`
5. GitHub Actions deploys to GitHub Pages in ~2 minutes

---

## How to Run Locally

### 1. Configure environment variables

Copy `.env.example` and fill in your own values — **never commit real keys**:

```bash
cp .env.example .env
# then edit .env, or export the variables in your shell
```

```bash
# Option 1 — Anthropic public API
export ANTHROPIC_API_KEY="sk-ant-..."

# Option 2 — Google Vertex AI (enterprise)
export ANTHROPIC_VERTEX_PROJECT_ID="your-gcp-project-id"
export VERTEX_REGION_CLAUDE_4_6_SONNET="europe-west1"

# Wiki location (optional — defaults to the repo directory)
export WIKI_VAULT="/path/to/your/wiki"
```

### 2. Build and preview

```bash
pip install anthropic pymupdf markdown pyyaml
python build_wiki.py   # process new PDFs → wiki/*.md
python build.py        # regenerate docs/ from wiki/
open docs/index.html   # preview in browser
```

For the knowledge graph, serve locally to avoid CORS issues:

```bash
python -m http.server 8000
# then open http://localhost:8000/docs/graph.html
```

---

## Tech Stack

| Component | Tool |
|---|---|
| Note generation | Claude API (`claude-sonnet-4-6`) |
| PDF extraction | PyMuPDF |
| Markdown → HTML | Python `markdown` library |
| Math rendering | KaTeX (client-side) |
| Knowledge graph | D3.js v7 (force simulation) |
| Static site | Vanilla HTML / CSS / JS — zero build step, zero dependencies |
| Hosting | GitHub Pages |
| Local knowledge base | Obsidian (Dataview plugin) |
| Version control | Git + GitHub |
| Auto-deploy | GitHub Actions |
| Reading tracker | `localStorage` API | Per-note progress persistence (Unread / Reading / Done) |
| Timeline view | Vanilla JS + CSS flex | Year-based chronological paper layout |
| Appearance modes | CSS `prefers-color-scheme` + JS | System / Dark / Light theming |
| Research agent | Claude API tool use | Autonomous paper ingestion |
| arXiv API | urllib + XML parsing | Paper search and download |
| Semantic Scholar | REST API | Citation tracking |
| yt-dlp | Python library | YouTube transcript extraction |
| GitHub API | REST API | Repo README extraction |

---

## 🏛️ Institution

Designed by the **Artificial Intelligence Research Center**

[Hon Hai Research Institute](https://hhri.foxconn.com/en)

---

## 📊 Analytics

Privacy-friendly, cookie-free visitor tracking. No personal data collected.

---

## Author

**Muhammad Saqlain Aslam**
GitHub: https://github.com/MuhammadSaqlainAslam
Repository: https://github.com/MuhammadSaqlainAslam/my-llm-wiki
Web Demo: https://MuhammadSaqlainAslam.github.io/my-llm-wiki
