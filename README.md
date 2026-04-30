# LLM Wiki

> AI research knowledge base — papers, concepts & intuitions  
> Curated by Muhammad Saqlain Aslam

**Live Web Demo:** https://MuhammadSaqlainAslam.github.io/my-llm-wiki

---

## What Is This?

A personal AI/ML research wiki built from academic papers, concept glossaries, and cross-linked notes. Every note is written in intuition-first style — concrete numbers, no fluff. The wiki is browsable as a web demo with a knowledge graph showing how all concepts connect.

---

## Live Demo Features

- **Browse & search** all notes by title, tag, or keyword
- **Interactive knowledge graph** — see how papers and concepts link
- Click any node in the graph to read the full note
- Filter notes by theme: Foundations, Efficiency, Scaling, Synthesis

---

## Repository Structure

```
/raw/                  → original PDF papers (source material)
/wiki/                 → processed markdown notes (24 files)
  000 Index.md               → master index with 4 reading paths
  Home.md                    → concept map and entry point
  DeepSeek_V4.md             → paper note example
  KV Cache.md                → concept glossary example
  ...
/docs/                 → generated web demo (auto-deployed to GitHub Pages)
  index.html                 → browse & search interface
  graph.html                 → D3 knowledge graph
  notes.json                 → all notes as structured data
/build.py              → converts wiki/ markdown into docs/ web files
/.github/
  workflows/
    deploy.yml               → auto-deploys to GitHub Pages on every push
```

---

## Wiki Contents

### Papers

| Title | Year | Summary |
|---|---|---|
| Attention Is All You Need | 2017 | Self-attention replaces recurrence; any two tokens connect in one step, enabling parallel training and O(1) path length between positions. |
| Switch Transformers / Mixtral of Experts | 2022 | Route each token to a sparse subset of expert FFNs; parameters scale cheaply while per-token compute stays constant. |
| Mamba: Linear-Time Sequence Modeling with Selective State Spaces | 2024 | Make SSM parameters (B, C, Δ) functions of the input so the model selectively compresses context. 5× inference throughput over Transformers. |
| Nemotron 3: Efficient and Open Intelligence | 2025 | Hybrid Mamba-2 + sparse attention + LatentMoE, trained with NVFP4 precision. 7.5× throughput over Qwen3.5-122B at 1M context. |
| DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence | 2026 | CSA + HCA compressed attention cuts KV cache 10× at 1M tokens. Muon optimizer + on-policy distillation. SOTA open model. |

### DeepSeek-V4 Concept Stubs

New mechanisms introduced in DeepSeek-V4, each with its own note:

| Concept | Summary |
|---|---|
| Compressed Sparse Attention (CSA) | Compress every m tokens into one KV entry via learned weighted sum, then sparse top-k block selection. |
| Heavily Compressed Attention (HCA) | Compress even more aggressively than CSA, then run full dense attention over the tiny resulting sequence. |
| Manifold-Constrained Hyper-Connections (mHC) | Expand the residual stream to [n_hc × d] with a doubly-stochastic mixing matrix; guarantees spectral norm ≤ 1. |
| Muon Optimizer | Orthogonalize the gradient matrix via Newton-Schulz iterations before applying; faster convergence than AdamW on matrix weights. |
| On-Policy Distillation (OPD) | Student generates its own rollouts then minimises reverse KL vs. specialist teacher ensemble; no distribution shift. |
| Group Relative Policy Optimization (GRPO) | PPO without a value function; normalises advantages within a group of G rollouts per prompt. |

### Concept Glossary

Sub-concepts with their own stub notes:

| Concept | Summary |
|---|---|
| KV Cache | Cached key-value tensors from past attention steps; eliminates redundant recomputation but grows linearly with sequence length. |
| State Space Model (SSM) | A continuous-time linear dynamical system discretized for sequences, with dual recurrent and convolutional computation modes. |
| Hardware-Aware Scan | Kernel fusion that keeps the Mamba SSM recurrence entirely in fast SRAM, never materialising intermediate states in HBM. |
| Load Balancing Loss | Auxiliary loss that penalises unequal token distribution across MoE experts, preventing router collapse. |
| LatentMoE | Project tokens to a smaller latent dimension before MoE routing; cuts all-to-all communication by d/ℓ, allowing more experts. |
| Multi-Token Prediction (MTP) | Auxiliary heads predict 2, 3… tokens ahead simultaneously; richer training signal and free speculative decoding at inference. |
| NVFP4 | NVIDIA's 4-bit floating point training format with E2M1 elements and 16-element micro-block scaling; 3× peak throughput vs BF16. |
| Speculative Decoding | A fast draft model proposes K tokens; the large model verifies all K in one parallel pass, accepting only consistent tokens. |
| Grouped Query Attention (GQA) | Partition Q heads into groups that share K and V heads, cutting KV cache size by the grouping factor. |
| RLVR | Reinforcement Learning with Verifiable Rewards — restrict RL to ground-truth verifiable signals (correct answer, passing test suite). |

### Navigation

| File | Purpose |
|---|---|
| `000 Index.md` | Master hub: all notes grouped by theme, concept glossary, and 4 reading paths |
| `Home.md` | Concept map entry point with dependency graph |

---

## Workflow

```
PDF papers in raw/
      ↓
build_wiki.py — Claude API extracts & summarizes
      ↓
wiki/*.md — structured notes with YAML frontmatter
      ↓
build.py — converts markdown to web files
      ↓
docs/ — static site (index.html + graph.html + notes.json)
      ↓
GitHub Actions — auto-deploys to GitHub Pages on push
      ↓
https://MuhammadSaqlainAslam.github.io/my-llm-wiki
```

---

## How to Add a New Paper

1. Drop the PDF into `raw/`
2. Run: `python build_wiki.py`
3. A new note appears in `wiki/`
4. Run: `python build.py`
5. `git add . && git commit -m "Add new paper" && git push`
6. GitHub Actions auto-deploys — site updates in ~2 minutes
7. `git pull` on local PC → Obsidian updates instantly

---

## How to Run Locally

```bash
pip install anthropic pymupdf markdown pyyaml
export ANTHROPIC_API_KEY="sk-ant-..."
python build_wiki.py   # process new PDFs → wiki/*.md
python build.py        # regenerate docs/ from wiki/
open docs/index.html   # preview in browser
```

---

## Tech Stack

| Component | Tool |
|---|---|
| Note generation | Claude API (`claude-sonnet-4-6`) |
| PDF extraction | PyMuPDF |
| Knowledge graph | D3.js v7 |
| Static site | Vanilla HTML / CSS / JS |
| Hosting | GitHub Pages |
| Local knowledge base | Obsidian |
| Version control | Git + GitHub |
| Auto-deploy | GitHub Actions |

---

## Author

**Muhammad Saqlain Aslam**  
GitHub: https://github.com/MuhammadSaqlainAslam  
Repository: https://github.com/MuhammadSaqlainAslam/my-llm-wiki  
Web Demo: https://MuhammadSaqlainAslam.github.io/my-llm-wiki
