# 🎬 Promo Finder — Microdrama Promo Clip Selection Engine

> **Automated, High-Velocity 20–60s Promo Clip Extraction from Screenplays for Vertical Drama Platforms (ReelShort, DramaBox, ShortMax, Kuku FM).**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/unit%20tests-22%2F22%20passing-brightgreen.svg)]()
[![Model-Agnostic](https://img.shields.io/badge/LLM-OpenAI%20%7C%20Gemini%20%7C%20vLLM-purple.svg)]()
[![Speed](https://img.shields.io/badge/batch%20runtime-%3C%201%20hour%20for%2050%20episodes-orange.svg)]()
[![Cost](https://img.shields.io/badge/cost-%240.007%20%2F%20episode-success.svg)]()

---

## 📌 Table of Contents
1. [Overview](#-overview)
2. [Why Naive LLM Approaches Fail](#-why-naive-llm-approaches-fail)
3. [Architecture & Rebuild V2 Highlights](#-architecture--rebuild-v2-highlights)
4. [Installation & Setup](#-installation--setup)
5. [How to Run the Project](#-how-to-run-the-project)
   - [Run Unit Tests](#1-run-unit-tests)
   - [Dry-Run Validation](#2-dry-run-validation-0-llm-calls)
   - [Run a Single Episode](#3-run-a-single-episode)
   - [Run an Episode Range](#4-run-an-episode-range)
   - [Run the Full 50-Episode Series Batch](#5-run-the-full-50-episode-series-batch)
6. [Pipeline Stages Explained](#-pipeline-stages-explained)
7. [Output Files & Deliverables](#-output-files--deliverables)
8. [Configuration & Environment Variables](#-configuration--environment-variables)
9. [Repository Structure](#-repository-structure)
10. [Troubleshooting & FAQs](#-troubleshooting--faqs)

---

## 📖 Overview

In the vertical microdrama market, organic and paid user acquisition lives and dies by **20–60 second promotional video ads**. An effective promo must:
1. **Hook cold viewers within 1.5–3.0 seconds** (on mute or with sound) to stop vertical scrolling.
2. **Escalate dramatic conflict rapidly** without expositional filler.
3. **End on an irresistible cliffhanger** (a high-stakes curiosity gap) that compels the viewer to install the app or unlock the next episode.
4. **Be 100% understandable to a stranger** who has never heard of the show.

**Promo Finder** automates this entire discovery and curation pipeline directly from raw screenplays. It parses multi-event screenplay text, discovers high-tension narrative windows, enforces strict standalone comprehension gates, scores dramatic intensity, ranks candidates via a calibrated 0–10 absolute rubric, and selects a mathematically diverse 5-clip portfolio using semantic embeddings and Maximal Marginal Relevance (MMR).

---

## ⚠️ Why Naive LLM Approaches Fail

Most generative AI solutions fail on microdrama screenplays because they try to use a **single-shot prompt** asking an LLM to *"read this script and pick the best 5 promo clips with timestamps"*. This fails in 5 ways:

* ❌ **Timestamp Fabrication:** LLMs cannot calculate cumulative floating-point timing from text; they hallucinate timestamps that cut mid-sentence or cross scene transitions.
* ❌ **Duration Hallucination:** Models trade dramatic tension for approximate length, outputting 10s fragments or 90s monologues.
* ❌ **Cognitive Overload:** Forcing one model call to perform candidate generation, character legibility verification, spoiler classification, and ranking collapses model reasoning into generic tropes.
* ❌ **Duplicate Clones:** Without diversity constraints, models output 5 minor variations of the exact same scene.
* ❌ **Prohibitive Latency:** Pairwise comparisons ($O(N^2)$ round-robin tournaments) take 10+ minutes per episode and hundreds of API calls.

**Promo Finder solves this with strict separation of concerns:** LLMs are restricted to qualitative perception and ranking, while boundary snapping, duration filtering, eligibility gates, and portfolio diversity are governed by pure Python mathematics.

---

## 🚀 Architecture & Rebuild V2 Highlights

The pipeline was completely overhauled from Legacy V1 to Rebuilt V2, achieving dramatic performance and efficiency gains:

| Metric | Legacy V1 | Rebuilt V2 (Current) | Improvement |
|---|---|---|---|
| **LLM Calls / Episode** | ~280 calls | **11–13 calls** | **95.5% reduction** |
| **Wall-Clock Latency / Ep** | 10.5 minutes | **~2.2 minutes** | **78.6% faster** |
| **50-Episode Batch Runtime** | ~8.8 hours | **< 1 hour (55–65 min)** | **100% on schedule** |
| **Cost per Episode** | ~$0.21 USD | **~$0.007 USD** | **96.8% cheaper** |
| **Full 50-Episode Series Cost** | ~$10.50 USD | **~$0.35 USD total** | Entire series costs 35¢ |
| **Stage 4 Ranking** | 120-call pairwise tournament | **1 batched absolute rubric call** | **0 round-robin matches** |
| **Portfolio Selection** | Heuristic ranking | **FastEmbed ONNX + MMR** | $\lambda=0.55$, $\text{Sim} < 0.90$ |

---

## ⚙️ Installation & Setup

### Prerequisites
* **OS:** Linux (Ubuntu/Debian recommended) or macOS
* **Python:** 3.10, 3.11, or 3.12
* **Git**

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/kuku-promo-finder.git
cd kuku-promo-finder
```

### 2. Set Up Virtual Environment & Dependencies
```bash
# Navigate to promo-finder engine directory
cd promo-finder
python3 -m venv .venv
source .venv/bin/activate

# Install required dependencies
pip install fastembed numpy scipy
cd ..
```

### 3. Configure LLM API Keys (Optional)
The client supports direct REST APIs (OpenAI, Gemini) as well as local CLI / sandbox runners:
```bash
# For OpenAI models (e.g. gpt-4o, gpt-4o-mini):
export OPENAI_API_KEY="your-openai-api-key"

# For Google Gemini models (e.g. gemini-2.5-flash):
export GEMINI_API_KEY="your-gemini-api-key"

# (Optional) Override the default model:
export PROMO_MODEL="gemini-2.5-flash"
```
*(If running within the AGY sandbox environment, the system automatically routes calls via the authenticated local sandbox runner without external keys).*

---

## 🏃 How to Run the Project

The root runner `run_pipeline.py` (and convenience alias `run_all_episodes.py`) provides an intuitive CLI for all modes of execution.

### 1. Run Unit Tests
Verify all 22 deterministic unit tests pass cleanly (takes < 0.01s):
```bash
PYTHONPATH=promo-finder promo-finder/.venv/bin/python3 -m unittest discover -s promo-finder -p "test_*.py" -t promo-finder
```
*Expected output: `Ran 22 tests in 0.007s — OK`*

---

### 2. Dry-Run Validation (0 LLM Calls)
Validate screenplay parsing, simultaneous event grouping, beat edge snapping, and capped window search across all 50 episodes with **zero API calls and zero cost**:
```bash
# Dry run for a single episode:
python3 run_pipeline.py --episode 1 --dry-run

# Dry run across all 50 episodes:
python3 run_pipeline.py --all --start-ep 1 --end-ep 50 --dry-run
```

---

### 3. Run a Single Episode
Execute the full 5-stage pipeline for Episode 1:
```bash
python3 run_pipeline.py --episode 1 --outputs-dir outputs_final_batch --concurrency 4
```
*Outputs are saved to `outputs_final_batch/`:*
* `episode_001_golden.json`: Structured beat map with simultaneous events
* `episode_001_signals.json`: Batched hook and cliffhanger signals
* `episode_001_candidates.json`: Capped window search candidates
* `episode_001_stage3_scored.json`: Comprehension gate, emotional salience, and spoiler scores
* `episode_001_stage4_ranking.json`: Pure 0–10 absolute rubric rankings
* `episode_001_final_portfolio.json`: Top-5 diverse promo portfolio
* `episode_001_rebuild_summary.json`: Timing, call counts, token usage, and cost breakdown

---

### 4. Run an Episode Range
Process a specific slice of episodes (e.g., Episodes 1 to 5):
```bash
python3 run_pipeline.py --all --start-ep 1 --end-ep 5 --outputs-dir outputs_final_batch --concurrency 4
```

---

### 5. Run the Full 50-Episode Series Batch
Execute the end-to-end pipeline across the entire series with automatic checkpointing and telemetry logging:
```bash
python3 run_pipeline.py --all --start-ep 1 --end-ep 50 --outputs-dir outputs_final_batch --concurrency 4
```
Or use the pre-configured batch shortcut:
```bash
python3 run_all_episodes.py
```
*(Tip: Append `2>&1 | tee outputs_final_batch/batch_execution.log` to stream logs directly to disk).*

#### Checkpointing & Idempotent Resume:
The pipeline features automatic stage-level and episode-level checkpointing:
* By default, `--resume` is enabled. If a run is interrupted, re-running the command **skips completed episodes and stages**, saving time and tokens.
* To force complete re-computation from scratch, pass `--force`.

---

## 🧩 Pipeline Stages Explained

```
Screenplay (.txt)
  │
  ▼
[Stage 0: Deterministic Parser] ────────── 0 Calls | 0.004s
  │  • Regex timestamp extraction
  │  • Simultaneous event containerization (CAMERA + ACTION + DIALOGUE)
  │  • Scene boundary tracking
  ▼
[Stage 1: Batched LLM Detection] ──────── 2 Calls | ~35s
  │  • Call #1: Visual & dialogue opening hook scores (1–5)
  │  • Call #2: Curiosity gap & anti-defusion cliffhanger scores (1–5)
  ▼
[Stage 2: Capped Window Search] ────────── 0 Calls | 0.003s
  │  • Pure Python beat-edge snapping (start = hook.start, end = cliff.end)
  │  • Scene boundary lock (no cross-scene cuts)
  │  • Duration filter: 20.0s ≤ duration ≤ 60.0s
  │  • Greedy candidate cap: max 3 per hook, ≤ 40 total pool
  │  • Floor-relaxation fallback if < 5 candidates found
  ▼
[Stage 3: Merged 3-in-1 Scoring] ──────── ~4-8 Calls (Batches of 4) | ~50s
  │  • Binary Situational Comprehension Gate (open_legible ∧ arc_legible)
  │  • Emotional Salience Score (1–5 intensity)
  │  • Spoiler Disclosure Level (0–3 tier)
  ▼
[Stage 4: Absolute Rubric Ranking] ────── 1 Call | ~25s
  │  • 4 Narrative Dimensions (2.5 pts each = 10.0 max)
  │  • Screenplay dialogue highlight injection (start, mid-arc, exit)
  │  • Anchor-calibrated grounding (Episode 1 worked examples)
  │  • Deterministic Python tie-breaking (duration closeness to 35s)
  ▼
[Stage 5: Boundary & Portfolio MMR] ───── 0 Calls | ~1.2s
  │  • Rule-based dialogue boundary cleanliness check
  │  • Local FastEmbed ONNX dense embeddings (BAAI/bge-small-en-v1.5)
  │  • Maximal Marginal Relevance portfolio selection (λ=0.55, Sim < 0.90)
  ▼
Top-5 Diverse Promo Portfolio (.json)
```

---

## 📂 Output Files & Deliverables

All outputs are saved to `outputs_final_batch/` (or your chosen `--outputs-dir`):

| File | Content |
|---|---|
| `episode_00X_golden.json` | Parsed ground-truth screenplay structure with beats and simultaneous events |
| `episode_00X_signals.json` | Detected hook signals and cliffhanger signals with rationales |
| `episode_00X_candidates.json` | Valid candidate clip intervals snapped to beat boundaries (20–60s) |
| `episode_00X_stage3_scored.json` | Comprehension gate passes, emotional salience, and spoiler classifications |
| `episode_00X_stage4_ranking.json` | Pure absolute rubric rankings (0–10 scale) across 4 narrative dimensions |
| `episode_00X_final_portfolio.json` | **Final Deliverable:** The top-5 diverse promos with opening quotes, exit quotes, cleanliness scores, and MMR similarity metrics |
| `episode_00X_rebuild_summary.json` | Episode wall-clock latency, API call count, tokens, and cost breakdown |
| `batch_summary.json` | Series-wide aggregated performance, average timing, total cost, and target verification |
| `llm_calls.jsonl` | Full telemetry audit log recording latency, tokens, cost, and prompts for every call |

### Sample Final Promo JSON Structure
```json
{
  "portfolio_rank": 1,
  "candidate_id": "CAND_EP001_09",
  "absolute_rubric_score": 9.30,
  "subscores": {
    "hook_scroll_stop": 2.4,
    "mid_arc_escalation": 2.3,
    "cliffhanger_curiosity": 2.4,
    "standalone_payoff": 2.2
  },
  "mmr_score": 0.884,
  "boundary": {
    "start_beat_id": "B014",
    "end_beat_id": "B022",
    "start_time": 35.8,
    "end_time": 63.4,
    "duration": 27.6,
    "composite_cleanliness": 5.0
  },
  "hook": {
    "opening_quote": "Why are you holding her hand?",
    "opening_character": "LADY JULIANA",
    "rationale": "Immediate personal confrontation at the altar with zero warmup."
  },
  "cliffhanger": {
    "final_quote": "Escort her out.",
    "final_character": "LORD STERLING",
    "specific_question": "Will Juliana be publicly arrested or reveal the Duke's letter?"
  },
  "situational_summary": "A bride walking down the aisle discovers her groom holding another woman's hand and is told the wedding is off.",
  "emotional_salience_score": 5,
  "spoiler_level": 1
}
```

---

## 🔧 Configuration & Environment Variables

| Variable | Default | Description |
|---|---|---|
| `PROMO_MODEL` | `gemini-3.8-flash-low` | LLM model identifier (e.g. `gpt-4o-mini`, `gemini-2.5-flash`) |
| `OPENAI_API_KEY` | *(None)* | API key for OpenAI-compatible REST endpoints |
| `GEMINI_API_KEY` | *(None)* | API key for Google Gemini REST endpoints |
| `LLM_API_BASE_URL`| *(Auto)* | Custom base URL for self-hosted models (vLLM / Ollama) |
| `AGY_EXE` | *(System path)* | Path to local sandbox CLI runner bridge |

---

## 📁 Repository Structure

```
├── README.md                           # Main project documentation (this file)
├── SOLUTION_ARCHITECTURE_DOC.md        # Exhaustive architectural specification & engineering report
├── run_pipeline.py                     # Master pipeline CLI runner (symlinked at root)
├── run_all_episodes.py                 # Full batch series runner shortcut
├── outputs_final_batch/                # Unedited batch outputs for all 50 episodes
│   ├── episode_001_final_portfolio.json
│   ├── episode_050_final_portfolio.json
│   ├── batch_summary.json              # Aggregated batch metrics and cost report
│   └── llm_calls.jsonl                 # Detailed LLM telemetry audit log
│
└── promo-finder/                       # Core python engine package
    ├── parser/                         # Stage 0: Deterministic screenplay parser
    │   ├── parser.py                   # Pure Python regex & event containerization
    │   └── schema.py                   # Data contracts (Episode, Beat, Event)
    │
    ├── candidates/                     # Stage 1 & 2: Detection & Window Search
    │   ├── llm_client.py               # Model-agnostic REST & sandbox LLM client
    │   ├── llm_detector.py             # Batched concurrent hook & cliffhanger detector
    │   └── window_search.py            # Capped window search & floor-relaxation fallback
    │
    ├── scoring/                        # Stage 3: Merged 3-in-1 Candidate Scorer
    │   └── merged_scorer.py            # Comprehension gate, emotion & spoiler scoring
    │
    ├── ranking/                        # Stage 4 & 5: Absolute Ranking & Portfolio MMR
    │   ├── batched_ranker.py           # Pure absolute rubric ranker (0-10 scale in 1 call)
    │   └── mmr.py                      # Rule-based cleanliness & FastEmbed MMR
    │
    ├── prompts/                        # Versioned, rubric-grounded LLM prompts
    │   ├── hook_detector_v1.txt        # Stage 1A Hook detection prompt
    │   ├── cliffhanger_detector_v1.txt # Stage 1B Cliffhanger detection prompt
    │   ├── merged_scoring_v1.txt       # Stage 3 Merged 3-in-1 scoring prompt
    │   └── batched_ranking_v1.txt      # Stage 4 Pure absolute rubric ranking prompt
    │
    └── eval/                           # Ground truth eval set and validation scripts
```

---

## ❓ Troubleshooting & FAQs

#### Q1: "ModuleNotFoundError: No module named 'fastembed'"
The project includes automatic `.venv` path injection in `ranking/mmr.py` and `run_pipeline.py`. If running outside the virtual environment, simply activate it or run via:
```bash
promo-finder/.venv/bin/python3 run_pipeline.py --episode 1
```

#### Q2: "What happens if an episode has no strong hooks or cliffhangers?"
`candidates/window_search.py` features an automatic **Floor-Relaxation Fallback**. If strict filtering ($\text{score} \ge 4$) produces fewer than 5 candidate pairs, the threshold automatically relaxes to $\text{score} \ge 3$, guaranteeing downstream pipeline viability even on dialogue-light or expositional filler episodes.

#### Q3: "Can I replace Stage 4's LLM call with a pure Python formula?"
Yes. While the 1-call Stage 4 ranker provides qualitative awareness of mid-arc escalation and dramatic synergy, you can compute a deterministic composite score in Python:
```python
score = (hook_score * 0.35 + cliff_score * 0.35 + emotion_score * 0.30) * 2.0 - abs(duration - 35.0) * 0.05
```
See Section 2 of [`SOLUTION_ARCHITECTURE_DOC.md`](./SOLUTION_ARCHITECTURE_DOC.md) for a detailed comparison.

---

*Authored by the Promo Finder Engineering Team | Built for Kuku FM Microdrama Studio*

