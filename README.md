# Promo Finder — Microdrama Promo Clip Selection Engine

> **High-Velocity, Cost-Calibrated 20–60s Promo Clip Extraction from Screenplays for Vertical Drama Platforms (ReelShort, DramaBox, ShortMax, Kuku FM).**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/unit%20tests-56%2F56%20passing-brightgreen.svg)]()
[![Code Style](https://img.shields.io/badge/code%20style-clean%20architecture-success.svg)]()
[![Model-Agnostic](https://img.shields.io/badge/LLM-OpenAI%20%7C%20Gemini%20%7C%20vLLM-purple.svg)]()
[![Calls](https://img.shields.io/badge/calls-7.5%20%2F%20episode-brightgreen.svg)]()
[![Batch Cost](https://img.shields.io/badge/50--ep%20batch%20cost-%240.188%20USD-success.svg)]()

---

## Quick Start (< 60 Seconds)

You can clone, install, test, and dry-run the entire pipeline in under a minute:

```bash
# 1. Clone the repository
git clone https://github.com/harshpsi9194/KukuFM-submission.git
cd KukuFM-submission

# 2. Set up virtual environment and install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Run comprehensive unit test suite (56 tests in < 0.2s)
python3 run_tests.py

# 4. Dry-run Stage 0 to Stage 2 pipeline on Episode 1 (0 API calls, $0.00 cost)
python3 run_pipeline.py --episode 1 --dry-run
```

---

## Table of Contents
1. [Overview](#overview)
2. [Why Naive LLM Approaches Fail](#why-naive-llm-approaches-fail)
3. [Architecture & Rebuilt V2 Performance](#architecture--rebuilt-v2-performance)
4. [Installation & Setup](#installation--setup)
5. [Running the Unit Tests](#running-the-unit-tests)
6. [How to Run the Pipeline](#how-to-run-the-pipeline)
   - [Dry-Run Structural Validation (0 LLM Calls)](#1-dry-run-structural-validation-0-llm-calls)
   - [Run a Single Episode](#2-run-a-single-episode)
   - [Run an Episode Range](#3-run-an-episode-range)
   - [Run the Full 50-Episode Series Batch](#4-run-the-full-50-episode-series-batch)
7. [Five-Stage Architecture In-Depth](#five-stage-architecture-in-depth)
8. [The 14 Non-Negotiable Pipeline Invariants](#the-14-non-negotiable-pipeline-invariants)
9. [Output Files & Deliverables](#output-files--deliverables)
10. [Configuration & Environment Variables](#configuration--environment-variables)
11. [Repository Structure](#repository-structure)
12. [Troubleshooting & FAQs](#troubleshooting--faqs)

---

## Overview

In vertical microdrama applications, organic user acquisition and ad monetization depend critically on **20–60 second promotional video ads**. An effective promo must:
1. **Hook cold viewers within 1.5–3.0 seconds** (on mute or with audio) to stop vertical feed scrolling.
2. **Escalate dramatic conflict rapidly** without expositional filler or dragging dialogue.
3. **Cut on an irresistible cliffhanger** (a high-stakes curiosity gap) that forces the user to install the app or unlock the next episode.
4. **Remain 100% legible to a stranger** who has never heard of the characters or plot.

**Promo Finder** automates this end-to-end selection pipeline directly from raw screenplays. It parses multi-event screenplay text, discovers high-tension narrative windows, enforces strict standalone comprehension gates, scores dramatic intensity, ranks candidates via a calibrated 0–10 absolute rubric, and selects a mathematically diverse 5-clip portfolio using semantic embeddings and Maximal Marginal Relevance (MMR).

---

## Why Naive LLM Approaches Fail

Most generative AI approaches attempt a **single-shot prompt** asking an LLM to *"read this screenplay and pick the top 5 promo clips with timestamps"*. In practice, this fails systematically:

* **Timestamp Fabrication:** LLMs cannot calculate cumulative floating-point timing from text; they hallucinate timestamps that cut mid-sentence or cross scene transitions.
* **Duration Hallucination:** Models trade dramatic tension for approximate length, outputting 10s fragments or 90s monologues.
* **Cognitive Overload:** Forcing one model call to perform candidate generation, character legibility verification, spoiler classification, and ranking collapses model reasoning into generic tropes.
* **Duplicate Clones:** Without diversity constraints, models output 5 minor variations of the exact same scene.
* **Prohibitive Latency:** Pairwise comparisons ($O(N^2)$ round-robin tournaments) take 10+ minutes per episode and hundreds of API calls.

**Promo Finder solves this with strict separation of concerns:** LLMs are restricted to qualitative perception and ranking, while boundary snapping, duration filtering, eligibility gates, and portfolio diversity are governed by pure Python mathematics.

---

## Architecture & Rebuilt V2 Performance

The pipeline was completely overhauled from Legacy V1 to Rebuilt V2, achieving dramatic performance and efficiency gains (empirically measured across the full 50-episode series batch):

| Metric | Legacy V1 | Rebuilt V2 (Empirical 50-Ep Batch) | Improvement |
|---|---|---|---|
| **Average Calls / Episode** | ~280 calls | **7.5 calls** | **37.3× fewer calls** |
| **Total Calls (50 Episodes)** | ~14,000 calls | **375 calls** | **97.3% call reduction** |
| **Average Wall-Clock / Ep** | 10.5 minutes (630s)| **89.8 seconds (1.50m)** | **7.0× faster** |
| **Full 50-Episode Batch** | ~8.8 hours | **74.8 minutes (1.25h)** | **~7 hours saved** |
| **Average Cost / Episode** | ~$0.215 USD | **$0.00376 USD** | **57.2× cheaper** |
| **Full 50-Episode Batch Cost** | ~$10.75 USD | **$0.188 USD total (< 19¢)** | **98.3% cost reduction** |
| **Stage 4 Ranking** | 120-call pairwise tournament | **1 batched absolute rubric call** | **0 round-robin matches** |
| **Portfolio Selection** | Heuristic ranking | **FastEmbed ONNX + MMR** | $\lambda=0.55$, $\text{Sim} < 0.90$ |

---

## Installation & Setup

### Prerequisites
* **OS:** Linux (Ubuntu 20.04+, Debian, Fedora) or macOS
* **Python:** 3.10, 3.11, or 3.12
* **Git**

### 1. Clone the Repository
```bash
git clone https://github.com/harshpsi9194/KukuFM-submission.git
cd KukuFM-submission
```

### 2. Set Up Virtual Environment & Dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

*(Note: The codebase includes automatic virtualenv path resolution, so scripts and tests run cleanly whether your virtual environment is active or invoked directly).*

### 3. Configure LLM API Keys (For Live Pipeline Runs)
The model client supports OpenAI, Google Gemini, and local vLLM/Ollama endpoints:
```bash
# For OpenAI models (e.g. gpt-4o, gpt-4o-mini):
export OPENAI_API_KEY="your-openai-api-key"

# For Google Gemini models (e.g. gemini-2.5-flash):
export GEMINI_API_KEY="your-gemini-api-key"

# (Optional) Override the default model:
export PROMO_MODEL="gemini-2.5-flash"
```
*(If running within the AGY sandbox environment, the system automatically routes calls via the authenticated local runner without external keys).*

---

## Running the Unit Tests

The test suite contains **56 comprehensive, deterministic unit tests** covering all stages and pipeline invariants. Every test executes in pure Python with zero network calls and runs in < 0.2 seconds.

### Option A: Universal Test Runner (Recommended)
```bash
python3 run_tests.py
```
For detailed test-by-test output:
```bash
python3 run_tests.py -v
```

### Option B: Pytest
```bash
pytest
```

### Option C: Standard Library Unittest
```bash
python3 -m unittest
# or explicitly:
python3 -m unittest discover -s tests
```

### Test Suite Coverage Breakdown

| Test Module | Stage | Covered Requirements & Invariants |
|---|---|---|
| `tests/test_stage0_parser.py` | Stage 0 | Strict non-defaulted `timestamp_source`, simultaneous event merging (CAMERA + ACTION + DIALOGUE), genuine overlap vs endpoint touching, ~2.5 words/sec fallback heuristic, scene boundary tracking, golden snapshot test. |
| `tests/test_stage1_detection.py` | Stage 1 | `HookSignal` dialogue vs visual separation, `CliffhangerSignal` curiosity gap and anti-defusion check, token/cost tracking, robust JSON extraction from markdown. |
| `tests/test_stage2_window_search.py` | Stage 2 | Python duration filter [20.0s, 60.0s], candidate duration as sum of segment durations, `timestamp_source` inheritance, strict halt at scene boundaries, candidate capping. |
| `tests/test_stage3_scoring.py` | Stage 3 | Standalone comprehension binary gate (`opening_legibility ∧ arc_legibility`), ban on retrospective justification, Python override of hallucinated passes, emotional salience (1-5), spoiler reporting (0-3), missing candidate fallback. |
| `tests/test_stage4_ranking.py` | Stage 4 | 4-dimension absolute rubric (0-10), strict score-based ordering, deterministic tie-breaking (duration closeness to 35.0s sweet spot), beat progression formatting with dialogue lines and camera cues. |
| `tests/test_stage5_portfolio.py` | Stage 5 | Narrative signature synthesis, cosine similarity matrix symmetry, MMR portfolio selection ($\lambda=0.55$), hard deduplication (similarity cutoff $\ge 0.90$), rule-based boundary cleanliness check. |
| `tests/test_cli_e2e.py` | End-to-End | CLI argument parsing, script file resolution, full dry-run subprocess execution on real Episode 1 script with 0 API calls. |

---

## How to Run the Pipeline

The master orchestrator `run_pipeline.py` (and convenience alias `run_all_episodes.py`) provides a clean CLI for single-episode and batch executions.

### 1. Dry-Run Structural Validation (0 LLM Calls)
Validates screenplay parsing, beat edge snapping, and capped window search across any episode with **zero API calls and zero cost**:
```bash
# Dry run for Episode 1:
python3 run_pipeline.py --episode 1 --dry-run

# Dry run across all 50 episodes:
python3 run_pipeline.py --all --start-ep 1 --end-ep 50 --dry-run
```

### 2. Run a Single Episode
Execute the full 5-stage pipeline for Episode 1 (the runner automatically resolves the production script in the series scripts folder):
```bash
python3 run_pipeline.py --episode 1 --outputs-dir outputs_final_batch --concurrency 4
```

### 3. Run on an Explicit Script File (`--script`)
You can pass any screenplay file directly from anywhere on disk without relying on predefined naming or directories:
```bash
# Run on an explicit script file (episode number automatically inferred from filename):
python3 run_pipeline.py --script "path/to/any_episode_script.txt"

# Or with an explicit episode number:
python3 run_pipeline.py --script "path/to/script.txt" --episode 1

# Or in dry-run mode:
python3 run_pipeline.py --script "path/to/script.txt" --dry-run
```

### 4. Custom Scripts Directory (`--scripts-dir`)
If your screenplay files live in a custom folder:
```bash
python3 run_pipeline.py --scripts-dir "path/to/my_scripts" --episode 5
```

### 5. Run an Episode Range
Process a specific slice of episodes (e.g., Episodes 1 to 5):
```bash
python3 run_pipeline.py --all --start-ep 1 --end-ep 5 --outputs-dir outputs_final_batch --concurrency 4
```

### 6. Run the Full 50-Episode Series Batch
Execute the end-to-end pipeline across the entire series with automatic checkpointing and telemetry logging:
```bash
python3 run_pipeline.py --all --start-ep 1 --end-ep 50 --outputs-dir outputs_final_batch --concurrency 4
```
Or use the shortcut:
```bash
python3 run_all_episodes.py
```

#### Checkpointing & Idempotent Resume:
The pipeline includes automatic stage-level and episode-level checkpointing:
* By default, `--resume` is enabled. If a run is interrupted, re-running the command **skips completed episodes and stages**, saving time and tokens.
* To force complete re-computation from scratch, pass `--force`.

---

## Five-Stage Architecture In-Depth

```
Screenplay (.txt)
  │
  ▼
[Stage 0: Deterministic Parser] ────────── 0 Calls | 0.004s
  │  • Regex timestamp extraction ("given" vs "estimated" ~2.5 wps)
  │  • Simultaneous event containerization (CAMERA + ACTION + DIALOGUE)
  │  • Scene boundary tracking (no implicit cross-scene merges)
  ▼
[Stage 1: Batched LLM Detection] ──────── 2 Calls | ~35s
  │  • Call #1: Visual & dialogue opening hook scores (1–5) separately
  │  • Call #2: Curiosity gap (1–5) & anti-defusion cliffhanger check
  ▼
[Stage 2: Capped Window Search] ────────── 0 Calls | 0.003s
  │  • Pure Python beat-edge snapping (start = hook.start, end = cliff.end)
  │  • Scene boundary lock (single-segment candidates never cross scenes)
  │  • Deterministic duration filter: 20.0s ≤ duration ≤ 60.0s
  │  • Capped matching: keep best cliffhanger per hook & best hook per cliffhanger
  │  • Floor-relaxation fallback: relaxes floor by 1 pt if < 5 pairs found
  ▼
[Stage 3: Merged 3-in-1 Scoring] ──────── ~4-8 Calls (Batches of 4) | ~50s
  │  • Binary Comprehension Gate: (opening_legibility ∧ arc_legibility)
  │  • Ban on retrospective justification: opening judged on beat 0 alone
  │  • Emotional Salience Score (1–5 intensity) + core emotion extraction
  │  • Spoiler Disclosure Level (0–3 tier, reported never dropped)
  ▼
[Stage 4: Absolute Rubric Ranking] ────── 1 Call | ~25s
  │  • 4 Narrative Dimensions (2.5 pts each = 10.0 max):
  │    1. Hook Scroll-Stop Power
  │    2. Mid-Arc Escalation & Conflict Density
  │    3. Cliffhanger Irresolution & Curiosity Gap
  │    4. Standalone Viewer Payoff & Emotional Resonance
  │  • Screenplay dialogue highlight injection (start, mid-arc, exit)
  │  • Anchor-calibrated grounding (worked examples)
  │  • Deterministic Python tie-breaking (duration closeness to 35.0s sweet spot)
  ▼
[Stage 5: Boundary Cleanliness & MMR] ─── 0 Calls | ~1.2s
  │  • Rule-based dialogue boundary cleanliness check
  │  • Local FastEmbed ONNX dense embeddings (BAAI/bge-small-en-v1.5)
  │  • Maximal Marginal Relevance portfolio selection (λ=0.55)
  │  • Hard deduplication: cosine similarity < 0.90
  ▼
Top-5 Diverse Promo Portfolio (.json)
```

---

## The 14 Non-Negotiable Pipeline Invariants

The rebuild strictly enforces the 14 core rules established in `AGENTS.md`:

1. **No Invented Timestamps:** Every clip boundary resolves to a real beat start/end time from the parser. LLMs output beat IDs, never raw timestamps.
2. **Beat-Edge Snapping:** Clip boundaries snap strictly to beat edges (`start = start_beat.start`, `end = end_beat.end`). No manual offsets, no smoothing.
3. **Deterministic Duration Filtering:** 20–60s duration filtering is a pure Python filter, never an LLM judgment call or soft prompt preference.
4. **Mandatory `timestamp_source`:** Required, non-defaulted field (`"given"` or `"estimated"`) on every Event, Beat, and Segment. Candidates mixing sources resolve to `"estimated"`.
5. **Scene Boundary Safety:** Single-segment candidates must never cross scene boundaries.
6. **Binary Standalone Comprehension Gate:** `passes_comprehension_gate` = `opening_legibility ∧ arc_legibility`. Retrospective justification is strictly forbidden (beat 0 must make sense alone). Failed candidates are excluded from ranking entirely.
7. **Transparent Spoiler Reporting:** Spoiler levels (0–3) are reported, never auto-dropped.
8. **Absolute Rubric Ranking:** Grounded in a calibrated 4-dimension 10.0-point rubric, replacing non-scalable pairwise round-robin comparisons.
9. **Portfolio MMR & Hard Deduplication:** Near-identical hooks are deduplicated by semantic embedding similarity ($\text{Sim} < 0.90$).
10. **Zero Mock Values in Production:** Scored objects come from actual model calls or deterministic algorithms.
11. **No Hallucinated Plot Facts:** Examples in prompts are grounded in real episode text or explicitly labeled as illustrative hypotheticals.
12. **Relative Paths Only:** All file paths are resolved relative to the repository via `__file__`.
13. **Schema Integrity:** Schema changes regenerate affected output files by re-running stages, never by hand-editing JSON.
14. **Position Bias Mitigation:** Randomize candidate order in prompts where items are compared.

---

## Output Files & Deliverables

All deliverables and audit artifacts are pre-computed and stored in `outputs_final_batch/`:

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

## Configuration & Environment Variables

| Variable | Default | Description |
|---|---|---|
| `PROMO_MODEL` | `gemini-3.8-flash-low` | LLM model identifier (e.g. `gpt-4o-mini`, `gemini-2.5-flash`) |
| `OPENAI_API_KEY` | *(None)* | API key for OpenAI-compatible REST endpoints |
| `GEMINI_API_KEY` | *(None)* | API key for Google Gemini REST endpoints |
| `LLM_API_BASE_URL`| *(Auto)* | Custom base URL for self-hosted models (vLLM / Ollama) |
| `AGY_EXE` | *(System path)* | Path to local sandbox CLI runner bridge |

---

## Repository Structure

```
KukuFM-submission/
├── README.md                           # Main project documentation (this file)
├── SOLUTION_ARCHITECTURE_DOC.md        # Exhaustive architectural specification & engineering report
├── AGENTS.md                           # Core behavioral invariants & prompt engineering rules
├── requirements.txt                    # Project dependencies (FastEmbed, NumPy, Pytest, etc.)
├── pytest.ini                          # Pytest configuration
├── run_tests.py                        # Universal unit test runner (55 tests)
├── run_pipeline.py                     # Master pipeline CLI runner
├── run_all_episodes.py                 # Full batch series runner shortcut
├── outputs_final_batch/                # Complete pre-computed outputs for all 50 episodes
│   ├── episode_001_final_portfolio.json
│   ├── episode_050_final_portfolio.json
│   ├── batch_summary.json              # Aggregated batch metrics and cost report
│   └── llm_calls.jsonl                 # Detailed LLM telemetry audit log
│
├── tests/                              # Comprehensive top-level unit test suite
│   ├── conftest.py                     # Test environment & path setup
│   ├── test_stage0_parser.py           # Stage 0 deterministic parser tests
│   ├── test_stage1_detection.py        # Stage 1 batched signal detector tests
│   ├── test_stage2_window_search.py    # Stage 2 capped window search tests
│   ├── test_stage3_scoring.py          # Stage 3 comprehension gate & emotion tests
│   ├── test_stage4_ranking.py          # Stage 4 absolute rubric ranking tests
│   ├── test_stage5_portfolio.py        # Stage 5 MMR portfolio & cleanliness tests
│   └── test_cli_e2e.py                 # End-to-end CLI dry-run integration tests
│
└── promo-finder/                       # Core python engine package
    ├── parser/                         # Stage 0: Deterministic screenplay parser
    │   ├── parser.py                   # Pure Python regex & event containerization
    │   ├── schema.py                   # Data contracts (Episode, Beat, Event)
    │   └── test_parser.py              # Parser unit tests
    │
    ├── candidates/                     # Stage 1 & 2: Detection & Window Search
    │   ├── llm_client.py               # Model-agnostic REST & sandbox LLM client
    │   ├── llm_detector.py             # Batched concurrent hook & cliffhanger detector
    │   ├── window_search.py            # Capped window search & floor-relaxation fallback
    │   └── test_candidates.py          # Candidate generation unit tests
    │
    ├── scoring/                        # Stage 3: Merged 3-in-1 Candidate Scorer
    │   ├── merged_scorer.py            # Comprehension gate, emotion & spoiler scoring
    │   └── test_merged_scorer.py       # Scorer unit tests
    │
    ├── ranking/                        # Stage 4 & 5: Absolute Ranking & Portfolio MMR
    │   ├── batched_ranker.py           # Pure absolute rubric ranker (0-10 scale in 1 call)
    │   ├── mmr.py                      # Rule-based cleanliness & FastEmbed MMR
    │   ├── test_batched_ranker.py      # Ranker unit tests
    │   └── test_mmr.py                 # MMR unit tests
    │
    └── prompts/                        # Versioned, rubric-grounded LLM prompts
        ├── hook_detector_v1.txt        # Stage 1A Hook detection prompt
        ├── cliffhanger_detector_v1.txt # Stage 1B Cliffhanger detection prompt
        ├── merged_scoring_v1.txt       # Stage 3 Merged 3-in-1 scoring prompt
        └── batched_ranking_v1.txt      # Stage 4 Pure absolute rubric ranking prompt
```

---

## Troubleshooting & FAQs

#### Q1: How do I run the unit tests?
Simply run:
```bash
python3 run_tests.py
```
This requires no external services or API keys and finishes in ~0.15 seconds.

#### Q2: "ModuleNotFoundError: No module named 'fastembed'"
The project automatically resolves virtual environment site-packages in `ranking/mmr.py` and `run_pipeline.py`. If running in a fresh shell, activate the virtual environment:
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

#### Q3: "What happens if an episode has no strong hooks or cliffhangers?"
`candidates/window_search.py` features an automatic **Floor-Relaxation Fallback**. If strict filtering ($\text{score} \ge 4$) produces fewer than 5 candidate pairs, the threshold automatically relaxes to $\text{score} \ge 3$, guaranteeing downstream pipeline viability even on dialogue-light or expositional filler episodes.

#### Q4: "Can I run the pipeline without incurring API costs?"
Yes! Run the dry-run command:
```bash
python3 run_pipeline.py --episode 1 --dry-run
```
This validates the parser, event containerization, beat boundaries, and window search with zero API calls.

---

*Authored by the Promo Finder Engineering Team | Built for Kuku FM Microdrama Studio*
