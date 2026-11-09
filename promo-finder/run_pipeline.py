"""
run_pipeline.py — Master Unified Pipeline for Promo Finder.

Implements the rebuilt 5-stage microdrama promo selection architecture:
- Stage 0: Deterministic screenplay parsing (parser/parser.py) [0 calls]
- Stage 1: Batched concurrent Hook & Cliffhanger detection (candidates/llm_detector.py) [2 calls]
- Stage 2: Capped bidirectional window search + floor-relaxation fallback (candidates/window_search.py) [0 calls]
- Stage 3: Merged 3-in-1 concurrent candidate scoring (scoring/merged_scorer.py) [~4-8 calls]
- Stage 4: Pure 4-dimension absolute rubric ranking (ranking/batched_ranker.py) [1 call]
- Stage 5: Rule-based boundary cleanliness check + MMR portfolio selection (ranking/mmr.py) [0 calls]

Total LLM calls: ~7-12 calls per episode (down from ~280).
Total wall-clock: ~2-3 minutes per episode (down from ~10.5m).
"""

import argparse
import datetime
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Ensure venv site-packages are accessible even when invoked via system python
venv_site = ROOT_DIR / ".venv" / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
if venv_site.exists() and str(venv_site) not in sys.path:
    sys.path.insert(0, str(venv_site))

from parser.parser import parse_file
from parser.schema import Episode, Beat
from candidates.schema import HookSignal, CliffhangerSignal
from candidates.llm_detector import detect_episode_signals
from candidates.llm_client import set_log_dir
from candidates.window_search import bidirectional_window_search
from scoring.merged_scorer import score_all_candidates_concurrent
from ranking.batched_ranker import run_batched_ranking
from ranking.mmr import (
    build_narrative_signature,
    compute_embeddings,
    compute_cosine_similarity_matrix,
    select_mmr_portfolio,
)


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def evaluate_boundary_cleanliness(beats_map: Dict[str, Dict], start_b_id: str, end_b_id: str) -> Tuple[float, float, float]:
    """Pure Python rule-based boundary cleanliness assessment."""
    start_b = beats_map[start_b_id]
    end_b = beats_map[end_b_id]

    open_score = 4.0
    for ev in start_b.get("events", []):
        if ev.get("type") in ("camera", "action"):
            open_score = 5.0
            break

    cliff_score = 4.0
    dialogue_events = [ev for ev in end_b.get("events", []) if ev.get("type") == "dialogue"]
    if dialogue_events:
        last_text = dialogue_events[-1].get("description", "").strip()
        if last_text.endswith((".", "!", "?", '"', "'")):
            cliff_score = 5.0
        elif last_text.endswith((",", "-", "--")):
            cliff_score = 2.0

    comp_clean = round((open_score + cliff_score) / 2.0, 2)
    return open_score, cliff_score, comp_clean


def get_repo_root() -> Path:
    """Resolves the true repository root path without hardcoding.
    
    Works whether run_pipeline.py is invoked from repository root,
    within the promo-finder directory, or via symlink.
    Strictly stays within repository boundary per AGENTS.md Rule 12.
    """
    file_path = Path(__file__).resolve()
    current = file_path.parent
    if (current / "promo-finder").exists():
        return current
    if current.name == "promo-finder":
        return current.parent
    for parent in [current] + list(current.parents):
        if (parent / ".git").exists() or (parent / "script_directory_for_promo").exists() or (parent / "tests").exists():
            return parent
    return current


def discover_script_directories(repo_root: Path, scripts_dir: Optional[str] = None) -> List[Path]:
    """Dynamically discovers script directories within the repository.
    
    Zero hardcoded show titles. Works with ANY drama directory structure.
    Search priority:
    1. Custom scripts_dir if explicitly provided by the user.
    2. Any directories inside repo_root containing episode script files (.txt).
    3. Standard directory names ('scripts', 'script_directory_for_promo', etc.).
    4. Parser fixtures sample_scripts (secondary fallback for unit tests).
    """
    search_dirs: List[Path] = []
    
    # 1. Custom scripts_dir if provided
    if scripts_dir:
        p = Path(scripts_dir) if Path(scripts_dir).is_absolute() else (repo_root / scripts_dir)
        if p.exists() and p.is_dir():
            search_dirs.append(p)
            for sub in p.iterdir():
                if sub.is_dir():
                    search_dirs.append(sub)

    # 2. Dynamic in-repo discovery: find any directories with episode .txt files
    exclude_parts = {".git", ".venv", "venv", "__pycache__", "outputs", "outputs_final_batch", "outputs_all_episodes", "fixtures"}
    try:
        for txt_file in repo_root.glob("**/*.txt"):
            if any(ex in txt_file.parts for ex in exclude_parts):
                continue
            if re.search(r"episode[-_\s]*\d+", txt_file.name, re.IGNORECASE):
                parent_dir = txt_file.parent
                if parent_dir not in search_dirs:
                    search_dirs.append(parent_dir)
    except Exception:
        pass

    # 3. Standard fallback script folder names within repo
    standard_names = ["scripts", "script_directory_for_promo", "episodes", "data"]
    for s_name in standard_names:
        cand = repo_root / s_name
        if cand.exists() and cand.is_dir():
            if cand not in search_dirs:
                search_dirs.append(cand)
            for sub in cand.iterdir():
                if sub.is_dir() and sub not in search_dirs:
                    search_dirs.append(sub)

    # 4. Secondary fallback: parser unit test fixtures
    fixture_dirs = [
        repo_root / "promo-finder" / "parser" / "fixtures" / "sample_scripts",
        repo_root / "parser" / "fixtures" / "sample_scripts",
    ]
    for fd in fixture_dirs:
        if fd.exists() and fd not in search_dirs:
            search_dirs.append(fd)

    return search_dirs


def find_script_file(ep_num: int, repo_root: Path, scripts_dir: Optional[str] = None) -> Optional[Path]:
    """Finds script file for a given episode number across dynamically discovered locations.
    
    Strictly adheres to AGENTS.md Rule 12 (never traverses outside repo_root).
    """
    search_dirs = discover_script_directories(repo_root, scripts_dir=scripts_dir)

    patterns = [
        f"episode-{ep_num:03d}-*.txt",
        f"episode_{ep_num:03d}_*.txt",
        f"episode-{ep_num:03d}.txt",
        f"episode_{ep_num:03d}.txt",
        f"episode-{ep_num:02d}-*.txt",
        f"episode_{ep_num:02d}_*.txt",
        f"episode-{ep_num:02d}.txt",
        f"episode_{ep_num:02d}.txt",
        f"episode-{ep_num}-*.txt",
        f"episode_{ep_num}_*.txt",
        f"episode-{ep_num}.txt",
        f"episode_{ep_num}.txt",
        f"*{ep_num:03d}*.txt",
    ]

    for d in search_dirs:
        if not d.exists():
            continue
        for pat in patterns:
            matches = sorted(list(d.glob(pat)))
            if matches:
                return matches[0]

        # Regex fallback over directory contents
        try:
            for child in sorted(d.iterdir()):
                if child.is_file() and child.suffix.lower() == ".txt":
                    m = re.search(r"episode[-_\s]*0*(\d+)", child.name, re.IGNORECASE)
                    if m and int(m.group(1)) == ep_num:
                        return child
        except Exception:
            pass

    return None


def run_single_episode(
    ep_num: int,
    script_path: Path,
    output_dir: Path,
    dry_run: bool = False,
    model: Optional[str] = None,
    concurrency: int = 4,
    resume: bool = True,
) -> Dict[str, Any]:
    """Executes the full 5-stage pipeline for a single episode."""
    set_log_dir(output_dir)
    print("=" * 68)
    print(f"PROMO FINDER PIPELINE — EPISODE {ep_num:03d} {'[DRY RUN]' if dry_run else ''}")
    try:
        display_path = script_path.relative_to(get_repo_root())
    except ValueError:
        try:
            display_path = script_path.relative_to(Path.cwd())
        except ValueError:
            display_path = script_path.name
    print(f"Source Script: {display_path}")
    print("=" * 68)

    # Episode-level checkpointing: return existing results if full episode already completed
    summary_file = output_dir / f"episode_{ep_num:03d}_rebuild_summary.json"
    portfolio_file = output_dir / f"episode_{ep_num:03d}_final_portfolio.json"
    if resume and not dry_run and summary_file.exists() and portfolio_file.exists():
        try:
            summary_data = load_json(summary_file)
            print(f"[Checkpoint] Episode {ep_num:03d} already completed. Resuming (skipping execution)...")
            return summary_data
        except Exception:
            pass

    pipeline_start = time.time()
    run_start_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
    stage_metrics: Dict[str, Any] = {}

    # Stage 0: Parse
    t0 = time.time()
    ep = parse_file(str(script_path))
    ep.episode_number = ep_num
    beats_map = {b.beat_id: b.to_dict() for b in ep.beats}
    save_json(ep.to_dict(), output_dir / f"episode_{ep_num:03d}_golden.json")
    stage_metrics["Stage 0: Parsing"] = {
        "calls": 0,
        "latency_seconds": round(time.time() - t0, 3),
        "summary": f"{len(ep.beats)} beats parsed, duration: {ep.beats[-1].end:.1f}s",
    }
    print(f"\n[Stage 0] Screenplay parsed: {len(ep.beats)} beats in {stage_metrics['Stage 0: Parsing']['latency_seconds']}s.")

    if dry_run:
        # In dry-run mode, construct synthetic signals to test downstream window search & geometry
        print("[Dry Run] Simulating Stage 1 signals for structural validation...")
        sim_hooks = {}
        sim_cliffs = {}
        for b in ep.beats:
            has_dial = any(ev.type == "dialogue" for ev in b.events)
            sim_hooks[b.beat_id] = HookSignal(
                beat_id=b.beat_id,
                dialogue_hook_score=4 if has_dial else 1,
                dialogue_hook_rationale="Simulated dry-run hook",
                visual_hook_score=2,
                visual_hook_rationale="Simulated dry-run visual",
                is_strong_hook=has_dial,
            )
            sim_cliffs[b.beat_id] = CliffhangerSignal(
                beat_id=b.beat_id,
                curiosity_gap_score=4 if has_dial else 1,
                specific_question="Simulated dry-run cliffhanger?",
                stakes_importance="High" if has_dial else "Low",
                defuses_tension=False,
                cliffhanger_type="dialogue_question" if has_dial else "visual_cut",
                is_strong_cliffhanger=has_dial,
            )

        cands = bidirectional_window_search(ep, sim_hooks, sim_cliffs)
        print(f"[Dry Run] Stage 2 Capped search produced {len(cands)} valid candidates.")
        for i, c in enumerate(cands[:5], 1):
            print(f"  • Candidate {i}: {c.start_beat_id} -> {c.end_beat_id} ({c.duration:.1f}s, Scene {beats_map[c.start_beat_id]['scene_number']})")
        print("\n[Dry Run] Structural validation successful: all paths, parsing, schemas, and window search operational.")
        return {"dry_run": True, "beats_count": len(ep.beats), "candidates_count": len(cands)}

    # Stage 1: Detection (with Checkpointing)
    signals_file = output_dir / f"episode_{ep_num:03d}_signals.json"
    hook_signals = None
    cliff_signals = None

    if resume and not dry_run and signals_file.exists():
        try:
            sig_data = load_json(signals_file)
            if "hook_signals" in sig_data and "cliffhanger_signals" in sig_data:
                hook_signals = {h["beat_id"]: HookSignal(**h) for h in sig_data["hook_signals"]}
                cliff_signals = {c["beat_id"]: CliffhangerSignal(**c) for c in sig_data["cliffhanger_signals"]}
                stage_metrics["Stage 1: Detection"] = {
                    "calls": 0,
                    "latency_seconds": 0.0,
                    "summary": f"{len(hook_signals)} hook signals, {len(cliff_signals)} cliffhanger signals (from checkpoint)",
                }
                print(f"\n[Stage 1 Checkpoint] Loaded {len(hook_signals)} hook signals and {len(cliff_signals)} cliffhanger signals from {signals_file.name}.")
        except Exception as e:
            print(f"  [Checkpoint Warning] Failed to parse {signals_file.name}: {e}. Re-running Stage 1 detection...")
            hook_signals = None
            cliff_signals = None

    if hook_signals is None or cliff_signals is None:
        t0 = time.time()
        print(f"\n[Stage 1] Running batched hook and cliffhanger detection (2 concurrent calls)...")
        hook_signals, cliff_signals = detect_episode_signals(ep, model=model) if model else detect_episode_signals(ep)

        save_json({
            "episode_number": ep_num,
            "hook_signals": [h.to_dict() for h in hook_signals.values()],
            "cliffhanger_signals": [c.to_dict() for c in cliff_signals.values()],
        }, signals_file)
        stage_metrics["Stage 1: Detection"] = {
            "calls": 2,
            "latency_seconds": round(time.time() - t0, 3),
            "summary": f"{len(hook_signals)} hook signals, {len(cliff_signals)} cliffhanger signals",
        }
        print(f"[Stage 1] Completed in {stage_metrics['Stage 1: Detection']['latency_seconds']}s (2 calls).")

    # Stage 2: Capped Window Search with Floor-Relaxation Fallback
    t0 = time.time()
    raw_cands = bidirectional_window_search(ep, hook_signals, cliff_signals)

    candidates_data = []
    for i, c in enumerate(raw_cands, 1):
        cid = f"CAND_EP{ep_num:03d}_{i:02d}"
        c.candidate_id = cid
        c.episode_number = ep_num
        d = c.to_dict()
        d["candidate_id"] = cid
        candidates_data.append(d)

    save_json({
        "episode_number": ep_num,
        "total_candidates": len(candidates_data),
        "candidates": candidates_data,
    }, output_dir / f"episode_{ep_num:03d}_candidates.json")

    stage_metrics["Stage 2: Window Search"] = {
        "calls": 0,
        "latency_seconds": round(time.time() - t0, 3),
        "summary": f"Capped candidate count: {len(candidates_data)}",
    }
    print(f"[Stage 2] Capped search generated {len(candidates_data)} candidates in {stage_metrics['Stage 2: Window Search']['latency_seconds']}s.")

    # Stage 3: Merged 3-in-1 Scoring (with Checkpointing)
    scored_file = output_dir / f"episode_{ep_num:03d}_stage3_scored.json"
    scored_candidates = None

    if resume and not dry_run and scored_file.exists():
        try:
            s3_data = load_json(scored_file)
            if "scored_candidates" in s3_data and len(s3_data["scored_candidates"]) == len(candidates_data):
                scored_candidates = s3_data["scored_candidates"]
                stage_metrics["Stage 3: Merged Scoring"] = {
                    "calls": 0,
                    "latency_seconds": 0.0,
                    "summary": f"{len(scored_candidates)} candidates scored (from checkpoint)",
                }
                print(f"\n[Stage 3 Checkpoint] Loaded {len(scored_candidates)} scored candidates from {scored_file.name}.")
        except Exception as e:
            print(f"  [Checkpoint Warning] Failed to parse {scored_file.name}: {e}. Re-running Stage 3 scoring...")
            scored_candidates = None

    if scored_candidates is None:
        t0 = time.time()
        num_s3_calls = (len(candidates_data) + 3) // 4
        print(f"\n[Stage 3] Merged scoring on {len(candidates_data)} candidates in batches of 4 ({concurrency} concurrent workers)...")
        scored_candidates = score_all_candidates_concurrent(
            candidates=candidates_data,
            beats_map=beats_map,
            episode_number=ep_num,
            batch_size=4,
            max_workers=concurrency,
            model=model,
        )

        save_json({
            "episode_number": ep_num,
            "total_scored": len(scored_candidates),
            "scored_candidates": scored_candidates,
        }, scored_file)

        stage_metrics["Stage 3: Merged Scoring"] = {
            "calls": num_s3_calls,
            "latency_seconds": round(time.time() - t0, 3),
            "summary": f"{len(scored_candidates)} candidates scored in {num_s3_calls} calls",
        }
        print(f"[Stage 3] Completed in {stage_metrics['Stage 3: Merged Scoring']['latency_seconds']}s ({num_s3_calls} calls).")

    scored_map = {c["candidate_id"]: c for c in scored_candidates}
    cands_map = {c["candidate_id"]: c for c in candidates_data}
    gate_survivors = [c for c in candidates_data if scored_map[c["candidate_id"]]["comprehension"]["passes_gate"]]
    pass_pct = (len(gate_survivors) / max(1, len(candidates_data))) * 100.0
    print(f"  [Comprehension Gate] {len(gate_survivors)}/{len(candidates_data)} passed binary gate ({pass_pct:.1f}%).")

    # Stage 4: Pure Absolute Rubric Ranking (with Checkpointing)
    ranking_file = output_dir / f"episode_{ep_num:03d}_stage4_ranking.json"
    rankings = None

    if resume and not dry_run and ranking_file.exists():
        try:
            r_data = load_json(ranking_file)
            if "rankings" in r_data and len(r_data["rankings"]) == len(gate_survivors):
                rankings = r_data["rankings"]
                stage_metrics["Stage 4: Absolute Ranking"] = {
                    "calls": 0,
                    "latency_seconds": 0.0,
                    "summary": f"Ranked {len(rankings)} candidates (from checkpoint)",
                }
                print(f"\n[Stage 4 Checkpoint] Loaded Stage 4 ranking from {ranking_file.name}.")
        except Exception as e:
            print(f"  [Checkpoint Warning] Failed to parse {ranking_file.name}: {e}. Re-running Stage 4 ranking...")
            rankings = None

    if rankings is None:
        t0 = time.time()
        print(f"\n[Stage 4] Pure absolute rubric ranking across {len(gate_survivors)} gate survivors (1 call)...")
        rankings, _ = run_batched_ranking(
            passing_candidates=gate_survivors,
            scored_map=scored_map,
            beats_map=beats_map,
            episode_number=ep_num,
            model=model,
        )

        save_json({
            "episode_number": ep_num,
            "total_ranked": len(rankings),
            "rankings": rankings,
        }, ranking_file)

        stage_metrics["Stage 4: Absolute Ranking"] = {
            "calls": 1,
            "latency_seconds": round(time.time() - t0, 3),
            "summary": f"Ranked {len(rankings)} candidates in 1 call (pure absolute rubric)",
        }
        print(f"[Stage 4] Completed in {stage_metrics['Stage 4: Absolute Ranking']['latency_seconds']}s (1 call).")

    # Stage 5: Rule-Based Boundary Cleanliness Check + MMR Portfolio Selection
    t0 = time.time()
    print(f"\n[Stage 5] Running rule-based dialogue boundary check and constrained MMR portfolio selection...")

    if not rankings:
        print("[Stage 5] No candidates available for MMR portfolio selection; portfolio is empty.")
        picks = []
        disqualifications = []
        sim_matrix = np.empty((0, 0), dtype=np.float32)
        enriched_promos = []
    else:
        signatures = []
        for item in rankings:
            cid = item["candidate_id"]
            c_base = cands_map.get(cid, {})
            c_s3 = scored_map.get(cid, {})
            signatures.append(build_narrative_signature(c_base, c_s3))

        norm_embs = compute_embeddings(signatures, model_name="BAAI/bge-small-en-v1.5")
        sim_matrix = compute_cosine_similarity_matrix(norm_embs)

        target_k = min(5, len(rankings))
        picks, disqualifications = select_mmr_portfolio(
            rankings=rankings,
            sim_matrix=sim_matrix,
            k=target_k,
            lambda_param=0.55,
            max_sim_threshold=0.90,
        )

        enriched_promos = []
        for pick in picks:
            cid = pick["candidate_id"]
            c_base = cands_map.get(cid, {})
            c_s3 = scored_map.get(cid, {})
            s_b = c_base.get("start_beat_id", "")
            e_b = c_base.get("end_beat_id", "")
            start_beat = beats_map.get(s_b, {})
            end_beat = beats_map.get(e_b, {})

            open_c, cliff_c, comp_c = evaluate_boundary_cleanliness(beats_map, s_b, e_b)

            hook_dialogues = [ev["description"] for ev in start_beat.get("events", []) if ev.get("type") == "dialogue"]
            hook_chars = [ev.get("character") for ev in start_beat.get("events", []) if ev.get("type") == "dialogue"]
            cliff_dialogues = [ev["description"] for ev in end_beat.get("events", []) if ev.get("type") == "dialogue"]
            cliff_chars = [ev.get("character") for ev in end_beat.get("events", []) if ev.get("type") == "dialogue"]

            ranking_entry = next((r for r in rankings if r["candidate_id"] == cid), {})

            enriched_promos.append({
                "portfolio_rank": pick["selection_order"],
                "candidate_id": cid,
                "absolute_rubric_score": ranking_entry.get("absolute_score", 0.0),
                "subscores": ranking_entry.get("subscores", {}),
                "mmr_score": pick["mmr_score"],
                "max_similarity_to_selected": pick["max_similarity_to_selected"],
                "boundary": {
                    "start_beat_id": s_b,
                    "end_beat_id": e_b,
                    "start_time": c_base.get("start_time", 0.0),
                    "end_time": c_base.get("end_time", 0.0),
                    "duration": c_base.get("duration", 0.0),
                    "timestamp_source": c_base.get("timestamp_source", "given"),
                    "composite_cleanliness": comp_c,
                },
                "hook": {
                    "opening_quote": hook_dialogues[0] if hook_dialogues else "None (Visual Opening)",
                    "opening_character": hook_chars[0] if hook_chars else None,
                    "rationale": c_base.get("hook_signal", {}).get("dialogue_hook_rationale") or c_base.get("hook_signal", {}).get("visual_hook_rationale"),
                },
                "cliffhanger": {
                    "final_quote": cliff_dialogues[-1] if cliff_dialogues else "None (Visual Cliffhanger)",
                    "final_character": cliff_chars[-1] if cliff_chars else None,
                    "specific_question": c_base.get("cliffhanger_signal", {}).get("specific_question"),
                },
                "situational_summary": c_s3.get("comprehension", {}).get("situational_summary", ""),
                "emotional_salience_score": c_s3.get("emotional_salience", {}).get("emotional_salience_score", 1),
                "spoiler_level": c_s3.get("spoiler", {}).get("spoiler_level", 0),
            })

    portfolio_data = {
        "episode_number": ep_num,
        "mmr_parameters": {
            "lambda": 0.55,
            "max_sim_threshold": 0.90,
            "target_size": target_k if rankings else 0,
        },
        "selected_promos": enriched_promos,
        "disqualification_log": disqualifications,
        "similarity_matrix": sim_matrix.tolist() if hasattr(sim_matrix, "tolist") else [],
    }
    save_json(portfolio_data, output_dir / f"episode_{ep_num:03d}_final_portfolio.json")

    stage_metrics["Stage 5: Portfolio Selection"] = {
        "calls": 0,
        "latency_seconds": round(time.time() - t0, 3),
        "summary": f"{len(enriched_promos)} promos selected via pure Python MMR",
    }
    print(f"[Stage 5] Completed in {stage_metrics['Stage 5: Portfolio Selection']['latency_seconds']}s (0 calls).")

    total_pipeline_time = round(time.time() - pipeline_start, 2)
    total_calls = sum(m["calls"] for m in stage_metrics.values())

    # Read token and cost metrics
    total_in_tokens = 0
    total_out_tokens = 0
    total_cost_usd = 0.0
    calls_log_path = output_dir / "llm_calls.jsonl"
    if not calls_log_path.exists():
        calls_log_path = ROOT_DIR / "outputs" / "llm_calls.jsonl"

    if calls_log_path.exists():
        with open(calls_log_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                    if entry.get("episode_number") == ep_num and entry.get("timestamp", "") >= run_start_iso:
                        total_in_tokens += entry.get("input_tokens", 0)
                        total_out_tokens += entry.get("output_tokens", 0)
                        total_cost_usd += entry.get("cost_usd", 0.0)
                except Exception:
                    pass

    summary_data = {
        "episode_number": ep_num,
        "total_llm_calls": total_calls,
        "total_wall_clock_seconds": total_pipeline_time,
        "total_wall_clock_minutes": round(total_pipeline_time / 60.0, 2),
        "total_input_tokens": total_in_tokens,
        "total_output_tokens": total_out_tokens,
        "total_estimated_cost_usd": round(total_cost_usd, 6),
        "stage_breakdown": stage_metrics,
    }
    save_json(summary_data, output_dir / f"episode_{ep_num:03d}_rebuild_summary.json")

    print("\n" + "=" * 68)
    print(f"EPISODE {ep_num:03d} TIMING & COST SUMMARY")
    print("=" * 68)
    print(f"Total LLM Calls: {total_calls} (Rebuild Target: <15-20 calls)")
    print(f"Total Wall-Clock Time: {total_pipeline_time:.1f}s ({total_pipeline_time/60.0:.2f}m)")
    print(f"Total Tokens: {total_in_tokens:,} in / {total_out_tokens:,} out")
    print(f"Total Estimated Cost: ${total_cost_usd:.5f} USD")
    for stg, val in stage_metrics.items():
        print(f"  • {stg}: {val['calls']} calls | {val['latency_seconds']}s | {val['summary']}")
    print("=" * 68)

    return summary_data


def main():
    parser = argparse.ArgumentParser(description="Master Promo Finder Rebuilt Pipeline")
    parser.add_argument("--script", type=str, default=None, help="Explicit path to a screenplay script file (.txt)")
    parser.add_argument("--scripts-dir", type=str, default=None, help="Custom directory containing episode screenplay files")
    parser.add_argument("--episode", type=int, default=None, help="Episode number to process (default: 1, or inferred from --script)")
    parser.add_argument("--all", action="store_true", help="Process all available episodes in batch")
    parser.add_argument("--start-ep", type=int, default=1, help="Start episode for batch")
    parser.add_argument("--end-ep", type=int, default=50, help="End episode for batch")
    parser.add_argument("--dry-run", action="store_true", help="Dry run: validate parsing and geometry without LLM calls")
    parser.add_argument("--outputs-dir", type=str, default="outputs", help="Directory for output files")
    parser.add_argument("--model", type=str, default=None, help="LLM model override")
    parser.add_argument("--list-models", action="store_true", help="List all supported models across providers with pricing and exit")
    parser.add_argument("--concurrency", type=int, default=4, help="Concurrency for batched scoring")
    parser.add_argument("--resume", action="store_true", default=True, help="Resume execution from existing checkpoints (default: True)")
    parser.add_argument("--force", action="store_true", default=False, help="Force recomputation and ignore checkpoints")
    args = parser.parse_args()

    if args.list_models:
        from candidates.llm_client import MODEL_CATALOG, get_available_cli_models, AGY_EXE
        cli_models = get_available_cli_models()
        print("=" * 92)
        print("PROMO-FINDER MULTI-PROVIDER MODEL CATALOG & PRICING")
        print("=" * 92)
        print(f"{'Model Identifier':<34} {'Provider':<10} {'In $/1M':<10} {'Out $/1M':<10} {'CLI Equiv':<24}")
        print("-" * 92)
        for name, info in MODEL_CATALOG.items():
            in_cost = f"${info.get('input_cost_per_million', 0.0):.4f}"
            out_cost = f"${info.get('output_cost_per_million', 0.0):.4f}"
            provider = info.get("provider", "other")
            cli_eq = info.get("cli_equivalent", "-")
            print(f"{name:<34} {provider:<10} {in_cost:<10} {out_cost:<10} {cli_eq:<24}")
        print("-" * 92)
        print(f"\nActive CLI Bridge Binary: {AGY_EXE}")
        print(f"Detected Native CLI Models ({len(cli_models)}): {', '.join(sorted(cli_models))}")
        print("\nUsage Tips:")
        print("  • REST API Mode: Set OPENAI_API_KEY / GEMINI_API_KEY and pass any catalog model via --model <name>.")
        print("  • Local CLI Bridge Mode: Any model string is dynamically resolved to its native CLI bridge equivalent.")
        print("=" * 92)
        sys.exit(0)

    resume = (args.resume and not args.force)
    repo_root = get_repo_root()
    output_dir = Path(args.outputs_dir) if Path(args.outputs_dir).is_absolute() else (ROOT_DIR / args.outputs_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_log_dir(output_dir)

    if args.script:
        raw_path = Path(args.script)
        if raw_path.exists():
            script = raw_path
        elif (repo_root / args.script).exists():
            script = repo_root / args.script
        else:
            print(f"Error: Script file not found at: {args.script}")
            sys.exit(1)

        # Invert or infer episode number
        if args.episode is not None:
            ep_num = args.episode
        else:
            match = re.search(r"episode[-_\s]*0*(\d+)", script.name, re.IGNORECASE)
            ep_num = int(match.group(1)) if match else 1

        run_single_episode(
            ep_num=ep_num,
            script_path=script,
            output_dir=output_dir,
            dry_run=args.dry_run,
            model=args.model,
            concurrency=args.concurrency,
            resume=resume,
        )
    elif not args.all:
        ep_num = args.episode if args.episode is not None else 1
        script = find_script_file(ep_num, repo_root, scripts_dir=args.scripts_dir)
        if not script:
            print(f"Error: Could not locate screenplay script file for Episode {ep_num}.")
            sys.exit(1)
        run_single_episode(
            ep_num=ep_num,
            script_path=script,
            output_dir=output_dir,
            dry_run=args.dry_run,
            model=args.model,
            concurrency=args.concurrency,
            resume=resume,
        )
    else:
        print(f"Launching batch execution for Episodes {args.start_ep} to {args.end_ep}...")
        batch_summaries = []
        for ep in range(args.start_ep, args.end_ep + 1):
            script = find_script_file(ep, repo_root, scripts_dir=args.scripts_dir)
            if not script:
                print(f"Skipping Episode {ep:03d}: script file not found.")
                continue
            try:
                res = run_single_episode(
                    ep_num=ep,
                    script_path=script,
                    output_dir=output_dir,
                    dry_run=args.dry_run,
                    model=args.model,
                    concurrency=args.concurrency,
                    resume=resume,
                )
                batch_summaries.append(res)
            except Exception as ep_err:
                print(f"\n[ERROR] Episode {ep:03d} failed with error: {ep_err}")
                import traceback
                traceback.print_exc()

        print(f"\nBatch processing completed for {len(batch_summaries)} episodes.")
        
        # Batch summary calculation
        if not args.dry_run and batch_summaries:
            total_time = sum(s.get("total_wall_clock_seconds", 0.0) for s in batch_summaries)
            total_calls = sum(s.get("total_llm_calls", 0) for s in batch_summaries)
            total_in = sum(s.get("total_input_tokens", 0) for s in batch_summaries)
            total_out = sum(s.get("total_output_tokens", 0) for s in batch_summaries)
            total_cost = sum(s.get("total_estimated_cost_usd", 0.0) for s in batch_summaries)
            avg_time = total_time / max(1, len(batch_summaries))

            batch_summary = {
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "total_episodes_processed": len(batch_summaries),
                "start_episode": args.start_ep,
                "end_episode": args.end_ep,
                "total_wall_clock_seconds": round(total_time, 2),
                "total_wall_clock_minutes": round(total_time / 60.0, 2),
                "total_wall_clock_hours": round(total_time / 3600.0, 3),
                "average_time_per_episode_seconds": round(avg_time, 2),
                "average_time_per_episode_minutes": round(avg_time / 60.0, 2),
                "total_llm_calls": total_calls,
                "average_calls_per_episode": round(total_calls / max(1, len(batch_summaries)), 1),
                "total_input_tokens": total_in,
                "total_output_tokens": total_out,
                "total_estimated_cost_usd": round(total_cost, 6),
                "target_met_under_1_hour": (total_time / 3600.0) <= 1.0,
                "episode_summaries": batch_summaries,
            }
            save_json(batch_summary, output_dir / "batch_summary.json")

            print("\n" + "=" * 68)
            print("FULL BATCH EXECUTION SUMMARY")
            print("=" * 68)
            print(f"Total Episodes Processed: {len(batch_summaries)}")
            print(f"Total Wall-Clock Time: {total_time:.1f}s ({total_time/60.0:.2f}m / {total_time/3600.0:.2f}h)")
            print(f"Average Time Per Episode: {avg_time:.1f}s ({avg_time/60.0:.2f}m)")
            print(f"Total LLM Calls: {total_calls} (Avg {total_calls/max(1, len(batch_summaries)):.1f}/ep)")
            print(f"Total Tokens: {total_in:,} in / {total_out:,} out")
            print(f"Total Estimated Cost: ${total_cost:.5f} USD")
            print(f"Under 1 Hour Target Met: {batch_summary['target_met_under_1_hour']}")
            print("=" * 68)


if __name__ == "__main__":
    main()

