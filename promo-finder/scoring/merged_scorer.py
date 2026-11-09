"""
merged_scorer.py — Merged Stage 3 Batch Scorer (Comprehension Gate + Emotional Salience + Spoiler).

Evaluates batches of 4-5 candidates in a single LLM call per AGENTS.md and rebuild spec:
- Standalone comprehension as a binary gate (opening_legibility + arc_legibility)
- Emotional salience score (1-5)
- Spoiler level (0-3)

Calls candidates/llm_client.py with unified telemetry and cost tracking.
"""

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from candidates.llm_client import call_llm, extract_json_payload, DEFAULT_MODEL

PROMPT_FILE = "prompts/merged_scoring_v1.txt"


def format_candidate_for_scoring(cand: Dict[str, Any], beats_map: Dict[str, Dict]) -> str:
    """Formats a single candidate's metadata and multi-event screenplay excerpt."""
    cid = cand["candidate_id"]
    s_b = cand["start_beat_id"]
    e_b = cand["end_beat_id"]
    dur = cand["duration"]
    hook_d = cand.get("hook_signal", {})
    cliff_d = cand.get("cliffhanger_signal", {})

    lines = [
        f"### Candidate {cid} ({s_b} -> {e_b}, {dur:.1f}s)",
        f"- Hook Signal: Dialogue {hook_d.get('dialogue_hook_score', 0)}/5, Visual {hook_d.get('visual_hook_score', 0)}/5",
        f"  Hook Rationale: {hook_d.get('dialogue_hook_rationale') or hook_d.get('visual_hook_rationale')}",
        f"- Cliffhanger Signal: Curiosity Gap {cliff_d.get('curiosity_gap_score', 0)}/5, Type: {cliff_d.get('cliffhanger_type', 'none')}",
        f"  Specific Question: {cliff_d.get('specific_question', 'None')}",
        "",
        "Screenplay Excerpt:",
    ]

    beat_ids = cand.get("beat_ids", [])
    for b_id in beat_ids:
        b = beats_map.get(b_id)
        if not b:
            continue
        lines.append(f"--- Beat {b_id} ({b['start']:.1f}s - {b['end']:.1f}s) ---")
        for ev in b.get("events", []):
            ev_type = ev.get("type", "").lower()
            if ev_type == "camera":
                lines.append(f"[CAMERA]: {ev.get('description', '')}")
            elif ev_type == "action":
                lines.append(f"[ACTION]: {ev.get('description', '')}")
            elif ev_type == "dialogue":
                char = ev.get("character", "SPEAKER")
                lines.append(f"[{char}]: \"{ev.get('description', '')}\"")
            elif ev_type in ("sound", "sfx"):
                lines.append(f"[SOUND]: {ev.get('description', '')}")
        lines.append("")

    return "\n".join(lines)


def build_batch_scoring_prompt(candidate_batch: List[Dict[str, Any]], beats_map: Dict[str, Dict]) -> str:
    """Builds prompt containing 4-5 candidates for merged scoring."""
    prompt_path = ROOT_DIR / PROMPT_FILE
    with open(prompt_path, "r", encoding="utf-8") as f:
        template = f.read()

    formatted_cands = []
    for cand in candidate_batch:
        formatted_cands.append(format_candidate_for_scoring(cand, beats_map))

    cands_text = "\n\n" + ("=" * 50) + "\n\n".join(formatted_cands)
    return template + "\n\n## CANDIDATES TO EVALUATE IN THIS BATCH:\n" + cands_text


def score_candidate_batch(
    candidate_batch: List[Dict[str, Any]],
    beats_map: Dict[str, Dict],
    episode_number: int = 1,
    model: str = DEFAULT_MODEL,
) -> List[Dict[str, Any]]:
    """Evaluates a batch of candidates in a single LLM call."""
    prompt = build_batch_scoring_prompt(candidate_batch, beats_map)
    c_ids = [c["candidate_id"] for c in candidate_batch]
    all_b_ids = []
    for c in candidate_batch:
        all_b_ids.extend(c.get("beat_ids", []))

    res = call_llm(
        prompt=prompt,
        prompt_file=PROMPT_FILE,
        stage="stage3_merged_scoring",
        episode_number=episode_number,
        candidate_ids=c_ids,
        beat_ids=list(set(all_b_ids)),
        model=model,
    )

    if isinstance(res, dict):
        if "raw_response" in res:
            parsed = extract_json_payload(res["raw_response"])
        else:
            parsed = res
        if isinstance(parsed, list):
            scored_list = parsed
        elif isinstance(parsed, dict):
            for key in ["scored_candidates", "candidates", "results", "scores", "data"]:
                if key in parsed and isinstance(parsed[key], list):
                    scored_list = parsed[key]
                    break
            else:
                scored_list = list(parsed.values()) if all(isinstance(v, dict) for v in parsed.values()) else []
        else:
            scored_list = []
    elif isinstance(res, list):
        scored_list = res
    else:
        raise ValueError(f"Unexpected response format from merged scorer: {type(res)}")

    # Map by candidate_id
    scored_map = {item.get("candidate_id"): item for item in scored_list if isinstance(item, dict)}

    # Ensure all candidates in batch have results, backfilling if necessary
    results = []
    for cand in candidate_batch:
        cid = cand["candidate_id"]
        if cid in scored_map:
            item = scored_map[cid]
            open_leg = bool(item.get("opening_legibility", False))
            arc_leg = bool(item.get("arc_legibility", False))
            passes_gate = bool(open_leg and arc_leg and item.get("passes_comprehension_gate", True))
            results.append({
                "candidate_id": cid,
                "start_beat_id": cand["start_beat_id"],
                "end_beat_id": cand["end_beat_id"],
                "start_time": cand["start_time"],
                "end_time": cand["end_time"],
                "duration": cand["duration"],
                "timestamp_source": cand.get("timestamp_source", "given"),
                "comprehension": {
                    "opening_legibility": open_leg,
                    "opening_rationale": str(item.get("opening_rationale", "")),
                    "arc_legibility": arc_leg,
                    "arc_rationale": str(item.get("arc_rationale", "")),
                    "passes_gate": passes_gate,
                    "situational_summary": str(item.get("situational_summary", "")),
                },
                "emotional_salience": {
                    "emotional_salience_score": int(item.get("emotional_salience_score", 3)),
                    "emotional_core": str(item.get("emotional_core", "")),
                },
                "spoiler": {
                    "spoiler_level": int(item.get("spoiler_level", 1)),
                    "spoiler_rationale": str(item.get("spoiler_rationale", "")),
                },
            })
        else:
            # Fallback for missing candidate in batch
            results.append({
                "candidate_id": cid,
                "start_beat_id": cand["start_beat_id"],
                "end_beat_id": cand["end_beat_id"],
                "start_time": cand["start_time"],
                "end_time": cand["end_time"],
                "duration": cand["duration"],
                "timestamp_source": cand.get("timestamp_source", "given"),
                "comprehension": {
                    "opening_legibility": False,
                    "opening_rationale": "Missing from batch response",
                    "arc_legibility": False,
                    "arc_rationale": "Missing from batch response",
                    "passes_gate": False,
                    "situational_summary": "Scoring failed",
                },
                "emotional_salience": {
                    "emotional_salience_score": 1,
                    "emotional_core": "None",
                },
                "spoiler": {
                    "spoiler_level": 0,
                    "spoiler_rationale": "None",
                },
            })

    return results


def score_all_candidates_concurrent(
    candidates: List[Dict[str, Any]],
    beats_map: Dict[str, Dict],
    episode_number: int = 1,
    batch_size: int = 4,
    max_workers: int = 4,
    model: str = DEFAULT_MODEL,
) -> List[Dict[str, Any]]:
    """
    Batches candidates into groups of 4-5 and scores them concurrently.
    Total calls for ~30 candidates: 6-8 calls.
    At 4 workers, completes in ~15-20 seconds.
    """
    if not candidates:
        print("Stage 3 scoring: 0 candidates to score.")
        return []

    batches = [candidates[i:i + batch_size] for i in range(0, len(candidates), batch_size)]
    print(f"Divided {len(candidates)} candidates into {len(batches)} batches of ~{batch_size} (concurrency: {max_workers})...")

    all_scored = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(score_candidate_batch, batch, beats_map, episode_number, model): (i, batch)
            for i, batch in enumerate(batches, 1)
        }
        for future in as_completed(future_map):
            b_idx, batch = future_map[future]
            try:
                batch_res = future.result()
                all_scored.extend(batch_res)
                print(f"  Batch {b_idx}/{len(batches)} scored ({len(batch_res)} candidates).")
            except Exception as e:
                print(f"  Error scoring batch {b_idx}: {e}. Creating fallback unpassed candidate records.")
                for cand in batch:
                    all_scored.append({
                        "candidate_id": cand["candidate_id"],
                        "start_beat_id": cand["start_beat_id"],
                        "end_beat_id": cand["end_beat_id"],
                        "start_time": cand.get("start_time", 0.0),
                        "end_time": cand.get("end_time", 0.0),
                        "duration": cand.get("duration", 0.0),
                        "timestamp_source": cand.get("timestamp_source", "given"),
                        "comprehension": {
                            "opening_legibility": False,
                            "opening_rationale": f"Batch scoring failed: {e}",
                            "arc_legibility": False,
                            "arc_rationale": f"Batch scoring failed: {e}",
                            "passes_gate": False,
                            "situational_summary": "Scoring failed",
                        },
                        "emotional_salience": {
                            "emotional_salience_score": 1,
                            "emotional_core": "Unknown",
                        },
                        "spoiler": {
                            "spoiler_level": 0,
                            "spoiler_rationale": "Unknown",
                        },
                    })

    all_scored.sort(key=lambda x: (x["start_beat_id"], x["end_beat_id"]))
    elapsed = time.time() - t0
    passes = sum(1 for c in all_scored if c["comprehension"]["passes_gate"])
    pct = (passes / len(all_scored) * 100.0) if all_scored else 0.0
    print(f"Stage 3 scoring completed in {elapsed:.2f}s. Passed gate: {passes}/{len(all_scored)} ({pct:.1f}%)")
    return all_scored

