"""
batched_ranker.py — Stage 4 Pure Absolute Rubric Ranking.

Replaces full O(n^2) round-robin tournament with 1 batched absolute rubric evaluation:
- Evaluates all comprehension-gate survivors in 1 call against a 4-dimension 10.0-point rubric.
- Four explicit sub-dimensions (2.5 pts each):
  1. Hook Scroll-Stop Power (0.0 - 2.5)
  2. Mid-Arc Escalation & Conflict Density (0.0 - 2.5)
  3. Cliffhanger Irresolution & Curiosity Gap (0.0 - 2.5)
  4. Standalone Viewer Payoff & Emotional Resonance (0.0 - 2.5)
- Strictly ABSOLUTE scoring: no pairwise comparisons, no round-robin matches.
- Pure Python deterministic tie-breaking:
  1. Primary: absolute_score (descending)
  2. Secondary: Duration closeness to 35.0s vertical short-form sweet spot (-abs(duration - 35.0))
  3. Tertiary: Stage 3 emotional_salience_score (descending)
  4. Quaternary: Hook score (descending)
- Total calls per episode for Stage 4: Exactly 1 call.
"""

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from candidates.llm_client import call_llm, extract_json_payload, DEFAULT_MODEL

RANKING_PROMPT_FILE = "prompts/batched_ranking_v1.txt"


def format_candidate_beat_progression(cand: Dict[str, Any], beats_map: Dict[str, Dict]) -> List[str]:
    """Extracts concise dramatic script progression for cold performance evaluation."""
    s_b = cand["start_beat_id"]
    e_b = cand["end_beat_id"]
    beat_ids = cand.get("beat_ids", [])
    if not beat_ids and s_b in beats_map and e_b in beats_map:
        beat_ids = [s_b, e_b]

    lines = []
    # Opening beat lines
    s_beat = beats_map.get(s_b, {})
    for ev in s_beat.get("events", []):
        if ev.get("type") == "dialogue":
            lines.append(f'  • Start [{s_b}] {ev.get("character", "SPEAKER")}: "{ev.get("description", "")}"')
        elif ev.get("type") in ("camera", "action") and len(lines) < 2:
            lines.append(f'  • Start [{s_b}] {ev.get("type").upper()}: {ev.get("description", "")[:90]}')

    # Middle arc dialogue highlights (up to 3)
    if len(beat_ids) > 2:
        mid_dialogues = []
        for b_id in beat_ids[1:-1]:
            b = beats_map.get(b_id, {})
            for ev in b.get("events", []):
                if ev.get("type") == "dialogue":
                    mid_dialogues.append(f'  • Mid [{b_id}] {ev.get("character", "SPEAKER")}: "{ev.get("description", "")}"')
                    break
            if len(mid_dialogues) >= 3:
                break
        lines.extend(mid_dialogues)

    # Cliffhanger exit lines
    e_beat = beats_map.get(e_b, {})
    for ev in e_beat.get("events", []):
        if ev.get("type") == "dialogue":
            lines.append(f'  • Exit [{e_b}] {ev.get("character", "SPEAKER")}: "{ev.get("description", "")}"')
        elif ev.get("type") in ("camera", "action"):
            lines.append(f'  • Exit [{e_b}] {ev.get("type").upper()}: {ev.get("description", "")[:90]}')

    return lines


def build_batched_ranking_prompt(
    candidates: List[Dict[str, Any]],
    scored_map: Dict[str, Dict[str, Any]],
    beats_map: Dict[str, Dict],
) -> str:
    """Formats all candidates into a single batched absolute rubric ranking prompt with dramatic script context."""
    prompt_path = ROOT_DIR / RANKING_PROMPT_FILE
    with open(prompt_path, "r", encoding="utf-8") as f:
        template = f.read()

    lines = []
    for cand in candidates:
        cid = cand["candidate_id"]
        sc = scored_map.get(cid, {})
        s_b = cand["start_beat_id"]
        e_b = cand["end_beat_id"]
        dur = cand["duration"]
        comp = sc.get("comprehension", {})
        emot = sc.get("emotional_salience", {})
        spoil = sc.get("spoiler", {})

        hook_d = cand.get("hook_signal", {})
        cliff_d = cand.get("cliffhanger_signal", {})

        lines.append(f"### Candidate {cid} ({s_b} -> {e_b}, {dur:.1f}s)")
        lines.append(f"- Situational Summary: {comp.get('situational_summary', 'N/A')}")
        lines.append(f"- Emotional Core: {emot.get('emotional_core', '')} (Stage 3 Salience: {emot.get('emotional_salience_score', 3)}/5)")
        lines.append(f"- Hook Signal ({s_b}): {hook_d.get('dialogue_hook_rationale') or hook_d.get('visual_hook_rationale')}")
        lines.append(f"- Cliffhanger Signal ({e_b}): {cliff_d.get('specific_question')} (Type: {cliff_d.get('cliffhanger_type', 'unresolved')})")
        lines.append(f"- Spoiler Level: Tier {spoil.get('spoiler_level', 1)} ({spoil.get('spoiler_rationale', '')})")
        
        # Dramatic script flow
        progression = format_candidate_beat_progression(cand, beats_map)
        if progression:
            lines.append("- Dramatic Script Highlights:")
            lines.extend(progression)
        lines.append("")

    cands_text = "\n".join(lines)
    return template + "\n\n## CANDIDATES TO RANK:\n\n" + cands_text


def run_batched_ranking(
    passing_candidates: List[Dict[str, Any]],
    scored_map: Dict[str, Dict[str, Any]],
    beats_map: Dict[str, Dict],
    episode_number: int = 1,
    model: str = DEFAULT_MODEL,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Ranks candidates in 1 single batched absolute rubric call.
    Returns:
        (ranked_candidates, []) — tiebreak list is empty because ranking is purely absolute.
    """
    if not passing_candidates:
        return [], []

    prompt = build_batched_ranking_prompt(passing_candidates, scored_map, beats_map)
    c_ids = [c["candidate_id"] for c in passing_candidates]

    print(f"Executing Stage 4 absolute rubric ranking for {len(passing_candidates)} candidates in 1 call...")
    res = call_llm(
        prompt=prompt,
        prompt_file=RANKING_PROMPT_FILE,
        stage="stage4_absolute_ranking",
        episode_number=episode_number,
        candidate_ids=c_ids,
        model=model,
    )

    if isinstance(res, dict) and "raw_response" in res:
        ranked_list = extract_json_payload(res["raw_response"])
    elif isinstance(res, list):
        ranked_list = res
    else:
        raise ValueError(f"Unexpected ranking response: {type(res)}")

    score_dict: Dict[str, Dict[str, Any]] = {}
    for item in ranked_list:
        cid = item.get("candidate_id")
        if cid:
            h_score = float(item.get("hook_score", 1.5))
            t_score = float(item.get("tension_score", item.get("arc_score", 1.5)))
            c_score = float(item.get("cliffhanger_score", 1.5))
            r_score = float(item.get("resonance_score", item.get("pacing_score", 1.5)))
            # Calculate or use provided absolute_score
            abs_score = float(item.get("absolute_score", h_score + t_score + c_score + r_score))
            score_dict[cid] = {
                "hook_score": round(h_score, 2),
                "tension_score": round(t_score, 2),
                "cliffhanger_score": round(c_score, 2),
                "resonance_score": round(r_score, 2),
                "absolute_score": round(abs_score, 2),
                "rank_justification": str(item.get("rank_justification", "")),
            }

    # Enrich candidates with scores
    enriched = []
    for cand in passing_candidates:
        cid = cand["candidate_id"]
        sc = score_dict.get(cid, {
            "hook_score": 1.5,
            "tension_score": 1.5,
            "cliffhanger_score": 1.5,
            "resonance_score": 1.5,
            "absolute_score": 6.0,
            "rank_justification": "Default baseline score",
        })
        s3_emot = scored_map.get(cid, {}).get("emotional_salience", {}).get("emotional_salience_score", 3)

        enriched.append({
            "candidate_id": cid,
            "absolute_score": sc["absolute_score"],
            "subscores": {
                "hook_score": sc["hook_score"],
                "tension_score": sc["tension_score"],
                "cliffhanger_score": sc["cliffhanger_score"],
                "resonance_score": sc["resonance_score"],
            },
            "rank_justification": sc["rank_justification"],
            "duration": cand["duration"],
            "start_beat_id": cand["start_beat_id"],
            "end_beat_id": cand["end_beat_id"],
            "emotional_salience_score": s3_emot,
        })

    # Pure Python deterministic total ordering:
    # 1. absolute_score descending
    # 2. duration closeness to 35.0s sweet spot (-abs(duration - 35.0))
    # 3. emotional_salience_score descending
    # 4. hook_score descending
    enriched.sort(
        key=lambda x: (
            round(x["absolute_score"], 2),
            round(-abs(x["duration"] - 35.0), 2),
            x["emotional_salience_score"],
            round(x["subscores"]["hook_score"], 2),
        ),
        reverse=True,
    )

    # Assign sequential ranks
    for rank, item in enumerate(enriched, 1):
        item["rank"] = rank

    leader = enriched[0]
    print(f"Stage 4 absolute ranking complete! Leader: {leader['candidate_id']} (score: {leader['absolute_score']:.2f}/10.0)")
    return enriched, []
