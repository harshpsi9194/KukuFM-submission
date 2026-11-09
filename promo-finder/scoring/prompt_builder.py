"""
prompt_builder.py — Formats candidate screenplay text (all beats and simultaneous events)
and constructs versioned prompts for Stage 3 scoring.
Separates Opening Beat (Section A) from Full Screenplay Arc (Section B) for the comprehension gate.
"""

import json
import os
from typing import Any

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_episode_beats(golden_path: str = None) -> dict[str, dict]:
    if golden_path is None:
        golden_path = os.path.join(BASE_DIR, "parser", "fixtures", "episode_001_golden.json")
    with open(golden_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {b["beat_id"]: b for b in data["beats"]}


def format_single_beat_text(beat_id: str, episode_beats: dict[str, dict]) -> str:
    """
    Renders a single beat's events preserving simultaneous CAMERA, ACTION, DIALOGUE, SFX events.
    """
    b = episode_beats.get(beat_id)
    if not b:
        return f"[Beat {beat_id} not found in episode script]"
    lines = [f"--- [Beat {beat_id}] ({b['start']:.1f}s - {b['end']:.1f}s) ---"]
    for ev in b.get("events", []):
        ev_type = ev.get("type", "EVENT").upper()
        desc = ev.get("description", "")
        lines.append(f"  [{ev_type}] {desc}")
    return "\n".join(lines)


def format_candidate_screenplay(candidate: dict[str, Any], episode_beats: dict[str, dict]) -> str:
    """
    Renders the candidate's complete screenplay excerpt across all beats in beat_ids.
    Preserves all simultaneous events (CAMERA, ACTION, DIALOGUE, SFX) per AGENTS.md Rule 5.
    """
    lines = []
    lines.append(f"Candidate ID: {candidate['candidate_id']}")
    lines.append(f"Beat Interval: {candidate['start_beat_id']} -> {candidate['end_beat_id']}")
    lines.append(f"Time Range: {candidate['start_time']:.1f}s - {candidate['end_time']:.1f}s (Duration: {candidate['duration']:.1f}s)")
    lines.append("\nFull Candidate Screenplay Text (Every Beat and Event in Interval):")

    for b_id in candidate["beat_ids"]:
        lines.append("\n" + format_single_beat_text(b_id, episode_beats))

    return "\n".join(lines)


def build_comprehension_prompt(candidate: dict[str, Any], episode_beats: dict[str, dict]) -> str:
    """
    Builds the two-part comprehension gate prompt:
    - Section A: Opening Beat in Strict Isolation (ONLY beat_ids[0])
    - Section B: Full Candidate Screenplay Arc (all beat_ids)
    """
    prompt_file = os.path.join(BASE_DIR, "prompts", "comprehension_gate_v1.txt")
    with open(prompt_file, "r", encoding="utf-8") as f:
        template = f.read()

    opening_beat_id = candidate["beat_ids"][0]
    opening_text = format_single_beat_text(opening_beat_id, episode_beats)
    full_arc_text = format_candidate_screenplay(candidate, episode_beats)

    candidate_block = []
    candidate_block.append(f"Candidate ID: {candidate['candidate_id']}")
    candidate_block.append(f"Beat Interval: {candidate['start_beat_id']} -> {candidate['end_beat_id']}")
    candidate_block.append(f"Duration: {candidate['duration']:.1f}s")
    candidate_block.append("\n" + "="*60)
    candidate_block.append("### Section A: Opening Beat in Strict Isolation (ONLY Beat " + opening_beat_id + ")")
    candidate_block.append("Evaluate 'opening_legibility' using ONLY the following beat. FORBIDDEN: Do NOT use Section B to justify this opening.")
    candidate_block.append("="*60)
    candidate_block.append(opening_text)
    candidate_block.append("\n" + "="*60)
    candidate_block.append("### Section B: Full Candidate Screenplay Arc (Beats " + candidate['start_beat_id'] + " -> " + candidate['end_beat_id'] + ")")
    candidate_block.append("Evaluate 'arc_legibility' using the full narrative progression below.")
    candidate_block.append("="*60)
    candidate_block.append(full_arc_text)

    script_text = "\n".join(candidate_block)
    return (
        f"{template}\n\n"
        f"## Candidate Under Evaluation:\n\n{script_text}\n\n"
        f"Evaluate candidate '{candidate['candidate_id']}'. Return STRICT valid JSON matching the Output Schema."
    )


def build_emotional_prompt(candidate: dict[str, Any], episode_beats: dict[str, dict]) -> str:
    prompt_file = os.path.join(BASE_DIR, "prompts", "emotional_salience_v1.txt")
    with open(prompt_file, "r", encoding="utf-8") as f:
        template = f.read()
    script_text = format_candidate_screenplay(candidate, episode_beats)
    return f"{template}\n\n## Candidate Under Evaluation:\n\n{script_text}\n\nEvaluate candidate '{candidate['candidate_id']}'. Return STRICT valid JSON matching the Output Schema."


def build_spoiler_prompt(candidate: dict[str, Any], episode_beats: dict[str, dict]) -> str:
    prompt_file = os.path.join(BASE_DIR, "prompts", "spoiler_classifier_v1.txt")
    with open(prompt_file, "r", encoding="utf-8") as f:
        template = f.read()
    script_text = format_candidate_screenplay(candidate, episode_beats)
    return f"{template}\n\n## Candidate Under Evaluation:\n\n{script_text}\n\nEvaluate candidate '{candidate['candidate_id']}'. Return STRICT valid JSON matching the Output Schema."
