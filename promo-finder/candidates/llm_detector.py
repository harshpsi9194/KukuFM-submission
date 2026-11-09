"""
llm_detector.py — Real LLM-based Hook and Cliffhanger Detection (Stage 2).

Evaluates parsed screenplay beats for:
1. Hook Signals (dialogue punch, visual scroll-stop potential) via prompts/hook_detector_v1.txt
2. Cliffhanger Signals (curiosity gap, specific question, over-reveal check) via prompts/cliffhanger_detector_v1.txt

Per AGENTS.md Rule 1: Never let one LLM call do generation + scoring + ranking.
Per AGENTS.md Rule 2: Never let an LLM invent a timestamp.
Per AGENTS.md Rule 5: Events within a beat can overlap in time (CAMERA/ACTION/DIALOGUE/SOUND).

Logs all successful detection calls to outputs/llm_detector_calls.jsonl and failures to outputs/llm_detector_errors.jsonl.
"""

import datetime
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from parser.schema import Episode, Beat
from candidates.schema import HookSignal, CliffhangerSignal
from candidates.llm_client import call_llm, extract_json_payload, DEFAULT_MODEL, estimate_tokens


def estimate_tokens(text: str) -> int:
    """Estimates token count from text using standard 1.33 tokens/word heuristic."""
    words = len(text.split())
    chars = len(text)
    return max(1, int(words * 1.33)) if words > 0 else max(1, chars // 4)


def format_episode_beats(episode: Episode, beats: Optional[List[Beat]] = None) -> str:
    """
    Formats episode beats preserving simultaneous CAMERA, ACTION, DIALOGUE, and SOUND events.
    Does NOT flatten them sequentially; each beat is formatted as a distinct multi-event container.
    """
    beats_to_format = beats if beats is not None else episode.beats
    lines = []
    for b in beats_to_format:
        lines.append(f"=== Beat {b.beat_id} ({b.start:.1f}s - {b.end:.1f}s) ===")
        for ev in b.events:
            ev_type = ev.type.lower()
            if ev_type == "camera":
                lines.append(f"[CAMERA]: {ev.description}")
            elif ev_type == "action":
                lines.append(f"[ACTION]: {ev.description}")
            elif ev_type == "dialogue":
                char = ev.character or "SPEAKER"
                lines.append(f"[{char}]: \"{ev.description}\"")
            elif ev_type in ("sound", "sfx"):
                lines.append(f"[SOUND]: {ev.description}")
        lines.append("")
    return "\n".join(lines)


def chunk_beats(beats: List[Beat], chunk_size: int = 40, overlap: int = 3) -> List[List[Beat]]:
    """Splits beats into overlapping chunks if an episode exceeds chunk_size."""
    if len(beats) <= chunk_size:
        return [beats]
    chunks = []
    start = 0
    while start < len(beats):
        end = min(start + chunk_size, len(beats))
        chunks.append(beats[start:end])
        if end >= len(beats):
            break
        start += (chunk_size - overlap)
    return chunks


def log_detector_call(
    log_path: Path,
    entry: Dict[str, Any]
):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def detect_hook_signals(
    episode: Episode,
    model: str = DEFAULT_MODEL,
) -> Dict[str, HookSignal]:
    """
    Evaluates all beats in an episode for hook signals using prompts/hook_detector_v1.txt.
    Supports chunked batched calls if episode exceeds 40 beats.
    Returns a mapping of beat_id -> HookSignal.
    """
    prompt_file_rel = "prompts/hook_detector_v1.txt"
    prompt_path = ROOT_DIR / prompt_file_rel
    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt_template = f.read()

    chunks = chunk_beats(episode.beats, chunk_size=40, overlap=3)
    all_hook_items = []

    for c_idx, beat_chunk in enumerate(chunks, 1):
        beats_text = format_episode_beats(episode, beat_chunk)
        full_prompt = prompt_template + "\n\n## SCREENPLAY BEATS TO EVALUATE:\n\n" + beats_text
        beat_ids = [b.beat_id for b in beat_chunk]
        cid = f"EPISODE_{episode.episode_number:03d}_CHUNK_{c_idx}_HOOKS"

        res = call_llm(
            prompt=full_prompt,
            prompt_file=prompt_file_rel,
            stage="stage1_hook_detection",
            episode_number=episode.episode_number,
            candidate_ids=[cid],
            beat_ids=beat_ids,
            model=model,
        )

        if isinstance(res, dict) and "raw_response" in res:
            chunk_list = extract_json_payload(res["raw_response"])
        elif isinstance(res, list):
            chunk_list = res
        else:
            raise ValueError(f"Unexpected Hook Detector response type: {type(res)}")
        all_hook_items.extend(chunk_list)

    hook_dict: Dict[str, HookSignal] = {}
    for item in all_hook_items:
        b_id = str(item.get("beat_id", "")).strip()
        if not b_id:
            continue
        d_score = int(item.get("dialogue_hook_score", 0))
        d_rat = str(item.get("dialogue_hook_rationale", ""))
        v_score = int(item.get("visual_hook_score", 0))
        v_rat = str(item.get("visual_hook_rationale", ""))
        is_strong = bool(d_score >= 4 or v_score >= 4)

        if b_id in hook_dict:
            # If overlapping chunk, keep highest scoring
            existing = hook_dict[b_id]
            if max(d_score, v_score) > max(existing.dialogue_hook_score, existing.visual_hook_score):
                hook_dict[b_id] = HookSignal(
                    beat_id=b_id,
                    dialogue_hook_score=d_score,
                    dialogue_hook_rationale=d_rat,
                    visual_hook_score=v_score,
                    visual_hook_rationale=v_rat,
                    is_strong_hook=is_strong,
                )
        else:
            hook_dict[b_id] = HookSignal(
                beat_id=b_id,
                dialogue_hook_score=d_score,
                dialogue_hook_rationale=d_rat,
                visual_hook_score=v_score,
                visual_hook_rationale=v_rat,
                is_strong_hook=is_strong,
            )

    # Backfill any beats missing from output
    for b in episode.beats:
        if b.beat_id not in hook_dict:
            hook_dict[b.beat_id] = HookSignal(
                beat_id=b.beat_id,
                dialogue_hook_score=0,
                dialogue_hook_rationale="Not scored by LLM (fallback)",
                visual_hook_score=0,
                visual_hook_rationale="Not scored by LLM (fallback)",
                is_strong_hook=False,
            )

    return hook_dict


def detect_cliffhanger_signals(
    episode: Episode,
    model: str = DEFAULT_MODEL,
) -> Dict[str, CliffhangerSignal]:
    """
    Evaluates all beats in an episode for cliffhanger signals using prompts/cliffhanger_detector_v1.txt.
    Supports chunked batched calls if episode exceeds 40 beats.
    Returns a mapping of beat_id -> CliffhangerSignal.
    """
    prompt_file_rel = "prompts/cliffhanger_detector_v1.txt"
    prompt_path = ROOT_DIR / prompt_file_rel
    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt_template = f.read()

    chunks = chunk_beats(episode.beats, chunk_size=40, overlap=3)
    all_cliff_items = []

    for c_idx, beat_chunk in enumerate(chunks, 1):
        beats_text = format_episode_beats(episode, beat_chunk)
        full_prompt = prompt_template + "\n\n## SCREENPLAY BEATS TO EVALUATE:\n\n" + beats_text
        beat_ids = [b.beat_id for b in beat_chunk]
        cid = f"EPISODE_{episode.episode_number:03d}_CHUNK_{c_idx}_CLIFFHANGERS"

        res = call_llm(
            prompt=full_prompt,
            prompt_file=prompt_file_rel,
            stage="stage1_cliffhanger_detection",
            episode_number=episode.episode_number,
            candidate_ids=[cid],
            beat_ids=beat_ids,
            model=model,
        )

        if isinstance(res, dict) and "raw_response" in res:
            chunk_list = extract_json_payload(res["raw_response"])
        elif isinstance(res, list):
            chunk_list = res
        else:
            raise ValueError(f"Unexpected Cliffhanger Detector response type: {type(res)}")
        all_cliff_items.extend(chunk_list)

    cliff_dict: Dict[str, CliffhangerSignal] = {}
    for item in all_cliff_items:
        b_id = str(item.get("beat_id", "")).strip()
        if not b_id:
            continue
        gap_score = int(item.get("curiosity_gap_score", 0))
        question = str(item.get("specific_question", ""))
        stakes = str(item.get("stakes_importance", "Medium"))
        defuses = bool(item.get("defuses_tension", False))
        c_type = str(item.get("cliffhanger_type", "none"))
        is_strong = bool(gap_score >= 4 and not defuses)

        if b_id in cliff_dict:
            existing = cliff_dict[b_id]
            if gap_score > existing.curiosity_gap_score:
                cliff_dict[b_id] = CliffhangerSignal(
                    beat_id=b_id,
                    curiosity_gap_score=gap_score,
                    specific_question=question,
                    stakes_importance=stakes,
                    defuses_tension=defuses,
                    cliffhanger_type=c_type,
                    is_strong_cliffhanger=is_strong,
                )
        else:
            cliff_dict[b_id] = CliffhangerSignal(
                beat_id=b_id,
                curiosity_gap_score=gap_score,
                specific_question=question,
                stakes_importance=stakes,
                defuses_tension=defuses,
                cliffhanger_type=c_type,
                is_strong_cliffhanger=is_strong,
            )

    # Backfill any beats missing from output
    for b in episode.beats:
        if b.beat_id not in cliff_dict:
            cliff_dict[b.beat_id] = CliffhangerSignal(
                beat_id=b.beat_id,
                curiosity_gap_score=0,
                specific_question="None",
                stakes_importance="Low",
                defuses_tension=False,
                cliffhanger_type="none",
                is_strong_cliffhanger=False,
            )

    return cliff_dict


import concurrent.futures

def detect_episode_signals(
    episode: Episode,
    model: str = DEFAULT_MODEL,
) -> Tuple[Dict[str, HookSignal], Dict[str, CliffhangerSignal]]:
    """
    Runs both Hook Detector and Cliffhanger Detector models on an episode concurrently.
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        f_hooks = executor.submit(detect_hook_signals, episode, model)
        f_cliffs = executor.submit(detect_cliffhanger_signals, episode, model)
        hooks = f_hooks.result()
        cliffs = f_cliffs.result()
    return hooks, cliffs

