"""
window_search.py — Bidirectional window search for promo candidates (Stage 2).

Implements Rule 2, Rule 3, Rule 4 of AGENTS.md:
- Candidate boundaries are beat IDs, not invented timestamps.
- Clip boundaries snap strictly to beat edges (start = start_beat.start, end = end_beat.end).
- Pure Python deterministic timestamp resolution.
- Duration (20–60s) is a downstream Python filter, NOT an LLM prompt tradeoff.
- Capped candidate explosion: keeps single best-scoring cliffhanger per strong hook,
  and single best-scoring hook per strong cliffhanger.
- Floor-relaxation fallback: if strict floor (<5 candidates) underflows, relaxes by 1 point.
"""

from typing import Optional, List, Dict, Set, Tuple
try:
    from ..parser.schema import Episode, Beat
    from .schema import HookSignal, CliffhangerSignal, PromoCandidate, Segment
except (ImportError, ValueError):
    from parser.schema import Episode, Beat
    from candidates.schema import HookSignal, CliffhangerSignal, PromoCandidate, Segment

MIN_PROMO_DURATION = 20.0
MAX_PROMO_DURATION = 60.0


def _search_pairs(
    beat_list: List[Beat],
    beat_index_map: Dict[str, int],
    hooks: List[HookSignal],
    cliffhangers: List[CliffhangerSignal],
    hook_signals: Dict[str, HookSignal],
    cliffhanger_signals: Dict[str, CliffhangerSignal],
    min_duration: float,
    max_duration: float,
) -> Set[Tuple[str, str]]:
    candidate_pairs: Set[Tuple[str, str]] = set()
    target_cliff_ids = {c.beat_id for c in cliffhangers}
    target_hook_ids = {h.beat_id for h in hooks}

    # 1. Forward Search: For each hook, keep only its single best-scoring valid cliffhanger match
    for hook in hooks:
        start_idx = beat_index_map[hook.beat_id]
        start_beat = beat_list[start_idx]
        valid_cliff_matches = []

        for end_idx in range(start_idx, len(beat_list)):
            end_beat = beat_list[end_idx]

            # Scene boundary check: stop expansion immediately
            if start_beat.scene_number != end_beat.scene_number:
                break

            duration = end_beat.end - start_beat.start
            if duration > max_duration:
                break

            if duration >= min_duration and end_beat.beat_id in target_cliff_ids:
                cliff = cliffhanger_signals[end_beat.beat_id]
                dur_closeness = -abs(duration - 35.0)
                valid_cliff_matches.append((cliff.curiosity_gap_score, dur_closeness, end_beat.beat_id))

        if valid_cliff_matches:
            valid_cliff_matches.sort(key=lambda x: (x[0], x[1]), reverse=True)
            best_end_beat_id = valid_cliff_matches[0][2]
            candidate_pairs.add((start_beat.beat_id, best_end_beat_id))

    # 2. Backward Search: For each cliffhanger, keep only its single best-scoring valid hook match
    for cliff in cliffhangers:
        end_idx = beat_index_map[cliff.beat_id]
        end_beat = beat_list[end_idx]
        valid_hook_matches = []

        for start_idx in range(end_idx, -1, -1):
            start_beat = beat_list[start_idx]

            if start_beat.scene_number != end_beat.scene_number:
                break

            duration = end_beat.end - start_beat.start
            if duration > max_duration:
                break

            if duration >= min_duration and start_beat.beat_id in target_hook_ids:
                hook = hook_signals[start_beat.beat_id]
                max_hook_score = max(hook.dialogue_hook_score, hook.visual_hook_score)
                dur_closeness = -abs(duration - 35.0)
                valid_hook_matches.append((max_hook_score, dur_closeness, start_beat.beat_id))

        if valid_hook_matches:
            valid_hook_matches.sort(key=lambda x: (x[0], x[1]), reverse=True)
            best_start_beat_id = valid_hook_matches[0][2]
            candidate_pairs.add((best_start_beat_id, end_beat.beat_id))

    return candidate_pairs


def bidirectional_window_search(
    episode: Episode,
    hook_signals: Dict[str, HookSignal],
    cliffhanger_signals: Dict[str, CliffhangerSignal],
    min_duration: float = MIN_PROMO_DURATION,
    max_duration: float = MAX_PROMO_DURATION,
) -> List[PromoCandidate]:
    """
    Perform capped bidirectional window search:
    1. Forward: single best cliffhanger per strong hook.
    2. Backward: single best hook per strong cliffhanger.
    3. Floor-relaxation fallback: if strict floor produces <5 candidates, relaxes by 1 point.
    4. Deterministic Python filter strictly by min_duration and max_duration.
    """
    beat_map: Dict[str, Beat] = {b.beat_id: b for b in episode.beats}
    beat_list: List[Beat] = episode.beats
    beat_index_map: Dict[str, int] = {b.beat_id: i for i, b in enumerate(beat_list)}

    strong_hooks = [h for h in hook_signals.values() if h.is_strong_hook and h.beat_id in beat_index_map]
    strong_cliffhangers = [c for c in cliffhanger_signals.values() if c.is_strong_cliffhanger and c.beat_id in beat_index_map]

    candidate_pairs = _search_pairs(
        beat_list, beat_index_map, strong_hooks, strong_cliffhangers,
        hook_signals, cliffhanger_signals, min_duration, max_duration
    )

    # Floor-relaxation fallback: if strict floor produces < 5 candidates, relax by 1 point
    if len(candidate_pairs) < 5:
        print(f"  [Floor-Relaxation Triggered] Strict floor yielded {len(candidate_pairs)} pairs; relaxing floor by 1 point...")
        relaxed_hooks = [
            h for h in hook_signals.values()
            if (max(h.dialogue_hook_score, h.visual_hook_score) >= 3 or h.is_strong_hook)
            and h.beat_id in beat_index_map
        ]
        relaxed_cliffs = [
            c for c in cliffhanger_signals.values()
            if (c.curiosity_gap_score >= 3 or c.is_strong_cliffhanger) and not c.defuses_tension
            and c.beat_id in beat_index_map
        ]
        relaxed_pairs = _search_pairs(
            beat_list, beat_index_map, relaxed_hooks, relaxed_cliffs,
            hook_signals, cliffhanger_signals, min_duration, max_duration
        )
        candidate_pairs.update(relaxed_pairs)

    # Construct and filter PromoCandidates in Python
    candidates: List[PromoCandidate] = []
    cand_idx = 1

    sorted_pairs = sorted(
        candidate_pairs,
        key=lambda pair: (beat_index_map[pair[0]], beat_index_map[pair[1]])
    )

    for start_id, end_id in sorted_pairs:
        s_beat = beat_map[start_id]
        e_beat = beat_map[end_id]

        start_time = round(s_beat.start, 2)
        end_time = round(e_beat.end, 2)
        duration = round(end_time - start_time, 2)

        # Rule 4: Duration filter applied strictly in Python
        if min_duration <= duration <= max_duration:
            s_idx = beat_index_map[start_id]
            e_idx = beat_index_map[end_id]
            included_beats = [b.beat_id for b in beat_list[s_idx:e_idx + 1]]

            is_estimated = any(b.timestamp_source == "estimated" for b in beat_list[s_idx:e_idx + 1])
            ts_source = "estimated" if is_estimated else "given"

            cand_id = f"CAND_EP{episode.episode_number:03d}_{cand_idx:02d}"
            cand_idx += 1

            segment = Segment(
                segment_index=1,
                episode_number=episode.episode_number,
                start_beat_id=start_id,
                end_beat_id=end_id,
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                beat_ids=included_beats,
                timestamp_source=ts_source
            )

            candidates.append(PromoCandidate(
                candidate_id=cand_id,
                episode=episode.episode,
                episode_number=episode.episode_number,
                segments=[segment],
                hook_signal=hook_signals[start_id],
                cliffhanger_signal=cliffhanger_signals[end_id]
            ))

    return candidates
