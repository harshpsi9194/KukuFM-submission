"""
schema.py — Data structures for promo candidate generation (Stage 2).

Invariants:
- Clip boundaries snap strictly to beat edges (Rule 3).
- Candidate boundaries are beat IDs, not invented timestamps (Rule 2).
- Pure Python deterministic timestamp resolution.
- PromoCandidate holds `segments: list[Segment]` (1 for single-segment, >=2 for multi-segment).
- Computed properties (start_beat_id, end_beat_id, start_time, end_time, duration, beat_ids, timestamp_source)
  are derived from `segments` to maintain 100% backwards compatibility across tests and consumers.
"""

from dataclasses import dataclass
from typing import Optional, Literal

try:
    from ..parser.schema import TimestampSource
except (ImportError, ValueError):
    from parser.schema import TimestampSource


@dataclass
class HookSignal:
    """Evaluated hook signals for a candidate opening beat."""
    beat_id: str
    dialogue_hook_score: int         # 0–5
    dialogue_hook_rationale: str
    visual_hook_score: int           # 0–5 (Scroll-stop potential)
    visual_hook_rationale: str
    is_strong_hook: bool

    def to_dict(self) -> dict:
        return {
            "beat_id": self.beat_id,
            "dialogue_hook_score": self.dialogue_hook_score,
            "dialogue_hook_rationale": self.dialogue_hook_rationale,
            "visual_hook_score": self.visual_hook_score,
            "visual_hook_rationale": self.visual_hook_rationale,
            "is_strong_hook": self.is_strong_hook,
        }


@dataclass
class CliffhangerSignal:
    """Evaluated cliffhanger signals for a candidate closing beat."""
    beat_id: str
    curiosity_gap_score: int         # 0–5
    specific_question: str
    stakes_importance: str           # "High", "Medium", "Low"
    defuses_tension: bool            # Over-reveal check
    cliffhanger_type: str            # From taxonomy
    is_strong_cliffhanger: bool

    def to_dict(self) -> dict:
        return {
            "beat_id": self.beat_id,
            "curiosity_gap_score": self.curiosity_gap_score,
            "specific_question": self.specific_question,
            "stakes_importance": self.stakes_importance,
            "defuses_tension": self.defuses_tension,
            "cliffhanger_type": self.cliffhanger_type,
            "is_strong_cliffhanger": self.is_strong_cliffhanger,
        }


@dataclass
class Segment:
    """A single continuous segment within a promo clip."""
    segment_index: int               # 1, 2, ...
    episode_number: int              # Allows cross-episode stitching
    start_beat_id: str
    end_beat_id: str
    start_time: float
    end_time: float
    duration: float
    beat_ids: list[str]
    timestamp_source: TimestampSource

    def __post_init__(self):
        if self.timestamp_source not in ("given", "estimated"):
            raise ValueError(f"Invalid timestamp_source: {self.timestamp_source}")
        if not self.beat_ids:
            raise ValueError(f"Segment {self.segment_index} must contain at least one beat_id.")
        if self.end_time < self.start_time:
            raise ValueError(f"Segment {self.segment_index} end_time ({self.end_time}) < start_time ({self.start_time})")

    def to_dict(self) -> dict:
        return {
            "segment_index": self.segment_index,
            "episode_number": self.episode_number,
            "start_beat_id": self.start_beat_id,
            "end_beat_id": self.end_beat_id,
            "start_time": round(self.start_time, 2),
            "end_time": round(self.end_time, 2),
            "duration": round(self.duration, 2),
            "beat_ids": self.beat_ids,
            "timestamp_source": self.timestamp_source,
        }


@dataclass
class PromoCandidate:
    """
    A promo candidate composed of one or more segments.
    For single-segment promos: len(segments) == 1.
    For multi-segment promos: len(segments) >= 2.
    """
    candidate_id: str
    episode: str
    episode_number: int
    segments: list[Segment]
    hook_signal: HookSignal
    cliffhanger_signal: CliffhangerSignal

    def __post_init__(self):
        if not self.segments:
            raise ValueError(f"PromoCandidate {self.candidate_id} must have at least one segment.")

    @property
    def start_beat_id(self) -> str:
        return self.segments[0].start_beat_id

    @property
    def end_beat_id(self) -> str:
        return self.segments[-1].end_beat_id

    @property
    def start_time(self) -> float:
        return self.segments[0].start_time

    @property
    def end_time(self) -> float:
        return self.segments[-1].end_time

    @property
    def duration(self) -> float:
        return round(sum(s.duration for s in self.segments), 2)

    @property
    def beat_ids(self) -> list[str]:
        return [b_id for seg in self.segments for b_id in seg.beat_ids]

    @property
    def timestamp_source(self) -> TimestampSource:
        return "estimated" if any(s.timestamp_source == "estimated" for s in self.segments) else "given"

    def to_dict(self) -> dict:
        return {
            "candidate_id": self.candidate_id,
            "episode": self.episode,
            "episode_number": self.episode_number,
            "start_beat_id": self.start_beat_id,
            "end_beat_id": self.end_beat_id,
            "start_time": round(self.start_time, 2),
            "end_time": round(self.end_time, 2),
            "duration": round(self.duration, 2),
            "hook_signal": self.hook_signal.to_dict(),
            "cliffhanger_signal": self.cliffhanger_signal.to_dict(),
            "segments": [s.to_dict() for s in self.segments],
            "beat_ids": self.beat_ids,
            "timestamp_source": self.timestamp_source,
        }
