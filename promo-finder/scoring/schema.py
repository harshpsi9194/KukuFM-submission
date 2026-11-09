"""
schema.py — Data models for Stage 3 Scoring.
Strict schemas corresponding to:
- prompts/comprehension_gate_v1.txt (Two-part boolean gate: opening_legibility && arc_legibility)
- prompts/emotional_salience_v1.txt
- prompts/spoiler_classifier_v1.txt
"""

from dataclasses import dataclass, asdict
from typing import Optional, Any


@dataclass
class ComprehensionGateResult:
    candidate_id: str
    opening_legibility: bool
    opening_rationale: str
    arc_legibility: bool
    arc_rationale: str
    passes_gate: bool
    situational_summary: Optional[str]
    missing_context: Optional[str]
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EmotionalSalienceScore:
    candidate_id: str
    emotional_salience_score: int  # 1–5 relative scale
    emotional_core: str
    relative_rank_justification: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SpoilerClassification:
    candidate_id: str
    spoiler_level: int  # 0–3 tier
    spoiler_rationale: str
    ideal_ad_placement: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Stage3ScoredCandidate:
    candidate_id: str
    start_beat_id: str
    end_beat_id: str
    start_time: float
    end_time: float
    duration: float
    timestamp_source: str
    narrative_thread: str
    comprehension: ComprehensionGateResult
    emotional_salience: EmotionalSalienceScore
    spoiler: SpoilerClassification

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "start_beat_id": self.start_beat_id,
            "end_beat_id": self.end_beat_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "timestamp_source": self.timestamp_source,
            "narrative_thread": self.narrative_thread,
            "comprehension": self.comprehension.to_dict(),
            "emotional_salience": self.emotional_salience.to_dict(),
            "spoiler": self.spoiler.to_dict(),
        }
