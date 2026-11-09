"""
schema.py — Core data structures for the screenplay parser.

Invariants:
- A Beat is a time window containing one or more simultaneous events (CAMERA, ACTION, DIALOGUE, SFX).
- timestamp_source is a REQUIRED field on both Event and Beat ("given" or "estimated").
  Never defaulted. Never optional.
"""

from dataclasses import dataclass
from typing import Optional, Literal
import json

TimestampSource = Literal["given", "estimated"]


@dataclass
class Event:
    """A single atomic event in the screenplay (camera direction, action, dialogue line, or SFX)."""
    type: str                  # "camera", "action", "dialogue", "sfx", "music"
    description: str           # The action text, camera movement, or dialogue spoken
    start: float               # Start time in seconds
    end: float                 # End time in seconds
    timestamp_source: TimestampSource  # REQUIRED: "given" or "estimated"
    character: Optional[str] = None    # Character speaking (for dialogue)
    emotion: Optional[str] = None      # Emotional delivery / tone tag

    def __post_init__(self):
        if self.timestamp_source not in ("given", "estimated"):
            raise ValueError(
                f"Invalid timestamp_source: {self.timestamp_source!r}. Must be 'given' or 'estimated'."
            )
        if self.end < self.start:
            raise ValueError(
                f"Event end ({self.end}) cannot be less than start ({self.start}): {self.description[:40]}"
            )

    @property
    def duration(self) -> float:
        return round(self.end - self.start, 2)

    def to_dict(self) -> dict:
        d = {
            "type": self.type,
            "description": self.description,
            "start": round(self.start, 2),
            "end": round(self.end, 2),
            "timestamp_source": self.timestamp_source,
        }
        if self.character:
            d["character"] = self.character
        if self.emotion:
            d["emotion"] = self.emotion
        return d


@dataclass
class Beat:
    """
    A Beat is a container of one or more simultaneous/overlapping events.
    Boundary rule: beat.start = min(event.start), beat.end = max(event.end).
    Clip boundaries snap strictly to beat edges.
    """
    beat_id: str                       # e.g., "B001"
    start: float                       # Earliest start among constituent events
    end: float                         # Latest end among constituent events
    events: list[Event]                # Overlapping events in this beat
    timestamp_source: TimestampSource  # REQUIRED: "given" if all events given, else "estimated"
    scene_number: Optional[int] = None
    scene_heading: Optional[str] = None

    def __post_init__(self):
        if self.timestamp_source not in ("given", "estimated"):
            raise ValueError(
                f"Invalid timestamp_source: {self.timestamp_source!r}. Must be 'given' or 'estimated'."
            )
        if not self.events:
            raise ValueError(f"Beat {self.beat_id} must contain at least one event.")
        if self.end < self.start:
            raise ValueError(f"Beat {self.beat_id} end ({self.end}) cannot be less than start ({self.start})")

    @property
    def duration(self) -> float:
        return round(self.end - self.start, 2)

    def get_dialogue_events(self) -> list[Event]:
        return [e for e in self.events if e.type == "dialogue"]

    def get_visual_events(self) -> list[Event]:
        return [e for e in self.events if e.type in ("camera", "action")]

    def has_dialogue(self) -> bool:
        return any(e.type == "dialogue" for e in self.events)

    def has_visual_action(self) -> bool:
        return any(e.type in ("camera", "action") for e in self.events)

    def get_summary_text(self) -> str:
        """Produce readable text of all events in this beat for LLM evaluation."""
        lines = []
        for e in self.events:
            if e.type == "dialogue":
                char = e.character or "UNKNOWN"
                emo = f" ({e.emotion})" if e.emotion else ""
                lines.append(f'{char}{emo}: "{e.description}"')
            elif e.type == "camera":
                lines.append(f"[CAMERA]: {e.description}")
            elif e.type == "action":
                lines.append(f"[ACTION]: {e.description}")
            elif e.type == "sfx":
                lines.append(f"[SFX]: {e.description}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        d = {
            "beat_id": self.beat_id,
            "start": round(self.start, 2),
            "end": round(self.end, 2),
            "timestamp_source": self.timestamp_source,
            "events": [e.to_dict() for e in self.events],
        }
        if self.scene_number is not None:
            d["scene_number"] = self.scene_number
        if self.scene_heading:
            d["scene_heading"] = self.scene_heading
        return d


@dataclass
class Episode:
    """Full parsed episode composed of structured beats."""
    episode: str               # e.g., "Episode 1"
    episode_number: int        # e.g., 1
    logline: str
    continuity_notes: str
    beats: list[Beat]
    filepath: str = ""

    @property
    def duration(self) -> float:
        return max((b.end for b in self.beats), default=0.0)

    @property
    def total_beats(self) -> int:
        return len(self.beats)

    def get_beat(self, beat_id: str) -> Optional[Beat]:
        for b in self.beats:
            if b.beat_id == beat_id:
                return b
        return None

    def get_beats_between(self, start_beat_id: str, end_beat_id: str) -> list[Beat]:
        """Return slice of beats from start_beat_id to end_beat_id inclusive."""
        start_idx = None
        end_idx = None
        for i, b in enumerate(self.beats):
            if b.beat_id == start_beat_id:
                start_idx = i
            if b.beat_id == end_beat_id:
                end_idx = i
        if start_idx is None or end_idx is None or start_idx > end_idx:
            return []
        return self.beats[start_idx:end_idx + 1]

    def to_dict(self) -> dict:
        return {
            "episode": self.episode,
            "episode_number": self.episode_number,
            "logline": self.logline,
            "continuity_notes": self.continuity_notes,
            "duration": round(self.duration, 2),
            "total_beats": len(self.beats),
            "beats": [b.to_dict() for b in self.beats],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

