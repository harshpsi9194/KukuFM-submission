"""
parser.py — Screenplay parser for Kuku FM microdrama scripts.

Parses raw screenplay text into structured Episode, Beat, and Event objects.
Grouping rule:
- Group simultaneous/overlapping events into Beats.
- Merge events into the same beat only if their time intervals overlap meaningfully (> 0s).
- Touching at an endpoint (event A end == event B start) does NOT merge.
- Boundaries snap strictly to beat edges.
- Required timestamp_source ("given" or "estimated").
"""

import os
import re
from typing import Optional
try:
    from .schema import Event, Beat, Episode, TimestampSource
except ImportError:
    from schema import Event, Beat, Episode, TimestampSource

EPSILON = 1e-4  # Floating point tolerance for timestamp comparisons
DEFAULT_WORDS_PER_SECOND = 2.5


def parse_timestamp_str(ts_str: str) -> float:
    """Convert MM:SS.ss (or HH:MM:SS.ss) string into float seconds."""
    clean = ts_str.strip().strip("()[]")
    parts = clean.split(":")
    if len(parts) == 2:
        minutes, seconds = parts
        return round(float(minutes) * 60 + float(seconds), 2)
    elif len(parts) == 3:
        hours, minutes, seconds = parts
        return round(float(hours) * 3600 + float(minutes) * 60 + float(seconds), 2)
    return round(float(clean), 2)


def parse_header_metadata(lines: list[str]) -> tuple[str, int, str, str]:
    """Extract episode title, number, logline, and continuity notes from top of file."""
    episode_title = "Episode 1"
    episode_number = 1
    logline = ""
    continuity_notes = ""

    for line in lines[:20]:
        line_str = line.strip()
        if line_str.startswith("EPISODE NUMBER:"):
            episode_title = line_str.replace("EPISODE NUMBER:", "").strip()
            num_match = re.search(r'\d+', episode_title)
            if num_match:
                episode_number = int(num_match.group(0))
        elif line_str.startswith("EPISODE LOGLINE:"):
            logline = line_str.replace("EPISODE LOGLINE:", "").strip()
        elif line_str.startswith("CONTINUITY NOTES:"):
            continuity_notes = line_str.replace("CONTINUITY NOTES:", "").strip()

    return episode_title, episode_number, logline, continuity_notes


def parse_block_header(header_line: str) -> tuple[str, Optional[str], Optional[str]]:
    """
    Parse a block header line like:
      **CAMERA:**
      **ACTION:**
      **SOUND FX:**
      **LADY JULIANA (Confused, Breathless):**
      **MATRIARCH (O.S.) (Cold, Commanding):**

    Returns:
      (event_type, character_name, emotion)
      event_type is one of: "camera", "action", "dialogue", "sfx", "music"
    """
    clean = header_line.strip().strip("*").rstrip(":").strip()
    upper = clean.upper()

    if upper in ("CAMERA", "CAM"):
        return "camera", None, None
    if upper in ("ACTION", "ACT"):
        return "action", None, None
    if upper in ("SOUND FX", "SOUND EFFECT", "SOUND EFFECTS", "SFX"):
        return "sfx", None, None
    if upper in ("MUSIC", "SCORE"):
        return "music", None, None

    # Dialogue block — extract character name and emotion parentheticals
    parens = re.findall(r'\(([^)]+)\)', clean)
    char_name = re.sub(r'\s*\([^)]*\)', '', clean).strip()

    # Filter out production cues like O.S., V.O., CONT'D from emotion
    tech_tags = {"O.S.", "V.O.", "O.C.", "CONT'D", "CONTINUED", "OFF-SCREEN", "OFF SCREEN"}
    emotions = [p.strip() for p in parens if p.strip().upper() not in tech_tags]
    emotion = ", ".join(emotions) if emotions else None

    return "dialogue", char_name, emotion


def merge_events_into_beats(
    events: list[Event],
    start_beat_idx: int = 1,
    scene_number: Optional[int] = None,
    scene_heading: Optional[str] = None
) -> list[Beat]:
    """
    Group events into beats based on meaningful time overlap.
    Rules:
    - Events sorted chronologically by (start, end).
    - An event merges into current beat IF event.start < current_beat.end - EPSILON.
    - Touching at endpoint (event.start == current_beat.end) does NOT merge.
    - timestamp_source is "given" if all constituent events are "given", else "estimated".
    """
    if not events:
        return []

    # Ensure strictly sorted order
    sorted_events = sorted(events, key=lambda e: (e.start, e.end))

    beats: list[Beat] = []
    current_events: list[Event] = []
    current_start = sorted_events[0].start
    current_end = sorted_events[0].end

    for ev in sorted_events:
        if not current_events:
            current_events = [ev]
            current_start = ev.start
            current_end = ev.end
            continue

        # Meaningful overlap check: start strictly before current beat end
        if ev.start < current_end - EPSILON:
            # Merges into active beat
            current_events.append(ev)
            current_end = max(current_end, ev.end)
        else:
            # Touching at endpoint or gap — close current beat
            beat_source: TimestampSource = (
                "estimated" if any(e.timestamp_source == "estimated" for e in current_events) else "given"
            )
            beat_id = f"B{start_beat_idx + len(beats):03d}"
            beats.append(Beat(
                beat_id=beat_id,
                start=round(current_start, 2),
                end=round(current_end, 2),
                events=current_events,
                timestamp_source=beat_source,
                scene_number=scene_number,
                scene_heading=scene_heading
            ))
            # Start new beat
            current_events = [ev]
            current_start = ev.start
            current_end = ev.end

    # Flush final beat
    if current_events:
        beat_source: TimestampSource = (
            "estimated" if any(e.timestamp_source == "estimated" for e in current_events) else "given"
        )
        beat_id = f"B{start_beat_idx + len(beats):03d}"
        beats.append(Beat(
            beat_id=beat_id,
            start=round(current_start, 2),
            end=round(current_end, 2),
            events=current_events,
            timestamp_source=beat_source,
            scene_number=scene_number,
            scene_heading=scene_heading
        ))

    return beats


def parse_screenplay_text(raw_text: str, filepath: str = "") -> Episode:
    """Parse entire screenplay text into an Episode object with structured Beats."""
    lines = raw_text.splitlines()

    episode_title, episode_number, logline, continuity_notes = parse_header_metadata(lines)

    # Locate script start and end
    script_start = 0
    script_end = len(lines)
    for i, line in enumerate(lines):
        if "--- SCRIPT START ---" in line:
            script_start = i + 1
        elif "--- SCRIPT END ---" in line:
            script_end = i
            break

    script_lines = lines[script_start:script_end]

    # Parse scene by scene
    # Split by SCENE marker
    scene_splits: list[tuple[int, str, list[str]]] = []
    current_scene_num = 1
    current_scene_heading = ""
    current_scene_lines: list[str] = []

    scene_header_pattern = re.compile(r'^SCENE\s+(\d+)', re.IGNORECASE)

    for line in script_lines:
        m = scene_header_pattern.match(line.strip())
        if m:
            if current_scene_lines:
                scene_splits.append((current_scene_num, current_scene_heading, current_scene_lines))
                current_scene_lines = []
            current_scene_num = int(m.group(1))
            current_scene_heading = ""
        else:
            if line.strip().startswith("Scene heading:"):
                current_scene_heading = line.strip().replace("Scene heading:", "").strip()
            current_scene_lines.append(line)

    if current_scene_lines:
        scene_splits.append((current_scene_num, current_scene_heading, current_scene_lines))

    # Fallback if no explicit SCENE markers found
    if not scene_splits:
        scene_splits.append((1, "UNKNOWN", script_lines))

    all_beats: list[Beat] = []
    global_beat_idx = 1
    running_fallback_time = 0.0

    for sc_num, sc_heading, sc_lines in scene_splits:
        # Extract all event blocks in this scene
        scene_events: list[Event] = []
        i = 0
        while i < len(sc_lines):
            line = sc_lines[i].strip()
            # Look for block headers (**...**)
            if line.startswith("**") and (line.endswith("**") or line.endswith(":**") or line.endswith(":")):
                event_type, char_name, emotion = parse_block_header(line)

                timing_line = None
                desc_lines = []
                j = i + 1

                while j < len(sc_lines):
                    next_line = sc_lines[j].strip()
                    if not next_line:
                        # Check if empty line is followed by next block
                        if j + 1 < len(sc_lines) and sc_lines[j + 1].strip().startswith("**"):
                            j += 1
                            break
                        j += 1
                        continue

                    # Check for timing line
                    if "Start:" in next_line and "End:" in next_line:
                        timing_line = next_line
                        j += 1
                        # Collect description lines until next block or end
                        while j < len(sc_lines):
                            dl = sc_lines[j].strip()
                            if not dl:
                                if j + 1 < len(sc_lines) and sc_lines[j + 1].strip().startswith("**"):
                                    break
                                j += 1
                                continue
                            if dl.startswith("**") or dl.startswith("SCENE ") or dl.startswith("---"):
                                break
                            desc_lines.append(dl)
                            j += 1
                        break
                    elif next_line.startswith("**") or next_line.startswith("SCENE ") or next_line.startswith("---"):
                        break
                    else:
                        # Description line without timing line
                        desc_lines.append(next_line)
                        j += 1

                raw_desc = " ".join(desc_lines).strip()
                # Clean enclosing quotes from dialogue
                if event_type == "dialogue" and raw_desc.startswith('"') and raw_desc.endswith('"') and len(raw_desc) > 1:
                    raw_desc = raw_desc[1:-1]

                # Timestamp resolution
                if timing_line:
                    tm = re.search(r'Start:\s*\(([^)]+)\)\s*\|\s*End:\s*\(([^)]+)\)', timing_line)
                    if tm:
                        start_sec = parse_timestamp_str(tm.group(1))
                        end_sec = parse_timestamp_str(tm.group(2))
                        timestamp_source: TimestampSource = "given"
                        running_fallback_time = end_sec
                    else:
                        # Malformed timing line fallback
                        word_count = max(1, len(raw_desc.split()))
                        dur = max(1.0, round(word_count / DEFAULT_WORDS_PER_SECOND, 2))
                        start_sec = round(running_fallback_time, 2)
                        end_sec = round(start_sec + dur, 2)
                        timestamp_source: TimestampSource = "estimated"
                        running_fallback_time = end_sec
                else:
                    # Missing timing line fallback (~2.5 words/sec)
                    word_count = max(1, len(raw_desc.split()))
                    dur = max(1.0, round(word_count / DEFAULT_WORDS_PER_SECOND, 2))
                    start_sec = round(running_fallback_time, 2)
                    end_sec = round(start_sec + dur, 2)
                    timestamp_source: TimestampSource = "estimated"
                    running_fallback_time = end_sec

                if raw_desc:
                    scene_events.append(Event(
                        type=event_type,
                        description=raw_desc,
                        start=start_sec,
                        end=end_sec,
                        timestamp_source=timestamp_source,
                        character=char_name,
                        emotion=emotion
                    ))
                i = j
            else:
                i += 1

        # Group scene events into beats
        scene_beats = merge_events_into_beats(
            scene_events,
            start_beat_idx=global_beat_idx,
            scene_number=sc_num,
            scene_heading=sc_heading
        )
        all_beats.extend(scene_beats)
        global_beat_idx += len(scene_beats)

    return Episode(
        episode=episode_title,
        episode_number=episode_number,
        logline=logline,
        continuity_notes=continuity_notes,
        beats=all_beats,
        filepath=filepath
    )


def parse_file(filepath: str) -> Episode:
    """Read and parse a single screenplay file."""
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
    return parse_screenplay_text(text, filepath=filepath)
