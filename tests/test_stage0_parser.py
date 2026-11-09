"""
test_stage0_parser.py — Stage 0 Screenplay Parser Unit Tests.

Verifies:
1. Required, non-default timestamp_source ("given" or "estimated") on Event and Beat (Rule 4).
2. Meaningful overlap (>0s) merges into one beat; endpoint touching strictly splits into two beats (Rule 2).
3. Missing timestamps fallback (~2.5 words/sec heuristic, marked "estimated").
4. Out-of-order blocks properly sorted chronologically.
5. Dialogue-first scenes with no preceding action.
6. Consecutive CAMERA blocks with no intervening dialogue.
7. Scene boundary safety: beats never merge across scene boundaries (Rule 5).
8. Golden-file snapshot test for Episode 1.
"""

import json
import os
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PROMO_DIR = REPO_ROOT / "promo-finder"
for p in [REPO_ROOT, PROMO_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from parser.schema import Event, Beat, Episode
from parser.parser import (
    parse_file,
    parse_screenplay_text,
    merge_events_into_beats,
    parse_block_header,
    parse_timestamp_str,
)


class TestStage0Parser(unittest.TestCase):

    def test_schema_required_timestamp_source(self):
        """Rule 4: timestamp_source is REQUIRED on Event and Beat (no default values)."""
        with self.assertRaises(TypeError):
            Event(type="camera", description="Establishing shot.", start=0.0, end=5.0)

        with self.assertRaises(ValueError):
            Event(type="camera", description="Establishing shot.", start=0.0, end=5.0, timestamp_source="inferred")  # type: ignore

        ev = Event(type="camera", description="Establishing shot.", start=0.0, end=5.0, timestamp_source="given")
        self.assertEqual(ev.timestamp_source, "given")

        with self.assertRaises(TypeError):
            Beat(beat_id="B001", start=0.0, end=5.0, events=[ev])

        with self.assertRaises(ValueError):
            Beat(beat_id="B001", start=0.0, end=5.0, events=[ev], timestamp_source="approximate")  # type: ignore

        beat = Beat(beat_id="B001", start=0.0, end=5.0, events=[ev], timestamp_source="given")
        self.assertEqual(beat.timestamp_source, "given")

    def test_overlap_merge_logic_meaningful_overlap_vs_endpoint_touching(self):
        """
        Rule 2:
        - Two events genuinely overlapping (>0s) merge into 1 beat.
        - Two events touching only at an endpoint (A end == B start) do NOT merge.
        """
        # Case A: Genuinely overlapping
        ev_a = Event(type="camera", description="Establishing shot.", start=0.0, end=5.0, timestamp_source="given")
        ev_b = Event(type="action", description="Juliana enters frame.", start=3.0, end=7.0, timestamp_source="given")
        merged = merge_events_into_beats([ev_a, ev_b], start_beat_idx=1)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].start, 0.0)
        self.assertEqual(merged[0].end, 7.0)
        self.assertEqual(len(merged[0].events), 2)

        # Case B: Endpoint touching
        ev_1 = Event(type="camera", description="Shot inside chapel.", start=0.0, end=10.0, timestamp_source="given")
        ev_2 = Event(type="dialogue", description="Stop right there.", start=10.0, end=15.0, timestamp_source="given", character="GUARD")
        touching = merge_events_into_beats([ev_1, ev_2], start_beat_idx=1)
        self.assertEqual(len(touching), 2)
        self.assertEqual(touching[0].end, touching[1].start)
        self.assertNotEqual(touching[0].beat_id, touching[1].beat_id)

    def test_missing_timestamps_fallback_estimated(self):
        """Test fallback when timestamps are absent: estimate ~2.5 words/sec, marked 'estimated'."""
        script_missing = """EPISODE NUMBER: Episode 99
EPISODE LOGLINE: Test episode without timestamps.
CONTINUITY NOTES: Test.

--- SCRIPT START ---

SCENE 01
Scene heading: INT. ROOM – DAY

**CAMERA:**
A wide establishing shot of the shadowy room.

**ACTION:**
Juliana steps cautiously across the stone tiles toward the hearth.

**LADY JULIANA (Whispering):**
"Is anyone there?"

--- SCRIPT END ---
"""
        ep = parse_screenplay_text(script_missing)
        self.assertGreater(len(ep.beats), 0)
        for beat in ep.beats:
            self.assertEqual(beat.timestamp_source, "estimated")
            for ev in beat.events:
                self.assertEqual(ev.timestamp_source, "estimated")
                self.assertGreater(ev.end, ev.start)

    def test_out_of_order_blocks(self):
        """Test that out-of-order blocks are chronologically sorted."""
        script_out_of_order = """EPISODE NUMBER: Episode 98
EPISODE LOGLINE: Out of order test.
CONTINUITY NOTES: None.

--- SCRIPT START ---

SCENE 01
Scene heading: INT. HALL – DAY

**ACTION:**
Start: (00:15.00) | End: (00:20.00)
Juliana turns and runs down the corridor.

**CAMERA:**
Start: (00:02.00) | End: (00:05.00)
Wide shot of the empty doorway.

**DIALOGUE (Urgent):**
Start: (00:04.00) | End: (00:08.00)
"Wait for me!"

--- SCRIPT END ---
"""
        ep = parse_screenplay_text(script_out_of_order)
        self.assertEqual(len(ep.beats), 2)
        self.assertEqual(ep.beats[0].start, 2.0)
        self.assertEqual(ep.beats[0].end, 8.0)
        self.assertEqual(ep.beats[1].start, 15.0)
        self.assertEqual(ep.beats[1].end, 20.0)

    def test_dialogue_with_no_preceding_action(self):
        """Test that a scene starting directly with dialogue parses cleanly."""
        script_dialogue_first = """EPISODE NUMBER: Episode 97
EPISODE LOGLINE: Dialogue first test.
CONTINUITY NOTES: None.

--- SCRIPT START ---

SCENE 01
Scene heading: INT. CONFESSIONAL – NIGHT

**FATHER THOMAS (Cold, Threatening):**
Start: (00:00.00) | End: (00:03.50)
"You cannot escape your penance."

**ACTION:**
Start: (00:03.50) | End: (00:06.00)
The wooden lattice slides shut.

--- SCRIPT END ---
"""
        ep = parse_screenplay_text(script_dialogue_first)
        self.assertEqual(len(ep.beats), 2)
        b1 = ep.beats[0]
        self.assertEqual(b1.start, 0.0)
        self.assertEqual(b1.end, 3.5)
        self.assertEqual(b1.events[0].type, "dialogue")
        self.assertEqual(b1.events[0].character, "FATHER THOMAS")

    def test_consecutive_camera_blocks_no_dialogue(self):
        """Test consecutive camera blocks: overlapping merge, non-overlapping don't."""
        script_consecutive_cam = """EPISODE NUMBER: Episode 96
EPISODE LOGLINE: Camera only test.
CONTINUITY NOTES: None.

--- SCRIPT START ---

SCENE 01
Scene heading: EXT. MOUNTAIN PASS – DAWN

**CAMERA:**
Start: (00:00.00) | End: (00:05.00)
Wide aerial crane shot sweeping across the snow-capped peaks.

**CAMERA:**
Start: (00:03.00) | End: (00:07.00)
Tracking shot descending toward a lone hooded rider.

**CAMERA:**
Start: (00:07.00) | End: (00:10.00)
Close insert on the horse's steaming breath.

--- SCRIPT END ---
"""
        ep = parse_screenplay_text(script_consecutive_cam)
        self.assertEqual(len(ep.beats), 2)
        self.assertEqual(ep.beats[0].start, 0.0)
        self.assertEqual(ep.beats[0].end, 7.0)
        self.assertEqual(len(ep.beats[0].events), 2)
        self.assertEqual(ep.beats[1].start, 7.0)
        self.assertEqual(ep.beats[1].end, 10.0)

    def test_beat_merging_never_crosses_scene_boundary(self):
        """Rule 5: Beat merging never crosses a scene boundary."""
        two_scene_script = """EPISODE NUMBER: Episode 95
EPISODE LOGLINE: Two scene boundary test.
CONTINUITY NOTES: None.

--- SCRIPT START ---

SCENE 01
Scene heading: INT. GREAT HALL – NIGHT

**CAMERA:**
Start: (00:00.00) | End: (00:08.00)
Wide shot of the great hall.

**ACTION:**
Start: (00:06.00) | End: (00:15.00)
Juliana slams the heavy oak doors shut.

SCENE 02
Scene heading: EXT. COURTYARD – NIGHT

**CAMERA:**
Start: (00:12.00) | End: (00:18.00)
Rain beats down on the stone courtyard.

**ACTION:**
Start: (00:16.00) | End: (00:25.00)
Alexander mounts his warhorse in haste.

--- SCRIPT END ---
"""
        ep = parse_screenplay_text(two_scene_script)
        self.assertEqual(len(ep.beats), 2)
        b1, b2 = ep.beats[0], ep.beats[1]
        self.assertEqual(b1.scene_number, 1)
        self.assertEqual(b2.scene_number, 2)
        self.assertNotEqual(b1.scene_number, b2.scene_number)

    def test_golden_snapshot_episode_001(self):
        """Rule 12 & 13: Golden-file snapshot test for Episode 1."""
        script_path = PROMO_DIR / "parser" / "fixtures" / "sample_scripts" / "episode_001.txt"
        golden_path = PROMO_DIR / "parser" / "fixtures" / "episode_001_golden.json"
        self.assertTrue(script_path.exists())
        self.assertTrue(golden_path.exists())

        ep = parse_file(str(script_path))
        ep_dict = ep.to_dict()

        with open(golden_path, "r", encoding="utf-8") as f:
            gold_dict = json.load(f)

        self.assertEqual(ep_dict["episode_number"], gold_dict["episode_number"])
        self.assertEqual(ep_dict["total_beats"], gold_dict["total_beats"])
        self.assertEqual(len(ep_dict["beats"]), len(gold_dict["beats"]))
        for cur, gold in zip(ep_dict["beats"], gold_dict["beats"]):
            self.assertEqual(cur["beat_id"], gold["beat_id"])
            self.assertEqual(cur["start"], gold["start"])
            self.assertEqual(cur["end"], gold["end"])
            self.assertEqual(cur["timestamp_source"], gold["timestamp_source"])


if __name__ == "__main__":
    unittest.main()
