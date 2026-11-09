"""
test_parser.py — Comprehensive unit tests for screenplay parser.

Covers:
1. Required timestamp_source validation on Event and Beat (TypeError/ValueError on missing/invalid).
2. Overlap-merge logic:
   - Genuine overlap -> merges into 1 beat.
   - Endpoint touching (A end == B start) -> strictly 2 beats.
3. Missing timestamps fallback (~2.5 words/sec, marked "estimated").
4. Out-of-order blocks properly sorted chronologically.
5. Dialogue with no preceding action.
6. Consecutive CAMERA blocks with no dialogue.
7. Golden-file snapshot test for Episode 1.
"""

import json
import os
import unittest
try:
    from .schema import Event, Beat, Episode
    from .parser import (
        parse_file,
        parse_screenplay_text,
        merge_events_into_beats,
        parse_block_header,
        parse_timestamp_str,
    )
except ImportError:
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from schema import Event, Beat, Episode
    from parser import (
        parse_file,
        parse_screenplay_text,
        merge_events_into_beats,
        parse_block_header,
        parse_timestamp_str,
    )


class TestParser(unittest.TestCase):

    def test_schema_required_timestamp_source(self):
        """Rule 1: timestamp_source is a REQUIRED field on every Event and Beat (no defaults)."""
        # Missing timestamp_source on Event -> TypeError
        with self.assertRaises(TypeError):
            Event(
                type="camera",
                description="Wide shot of cathedral.",
                start=0.0,
                end=5.0
                # timestamp_source omitted
            )

        # Invalid timestamp_source on Event -> ValueError
        with self.assertRaises(ValueError):
            Event(
                type="camera",
                description="Wide shot of cathedral.",
                start=0.0,
                end=5.0,
                timestamp_source="approximate"  # type: ignore
            )

        # Valid Event
        ev = Event(
            type="camera",
            description="Wide shot of cathedral.",
            start=0.0,
            end=5.0,
            timestamp_source="given"
        )
        self.assertEqual(ev.timestamp_source, "given")

        # Missing timestamp_source on Beat -> TypeError
        with self.assertRaises(TypeError):
            Beat(
                beat_id="B001",
                start=0.0,
                end=5.0,
                events=[ev]
                # timestamp_source omitted
            )

        # Invalid timestamp_source on Beat -> ValueError
        with self.assertRaises(ValueError):
            Beat(
                beat_id="B001",
                start=0.0,
                end=5.0,
                events=[ev],
                timestamp_source="inferred"  # type: ignore
            )

        # Valid Beat
        beat = Beat(
            beat_id="B001",
            start=0.0,
            end=5.0,
            events=[ev],
            timestamp_source="given"
        )
        self.assertEqual(beat.timestamp_source, "given")

    def test_overlap_merge_logic_meaningful_overlap_vs_endpoint_touching(self):
        """
        Rule 2:
        - Two events whose intervals genuinely overlap MUST merge into one beat.
        - Two events that only touch at an endpoint (e.g. A ends at 10.0, B starts at 10.0)
          MUST NOT merge into one beat (Rule 5: 'overlap meaningfully', not just touch).
        """
        # --- CASE A: Genuinely overlapping events ---
        # Event A: [0.0s, 5.0s]
        # Event B: [3.0s, 7.0s]
        # Overlap interval: [3.0s, 5.0s], duration = 2.0s > 0
        ev_a = Event(type="camera", description="Establishing shot.", start=0.0, end=5.0, timestamp_source="given")
        ev_b = Event(type="action", description="Juliana walks into frame.", start=3.0, end=7.0, timestamp_source="given")

        merged_beats = merge_events_into_beats([ev_a, ev_b], start_beat_idx=1)

        # Assertions for genuine overlap
        self.assertEqual(len(merged_beats), 1, "Genuinely overlapping events must merge into exactly 1 beat.")
        beat_overlap = merged_beats[0]
        self.assertEqual(beat_overlap.beat_id, "B001")
        self.assertEqual(beat_overlap.start, 0.0, "Merged beat start must match earliest event start.")
        self.assertEqual(beat_overlap.end, 7.0, "Merged beat end must match latest event end.")
        self.assertEqual(len(beat_overlap.events), 2)
        # Verify constituent events retain their own exact boundaries inside the beat
        self.assertEqual(beat_overlap.events[0].start, 0.0)
        self.assertEqual(beat_overlap.events[0].end, 5.0)
        self.assertEqual(beat_overlap.events[1].start, 3.0)
        self.assertEqual(beat_overlap.events[1].end, 7.0)

        # --- CASE B: Touching only at an endpoint ---
        # Event 1: [0.0s, 10.0s]
        # Event 2: [10.0s, 15.0s]
        # Overlap duration: max(0, min(10.0, 15.0) - max(0.0, 10.0)) = 10.0 - 10.0 = 0.0s
        ev_touch_1 = Event(type="camera", description="Shot inside chapel.", start=0.0, end=10.0, timestamp_source="given")
        ev_touch_2 = Event(type="dialogue", description="Stop right there.", start=10.0, end=15.0, timestamp_source="given", character="GUARD")

        touching_beats = merge_events_into_beats([ev_touch_1, ev_touch_2], start_beat_idx=1)

        # Assertions for endpoint touching
        self.assertEqual(len(touching_beats), 2, "Events touching only at endpoint must NOT merge into one beat.")
        b1, b2 = touching_beats[0], touching_beats[1]
        self.assertEqual(b1.beat_id, "B001")
        self.assertEqual(b1.start, 0.0)
        self.assertEqual(b1.end, 10.0)
        self.assertEqual(len(b1.events), 1)

        self.assertEqual(b2.beat_id, "B002")
        self.assertEqual(b2.start, 10.0)
        self.assertEqual(b2.end, 15.0)
        self.assertEqual(len(b2.events), 1)

        # Explicitly verify the boundary touching point
        self.assertEqual(b1.end, b2.start, "Beat 1 ends exactly where Beat 2 starts (touching, unmerged).")
        self.assertNotEqual(b1.beat_id, b2.beat_id)

    def test_missing_timestamps_fallback_estimated(self):
        """Test fallback when timestamps are absent: estimate ~2.5 words/sec and mark 'estimated'."""
        script_missing_timestamps = """EPISODE NUMBER: Episode 99
EPISODE LOGLINE: Test episode with missing timestamps.
CONTINUITY NOTES: Test.

--- SCRIPT START ---

SCENE 01
Scene heading: INT. ROOM – DAY

**CAMERA:**
A wide establishing shot of the shadowy room.

**ACTION:**
Juliana steps cautiously across the stone tiles toward the glowing hearth.

**LADY JULIANA (Whispering):**
"Is anyone there?"

--- SCRIPT END ---
"""
        ep = parse_screenplay_text(script_missing_timestamps)
        self.assertGreater(len(ep.beats), 0)

        for beat in ep.beats:
            # Rule: Must be marked "estimated"
            self.assertEqual(beat.timestamp_source, "estimated")
            for ev in beat.events:
                self.assertEqual(ev.timestamp_source, "estimated")
                self.assertGreater(ev.end, ev.start)
                self.assertGreater(ev.duration, 0.0)

    def test_out_of_order_blocks(self):
        """Test that out-of-order blocks in script are parsed and sorted into chronological beats."""
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
        # Sorted order should be:
        # [02.00 - 05.00] camera and [04.00 - 08.00] dialogue overlap -> Beat 1: [02.00 - 08.00]
        # [15.00 - 20.00] action -> Beat 2: [15.00 - 20.00]
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
        self.assertEqual(len(b1.events), 1)
        self.assertEqual(b1.events[0].type, "dialogue")
        self.assertEqual(b1.events[0].character, "FATHER THOMAS")
        self.assertEqual(b1.events[0].emotion, "Cold, Threatening")
        self.assertEqual(b1.events[0].description, "You cannot escape your penance.")

    def test_consecutive_camera_blocks_no_dialogue(self):
        """Test consecutive camera blocks with no dialogue: overlapping merge, non-overlapping don't."""
        script_consecutive_camera = """EPISODE NUMBER: Episode 96
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
        ep = parse_screenplay_text(script_consecutive_camera)
        # Cam 1 (0-5s) and Cam 2 (3-7s) overlap -> Beat 1: [0-7s], 2 events
        # Cam 3 (7-10s) touches endpoint 7s -> Beat 2: [7-10s], 1 event
        self.assertEqual(len(ep.beats), 2)
        self.assertEqual(ep.beats[0].start, 0.0)
        self.assertEqual(ep.beats[0].end, 7.0)
        self.assertEqual(len(ep.beats[0].events), 2)
        self.assertTrue(all(e.type == "camera" for e in ep.beats[0].events))

        self.assertEqual(ep.beats[1].start, 7.0)
        self.assertEqual(ep.beats[1].end, 10.0)
        self.assertEqual(len(ep.beats[1].events), 1)
        self.assertEqual(ep.beats[1].events[0].type, "camera")

    def test_beat_merging_never_crosses_scene_boundary(self):
        """
        Test that beat merging never crosses a scene boundary.
        Construct a two-scene fixture where the last block of scene 1 and the first
        block of scene 2 have timestamps that would overlap if scene boundaries were ignored,
        and assert they remain in separate beats with correct scene_number and scene_heading.
        """
        two_scene_script = """EPISODE NUMBER: Episode 95
EPISODE LOGLINE: Two scene boundary test.
CONTINUITY NOTES: None.

--- SCRIPT START ---

SCENE 01
Scene heading: INT. GREAT HALL – NIGHT
Scene start: 00:00.00
Scene end: 00:15.00

Actions and dialogue:

**CAMERA:**
Start: (00:00.00) | End: (00:08.00)
Wide shot of the great hall.

**ACTION:**
Start: (00:06.00) | End: (00:15.00)
Juliana slams the heavy oak doors shut.

SCENE 02
Scene heading: EXT. COURTYARD – NIGHT
Scene start: 00:12.00
Scene end: 00:25.00

Actions and dialogue:

**CAMERA:**
Start: (00:12.00) | End: (00:18.00)
Rain beats down on the stone courtyard as lightning flashes.

**ACTION:**
Start: (00:16.00) | End: (00:25.00)
Alexander mounts his warhorse in haste.

--- SCRIPT END ---
"""
        ep = parse_screenplay_text(two_scene_script)

        # In Scene 1:
        # Cam (0-8s) and Action (6-15s) overlap -> Beat B001: [0.0 - 15.0s], scene 1
        # In Scene 2:
        # Cam starts at 12.0s, which is < 15.0s (strictly overlaps Scene 1's action in time).
        # But because it is in SCENE 2, it MUST NOT merge with Beat B001!
        # In Scene 2: Cam (12-18s) and Action (16-25s) overlap -> Beat B002: [12.0 - 25.0s], scene 2

        self.assertEqual(len(ep.beats), 2, "Scene boundary must partition beats even if timestamps overlap across scenes.")

        b1 = ep.beats[0]
        b2 = ep.beats[1]

        self.assertEqual(b1.beat_id, "B001")
        self.assertEqual(b1.scene_number, 1)
        self.assertEqual(b1.scene_heading, "INT. GREAT HALL – NIGHT")
        self.assertEqual(b1.start, 0.0)
        self.assertEqual(b1.end, 15.0)
        self.assertEqual(len(b1.events), 2)

        self.assertEqual(b2.beat_id, "B002")
        self.assertEqual(b2.scene_number, 2)
        self.assertEqual(b2.scene_heading, "EXT. COURTYARD – NIGHT")
        self.assertEqual(b2.start, 12.0)
        self.assertEqual(b2.end, 25.0)
        self.assertEqual(len(b2.events), 2)

        # Assert distinct beats and different scene metadata
        self.assertNotEqual(b1.scene_number, b2.scene_number)
        self.assertNotEqual(b1.scene_heading, b2.scene_heading)

    def test_golden_snapshot_episode_001(self):
        """Rule 3: Golden-file snapshot test for Episode 1 using repo-relative fixture."""
        script_path = os.path.join(
            os.path.dirname(__file__), "fixtures", "sample_scripts", "episode_001.txt"
        )
        self.assertTrue(os.path.exists(script_path), f"Script not found at {script_path}")

        fixture_path = os.path.join(os.path.dirname(__file__), "fixtures", "episode_001_golden.json")
        self.assertTrue(os.path.exists(fixture_path), f"Golden fixture not found at {fixture_path}")

        # Parse current script
        current_ep = parse_file(script_path)
        current_dict = current_ep.to_dict()

        # Load golden snapshot
        with open(fixture_path, "r", encoding="utf-8") as f:
            golden_dict = json.load(f)

        # Assert structural invariants
        self.assertEqual(current_dict["episode"], golden_dict["episode"])
        self.assertEqual(current_dict["episode_number"], golden_dict["episode_number"])
        self.assertEqual(current_dict["duration"], golden_dict["duration"])
        self.assertEqual(current_dict["total_beats"], golden_dict["total_beats"])
        self.assertEqual(len(current_dict["beats"]), len(golden_dict["beats"]))

        # Assert beat-by-beat equality
        for i, (cur_beat, gold_beat) in enumerate(zip(current_dict["beats"], golden_dict["beats"])):
            self.assertEqual(cur_beat["beat_id"], gold_beat["beat_id"], f"Beat ID mismatch at index {i}")
            self.assertEqual(cur_beat["start"], gold_beat["start"], f"Beat start mismatch at {cur_beat['beat_id']}")
            self.assertEqual(cur_beat["end"], gold_beat["end"], f"Beat end mismatch at {cur_beat['beat_id']}")
            self.assertEqual(cur_beat["timestamp_source"], gold_beat["timestamp_source"])
            self.assertEqual(len(cur_beat["events"]), len(gold_beat["events"]), f"Event count mismatch at {cur_beat['beat_id']}")

            for j, (cur_ev, gold_ev) in enumerate(zip(cur_beat["events"], gold_beat["events"])):
                self.assertEqual(cur_ev["type"], gold_ev["type"], f"Event type mismatch at {cur_beat['beat_id']}[{j}]")
                self.assertEqual(cur_ev["start"], gold_ev["start"])
                self.assertEqual(cur_ev["end"], gold_ev["end"])
                self.assertEqual(cur_ev["timestamp_source"], gold_ev["timestamp_source"])
                self.assertEqual(cur_ev["description"], gold_ev["description"])


if __name__ == "__main__":
    unittest.main()
