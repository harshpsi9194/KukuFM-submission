"""
test_candidates.py — Unit tests for Stage 2 candidate generation.

Covers:
1. Bidirectional window search (forward from hook, backward from cliffhanger, union + dedupe).
2. Pure Python boundary resolution (snap to start_beat.start and end_beat.end).
3. Duration filtering (only 20s–60s clips survive).
4. No timestamp hallucinations.
"""

import unittest
import os
import sys

# Ensure parent directory is in path for package imports
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from parser.schema import Beat, Event, Episode
from candidates.schema import HookSignal, CliffhangerSignal, Segment, PromoCandidate
from candidates.window_search import bidirectional_window_search


class TestCandidateGeneration(unittest.TestCase):

    def setUp(self):
        # Construct synthetic beats:
        # B001: 0.0 - 5.0 (Strong hook)
        # B002: 5.0 - 15.0
        # B003: 15.0 - 25.0 (Strong cliffhanger) -> [0.0 - 25.0] = 25s (Valid)
        # B004: 25.0 - 40.0
        # B005: 40.0 - 55.0 (Strong cliffhanger) -> [0.0 - 55.0] = 55s (Valid)
        # B006: 55.0 - 70.0 (Strong cliffhanger) -> [0.0 - 70.0] = 70s (Invalid, > 60s)
        # B007: 70.0 - 80.0 (Strong hook)
        # B008: 80.0 - 90.0 (Strong cliffhanger) -> [70.0 - 90.0] = 20s (Valid)
        # B009: 90.0 - 100.0 (Strong hook)
        # B010: 100.0 - 105.0 (Strong cliffhanger) -> [90.0 - 105.0] = 15s (Invalid, < 20s)

        beats = []
        raw_intervals = [
            (0.0, 5.0),
            (5.0, 15.0),
            (15.0, 25.0),
            (25.0, 40.0),
            (40.0, 55.0),
            (55.0, 70.0),
            (70.0, 80.0),
            (80.0, 90.0),
            (90.0, 100.0),
            (100.0, 105.0),
        ]
        for i, (st, en) in enumerate(raw_intervals):
            b_id = f"B{i+1:03d}"
            ev = Event(type="dialogue", description=f"Beat {b_id} content", start=st, end=en, timestamp_source="given")
            beats.append(Beat(beat_id=b_id, start=st, end=en, events=[ev], timestamp_source="given"))

        self.episode = Episode(
            episode="Episode 1",
            episode_number=1,
            logline="Test",
            continuity_notes="None",
            beats=beats
        )

        # Hook signals
        self.hooks = {
            "B001": HookSignal("B001", 5, "Shock line", 4, "Cathedral reveal", True),
            "B007": HookSignal("B007", 4, "Threat", 3, "Confrontation", True),
            "B009": HookSignal("B009", 5, "Slap line", 5, "Slap action", True),
        }

        # Cliffhanger signals
        self.cliffhangers = {
            "B003": CliffhangerSignal("B003", 5, "Will she leave?", "High", False, "betrayal", True),
            "B005": CliffhangerSignal("B005", 5, "What will he do?", "High", False, "revenge", True),
            "B006": CliffhangerSignal("B006", 5, "Will the duel happen?", "High", False, "threat", True),
            "B008": CliffhangerSignal("B008", 4, "Is the secret out?", "High", False, "hidden_secret", True),
            "B010": CliffhangerSignal("B010", 4, "Who fired?", "High", False, "imminent_action", True),
        }

    def test_window_search_boundary_math_and_duration_filter(self):
        """Verify boundaries snap strictly to beat edges and duration 20s-60s filter works."""
        candidates = bidirectional_window_search(
            self.episode,
            self.hooks,
            self.cliffhangers,
            min_duration=20.0,
            max_duration=60.0
        )

        # Expected valid candidates:
        # 1. B001 -> B003: [0.0 - 25.0s] (25.0s) -> Valid
        # 2. B001 -> B005: [0.0 - 55.0s] (55.0s) -> Valid
        # (B001 -> B006 is 70s -> Dropped: > 60s)
        # 3. B007 -> B008: [70.0 - 90.0s] (20.0s) -> Valid
        # 4. B007 -> B010: [70.0 - 105.0s] (35.0s) -> Valid
        # (B009 -> B010 is 15s -> Dropped: < 20s)

        self.assertEqual(len(candidates), 4)

        cand1 = candidates[0]
        self.assertEqual(cand1.start_beat_id, "B001")
        self.assertEqual(cand1.end_beat_id, "B003")
        self.assertEqual(cand1.start_time, 0.0)
        self.assertEqual(cand1.end_time, 25.0)
        self.assertEqual(cand1.duration, 25.0)
        self.assertEqual(cand1.beat_ids, ["B001", "B002", "B003"])

        cand2 = candidates[1]
        self.assertEqual(cand2.start_beat_id, "B001")
        self.assertEqual(cand2.end_beat_id, "B005")
        self.assertEqual(cand2.start_time, 0.0)
        self.assertEqual(cand2.end_time, 55.0)
        self.assertEqual(cand2.duration, 55.0)
        self.assertEqual(cand2.beat_ids, ["B001", "B002", "B003", "B004", "B005"])

        cand3 = candidates[2]
        self.assertEqual(cand3.start_beat_id, "B007")
        self.assertEqual(cand3.end_beat_id, "B008")
        self.assertEqual(cand3.start_time, 70.0)
        self.assertEqual(cand3.end_time, 90.0)
        self.assertEqual(cand3.duration, 20.0)
        self.assertEqual(cand3.beat_ids, ["B007", "B008"])

        cand4 = candidates[3]
        self.assertEqual(cand4.start_beat_id, "B007")
        self.assertEqual(cand4.end_beat_id, "B010")
        self.assertEqual(cand4.start_time, 70.0)
        self.assertEqual(cand4.end_time, 105.0)
        self.assertEqual(cand4.duration, 35.0)
        self.assertEqual(cand4.beat_ids, ["B007", "B008", "B009", "B010"])

    def test_window_search_stops_at_scene_boundary(self):
        """
        Verify that window search stops expansion at scene boundaries.
        A strong hook in scene 1 and a strong cliffhanger in scene 2 MUST NEVER
        produce a candidate pair, even if their combined duration (e.g. 30s) is valid.
        """
        # Scene 1:
        # B001: 0.0 - 5.0 (Strong hook), scene_number=1
        # B002: 5.0 - 15.0, scene_number=1
        # Scene 2:
        # B003: 15.0 - 25.0, scene_number=2
        # B004: 25.0 - 30.0 (Strong cliffhanger), scene_number=2
        # Note: B001 to B004 is 30.0s (well within 20s-60s), but crosses scene 1 -> scene 2!
        ev1 = Event(type="action", description="Hook in scene 1", start=0.0, end=5.0, timestamp_source="given")
        b1 = Beat(beat_id="B001", start=0.0, end=5.0, events=[ev1], timestamp_source="given", scene_number=1, scene_heading="INT. HALL")

        ev2 = Event(type="dialogue", description="Walk in hall", start=5.0, end=15.0, timestamp_source="given")
        b2 = Beat(beat_id="B002", start=5.0, end=15.0, events=[ev2], timestamp_source="given", scene_number=1, scene_heading="INT. HALL")

        ev3 = Event(type="camera", description="Rain in yard", start=15.0, end=25.0, timestamp_source="given")
        b3 = Beat(beat_id="B003", start=15.0, end=25.0, events=[ev3], timestamp_source="given", scene_number=2, scene_heading="EXT. YARD")

        ev4 = Event(type="action", description="Cliffhanger in scene 2", start=25.0, end=30.0, timestamp_source="given")
        b4 = Beat(beat_id="B004", start=25.0, end=30.0, events=[ev4], timestamp_source="given", scene_number=2, scene_heading="EXT. YARD")

        ep = Episode(
            episode="Episode 2-Scene Test",
            episode_number=99,
            logline="Testing scene boundary stop",
            continuity_notes="",
            beats=[b1, b2, b3, b4]
        )

        hooks = {
            "B001": HookSignal("B001", 5, "Shock hook", 4, "Visual shock", True)
        }
        cliffhangers = {
            "B004": CliffhangerSignal("B004", 5, "Who escapes?", "High", False, "imminent_action", True)
        }

        candidates = bidirectional_window_search(ep, hooks, cliffhangers)

        # Assert that ZERO candidates are produced across the scene boundary
        self.assertEqual(len(candidates), 0, "Hook in scene 1 and cliffhanger in scene 2 must not produce a candidate across a scene boundary.")

    def test_two_segment_candidate_computed_properties(self):
        """
        Test that a 2-segment PromoCandidate resolves computed properties correctly across segment boundaries.
        Duration must be the sum of segment durations, not end_time - start_time.
        """
        seg1 = Segment(
            segment_index=1,
            episode_number=1,
            start_beat_id="B001",
            end_beat_id="B003",
            start_time=0.0,
            end_time=15.0,
            duration=15.0,
            beat_ids=["B001", "B002", "B003"],
            timestamp_source="given"
        )
        seg2 = Segment(
            segment_index=2,
            episode_number=2,  # Cross-episode stitch
            start_beat_id="B020",
            end_beat_id="B022",
            start_time=45.0,
            end_time=65.0,
            duration=20.0,
            beat_ids=["B020", "B021", "B022"],
            timestamp_source="given"
        )

        hook = HookSignal(
            beat_id="B001",
            dialogue_hook_score=5,
            dialogue_hook_rationale="Explosive line",
            visual_hook_score=4,
            visual_hook_rationale="Visual reveal",
            is_strong_hook=True
        )
        cliff = CliffhangerSignal(
            beat_id="B022",
            curiosity_gap_score=5,
            specific_question="Will he survive?",
            stakes_importance="High",
            defuses_tension=False,
            cliffhanger_type="threat",
            is_strong_cliffhanger=True
        )

        cand = PromoCandidate(
            candidate_id="CAND_MULTI_01",
            episode="Episode 1 & 2",
            episode_number=1,
            segments=[seg1, seg2],
            hook_signal=hook,
            cliffhanger_signal=cliff
        )

        # Assert computed properties
        self.assertEqual(cand.start_beat_id, "B001")
        self.assertEqual(cand.end_beat_id, "B022")
        self.assertEqual(cand.start_time, 0.0)
        self.assertEqual(cand.end_time, 65.0)
        self.assertEqual(cand.duration, 35.0)  # 15.0 + 20.0 = 35.0s, NOT 65.0 - 0.0 = 65.0s!
        self.assertEqual(cand.beat_ids, ["B001", "B002", "B003", "B020", "B021", "B022"])
        self.assertEqual(cand.timestamp_source, "given")
        self.assertEqual(len(cand.segments), 2)

        # Test to_dict serialization contains both computed properties and segments list
        d = cand.to_dict()
        self.assertEqual(d["start_beat_id"], "B001")
        self.assertEqual(d["end_beat_id"], "B022")
        self.assertEqual(d["duration"], 35.0)
        self.assertEqual(d["beat_ids"], ["B001", "B002", "B003", "B020", "B021", "B022"])
        self.assertEqual(len(d["segments"]), 2)
        self.assertEqual(d["segments"][0]["episode_number"], 1)
        self.assertEqual(d["segments"][1]["episode_number"], 2)

    def test_floor_relaxation_pairs_moderate_beats(self):
        """When strict floor yields < 5 pairs, floor relaxation activates and pairs score=3 beats."""
        # Setup: episode with only score=3 hooks and score=3 cliffhangers (0 strict pairs)
        b1 = Beat(beat_id="B001", start=0.0, end=5.0, events=[Event("dialogue", "Hook 3", 0.0, 5.0, "given")], timestamp_source="given", scene_number=1)
        b2 = Beat(beat_id="B002", start=5.0, end=30.0, events=[Event("dialogue", "Cliff 3", 5.0, 30.0, "given")], timestamp_source="given", scene_number=1)
        ep = Episode(episode="Ep 1", episode_number=1, logline="", continuity_notes="", beats=[b1, b2])

        hooks = {
            "B001": HookSignal("B001", dialogue_hook_score=3, dialogue_hook_rationale="", visual_hook_score=2, visual_hook_rationale="", is_strong_hook=False)
        }
        cliffs = {
            "B002": CliffhangerSignal("B002", curiosity_gap_score=3, specific_question="Q?", stakes_importance="Med", defuses_tension=False, cliffhanger_type="threat", is_strong_cliffhanger=False)
        }

        # Strict floor would produce 0 pairs, triggering floor relaxation
        cands = bidirectional_window_search(ep, hooks, cliffs, min_duration=20.0, max_duration=60.0)
        self.assertEqual(len(cands), 1)
        self.assertEqual(cands[0].start_beat_id, "B001")
        self.assertEqual(cands[0].end_beat_id, "B002")
        self.assertEqual(cands[0].duration, 30.0)


if __name__ == "__main__":
    unittest.main()
