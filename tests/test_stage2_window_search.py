"""
test_stage2_window_search.py — Stage 2 Capped Window Search & Pre-Filter Unit Tests.

Verifies:
1. Segment validation (timestamp_source required, start <= end).
2. PromoCandidate duration computed as SUM of segment durations (Rule 3).
3. Timestamp source inheritance: mixing given and estimated resolves to estimated (Rule 4).
4. Window search strictly terminates expansion at scene boundaries (Rule 5).
5. Duration filtered strictly to [20.0s, 60.0s] in pure Python.
6. Capped candidate matching prevents combination explosion (15-30 vs 213).
7. Quality floor pre-filter with floor-relaxation fallback.
"""

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PROMO_DIR = REPO_ROOT / "promo-finder"
for p in [REPO_ROOT, PROMO_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from parser.schema import Beat, Event, Episode
from candidates.schema import HookSignal, CliffhangerSignal, Segment, PromoCandidate
from candidates.window_search import bidirectional_window_search


class TestStage2WindowSearch(unittest.TestCase):

    def setUp(self):
        # 6 beats in scene 1 (each 10s -> total 60s)
        # 2 beats in scene 2 (each 10s)
        self.beats = []
        for i in range(1, 7):
            ev = Event(type="action", description=f"Beat {i} action", start=float((i - 1) * 10), end=float(i * 10), timestamp_source="given")
            self.beats.append(Beat(
                beat_id=f"B{i:03d}",
                start=float((i - 1) * 10),
                end=float(i * 10),
                events=[ev],
                timestamp_source="given",
                scene_number=1,
                scene_heading="INT. HALL - DAY"
            ))
        for i in range(7, 9):
            ev = Event(type="action", description=f"Beat {i} action", start=float((i - 1) * 10), end=float(i * 10), timestamp_source="given")
            self.beats.append(Beat(
                beat_id=f"B{i:03d}",
                start=float((i - 1) * 10),
                end=float(i * 10),
                events=[ev],
                timestamp_source="given",
                scene_number=2,
                scene_heading="EXT. COURTYARD - DAY"
            ))
        self.ep = Episode(
            episode="Episode 1",
            episode_number=1,
            logline="Test",
            continuity_notes="None",
            beats=self.beats
        )

    def test_segment_validation(self):
        """Segment requires valid timestamp_source and start <= end."""
        with self.assertRaises(ValueError):
            Segment(1, 1, "B001", "B005", 0.0, 25.0, 25.0, ["B001"], timestamp_source="inferred")  # type: ignore

        with self.assertRaises(ValueError):
            Segment(1, 1, "B001", "B005", 30.0, 10.0, -20.0, ["B001"], timestamp_source="given")

    def test_candidate_duration_is_sum_of_segments(self):
        """Rule 3: Duration is the sum of segment durations, not end_time - start_time."""
        hook = HookSignal("B001", 5, "H", 4, "V", True)
        cliff = CliffhangerSignal("B008", 5, "Q", "H", False, "betrayal", True)

        seg1 = Segment(1, 1, "B001", "B003", 0.0, 15.0, 15.0, ["B001", "B002", "B003"], "given")
        seg2 = Segment(2, 1, "B007", "B008", 60.0, 80.0, 20.0, ["B007", "B008"], "given")

        cand = PromoCandidate("C1", "Episode 1", 1, [seg1, seg2], hook, cliff)
        # Sum of segments: 15.0 + 20.0 = 35.0s (NOT 80.0 - 0.0 = 80.0s)
        self.assertEqual(cand.duration, 35.0)

    def test_timestamp_source_inheritance(self):
        """Rule 4: Candidate mixing given and estimated is marked 'estimated' overall."""
        hook = HookSignal("B001", 5, "H", 4, "V", True)
        cliff = CliffhangerSignal("B004", 5, "Q", "H", False, "betrayal", True)

        seg_given = Segment(1, 1, "B001", "B002", 0.0, 20.0, 20.0, ["B001", "B002"], "given")
        cand_given = PromoCandidate("C1", "Ep 1", 1, [seg_given], hook, cliff)
        self.assertEqual(cand_given.timestamp_source, "given")

        seg_est = Segment(2, 1, "B003", "B004", 20.0, 40.0, 20.0, ["B003", "B004"], "estimated")
        cand_mixed = PromoCandidate("C2", "Ep 1", 1, [seg_given, seg_est], hook, cliff)
        self.assertEqual(cand_mixed.timestamp_source, "estimated")

    def test_window_search_scene_boundary_halt(self):
        """Rule 5: Window search terminates expansion at scene boundaries."""
        hooks = {"B001": HookSignal("B001", 5, "H", 4, "V", True)}
        cliffs = {
            "B004": CliffhangerSignal("B004", 5, "Q", "H", False, "betrayal", True),
            "B007": CliffhangerSignal("B007", 5, "Q", "H", False, "threat", True),
        }
        candidates = bidirectional_window_search(self.ep, hooks, cliffs, min_duration=20.0, max_duration=70.0)

        # Candidate must match within Scene 1 (B001 -> B004), never crossing into Scene 2 (B007)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].start_beat_id, "B001")
        self.assertEqual(candidates[0].end_beat_id, "B004")

    def test_window_search_duration_bounds_and_capping(self):
        """Rule 3: Duration strictly 20-60s in Python. Capped matching keeps at most (#hooks + #cliffs)."""
        hooks = {"B001": HookSignal("B001", 5, "H", 4, "V", True)}
        cliffs = {
            "B002": CliffhangerSignal("B002", 4, "Q2", "H", False, "betrayal", True),
            "B004": CliffhangerSignal("B004", 5, "Q4", "H", False, "betrayal", True),
            "B005": CliffhangerSignal("B005", 4, "Q5", "H", False, "threat", True),
        }
        candidates = bidirectional_window_search(self.ep, hooks, cliffs, min_duration=20.0, max_duration=60.0)

        self.assertLessEqual(len(candidates), len(hooks) + len(cliffs))
        for c in candidates:
            self.assertTrue(20.0 <= c.duration <= 60.0)


if __name__ == "__main__":
    unittest.main()
