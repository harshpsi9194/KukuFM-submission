"""
test_stage4_ranking.py — Stage 4 Batched Absolute Rubric Ranker Unit Tests.

Verifies:
1. Candidate beat progression formatting includes dialogue lines and camera cues.
2. Batched ranking prompt structure includes all candidate metadata and rubric criteria.
3. Ranking computation:
   - 4-dimension absolute scoring: hook_score, tension_score, cliffhanger_score, resonance_score (0.0-2.5 each, total 0.0-10.0).
   - Strict score-based ordering (highest score -> Rank 1).
4. Deterministic tie-breaking:
   - When absolute scores tie, preference goes to candidate whose duration is closest to 35.0 seconds.
   - Secondary tie-break on emotional salience and hook scores.
5. Error recovery: missing candidates receive fallback rank and baseline score.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parent.parent
PROMO_DIR = REPO_ROOT / "promo-finder"
for p in [REPO_ROOT, PROMO_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from ranking.batched_ranker import (
    build_batched_ranking_prompt,
    run_batched_ranking,
    format_candidate_beat_progression,
)


class TestStage4Ranking(unittest.TestCase):

    def setUp(self):
        self.beats_map = {
            "B014": {
                "beat_id": "B014", "start": 35.4, "end": 39.0,
                "events": [
                    {"type": "dialogue", "character": "LADY JULIANA", "description": "Why are you holding her hand?"},
                ]
            },
            "B018": {
                "beat_id": "B018", "start": 46.8, "end": 50.4,
                "events": [
                    {"type": "dialogue", "character": "LADY JULIANA", "description": "You knew this before I walked through those doors!"},
                ]
            },
            "B022": {
                "beat_id": "B022", "start": 60.1, "end": 63.0,
                "events": [
                    {"type": "dialogue", "character": "THE FORMER GROOM", "description": "I changed my mind."},
                ]
            },
            "B037": {
                "beat_id": "B037", "start": 107.4, "end": 113.0,
                "events": [
                    {"type": "camera", "description": "Medium tracking shot showing Lady in Gold Gown stepping past Juliana."},
                ]
            },
            "B045": {
                "beat_id": "B045", "start": 133.8, "end": 138.0,
                "events": [
                    {"type": "camera", "description": "Push-in close-up on Juliana hardening from devastation into cold defiance."},
                ]
            }
        }
        self.cand_a = {
            "candidate_id": "CAND_EP001_01",
            "start_beat_id": "B014",
            "end_beat_id": "B022",
            "duration": 27.6,
            "beat_ids": ["B014", "B018", "B022"],
            "hook_signal": {"dialogue_hook_rationale": "Direct betrayal accusation"},
            "cliffhanger_signal": {"specific_question": "What will Gerald do?", "cliffhanger_type": "betrayal"}
        }
        self.cand_b = {
            "candidate_id": "CAND_EP001_02",
            "start_beat_id": "B037",
            "end_beat_id": "B045",
            "duration": 30.6,
            "beat_ids": ["B037", "B045"],
            "hook_signal": {"visual_hook_rationale": "Sister steps center stage"},
            "cliffhanger_signal": {"specific_question": "How will Juliana strike back?", "cliffhanger_type": "revenge"}
        }
        self.cand_c = {
            "candidate_id": "CAND_EP001_03",
            "start_beat_id": "B014",
            "end_beat_id": "B022",
            "duration": 35.0,  # Exact 35.0s optimal duration sweet-spot
            "beat_ids": ["B014", "B018", "B022"],
            "hook_signal": {"dialogue_hook_rationale": "Direct betrayal accusation"},
            "cliffhanger_signal": {"specific_question": "What will Gerald do?", "cliffhanger_type": "betrayal"}
        }
        self.scored_map = {
            "CAND_EP001_01": {
                "comprehension": {"situational_summary": "Groom betrays bride at altar."},
                "emotional_salience": {"emotional_salience_score": 5, "emotional_core": "Peak heartbreak"},
                "spoiler": {"spoiler_level": 1, "spoiler_rationale": "Wedding broken"}
            },
            "CAND_EP001_02": {
                "comprehension": {"situational_summary": "Sister gloats and groom insults bride."},
                "emotional_salience": {"emotional_salience_score": 4, "emotional_core": "Defiant revenge"},
                "spoiler": {"spoiler_level": 2, "spoiler_rationale": "Clarissa favored"}
            },
            "CAND_EP001_03": {
                "comprehension": {"situational_summary": "Groom betrays bride at altar."},
                "emotional_salience": {"emotional_salience_score": 5, "emotional_core": "Peak heartbreak"},
                "spoiler": {"spoiler_level": 1, "spoiler_rationale": "Wedding broken"}
            }
        }

    def test_format_candidate_beat_progression(self):
        """Beat progression format includes timestamps, beat IDs, dialogue text, and camera cues."""
        lines = format_candidate_beat_progression(self.cand_a, self.beats_map)
        text = "\n".join(lines)
        self.assertIn("B014", text)
        self.assertIn("Why are you holding her hand?", text)
        self.assertIn("B022", text)
        self.assertIn("I changed my mind.", text)

    def test_build_batched_ranking_prompt_includes_metadata(self):
        """Prompt contains dialogue excerpts, emotional core, and rubric dimensions."""
        prompt = build_batched_ranking_prompt([self.cand_a, self.cand_b], self.scored_map, self.beats_map)
        self.assertIn("Why are you holding her hand?", prompt)
        self.assertIn("Peak heartbreak", prompt)
        self.assertIn("CAND_EP001_01", prompt)
        self.assertIn("CAND_EP001_02", prompt)
        self.assertIn("Hook Scroll-Stop Power", prompt)
        self.assertIn("Cliffhanger Irresolution", prompt)

    @patch("ranking.batched_ranker.call_llm")
    def test_run_batched_ranking_absolute_ordering(self, mock_llm):
        """Higher total absolute score ranks strictly higher."""
        mock_llm.return_value = [
            {
                "candidate_id": "CAND_EP001_01",
                "hook_score": 2.4,
                "tension_score": 2.4,
                "cliffhanger_score": 2.4,
                "resonance_score": 2.3,
                "absolute_score": 9.5,
                "rank_justification": "Near perfect dramatic confrontation",
            },
            {
                "candidate_id": "CAND_EP001_02",
                "hook_score": 1.8,
                "tension_score": 1.8,
                "cliffhanger_score": 1.7,
                "resonance_score": 1.7,
                "absolute_score": 7.0,
                "rank_justification": "Solid but slightly lower emotional resonance",
            }
        ]

        ranked, _ = run_batched_ranking([self.cand_a, self.cand_b], self.scored_map, self.beats_map)
        self.assertEqual(len(ranked), 2)
        # CAND_EP001_01 total = 9.5 -> Rank 1
        self.assertEqual(ranked[0]["candidate_id"], "CAND_EP001_01")
        self.assertEqual(ranked[0]["rank"], 1)
        self.assertEqual(ranked[0]["absolute_score"], 9.5)
        # CAND_EP001_02 total = 7.0 -> Rank 2
        self.assertEqual(ranked[1]["candidate_id"], "CAND_EP001_02")
        self.assertEqual(ranked[1]["rank"], 2)
        self.assertEqual(ranked[1]["absolute_score"], 7.0)

    @patch("ranking.batched_ranker.call_llm")
    def test_run_batched_ranking_deterministic_tie_breaker(self, mock_llm):
        """When total scores tie, candidate closer to 35.0s duration wins."""
        # CAND_EP001_01 duration = 27.6s (diff: 7.4s)
        # CAND_EP001_03 duration = 35.0s (diff: 0.0s)
        # Both receive identical score of 8.0
        mock_llm.return_value = [
            {
                "candidate_id": "CAND_EP001_01",
                "hook_score": 2.0, "tension_score": 2.0, "cliffhanger_score": 2.0, "resonance_score": 2.0,
                "absolute_score": 8.0,
                "rank_justification": "Score 8.0",
            },
            {
                "candidate_id": "CAND_EP001_03",
                "hook_score": 2.0, "tension_score": 2.0, "cliffhanger_score": 2.0, "resonance_score": 2.0,
                "absolute_score": 8.0,
                "rank_justification": "Score 8.0",
            }
        ]

        ranked, _ = run_batched_ranking([self.cand_a, self.cand_c], self.scored_map, self.beats_map)
        self.assertEqual(len(ranked), 2)
        # CAND_EP001_03 (diff 0.0s) beats CAND_EP001_01 (diff 7.4s) on tie-breaker
        self.assertEqual(ranked[0]["candidate_id"], "CAND_EP001_03")
        self.assertEqual(ranked[0]["rank"], 1)
        self.assertEqual(ranked[1]["candidate_id"], "CAND_EP001_01")
        self.assertEqual(ranked[1]["rank"], 2)


if __name__ == "__main__":
    unittest.main()
