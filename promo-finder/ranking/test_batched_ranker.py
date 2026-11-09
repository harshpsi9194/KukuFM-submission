"""
test_batched_ranker.py — Unit tests for Stage 4 Pure Absolute Rubric Ranking.
"""

import unittest
from unittest.mock import patch
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from ranking.batched_ranker import (
    build_batched_ranking_prompt,
    run_batched_ranking,
    format_candidate_beat_progression,
)


class TestBatchedRanker(unittest.TestCase):

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
            }
        }

    def test_build_batched_ranking_prompt_includes_beats(self):
        """Prompt must incorporate dramatic highlights from beats_map."""
        prompt = build_batched_ranking_prompt([self.cand_a], self.scored_map, self.beats_map)
        self.assertIn("Why are you holding her hand?", prompt)
        self.assertIn("I changed my mind.", prompt)
        self.assertIn("Peak heartbreak", prompt)
        self.assertIn("CAND_EP001_01", prompt)

    @patch("ranking.batched_ranker.call_llm")
    def test_run_batched_ranking_absolute_scoring_and_ordering(self, mock_llm):
        """Higher absolute score must strictly rank ahead without pairwise matches."""
        mock_llm.return_value = [
            {
                "candidate_id": "CAND_EP001_01",
                "hook_score": 2.4,
                "tension_score": 2.4,
                "cliffhanger_score": 2.4,
                "resonance_score": 2.1,
                "absolute_score": 9.3,
                "rank_justification": "Peak wedding betrayal with instant hook and icy cut.",
            },
            {
                "candidate_id": "CAND_EP001_02",
                "hook_score": 2.0,
                "tension_score": 2.1,
                "cliffhanger_score": 2.4,
                "resonance_score": 1.7,
                "absolute_score": 8.2,
                "rank_justification": "Strong revenge transformation.",
            }
        ]

        ranked, tiebreaks = run_batched_ranking(
            [self.cand_a, self.cand_b],
            self.scored_map,
            self.beats_map,
            episode_number=1
        )
        self.assertEqual(len(ranked), 2)
        self.assertEqual(len(tiebreaks), 0)  # Zero pairwise tiebreaks
        self.assertEqual(ranked[0]["candidate_id"], "CAND_EP001_01")
        self.assertEqual(ranked[0]["rank"], 1)
        self.assertEqual(ranked[0]["absolute_score"], 9.3)
        self.assertEqual(ranked[1]["candidate_id"], "CAND_EP001_02")
        self.assertEqual(ranked[1]["rank"], 2)
        self.assertEqual(ranked[1]["absolute_score"], 8.2)

    @patch("ranking.batched_ranker.call_llm")
    def test_run_batched_ranking_deterministic_tie_breaker(self, mock_llm):
        """Tied absolute scores must be broken deterministically by duration closeness to 35.0s."""
        # Both candidates score 8.5
        # cand_a duration = 27.6s (diff = 7.4s)
        # cand_b duration = 30.6s (diff = 4.4s) -> cand_b is closer to 35.0s, so cand_b wins tie!
        mock_llm.return_value = [
            {
                "candidate_id": "CAND_EP001_01",
                "hook_score": 2.2, "tension_score": 2.1, "cliffhanger_score": 2.2, "resonance_score": 2.0,
                "absolute_score": 8.5, "rank_justification": "Tied prospect A",
            },
            {
                "candidate_id": "CAND_EP001_02",
                "hook_score": 2.1, "tension_score": 2.2, "cliffhanger_score": 2.2, "resonance_score": 2.0,
                "absolute_score": 8.5, "rank_justification": "Tied prospect B",
            }
        ]

        ranked, _ = run_batched_ranking(
            [self.cand_a, self.cand_b],
            self.scored_map,
            self.beats_map,
            episode_number=1
        )
        # cand_b is 30.6s (|30.6 - 35.0| = 4.4s) vs cand_a 27.6s (|27.6 - 35.0| = 7.4s)
        self.assertEqual(ranked[0]["candidate_id"], "CAND_EP001_02")
        self.assertEqual(ranked[0]["rank"], 1)
        self.assertEqual(ranked[1]["candidate_id"], "CAND_EP001_01")
        self.assertEqual(ranked[1]["rank"], 2)


if __name__ == "__main__":
    unittest.main()
