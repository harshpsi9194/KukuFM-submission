"""
test_merged_scorer.py — Unit tests for Stage 3 merged 3-in-1 scorer.
"""

import unittest
from unittest.mock import patch
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from scoring.merged_scorer import (
    format_candidate_for_scoring,
    build_batch_scoring_prompt,
    score_candidate_batch,
    score_all_candidates_concurrent,
)


class TestMergedScorer(unittest.TestCase):

    def setUp(self):
        self.beats_map = {
            "B014": {
                "beat_id": "B014",
                "start": 35.4,
                "end": 39.0,
                "events": [
                    {"type": "dialogue", "character": "LADY JULIANA", "description": "Why are you holding her hand?"},
                    {"type": "camera", "description": "Close-up insert of Gerald's fingers firmly interlocked with rival."},
                ]
            },
            "B022": {
                "beat_id": "B022",
                "start": 60.1,
                "end": 63.0,
                "events": [
                    {"type": "dialogue", "character": "THE FORMER GROOM", "description": "I changed my mind."},
                    {"type": "camera", "description": "Tight close-up on Gerald's hardened profile."},
                ]
            }
        }
        self.candidate = {
            "candidate_id": "CAND_EP001_01",
            "start_beat_id": "B014",
            "end_beat_id": "B022",
            "start_time": 35.4,
            "end_time": 63.0,
            "duration": 27.6,
            "beat_ids": ["B014", "B022"],
            "hook_signal": {
                "dialogue_hook_score": 4,
                "dialogue_hook_rationale": "Direct betrayal accusation",
                "visual_hook_score": 5,
                "visual_hook_rationale": "Insert interlocked fingers",
            },
            "cliffhanger_signal": {
                "curiosity_gap_score": 4,
                "specific_question": "How will Juliana respond to Gerald breaking vows?",
                "cliffhanger_type": "betrayal",
            }
        }

    def test_format_candidate_preserves_events(self):
        """Format candidate output must include CAMERA and DIALOGUE events distinctly."""
        text = format_candidate_for_scoring(self.candidate, self.beats_map)
        self.assertIn("Why are you holding her hand?", text)
        self.assertIn("Close-up insert of Gerald's fingers", text)
        self.assertIn("I changed my mind.", text)
        self.assertIn("B014", text)
        self.assertIn("B022", text)

    @patch("scoring.merged_scorer.call_llm")
    def test_score_candidate_batch_enforces_binary_gate(self, mock_llm):
        """Comprehension gate passes ONLY IF both opening_legibility and arc_legibility are True."""
        mock_llm.return_value = [
            {
                "candidate_id": "CAND_EP001_01",
                "opening_legibility": True,
                "opening_rationale": "Clear cathedral confrontation",
                "arc_legibility": True,
                "arc_rationale": "Self-contained wedding betrayal",
                "passes_comprehension_gate": True,
                "situational_summary": "Bride confronts unfaithful groom at altar.",
                "emotional_salience_score": 5,
                "emotional_core": "Brutal public heartbreak",
                "spoiler_level": 1,
                "spoiler_rationale": "Inciting incident",
            }
        ]

        results = score_candidate_batch([self.candidate], self.beats_map, episode_number=1)
        self.assertEqual(len(results), 1)
        res = results[0]
        self.assertTrue(res["comprehension"]["passes_gate"])
        self.assertEqual(res["emotional_salience"]["emotional_salience_score"], 5)
        self.assertEqual(res["spoiler"]["spoiler_level"], 1)

    @patch("scoring.merged_scorer.call_llm")
    def test_score_candidate_batch_rejects_hallucinated_gate(self, mock_llm):
        """If opening_legibility is False but LLM hallucinates passes_comprehension_gate=True, Python overrides to False."""
        mock_llm.return_value = [
            {
                "candidate_id": "CAND_EP001_01",
                "opening_legibility": False,
                "opening_rationale": "Ungrounded mid-sentence start",
                "arc_legibility": True,
                "arc_rationale": "Arc is fine",
                "passes_comprehension_gate": True,  # Hallucinated True by LLM!
                "situational_summary": "Confusing start.",
                "emotional_salience_score": 3,
                "emotional_core": "Mild drama",
                "spoiler_level": 1,
                "spoiler_rationale": "None",
            }
        ]

        results = score_candidate_batch([self.candidate], self.beats_map, episode_number=1)
        self.assertEqual(len(results), 1)
        res = results[0]
        # Python strict binary invariant must override hallucination
        self.assertFalse(res["comprehension"]["passes_gate"])

    @patch("scoring.merged_scorer.call_llm")
    def test_missing_candidate_fallback(self, mock_llm):
        """Candidates missing from batch response must receive safe fallback without crashing."""
        mock_llm.return_value = []  # Empty response from LLM

        results = score_candidate_batch([self.candidate], self.beats_map, episode_number=1)
        self.assertEqual(len(results), 1)
        res = results[0]
        self.assertFalse(res["comprehension"]["passes_gate"])
        self.assertEqual(res["comprehension"]["situational_summary"], "Scoring failed")


if __name__ == "__main__":
    unittest.main()
