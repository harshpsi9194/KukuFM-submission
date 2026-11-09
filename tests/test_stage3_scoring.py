"""
test_stage3_scoring.py — Stage 3 Merged 3-in-1 Candidate Scorer Unit Tests.

Verifies:
1. Candidate formatting: retains CAMERA, ACTION, and DIALOGUE blocks with character tags.
2. Comprehension Gate (Rule 6):
   - Binary gate: passes_gate = opening_legibility AND arc_legibility.
   - Retrospective justification is forbidden (opening legibility judged on beat 0 alone).
   - Hallucination rejection: code overrides LLM if it returns passes_gate=True with opening_legibility=False.
   - Failed candidates are excluded from ranking entirely.
3. Emotional Salience Scoring (1-5 scale) + emotional core extraction.
4. Spoiler Level Scoring (0-3 scale, Rule 7: reported, never auto-dropped).
5. Robust fallback handling for missing candidates or failed batches.
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

from scoring.merged_scorer import (
    format_candidate_for_scoring,
    build_batch_scoring_prompt,
    score_candidate_batch,
    score_all_candidates_concurrent,
)


class TestStage3Scoring(unittest.TestCase):

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

    def test_format_candidate_preserves_events_and_characters(self):
        """Format candidate output must include CAMERA and DIALOGUE events distinctly."""
        text = format_candidate_for_scoring(self.candidate, self.beats_map)
        self.assertIn("Why are you holding her hand?", text)
        self.assertIn("LADY JULIANA", text)
        self.assertIn("Close-up insert of Gerald's fingers", text)
        self.assertIn("I changed my mind.", text)
        self.assertIn("THE FORMER GROOM", text)
        self.assertIn("B014", text)
        self.assertIn("B022", text)

    def test_build_batch_scoring_prompt(self):
        """Prompt contains candidate text and clear schema requirements."""
        prompt = build_batch_scoring_prompt([self.candidate], self.beats_map)
        self.assertIn("CAND_EP001_01", prompt)
        self.assertIn("opening_legibility", prompt)
        self.assertIn("emotional_salience_score", prompt)
        self.assertIn("spoiler_level", prompt)

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
        self.assertEqual(res["emotional_salience"]["emotional_core"], "Brutal public heartbreak")
        # Rule 7: spoiler level reported
        self.assertEqual(res["spoiler"]["spoiler_level"], 1)

    @patch("scoring.merged_scorer.call_llm")
    def test_score_candidate_batch_rejects_hallucinated_gate(self, mock_llm):
        """Rule 6: If opening_legibility is False, code strictly overrides passes_gate to False."""
        mock_llm.return_value = [
            {
                "candidate_id": "CAND_EP001_01",
                "opening_legibility": False,
                "opening_rationale": "Ungrounded mid-sentence pronoun",
                "arc_legibility": True,
                "arc_rationale": "Conflict resolves clearly",
                "passes_comprehension_gate": True,  # Hallucinated True by LLM!
                "situational_summary": "Confusing opening.",
                "emotional_salience_score": 4,
                "emotional_core": "Shock",
                "spoiler_level": 0,
                "spoiler_rationale": "None",
            }
        ]

        results = score_candidate_batch([self.candidate], self.beats_map, episode_number=1)
        self.assertEqual(len(results), 1)
        res = results[0]
        # Invariant: passes_gate MUST be False
        self.assertFalse(res["comprehension"]["passes_gate"])

    @patch("scoring.merged_scorer.call_llm")
    def test_missing_candidate_fallback(self, mock_llm):
        """Candidates missing from batch response receive fail-safe record."""
        mock_llm.return_value = []

        results = score_candidate_batch([self.candidate], self.beats_map, episode_number=1)
        self.assertEqual(len(results), 1)
        res = results[0]
        self.assertFalse(res["comprehension"]["passes_gate"])
        self.assertEqual(res["comprehension"]["situational_summary"], "Scoring failed")
        self.assertEqual(res["emotional_salience"]["emotional_salience_score"], 1)
        self.assertEqual(res["spoiler"]["spoiler_level"], 0)

    def test_score_all_candidates_empty_list(self):
        """Zero candidates returns empty list without ZeroDivisionError."""
        res = score_all_candidates_concurrent([], self.beats_map, episode_number=1)
        self.assertEqual(res, [])

    @patch("scoring.merged_scorer.call_llm")
    def test_score_candidate_batch_dict_wrapped_response(self, mock_llm):
        """Unwraps dictionary responses containing 'candidates' or 'scored_candidates'."""
        mock_llm.return_value = {
            "scored_candidates": [
                {
                    "candidate_id": "CAND_EP001_01",
                    "opening_legibility": True,
                    "opening_rationale": "Clear start",
                    "arc_legibility": True,
                    "arc_rationale": "Clear conflict",
                    "passes_comprehension_gate": True,
                    "situational_summary": "Summary",
                    "emotional_salience_score": 4,
                    "emotional_core": "Tension",
                    "spoiler_level": 1,
                    "spoiler_rationale": "Minor",
                }
            ]
        }
        results = score_candidate_batch([self.candidate], self.beats_map, episode_number=1)
        self.assertEqual(len(results), 1)
        self.assertTrue(results[0]["comprehension"]["passes_gate"])

    @patch("scoring.merged_scorer.score_candidate_batch")
    def test_score_all_candidates_worker_failure_fallback(self, mock_batch):
        """Worker thread exceptions do not drop candidate IDs or crash the pipeline."""
        mock_batch.side_effect = RuntimeError("Worker failure / timeout")
        results = score_all_candidates_concurrent([self.candidate], self.beats_map, episode_number=1, batch_size=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["candidate_id"], "CAND_EP001_01")
        self.assertFalse(results[0]["comprehension"]["passes_gate"])
        self.assertIn("Batch scoring failed", results[0]["comprehension"]["opening_rationale"])


if __name__ == "__main__":
    unittest.main()
