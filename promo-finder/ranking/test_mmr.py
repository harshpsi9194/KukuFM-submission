"""
test_mmr.py — Unit tests for Stage 5 MMR portfolio selection.
"""

import unittest
import numpy as np
from ranking.mmr import (
    build_narrative_signature,
    compute_cosine_similarity_matrix,
    select_mmr_portfolio,
)


class TestMMRPortfolio(unittest.TestCase):

    def test_build_narrative_signature(self):
        cand = {
            "hook_signal": {
                "dialogue_hook_rationale": "Accusation line",
                "visual_hook_rationale": "Insert fingers",
            },
            "cliffhanger_signal": {
                "specific_question": "What happens next?",
                "cliffhanger_type": "betrayal",
            }
        }
        scored = {
            "comprehension": {
                "situational_summary": "A bride discovers groom betrayal."
            }
        }
        sig = build_narrative_signature(cand, scored)
        self.assertIn("Accusation line", sig)
        self.assertIn("Insert fingers", sig)
        self.assertIn("What happens next?", sig)
        self.assertIn("betrayal", sig)
        self.assertIn("A bride discovers groom betrayal.", sig)

    def test_compute_cosine_similarity_matrix(self):
        embs = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0]
        ], dtype=np.float32)
        sim = compute_cosine_similarity_matrix(embs)
        self.assertEqual(sim.shape, (3, 3))
        self.assertAlmostEqual(sim[0, 0], 1.0)
        self.assertAlmostEqual(sim[0, 1], 0.0)
        self.assertAlmostEqual(sim[0, 2], 1.0)

    def test_select_mmr_portfolio_hard_dedup(self):
        rankings = [
            {"candidate_id": "C1", "rank": 1, "recommended_boundary": "B1->B2", "narrative_thread": "T1", "duration": 25.0},
            {"candidate_id": "C2", "rank": 2, "recommended_boundary": "B1->B3", "narrative_thread": "T1", "duration": 40.0},
            {"candidate_id": "C3", "rank": 3, "recommended_boundary": "B4->B5", "narrative_thread": "T2", "duration": 30.0},
        ]
        # Sim: C1 and C2 are 0.95 (near-duplicates), C1 and C3 are 0.70, C2 and C3 are 0.70
        sim = np.array([
            [1.00, 0.95, 0.70],
            [0.95, 1.00, 0.70],
            [0.70, 0.70, 1.00]
        ], dtype=np.float32)

        portfolio, disq = select_mmr_portfolio(
            rankings=rankings,
            sim_matrix=sim,
            k=2,
            lambda_param=0.55,
            max_sim_threshold=0.90,
        )

        self.assertEqual(len(portfolio), 2)
        # Pick 1 should be C1 (Rank 1)
        self.assertEqual(portfolio[0]["candidate_id"], "C1")
        # Pick 2 should be C3 (C2 is disqualified due to Sim=0.95 >= 0.90)
        self.assertEqual(portfolio[1]["candidate_id"], "C3")
        self.assertEqual(len(disq), 1)
        self.assertEqual(disq[0]["candidate_id"], "C2")
        self.assertEqual(disq[0]["duplicate_of"], "C1")


if __name__ == "__main__":
    unittest.main()

