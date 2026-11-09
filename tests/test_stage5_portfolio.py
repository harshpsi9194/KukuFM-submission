"""
test_stage5_portfolio.py — Stage 5 MMR Portfolio Selection & Boundary Cleanliness Unit Tests.

Verifies:
1. Narrative signature synthesis from hook, cliffhanger, and situational summary.
2. Cosine similarity matrix computation over embedding representations.
3. MMR Portfolio Selection (Rule 9):
   - Balances rank quality with novelty using lambda parameter.
   - Hard deduplication: candidates with cosine similarity >= max_sim_threshold (0.90) are disqualified.
   - Disqualified candidates are logged with duplicate_of pointer and similarity.
4. Boundary Cleanliness Check:
   - Evaluates opening and closing beat cleanliness without LLM calls.
"""

import sys
import unittest
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
PROMO_DIR = REPO_ROOT / "promo-finder"
for p in [REPO_ROOT, PROMO_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from ranking.mmr import (
    build_narrative_signature,
    compute_cosine_similarity_matrix,
    select_mmr_portfolio,
)
from run_pipeline import evaluate_boundary_cleanliness


class TestStage5Portfolio(unittest.TestCase):

    def test_build_narrative_signature_structure(self):
        """Narrative signature contains hook rationale, cliffhanger question, type, and situational summary."""
        cand = {
            "hook_signal": {
                "dialogue_hook_rationale": "Juliana confronts Gerald on the dais.",
                "visual_hook_rationale": "Close-up insert of bridal bouquet falling.",
            },
            "cliffhanger_signal": {
                "specific_question": "Will Lord Raymond accept the broken bride?",
                "cliffhanger_type": "betrayal",
            }
        }
        scored = {
            "comprehension": {
                "situational_summary": "At the wedding altar, Gerald renounces his bride for Clarissa."
            }
        }
        sig = build_narrative_signature(cand, scored)
        self.assertIn("Juliana confronts Gerald", sig)
        self.assertIn("bridal bouquet falling", sig)
        self.assertIn("Will Lord Raymond accept", sig)
        self.assertIn("betrayal", sig)
        self.assertIn("Gerald renounces his bride", sig)

    def test_cosine_similarity_matrix_properties(self):
        """Cosine similarity matrix is symmetric with diagonal 1.0."""
        # Exact unit vectors (0.6^2 + 0.8^2 = 1.0)
        embs = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.6, 0.8, 0.0]
        ], dtype=np.float32)
        sim = compute_cosine_similarity_matrix(embs)
        self.assertEqual(sim.shape, (3, 3))
        # Diagonal is 1.0
        self.assertAlmostEqual(sim[0, 0], 1.0, places=5)
        self.assertAlmostEqual(sim[1, 1], 1.0, places=5)
        self.assertAlmostEqual(sim[2, 2], 1.0, places=5)
        # Symmetry
        self.assertAlmostEqual(sim[0, 1], sim[1, 0], places=5)
        self.assertAlmostEqual(sim[0, 2], sim[2, 0], places=5)
        # Orthogonal vectors have similarity 0.0
        self.assertAlmostEqual(sim[0, 1], 0.0, places=5)
        # Dot product with [0.6, 0.8, 0.0] is 0.6
        self.assertAlmostEqual(sim[0, 2], 0.6, places=5)

    def test_select_mmr_portfolio_hard_deduplication(self):
        """Rule 9: Candidates with similarity >= 0.90 to any selected candidate are disqualified."""
        rankings = [
            {"candidate_id": "C1", "rank": 1, "recommended_boundary": "B001->B005", "duration": 25.0},
            {"candidate_id": "C2", "rank": 2, "recommended_boundary": "B001->B006", "duration": 30.0},  # Duplicate of C1
            {"candidate_id": "C3", "rank": 3, "recommended_boundary": "B010->B015", "duration": 35.0},  # Diverse
        ]
        # Similarity matrix: C1 and C2 have 0.96 similarity (near identical)
        sim_matrix = np.array([
            [1.00, 0.96, 0.30],
            [0.96, 1.00, 0.35],
            [0.30, 0.35, 1.00]
        ], dtype=np.float32)

        portfolio, disqualified = select_mmr_portfolio(
            rankings=rankings,
            sim_matrix=sim_matrix,
            k=2,
            lambda_param=0.6,
            max_sim_threshold=0.90
        )

        self.assertEqual(len(portfolio), 2)
        # First pick is C1 (highest rank)
        self.assertEqual(portfolio[0]["candidate_id"], "C1")
        # Second pick must be C3 because C2 has similarity 0.96 >= 0.90
        self.assertEqual(portfolio[1]["candidate_id"], "C3")

        # C2 should be in disqualified list
        self.assertEqual(len(disqualified), 1)
        self.assertEqual(disqualified[0]["candidate_id"], "C2")
        self.assertEqual(disqualified[0]["duplicate_of"], "C1")
        self.assertGreaterEqual(disqualified[0]["similarity"], 0.90)

    def test_evaluate_boundary_cleanliness_action_start(self):
        """Opening beat with action/camera cue receives 5.0 cleanliness score."""
        beats_map = {
            "B001": {
                "events": [
                    {"type": "action", "description": "Juliana steps out of the carriage."}
                ]
            },
            "B005": {
                "events": [
                    {"type": "dialogue", "character": "GERALD", "description": "Who gives you the right to question me?"}
                ]
            }
        }
        open_score, cliff_score, total = evaluate_boundary_cleanliness(beats_map, "B001", "B005")
        self.assertEqual(open_score, 5.0)
        self.assertEqual(cliff_score, 5.0)  # Ends with question mark
        self.assertEqual(total, 5.0)

    def test_select_mmr_portfolio_empty_rankings(self):
        """Empty rankings returns empty portfolio and empty disqualification log without crashing."""
        sim_matrix = np.empty((0, 0), dtype=np.float32)
        portfolio, disqualifications = select_mmr_portfolio([], sim_matrix, k=5)
        self.assertEqual(portfolio, [])
        self.assertEqual(disqualifications, [])

    def test_compute_embeddings_empty_signatures(self):
        """Empty signatures list returns empty (0, 384) numpy array without AxisError."""
        from ranking.mmr import compute_embeddings
        embs = compute_embeddings([])
        self.assertEqual(embs.shape, (0, 384))

    def test_compute_cosine_similarity_empty(self):
        """Empty normalized embeddings returns empty (0, 0) similarity matrix."""
        sim_matrix = compute_cosine_similarity_matrix(np.empty((0, 384), dtype=np.float32))
        self.assertEqual(sim_matrix.shape, (0, 0))


if __name__ == "__main__":
    unittest.main()
