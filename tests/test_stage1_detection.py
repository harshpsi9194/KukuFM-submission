"""
test_stage1_detection.py — Stage 1 Hook & Cliffhanger Detection Unit Tests.

Verifies:
1. HookSignal & CliffhangerSignal data contracts and serialization.
2. Disentangled visual vs dialogue hook scoring.
3. Curiosity gap, specific question, over-reveal guard (defuses_tension), taxonomy.
4. Token estimation and cost calculation formulas.
5. Markdown code-block JSON extraction and fallback parsing.
"""

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PROMO_DIR = REPO_ROOT / "promo-finder"
for p in [REPO_ROOT, PROMO_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from candidates.schema import HookSignal, CliffhangerSignal
from candidates.llm_client import extract_json_payload, estimate_tokens, calculate_cost


class TestStage1Detection(unittest.TestCase):

    def test_hook_signal_disentangled_dimensions(self):
        """Dialogue hook and visual hook must be scored and stored separately."""
        hook = HookSignal(
            beat_id="B001",
            dialogue_hook_score=0,
            dialogue_hook_rationale="No dialogue present.",
            visual_hook_score=4,
            visual_hook_rationale="Cathedral doors part on bride in white silk.",
            is_strong_hook=True,
        )
        self.assertEqual(hook.dialogue_hook_score, 0)
        self.assertEqual(hook.visual_hook_score, 4)
        self.assertTrue(hook.is_strong_hook)

        d = hook.to_dict()
        self.assertIn("dialogue_hook_score", d)
        self.assertIn("visual_hook_score", d)
        self.assertIn("dialogue_hook_rationale", d)
        self.assertIn("visual_hook_rationale", d)

    def test_cliffhanger_signal_over_reveal_guard(self):
        """Over-reveal guard (defuses_tension) prevents defused beats from being strong cliffhangers."""
        cliff_strong = CliffhangerSignal(
            beat_id="B015",
            curiosity_gap_score=5,
            specific_question="How will Juliana react to the sister marrying her groom?",
            stakes_importance="High",
            defuses_tension=False,
            cliffhanger_type="betrayal",
            is_strong_cliffhanger=True,
        )
        self.assertTrue(cliff_strong.is_strong_cliffhanger)
        self.assertFalse(cliff_strong.defuses_tension)

        # Defused beat
        cliff_defused = CliffhangerSignal(
            beat_id="B020",
            curiosity_gap_score=4,
            specific_question="Does Gerald love Clarissa?",
            stakes_importance="Medium",
            defuses_tension=True,
            cliffhanger_type="none",
            is_strong_cliffhanger=False,
        )
        self.assertTrue(cliff_defused.defuses_tension)
        self.assertFalse(cliff_defused.is_strong_cliffhanger)

    def test_token_and_cost_estimation(self):
        """Token and cost estimation calculations."""
        text = "This is a test prompt containing eight words."
        tokens = estimate_tokens(text)
        self.assertGreater(tokens, 0)

        # Calculate cost for 1,000,000 input tokens and 1,000,000 output tokens
        # Default Gemini Flash pricing: $0.075 / 1M in, $0.30 / 1M out -> $0.375 total
        cost = calculate_cost(1_000_000, 1_000_000)
        self.assertAlmostEqual(cost, 0.375, places=4)

    def test_extract_json_payload_markdown_blocks(self):
        """Extracts JSON from markdown fences or raw JSON strings."""
        md_text = """```json
[
  {"beat_id": "B001", "dialogue_hook_score": 0, "visual_hook_score": 4}
]
```"""
        result = extract_json_payload(md_text)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["beat_id"], "B001")


if __name__ == "__main__":
    unittest.main()
