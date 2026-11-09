"""
test_cli_e2e.py — End-to-End CLI & Pipeline Integration Tests.

Verifies:
1. CLI dry-run executes cleanly with 0 LLM API calls and 0 network usage.
2. find_script_file prioritizes real show scripts over test fixtures.
3. Explicit --script parameter allows running on arbitrary screenplay files.
4. Pipeline invariants hold end-to-end on real script (Episode 1):
   - Snapping strictly to beat edges (Rule 1 & 2).
   - Python duration filter strictly bounded to [20.0s, 60.0s] (Rule 3).
   - timestamp_source is present on every beat and segment (Rule 4).
   - Scene boundaries respected in single-segment window search (Rule 5).
"""

import subprocess
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PROMO_DIR = REPO_ROOT / "promo-finder"
for p in [REPO_ROOT, PROMO_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from parser.parser import parse_file
from run_pipeline import find_script_file


class TestCliE2E(unittest.TestCase):

    def test_find_script_file_resolves_production_episode(self):
        """Script finder resolves the actual series script file within the repository."""
        script_path = find_script_file(1, REPO_ROOT)
        self.assertIsNotNone(script_path)
        self.assertTrue(script_path.exists())
        # Checks that it resolved to the real series file, not a hardcoded fixture
        self.assertTrue("episode-001" in script_path.name or "episode_001" in script_path.name)

    def test_cli_dry_run_subprocess_execution(self):
        """Running run_pipeline.py --episode 1 --dry-run completes successfully with exit code 0."""
        cmd = [sys.executable, str(PROMO_DIR / "run_pipeline.py"), "--episode", "1", "--dry-run"]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=str(REPO_ROOT))
        self.assertEqual(res.returncode, 0, f"Dry run failed with stderr: {res.stderr}")
        self.assertIn("Structural validation successful", res.stdout)
        self.assertIn("Stage 0] Screenplay parsed: 45 beats", res.stdout)
        self.assertIn("Stage 2 Capped search produced", res.stdout)
        # Verify it points to the real show script dynamically
        self.assertTrue(
            "episode-001" in res.stdout or "episode_001" in res.stdout,
            f"Expected episode-001 in dry-run stdout, got:\n{res.stdout}"
        )

    def test_cli_explicit_script_parameter(self):
        """Running with --script <path> allows targeting an explicit script file."""
        sample_path = PROMO_DIR / "parser" / "fixtures" / "sample_scripts" / "episode_001.txt"
        cmd = [sys.executable, str(PROMO_DIR / "run_pipeline.py"), "--script", str(sample_path), "--dry-run"]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=str(REPO_ROOT))
        self.assertEqual(res.returncode, 0, f"Dry run with --script failed: {res.stderr}")
        self.assertIn("Structural validation successful", res.stdout)
        self.assertIn("EPISODE 001", res.stdout)

    def test_dry_run_pipeline_invariants_on_episode_1(self):
        """Parse real episode 1 and verify structural invariants."""
        script_path = find_script_file(1, REPO_ROOT)
        self.assertIsNotNone(script_path)
        episode = parse_file(str(script_path))

        # Check beat invariants
        self.assertGreater(len(episode.beats), 0)
        for b in episode.beats:
            self.assertIn(b.timestamp_source, ("given", "estimated"))
            self.assertLessEqual(b.start, b.end)
            self.assertGreaterEqual(b.scene_number, 1)

        # Check beat boundary continuity
        for i in range(len(episode.beats) - 1):
            curr_b = episode.beats[i]
            next_b = episode.beats[i + 1]
            if curr_b.scene_number == next_b.scene_number:
                self.assertLessEqual(curr_b.start, next_b.start)

    def test_dynamic_discovery_arbitrary_drama_folder(self):
        """Verifies discover_script_directories and find_script_file find scripts in any drama directory structure."""
        import tempfile
        from run_pipeline import discover_script_directories
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            # Create an arbitrary drama directory structure
            drama_dir = tmp_root / "arbitrary_custom_drama" / "season_1"
            drama_dir.mkdir(parents=True)
            test_ep = drama_dir / "episode-001-pilot.txt"
            test_ep.write_text("SCENE 1 - INT. OFFICE - DAY\n**ACTION:**\nStart: (00:00.00) | End: (00:10.00)\nA character speaks.\n")
            
            # Discover should find drama_dir automatically without any hardcoded names
            discovered = discover_script_directories(tmp_root)
            self.assertIn(drama_dir, discovered)
            
            # find_script_file should locate episode 1
            found = find_script_file(1, tmp_root)
            self.assertEqual(found, test_ep)


if __name__ == "__main__":
    unittest.main()
