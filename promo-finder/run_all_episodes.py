"""
run_all_episodes.py — Batch Runner for Rebuilt Promo Finder Pipeline.

Executes the rebuilt 5-stage pipeline across all episodes in the scripts folder:
- Rebuilt Stage 1: Batched concurrent detection (2 calls per episode)
- Rebuilt Stage 2: Capped window search with floor-relaxation fallback (0 calls)
- Rebuilt Stage 3: Merged 3-in-1 scoring (comprehension gate + emotional + spoiler) in batches of 4
- Rebuilt Stage 4: Pure 4-dimension absolute rubric ranking (1 call, 0 round-robin, 0 tie-breaks)
- Rebuilt Stage 5: Rule-based boundary cleanliness check + MMR portfolio selection via fastembed (0 calls)
"""

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from run_pipeline import main as pipeline_main


if __name__ == "__main__":
    # Ensure --all is active unless a single --episode was explicitly requested
    if "--all" not in sys.argv and "--episode" not in sys.argv:
        sys.argv.append("--all")
    # Default outputs directory to outputs_final_batch if not explicitly provided
    if "--outputs-dir" not in sys.argv:
        sys.argv.extend(["--outputs-dir", "outputs_final_batch"])
    pipeline_main()
