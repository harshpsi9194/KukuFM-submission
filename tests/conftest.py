"""
conftest.py — Pytest configuration for Promo Finder.
Ensures promo-finder and root paths are properly added to sys.path.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PROMO_DIR = REPO_ROOT / "promo-finder"

for p in [REPO_ROOT, PROMO_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

