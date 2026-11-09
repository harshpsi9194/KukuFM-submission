#!/usr/bin/env python3
"""
run_tests.py — Universal Test Runner for Promo-Finder Pipeline.

Executes all unit and integration test suites across:
  - tests/ (Stage 0 to Stage 5, CLI E2E)
  - promo-finder/ (module-level unit tests)

Usage:
  python3 run_tests.py
  python3 run_tests.py -v
  python3 run_tests.py test_stage0_parser
"""

import os
import sys
import time
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
PROMO_DIR = REPO_ROOT / "promo-finder"

# Ensure repo root and promo-finder are on sys.path
for p in [REPO_ROOT, PROMO_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# Ensure venv site-packages are accessible even when invoked via system python
for venv_candidate in [PROMO_DIR / ".venv", REPO_ROOT / ".venv"]:
    site_packages = venv_candidate / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    if site_packages.exists() and str(site_packages) not in sys.path:
        sys.path.insert(0, str(site_packages))


def run_all_tests(verbosity: int = 2) -> bool:
    print("=" * 72)
    print("PROMO-FINDER PIPELINE — COMPREHENSIVE UNIT TEST SUITE")
    print("=" * 72)
    print(f"Repository Root: {REPO_ROOT}")
    print(f"Python Runtime:  {sys.executable} (Python {sys.version.split()[0]})")
    print("-" * 72)

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Discover tests in tests/
    tests_dir = REPO_ROOT / "tests"
    if tests_dir.exists():
        discovered_tests = loader.discover(start_dir=str(tests_dir), pattern="test_*.py")
        suite.addTests(discovered_tests)

    # Discover module tests in promo-finder/
    for sub in ["parser", "candidates", "scoring", "ranking"]:
        sub_dir = PROMO_DIR / sub
        if sub_dir.exists():
            discovered = loader.discover(start_dir=str(sub_dir), pattern="test_*.py", top_level_dir=str(PROMO_DIR))
            suite.addTests(discovered)

    start_time = time.time()
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    elapsed = time.time() - start_time

    print("\n" + "=" * 72)
    print("TEST SUITE EXECUTION SUMMARY")
    print("=" * 72)
    print(f"Total Tests Executed: {result.testsRun}")
    print(f"Successes:            {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures:             {len(result.failures)}")
    print(f"Errors:               {len(result.errors)}")
    print(f"Skipped:              {len(result.skipped)}")
    print(f"Execution Time:       {elapsed:.3f} seconds")
    print("=" * 72)

    if result.wasSuccessful():
        print(">>> ALL TESTS PASSED SUCCESSFULLY (100% PASS RATE)")
        return True
    else:
        print(">>> SOME TESTS FAILED")
        return False


if __name__ == "__main__":
    verbosity = 2 if "-v" in sys.argv or "--verbose" in sys.argv else 1
    # Check if a specific test module was requested
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if args:
        # Run specific module
        suite = unittest.TestLoader().loadTestsFromName(args[0])
        runner = unittest.TextTestRunner(verbosity=2)
        res = runner.run(suite)
        sys.exit(0 if res.wasSuccessful() else 1)
    else:
        success = run_all_tests(verbosity=verbosity)
        sys.exit(0 if success else 1)
