#!/usr/bin/env python
"""Run the same essential checks as CI, from the repository root: ruff, pytest with coverage, strict docs build.

    python scripts/check_project.py              # all checks, stops at the first failure
    python scripts/check_project.py --no-docs    # skip the MkDocs build

Needs the development dependencies: pip install -r requirements-dev.txt && pip install -e . --no-deps
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CHECKS = {
    "ruff": ("Lint (ruff)", [sys.executable, "-m", "ruff", "check", "src", "scripts", "tests"]),
    "tests": ("Tests with coverage (pytest)", [sys.executable, "-m", "pytest", "--cov", "--cov-report=term-missing:skip-covered"]),
    "docs": ("Documentation (mkdocs build --strict)", [sys.executable, "-m", "mkdocs", "build", "--strict"]),
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-docs", action="store_true", help="skip the MkDocs build")
    args = ap.parse_args(argv)
    steps = [k for k in CHECKS if not (k == "docs" and args.no_docs)]
    for i, key in enumerate(steps, 1):
        title, cmd = CHECKS[key]
        print(f"\n[{i}/{len(steps)}] {title}\n$ {' '.join(cmd[1:])}", flush=True)
        try:
            code = subprocess.run(cmd, cwd=ROOT).returncode
        except OSError as exc:
            print(f"FAILED to start: {exc}", file=sys.stderr)
            return 1
        if code != 0:
            print(f"\nFAILED: {title} (exit code {code}). Later checks were not run.", file=sys.stderr)
            return code
    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
