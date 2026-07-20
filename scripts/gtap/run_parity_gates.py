"""Mandatory LOCAL integration gates: NLP-vs-NLP + MCP-vs-MCP full sweeps.

CI cannot run these (they need the local PATH C-API, IPOPT and the fixture
GDXs), so this script is the single pre-push/pre-PR ritual for any change that
can touch GTAP parity:

  1. Refuses to run if gate-relevant paths are dirty (house rule: every
     validation run happens at a clean commit).
  2. Runs the gate tests (integration markers):
       - tests/templates/gtap/test_gtap7_mcp_parity.py   (MCP vs MCP, 18 rows)
       - tests/templates/gtap/test_gtap7_nlp_parity.py   (NLP vs NLP, 14 rows)
       - tests/templates/gtap/test_gtap7_nl_parity.py    (.nl coefficient gate)
       - tests/templates/gtap/test_coverage_matrix.py    (floors/doc in sync)
  3. Regenerates the measured docs (coverage matrix .md + both HTML matrix
     pages) so the documentation ALWAYS reflects the measured state.
  4. On a fully green run, writes `.git/gtap-parity-gates.stamp` with the
     input-tree hash of HEAD. The Claude PreToolUse hook
     (scripts/gtap/claude_hooks/block_push_without_gates.py) blocks
     `git push` / `gh pr create` while this stamp is missing or stale.

Doc-only commits (the regenerated files under docs/site) do not invalidate the
stamp — the hash covers only gate-INPUT trees. If step 3 changes any doc, the
script tells you to commit it; the stamp stays valid.

Usage:
    uv run python scripts/gtap/run_parity_gates.py            # full (stamps)
    uv run python scripts/gtap/run_parity_gates.py --quick    # gates only, NO stamp
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_parity_gates_stamp import (  # noqa: E402
    GENERATED_DOCS, STAMP_REL, dirty_watched, input_hash, repo_root,
)

GATE_TESTS = [
    "tests/templates/gtap/test_gtap7_mcp_parity.py",
    "tests/templates/gtap/test_gtap7_nlp_parity.py",
    "tests/templates/gtap/test_gtap7_nl_parity.py",
    "tests/templates/gtap/test_coverage_matrix.py",
]

REGEN_CMDS = [
    ["scripts/gtap/gen_coverage_doc.py"],
    ["scripts/gtap/gen_nlp_matrix_page.py", "--gate", "mcp"],
    ["scripts/gtap/gen_nlp_matrix_page.py", "--gate", "nlp"],
]


def _run(repo: Path, argv: list[str]) -> int:
    print(f"\n=== {' '.join(argv)} ===", flush=True)
    return subprocess.run(argv, cwd=repo).returncode


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="run the gate tests only; skip doc regen and do NOT stamp")
    args = ap.parse_args()

    repo = repo_root(Path(__file__).resolve().parent)

    dirty = dirty_watched(repo)
    if dirty:
        print("ABORT: gate-relevant paths are dirty — commit first (git discipline):")
        print("  " + "\n  ".join(dirty[:15]))
        return 1

    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=repo,
                          capture_output=True, text=True).stdout.strip()
    print(f"Running mandatory GTAP parity gates at {head}")

    rc = _run(repo, [sys.executable, "-m", "pytest", *GATE_TESTS, "-q"])
    if rc != 0:
        print("\nGATES RED — no stamp written. Fix the regression before pushing.")
        return rc

    if args.quick:
        print("\nGates green (quick mode) — docs NOT regenerated, stamp NOT written.")
        print("Run without --quick before pushing.")
        return 0

    for cmd in REGEN_CMDS:
        rc = _run(repo, [sys.executable, *cmd])
        if rc != 0:
            print(f"\nDoc regeneration failed: {' '.join(cmd)} — no stamp written.")
            return rc

    (repo / STAMP_REL).write_text(input_hash(repo, "HEAD") + "\n")
    print(f"\nGates GREEN — stamp written to {STAMP_REL}")

    changed = subprocess.run(["git", "status", "--porcelain", "--", *GENERATED_DOCS],
                             cwd=repo, capture_output=True, text=True).stdout.strip()
    if changed:
        print("\nMeasured docs changed — COMMIT THEM (the stamp stays valid):")
        print("  " + changed.replace("\n", "\n  "))
    else:
        print("Measured docs already in sync.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
