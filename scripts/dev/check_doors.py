"""Every model template keeps behind a public door (ADR-0012).

Runs ``equilibria.core.door``: each door in ``DOORS`` keeps the door contract
(``check_door``), and every template class in ``src/equilibria`` is the template of a
door or is listed in ``PENDING`` with the reason (``coverage_problems``), and every
door has its page in ``docs/site/api`` (``api_page_problems``). The same
checks run in CI (``tests/core/test_door_contract.py``); this is the pre-commit hook.

usage: python scripts/dev/check_doors.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from equilibria.core import door  # noqa: E402


def main() -> int:
    problems = list(door.coverage_problems())
    problems += door.api_page_problems(ROOT / "docs" / "site" / "api")
    for spec in door.DOORS:
        problems += door.check_door(spec)
    for p in problems:
        print(f"check_doors: {p}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
