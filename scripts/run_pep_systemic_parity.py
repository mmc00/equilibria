"""Legacy CLI entrypoint forwarded to the parity module."""
from __future__ import annotations

import sys
from pathlib import Path
from subprocess import run

SCRIPT = Path(__file__).resolve().parents[0] / "parity" / "run_pep_systemic_parity.py"


def main() -> int:
    if not SCRIPT.exists():
        raise FileNotFoundError(f"Parity script not found: {SCRIPT}")
    result = run([sys.executable, str(SCRIPT), *sys.argv[1:]])
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
