"""Capture the two-photo oracle of the Pyomo period prep for one dataset.

    <python> scripts/family/capture_prep.py gtap7_10x7 [--out DIR]

Runs ``_solve_shock(ds, 1, "capFix", solve_nlp=True)`` under ``NEWTON_MUMPS`` in a
child process with ``EQUILIBRIA_FAMILY_PREP_CAPTURE`` set. Uses MUMPS when pymumps
is importable, the exact scipy LU otherwise.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "gtap"))
from perf_budget_20x41 import NEWTON_MUMPS  # noqa: E402

_CHILD = """
import sys
sys.path[:0] = [{src!r}, {tests!r}, {root!r}]
from templates.gtap.test_gtap7_gempack_parity import _solve_shock
_, code = _solve_shock({ds!r}, {ifsub}, "{savf}", solve_nlp=True)
print("CODE", code)
"""


def closure_tag(ifsub: int, savf: str, mode: str) -> str:
    """Directory name of a closure's reference; today's one keeps its name."""
    if (ifsub, savf, mode) == (1, "capFix", "gtap"):
        return "tol_today"
    return f"{mode}_ifsub{ifsub}_{savf}"


GATE_FILES = ("final.pkl", "settled.pkl")


def capture(
    dataset: str,
    out: Path,
    *,
    ifsub: int = 1,
    savf: str = "capFix",
    mode: str = "gtap",
    keep: str = "all",
) -> int:
    """Run the Pyomo reference of one closure into ``out``.

    Each ``*_S_*.json`` gets a ``closure`` field. ``keep="gate"`` deletes, after a
    successful capture, everything but ``final.pkl``, ``settled.pkl`` and the S files.
    """
    if ifsub not in (0, 1) or savf not in ("capFix", "capFlex"):
        raise ValueError(f"unsupported closure ifsub={ifsub} savf={savf}")
    if mode != "gtap":
        raise NotImplementedError(f"mode={mode!r}: only 'gtap' until Phase 3")
    if keep not in ("all", "gate"):
        raise ValueError(f"keep must be 'all' or 'gate', got {keep!r}")
    try:
        import mumps  # noqa: F401

        linsolve = "mumps"
    except ImportError:
        linsolve = "direct"
    env = {
        **os.environ,
        **NEWTON_MUMPS,
        "EQUILIBRIA_GTAP_TR_LINSOLVE": linsolve,
        "EQUILIBRIA_FAMILY_PREP_CAPTURE": str(out),
    }
    env.pop("EQUILIBRIA_GTAP_MODEL_CACHE", None)
    code = _CHILD.format(
        src=str(ROOT / "src"),
        tests=str(ROOT / "tests"),
        root=str(ROOT),
        ds=dataset,
        ifsub=ifsub,
        savf=savf,
    )
    p = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True
    )
    line = next((s for s in p.stdout.splitlines() if s.startswith("CODE ")), None)
    if line is None:
        raise RuntimeError(p.stderr[-4000:])
    result = int(line.split()[1])
    for f in sorted(out.glob("*_S_*.json")):
        s = json.loads(f.read_text())
        s["closure"] = {
            "ifsub": ifsub,
            "savf": savf,
            "mode": mode,
            "linsolve": linsolve,
        }
        f.write_text(json.dumps(s, indent=1))
    if keep == "gate" and result == 1:
        for f in out.iterdir():
            if f.is_file() and f.name not in GATE_FILES and "_S_" not in f.name:
                f.unlink()
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    out = a.out or Path.home() / ".cache/equilibria/family/prep_capture" / a.dataset
    out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("*.pkl"):
        f.unlink()
    print("code", capture(a.dataset, out), "->", out)
