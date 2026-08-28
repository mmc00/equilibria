"""End-to-end byte-identical gate for the structural_matching cache (EQUILIBRIA_GTAP_STRUCT_CACHE).

Runs the real gtap7_3x3 base->check->shock solve (NLP mode) TWICE — cache OFF, cache ON —
with a 4-step shock continuation so the cache actually FIRES (3 hits expected: the 6-9
phase-invocation redundancy that motivates the cache only shows up across repeated
identical-signature calls, which the continuation provides). Asserts the two runs reach
the EXACT SAME solution (every free variable, byte-identical) and the same code/residual
per phase. This is the hard, non-negotiable gate from
docs/superpowers/specs/2026-08-28-structural-cache-design.md: if cache ON ever diverges
from cache OFF, the cache is unsound and must not ship enabled.

LOCAL-ONLY (like test_gtap7_nlp_parity.py): needs the gtap7_3x3 HAR fixtures + the local
NLP/ASL toolchain. SKIPS if either is missing.

Run:
    uv run pytest tests/templates/gtap/test_structural_cache_parity.py -v
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
DATASET = ROOT / "datasets" / "gtap7_3x3"
_RUNNER = Path(__file__).resolve().parent / "_structural_cache_3x3_runner.py"

pytestmark = pytest.mark.integration


def _dataset_missing() -> bool:
    return not (DATASET / "basedata.har").exists()


def _asl_missing() -> bool:
    try:
        from pyomo.contrib.pynumero.asl import AmplInterface

        return not AmplInterface.available()
    except Exception:
        return True


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(_dataset_missing(), reason="gtap7_3x3 HAR fixtures not found"),
    pytest.mark.skipif(_asl_missing(), reason="local ASL/NLP toolchain not available"),
]


def _run(cache_on: bool, continuation: str) -> dict:
    env = dict(os.environ)
    env["EQUILIBRIA_GTAP_STRUCT_CACHE"] = "1" if cache_on else "0"
    env["EQUILIBRIA_GTAP_SHOCK_CONTINUATION"] = continuation
    r = subprocess.run(
        [sys.executable, str(_RUNNER)],
        capture_output=True, text=True, timeout=300, env=env,
    )
    assert r.returncode == 0, f"runner failed:\nSTDOUT:\n{r.stdout[-3000:]}\nSTDERR:\n{r.stderr[-3000:]}"
    for line in reversed(r.stdout.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            out = json.loads(line)
            out["_stderr"] = r.stderr
            return out
        except json.JSONDecodeError:
            continue
    raise AssertionError(f"no JSON output from runner:\n{r.stdout[-3000:]}")


def test_cache_on_matches_off_no_continuation():
    """Trivial case: base->check->shock, no repeated phase -> cache never fires, must be a no-op."""
    off = _run(cache_on=False, continuation="1.0")
    on = _run(cache_on=True, continuation="1.0")
    assert off["result"] == on["result"]
    assert off["cells"] == on["cells"]


def test_cache_on_matches_off_with_continuation_and_actually_fires():
    """The real case: a 4-step continuation gives the cache repeated identical signatures.

    Asserts BOTH that it fires (>=1 hit for EACH of the two independent cache layers —
    the closure+squareness+fixing block and structural_matching — proving both mechanisms
    engage, not a vacuous pass) AND that the solution stays byte-identical (the hard
    correctness gate).
    """
    off = _run(cache_on=False, continuation="0.25,0.5,0.75,1.0")
    on = _run(cache_on=True, continuation="0.25,0.5,0.75,1.0")

    block_hits = on["_stderr"].count("reused closure+squareness+fixing block")
    matching_hits = on["_stderr"].count("reused structural_matching")
    assert block_hits >= 1, "closure+squareness+fixing cache never fired — test is vacuous"
    assert matching_hits >= 1, "structural_matching cache never fired — test is vacuous"

    assert off["result"] == on["result"], "phase code/residual diverged with cache ON"
    assert set(off["cells"]) == set(on["cells"]), "cache ON produced a different variable set"
    maxdiff = max(abs(off["cells"][k] - on["cells"][k]) for k in off["cells"])
    assert maxdiff == 0.0, f"cache ON diverged from OFF by {maxdiff} — NOT byte-identical"
