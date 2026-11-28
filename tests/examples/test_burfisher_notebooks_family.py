"""The four Burfisher notebooks that change the closure with ``@overwrite`` (TBL65B,
ME3C, TBL94, ME9A) give on the family (JAX) backend what they give on Pyomo (MAR-251):
every Var cell of every period at 1e-9, and the notebook prints what it printed on
main (``GOLDEN``; not the Pyomo run's stdout, which under these solver settings also
carries Pyomo's component-replacement warnings).

Each run is the notebook's own code in its own process (hooks last the whole process,
#214), with ``backend="family"`` added to its ``GTAP(...)`` for the family run. Both
runs use the Newton-TR + MUMPS settings the family route requires, so both solve the
same system (as ``tests/scripts/test_run_burfisher_family.py``)."""

from __future__ import annotations

import os
import pickle
import subprocess
import sys
from pathlib import Path

import pytest
from tests._printed import GOLDEN
from tests._printed import mismatches as printed_mismatches
from tests.examples.test_burfisher_notebooks import NB_DIR, _code
from tests.family._period_levels import mismatches
from tests.family._prep import _newton_mumps

pytest.importorskip("jax")
pytestmark = [pytest.mark.integration, pytest.mark.needs_path]

ROOT = Path(__file__).resolve().parents[2]
OVERWRITE = ("TBL65B", "ME3C", "TBL94", "ME9A")
# ME9A has all three hook kinds (endogenous, equation, target on rgdpmp, a cross-period
# Var); the others run with ``-m slow``.
IN_CI = {"ME9A"}
REL = 1e-9
_GTAP = "GTAP(p, gc,"
_SOLVED = "res = ex.run()\n"
_DUMP = """
import pickle as _pickle
from tests.family._period_levels import levels as _levels
with open({path!r}, "wb") as _f:
    _pickle.dump(_levels(ex.model), _f)
"""


def _run_notebook(exp: str, backend: str, tmp_path: Path) -> tuple[str, dict]:
    """The notebook's stdout and its levels right after the solve."""
    code = _code(NB_DIR / f"{exp.lower()}.ipynb")
    if backend == "family":
        assert _GTAP in code, exp
        code = code.replace(_GTAP, f'{_GTAP} backend="family",')
    assert code.count(_SOLVED) == 1, exp
    out = tmp_path / f"{backend}.pkl"
    code = code.replace(_SOLVED, _SOLVED + _DUMP.format(path=str(out)))
    script = tmp_path / f"notebook_{backend}.py"  # not pyomo.py: it would shadow pyomo
    script.write_text(code, encoding="utf-8")
    env = {
        **os.environ,
        **_newton_mumps(),
        "EQUILIBRIA_GTAP_SOLVE_NLP": "1",
        "EQUILIBRIA_FAMILY_CACHE": str(tmp_path / "family-cache"),
        "PYTHONPATH": os.pathsep.join(
            [str(ROOT), str(ROOT / "src"), os.environ.get("PYTHONPATH", "")]
        ),
    }
    r = subprocess.run(
        [sys.executable, str(script)],
        cwd=NB_DIR,
        capture_output=True,
        text=True,
        timeout=1800,
        env=env,
    )
    assert r.returncode == 0, f"{exp} {backend}: {r.stderr[-3000:]}"
    with open(out, "rb") as f:
        return r.stdout, pickle.load(f)


@pytest.mark.parametrize(
    "exp",
    [e if e in IN_CI else pytest.param(e, marks=pytest.mark.slow) for e in OVERWRITE],
)
def test_overwrite_notebook_on_family_equals_pyomo(exp: str, tmp_path: Path) -> None:
    _, want = _run_notebook(exp, "pyomo", tmp_path)
    f_out, got = _run_notebook(exp, "family", tmp_path)
    bad = mismatches(got, want, REL)
    assert not bad, f"{exp}: {len(bad)} of {len(want)} cells differ, first: {bad[:5]}"
    assert printed_mismatches(f_out.splitlines(), GOLDEN["notebooks"][exp]) == []
