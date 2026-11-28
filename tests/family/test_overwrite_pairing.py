"""Under ``@overwrite`` family pairs each hook row of the shock with the variable Pyomo
pairs it with (MAR-251): a different pairing can reach another root with the same
numbers elsewhere (the "acople estructural" note).

Only the hooks' rows: family's pairing of the other rows already differs from Pyomo's
without any hook (TBL65A, the same closure: 77 of 657 shock rows; measured 2026-11-27),
the known gap of ``test_squared_system::test_pairs_equal_pyomo_with_period_incidence``
(xfail), and the hooks move which rows those are. Every cell equals Pyomo's at 1e-9
all the same (``tests/examples/test_burfisher_notebooks_family.py``).

Pyomo's pairs come from path_capi's own dump (``EQUILIBRIA_DEBUG_DUMP_SQUARE_SYSTEM``:
``con_names`` and ``var_names`` in paired order, one file per period, the last solve
of the period kept); family's from the ``Squared`` of the shock's last system build.
Each run is the notebook's code in its own process (hooks last the process, #214)."""

from __future__ import annotations

import os
import pickle
import subprocess
import sys
from pathlib import Path

import pytest

from tests.examples.test_burfisher_notebooks import NB_DIR, _code
from tests.family._prep import _newton_mumps

pytest.importorskip("jax")
pytestmark = [pytest.mark.integration, pytest.mark.needs_path]

ROOT = Path(__file__).resolve().parents[2]
# the rows each notebook's hooks add (b.equation, b.target)
HOOK_ROWS = {
    "TBL65B": ("eq_wreal",),
    "ME3C": ("eq_wreal",),
    "TBL94": ("eq_qca_target",),
    "ME9A": ("eq_gdp_target", "eq_aoreg"),
}
_GTAP = "GTAP(p, gc,"
_RUN = "res = ex.run()\n"
_SPY = """
import pickle as _pickle
from equilibria.family import period_newton as _pn

_orig = _pn._period_system


def _spy(run, mp1, period):
    out = _orig(run, mp1, period)
    if period == "shock":
        with open({path!r}, "wb") as _f:
            _pickle.dump({{"con_names": out[0].row_names, "var_names": out[0].col_names}}, _f)
    return out


_pn._period_system = _spy
"""


def _pairs(exp: str, backend: str, tmp_path: Path) -> dict[str, str]:
    code = _code(NB_DIR / f"{exp.lower()}.ipynb")
    assert code.count(_RUN) == 1, exp
    code = code[: code.index(_RUN) + len(_RUN)]
    env = {
        **os.environ,
        **_newton_mumps(),
        "EQUILIBRIA_GTAP_SOLVE_NLP": "1",
        "PYTHONPATH": os.pathsep.join(
            [str(ROOT), str(ROOT / "src"), os.environ.get("PYTHONPATH", "")]
        ),
    }
    dump = tmp_path / backend
    if backend == "family":
        code = code.replace(_GTAP, f'{_GTAP} backend="family",')
        code = code.replace(_RUN, _SPY.format(path=str(dump)) + _RUN)
        path = dump
    else:
        env["EQUILIBRIA_DEBUG_DUMP_SQUARE_SYSTEM"] = str(dump)
        path = Path(f"{dump}.shock")
    script = tmp_path / f"pairs_{backend}.py"
    script.write_text(code, encoding="utf-8")
    r = subprocess.run(
        [sys.executable, str(script)],
        cwd=NB_DIR,
        capture_output=True,
        text=True,
        timeout=1800,
        env=env,
    )
    assert r.returncode == 0, f"{exp} {backend}: {r.stderr[-3000:]}"
    with open(path, "rb") as f:
        d = pickle.load(f)
    return dict(zip(d["con_names"], d["var_names"], strict=True))


@pytest.mark.parametrize(
    "exp",
    [
        "ME9A",
        *(pytest.param(e, marks=pytest.mark.slow) for e in ("TBL65B", "ME3C", "TBL94")),
    ],
)
def test_hook_rows_pair_as_on_pyomo(exp: str, tmp_path: Path) -> None:
    want = _pairs(exp, "pyomo", tmp_path)
    got = _pairs(exp, "family", tmp_path)
    assert got.keys() == want.keys(), sorted(got.keys() ^ want.keys())[:10]
    hook = {r: c for r, c in want.items() if r.split("[")[0] in HOOK_ROWS[exp]}
    assert {r.split("[")[0] for r in hook} == set(HOOK_ROWS[exp])
    assert {r: got[r] for r in hook} == hook
