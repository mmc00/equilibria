"""ALTERTAX on nus333 with a (numerically) zero shock: the shock period must come back
to the check point.

The shock is seeded from the check and solves the same rows over (base, shock) that the
check solved over (base, check), so with nothing shocked it must return the check's
levels, and its Newton must start on a zero residual on the cross-period index rows
(``eq_pwfact``, ``eq_pfact``, ``eq_pabs``): their Fisher aggregates ``mq_*`` are defined
sums and the shock reads the base-period ones (``mq_*_bb[base]``) that the check read.

A shock of exactly 0 makes ``_default_tariff`` write the +10 % import tariff, so the
cell gets 1e-12 % instead: its effect on any level is far below 1e-12. The solver runs
with an absolute stop of 1e-12 (no relative test) so the shock's Newton does not stop on
a start residual of 1e-8 and the levels are those of the shock's own rows.
"""

from __future__ import annotations

import numpy as np
import pytest

from equilibria._local_refs import nus333_dir
from equilibria.family.altertax.run import solve_altertax
from tests.family._prep import _newton_mumps

pytest.importorskip("pyomo")

ZERO = {"prdtx_rai": {("USA", "AGR", "AGR"): 1e-12}}
INDEX_ROWS = ("eq_pwfact[", "eq_pfact[", "eq_pabs[")
LEVEL_TOL = 1e-12  # relative: |shock - check| <= LEVEL_TOL * max(1, |check|)
START_TOL = 1e-12  # the shock's start residual on the index rows


@pytest.fixture(scope="module")
def zero_run(tmp_path_factory):
    import equilibria.family.driver as drv

    mp = pytest.MonkeyPatch()
    env = {**_newton_mumps(), "EQUILIBRIA_GTAP_TR_LINSOLVE": "direct"}
    env.update(EQUILIBRIA_GTAP_TR_FTOL="1e-12", EQUILIBRIA_GTAP_TR_RELTOL="0")
    for k, v in env.items():
        mp.setenv(k, v)
    mp.setenv("EQUILIBRIA_FAMILY_CACHE", str(tmp_path_factory.mktemp("cache")))
    starts: list[dict[str, float]] = []
    columns: list[list[str]] = []
    orig = drv._newton

    def spy(ss, sq):
        F0 = np.asarray(ss.F(ss.x0))
        columns.append(list(sq.col_names))
        starts.append(
            {
                n: float(F0[i])
                for i, n in enumerate(sq.row_names)
                if n.startswith(INDEX_ROWS)
            }
        )
        return orig(ss, sq)

    mp.setattr(drv, "_newton", spy)
    try:
        run = solve_altertax(
            nus333_dir(), shock=ZERO, cache_root=tmp_path_factory.mktemp("altax")
        )
    finally:
        mp.undo()
    return run, starts, columns[-1]


def test_zero_shock_converges(zero_run):
    run, _, _ = zero_run
    assert run.codes == {"check": 1, "shock": 1}


def test_zero_shock_starts_on_the_index_rows(zero_run):
    """The last Newton call is the shock's: at its start (the check point, with the
    aggregates refreshed) every index row holds."""
    _, starts, _ = zero_run
    shock_start = starts[-1]
    assert any(n.endswith(",shock]") or n == "eq_pwfact[shock]" for n in shock_start)
    worst = max(shock_start, key=lambda n: abs(shock_start[n]))
    assert abs(shock_start[worst]) < START_TOL, (
        f"{worst} = {shock_start[worst]:.3e} at the shock's start"
    )


def test_zero_shock_returns_the_check_point(zero_run):
    """Every cell the shock solves (a column of its squared system) ends at its check
    level. Cells the shock holds fixed are left out: dead cells (no flow) and report
    variables keep values no row of either period determines."""
    run, _, shock_columns = zero_run
    mp = run.result.mp
    worst = (0.0, "")
    for col in shock_columns:
        name, _, rest = col.partition("[")
        key = tuple(rest[:-1].split(","))
        assert key[-1] == "shock", col
        v = mp.vars[name]
        x = float(v.value[v.pos(key)])
        c = float(v.value[v.pos((*key[:-1], "check"))])
        rel = abs(x - c) / max(1.0, abs(c))
        if not rel <= worst[0]:
            worst = (rel, f"{col}: shock {x!r} check {c!r}")
    assert worst[0] <= LEVEL_TOL, f"worst level {worst[0]:.3e}: {worst[1]}"
