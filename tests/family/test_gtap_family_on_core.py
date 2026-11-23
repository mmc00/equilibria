"""The family route on the core driver leaves the arrays the old family driver left.

The old driver's final ``MPArrays`` of each case on gtap7_3x3 (the direct linear
solve) was recorded on main before PR 4 deleted it (``tests/_old_route_golden.py``):
every array (Var ``value``/``fixed``/``lb``/``ub``, row ``active``) must match it,
keys and flags exactly, values within the golden tolerance. The codes agree period
by period (the base is not solved). The run builds no Pyomo model (the spies of
``test_family_e2e``).
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import pytest

from tests.family._data import load_params
from tests.family._prep import _newton_mumps
from tests.family.test_sp_arrays import _params

DS = "gtap7_3x3"


def _closure(if_sub: bool, savf: Any) -> Any:
    from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig

    return GTAPClosureConfig(
        name="base",
        closure_type="MCP",
        capital_mobility="sluggish",
        fix_endowments=False,
        fix_taxes=False,
        fix_technology=False,
        if_sub=if_sub,
        savf_flag=savf,
        numeraire="pnum",
        va_subsidy_basis="gempack",
    )


def _gdx(if_sub: bool) -> Any:
    from tests.templates.gtap.test_gtap7_gempack_parity import _reference_gdx

    return _reference_gdx(DS, int(if_sub))


def _shock_cell(cache: Any) -> tuple:
    """The first ``imptx`` shock cell, as ``test_family_e2e`` picks it."""
    from equilibria.family.run_setup import prepare_run

    p = _params(DS)
    _, mp = prepare_run(
        p,
        _closure(True, "capFix"),
        list(p.sets.r)[-1],
        cache_root=cache,
        jax_cache=False,
    )
    return next(k[:-1] for k in mp.vars["imptx"].keys if k[-1] == "shock")


@dataclasses.dataclass(frozen=True)
class Case:
    name: str
    if_sub: bool = True
    savf: str = "capFix"
    trade_balance: str | None = None
    user_shock: bool = False


CASES = [
    Case("capfix_ifsub0", if_sub=False),
    Case("capfix_ifsub1"),
    Case("capflex_ifsub1", savf="capFlex"),
    Case("user_shock", user_shock=True),
    Case("nominal", trade_balance="nominal"),
    Case("ratio", trade_balance="ratio"),
]


@pytest.mark.parametrize(
    "case",
    [
        # capFlex reads rore/rorg from the reference GDX (gdxdump)
        pytest.param(c, marks=pytest.mark.needs_gdxdump) if c.savf == "capFlex" else c
        for c in CASES
    ],
    ids=lambda c: c.name,
)
def test_same_arrays_as_the_old_family_driver(
    case: Case, monkeypatch, tmp_path
) -> None:
    from equilibria.core.exercise import Exercise
    from equilibria.templates.gtap.recipe import gtap_static
    from equilibria.templates.gtap.template import GTAP
    from tests._old_route_golden import case as golden
    from tests._old_route_golden import mp_mismatches
    from tests.family.test_family_e2e import _spies

    load_params(DS)
    # the golden's linear solve (MUMPS is not bit-reproducible run to run)
    linsolve = "direct"
    for k, v in {**_newton_mumps(), "EQUILIBRIA_GTAP_TR_LINSOLVE": linsolve}.items():
        monkeypatch.setenv(k, v)
    gdx = _gdx(case.if_sub) if case.savf == "capFlex" else None
    if case.savf == "capFlex" and (gdx is None or not gdx.exists()):
        pytest.skip("capFlex needs the GAMS reference GDX")
    rr = list(_params(DS).sets.r)[-1]
    cache = tmp_path / "records"
    shock = {"imptx": {_shock_cell(cache): 5.0}} if case.user_shock else None

    template = GTAP(
        _params(DS),
        _closure(case.if_sub, case.savf),
        residual_region=rr,
        base_calibrated=True,
        ref_gdx=gdx,
        backend="family",
        cache_root=cache,
        jax_cache=False,
    )
    ex = Exercise(
        template,
        gtap_static(
            base_calibrated=True, solve_check=True, trade_balance=case.trade_balance
        ),
        shock=shock,
    )
    with _spies() as calls:
        new = ex.run()
    assert calls == {"ConcreteModel": 0, "PyomoNLP": 0, "identify_variables": 0}

    old = golden(f"family_{case.name}")
    assert old["base_skipped"] and new["base"]["code"] == "not solved"
    for period in ("check", "shock"):
        assert (old["codes"][period] == 1) == (new[period]["code"] == "optimal"), period
    diffs = mp_mismatches(ex.model.mp, old["mp"])  # type: ignore[union-attr]
    assert not diffs, "\n".join(diffs[:20])
