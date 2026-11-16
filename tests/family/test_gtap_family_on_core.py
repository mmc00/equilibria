"""The family route on the core driver leaves the same arrays as solve_gtap_family.

Both routes run in the same process on gtap7_3x3 (the direct linear solve, which
is bit-reproducible; same record cache), and every array of the final ``MPArrays`` (Var ``value``/``fixed``/``lb``/
``ub``, row ``active``) must be equal: ``==``, NaN equal to NaN. The codes agree
period by period (the base is not solved by either: ``skipped_solve`` against
``"not solved"``). The new run builds no Pyomo model (the spies of
``test_family_e2e``).
"""

from __future__ import annotations

import dataclasses
import sys
from typing import Any

import numpy as np
import pytest

from tests.family._data import load_params
from tests.family._prep import _newton_mumps
from tests.family.test_prep_recipe import _params

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
    from equilibria.family.driver import prepare_run

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


def _first_differences(a, b, n: int = 20) -> list[str]:
    out: list[str] = []
    for name, v in a.vars.items():
        w = b.vars[name]
        for f in ("value", "fixed", "lb", "ub"):
            x, y = getattr(v, f), getattr(w, f)
            same = (
                (x == y) | (np.isnan(x) & np.isnan(y))
                if x.dtype.kind == "f"
                else x == y
            )
            for i in np.flatnonzero(~same)[: n - len(out)]:
                out.append(f"{name}{v.keys[i]}.{f}: old {y[i]!r} new {x[i]!r}")
            if len(out) >= n:
                return out
    for name, r in a.rows.items():
        for i in np.flatnonzero(r.active != b.rows[name].active)[: n - len(out)]:
            out.append(
                f"{name}{r.keys[i]}.active: old {b.rows[name].active[i]} new {r.active[i]}"
            )
    return out


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_same_arrays_as_solve_gtap_family(case: Case, monkeypatch, tmp_path) -> None:
    from equilibria.core.exercise import Exercise
    from equilibria.family.driver import solve_gtap_family
    from equilibria.templates.gtap.recipe import gtap_static
    from equilibria.templates.gtap.template import GTAP
    from tests.family.test_family_e2e import _spies

    load_params(DS)
    # MUMPS is not bit-reproducible run to run (the old route against itself
    # differs in walras[check] at 1e-21); the direct solve is, so it is the referee
    linsolve = "direct"
    for k, v in {**_newton_mumps(), "EQUILIBRIA_GTAP_TR_LINSOLVE": linsolve}.items():
        monkeypatch.setenv(k, v)
    gdx = _gdx(case.if_sub) if case.savf == "capFlex" else None
    if case.savf == "capFlex" and (gdx is None or not gdx.exists()):
        pytest.skip("capFlex needs the GAMS reference GDX")
    rr = list(_params(DS).sets.r)[-1]
    cache = tmp_path / "records"
    shock = {"imptx": {_shock_cell(cache): 5.0}} if case.user_shock else None

    old = solve_gtap_family(
        _params(DS),
        _closure(case.if_sub, case.savf),
        rr,
        ref_gdx=gdx,
        shock=shock,
        cache_root=cache,
        jax_cache=False,
        nominal_trade_balance=case.trade_balance or False,
    )
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

    assert (
        old.results["base"].get("skipped_solve") and new["base"]["code"] == "not solved"
    )
    for period in ("check", "shock"):
        assert (old.results[period]["code"] == 1) == (
            new[period]["code"] == "optimal"
        ), period
    mp_new = ex.model.mp  # type: ignore[union-attr]
    assert mp_new.vars.keys() == old.mp.vars.keys()
    assert mp_new.rows.keys() == old.mp.rows.keys()
    diffs = _first_differences(mp_new, old.mp)
    assert not diffs, "\n".join(diffs)
