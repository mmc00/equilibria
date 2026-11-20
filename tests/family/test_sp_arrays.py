"""The single-period reference as arrays (``sp_reference_arrays``) equals the
``SPSnapshot`` the Pyomo period model reads from the reference model.

Also the shared helpers of the family tests: the test route's closure, its params
and its recipe flags (``ROUTE``).
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np
import pytest

from tests.family._data import load_params

SP_DATASETS = ("gtap7_3x3", "gtap7_10x7")


class Route(TypedDict):
    mute_welfare: bool
    seed_from_prior: bool
    base_calibrated: bool
    solve_check: bool


# The recipe flags of the test route (_solve_shock): F3.5 base, check solved.
ROUTE: Route = {
    "mute_welfare": True,
    "seed_from_prior": False,
    "base_calibrated": True,
    "solve_check": True,
}


def _closure():
    """The base closure (ifSUB=1) of the test route."""
    from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig

    return GTAPClosureConfig(
        name="base",
        closure_type="MCP",
        capital_mobility="sluggish",
        fix_endowments=False,
        fix_taxes=False,
        fix_technology=False,
        if_sub=True,
        numeraire="pnum",
    )


def _params(dataset: str):
    """The run's params: ``build_block_model`` recalibrates them to the GEMPACK
    factor-subsidy basis of the test route (in place), and the reference model
    reads them like that."""
    import copy

    p = copy.deepcopy(load_params(dataset))
    p.set_va_subsidy_basis("gempack")
    return p


def _key(idx) -> tuple:
    if idx is None:
        return ()
    return tuple(str(i) for i in idx) if isinstance(idx, tuple) else (str(idx),)


@pytest.mark.parametrize("dataset", SP_DATASETS)
def test_sp_ref_arrays_equal_pyomo_snapshot(dataset):
    from equilibria.family.sp_arrays import sp_reference_arrays
    from equilibria.templates.gtap.sp_reference import build_sp_reference, sp_snapshot

    p = _params(dataset)
    rr = list(p.sets.r)[-1]
    snap = sp_snapshot(build_sp_reference(p.sets, p, _closure(), rr))
    ref = sp_reference_arrays(p, _closure(), rr)

    bad = []
    for name, (_, cells) in snap.fixed.items():
        want = {_key(i): v for i, v in cells}
        v = ref.vars.get(name)
        got = (
            {}
            if v is None
            else {k: x for k, x, f in zip(v.keys, v.value, v.fixed, strict=True) if f}
        )
        if want.keys() != got.keys():
            bad.append(f"fixed cells of {name}: {len(want)} vs {len(got)}")
            continue
        off = [k for k in want if not np.isclose(want[k], got[k], rtol=1e-12, atol=0)]
        if off:
            bad.append(f"fixed values of {name}: {off[:3]}")
    for name, (_, cells) in snap.bounds.items():
        want = {
            _key(i): (-np.inf if lb is None else lb, np.inf if ub is None else ub)
            for i, lb, ub in cells
        }
        v = ref.vars.get(name)
        got = (
            {}
            if v is None
            else {
                k: (lo, hi)
                for k, lo, hi in zip(v.keys, v.lb, v.ub, strict=True)
                if np.isfinite(lo) or np.isfinite(hi)
            }
        )
        if want != got:
            diff = [k for k in want.keys() | got.keys() if want.get(k) != got.get(k)]
            bad.append(f"bounds of {name}: {len(diff)} cells, e.g. {diff[:3]}")
    extra = [
        n
        for n, v in ref.vars.items()
        if (v.fixed.any() or np.isfinite(v.lb).any() or np.isfinite(v.ub).any())
        and n not in snap.fixed
    ]
    assert not bad and not extra, f"{bad[:6]} extra components {extra[:6]}"
