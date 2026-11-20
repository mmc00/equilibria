"""Record what the old GTAP drivers give, for the tests that compared them to the core
(PR 4 oracle).

PR 4 of gtap-on-core deletes the old drivers. A few tests compared the old route
against the core route in the same process; run on ``main`` before the deletion, this
writes the old side of each, and those tests then compare the core against it. Each
case runs in its own child process, with the environment of the test it replaces.

usage: python scripts/dev/record_pr4_old_route.py --out FILE [--only CASE ...]
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]

# tests/family/test_gtap_family_on_core.py::CASES
FAMILY = (
    "capfix_ifsub0",
    "capfix_ifsub1",
    "capflex_ifsub1",
    "user_shock",
    "nominal",
    "ratio",
)
# tests/templates/gtap/test_gtap_on_core_solve.py
ON_CORE = tuple(
    f"static_{mode}_ifsub{int(s)}" for mode in ("gtap", "altertax") for s in (0, 1)
) + ("steps", "johansen", "overwrite_nus333")
CASES = (
    [f"family_{c}" for c in FAMILY]
    + ["template_build", "routes_3x3", "variant_sparse", "variant_loglevels"]
    + ["burfisher_nonlinear", "burfisher_johansen"]
    + [f"on_core_{c}" for c in ON_CORE]
)


def mp_arrays(mp: Any) -> dict:
    """Every array the old-vs-core tests compared: Var value/fixed/lb/ub, row active."""

    def floats(a: Any) -> list:
        return [None if math.isnan(x) else float(x) for x in a.tolist()]

    return {
        "vars": {
            n: {
                "keys": [repr(k) for k in v.keys],
                "value": floats(v.value),
                "fixed": [bool(x) for x in v.fixed.tolist()],
                "lb": floats(v.lb),
                "ub": floats(v.ub),
            }
            for n, v in mp.vars.items()
        },
        "rows": {
            n: {
                "keys": [repr(k) for k in r.keys],
                "active": [bool(x) for x in r.active],
            }
            for n, r in mp.rows.items()
        },
    }


def pyomo_levels(m: Any, *, active_only: bool = False) -> dict:
    """Every Var level of a Pyomo model, ``{repr((var, index)): float | None}``."""
    from pyomo.environ import Var

    out = {}
    for v in m.component_objects(Var, active=True if active_only else None):
        for k in v:
            x = v[k].value
            out[repr((v.name, k))] = None if x is None else float(x)
    return out


def _nested(d: Any) -> Any:
    """A nested ``{key: ... float}`` dict with ``repr`` keys (JSON keys are strings)."""
    if isinstance(d, dict):
        return {repr(k): _nested(v) for k, v in d.items()}
    return None if d is None else float(d)


def _env(mp: Any) -> None:
    from tests.family._prep import _newton_mumps

    for k, v in {**_newton_mumps(), "EQUILIBRIA_GTAP_TR_LINSOLVE": "direct"}.items():
        mp.setenv(k, v)


def case_family(name: str, mp: Any, tmp: Path) -> dict:
    from tests.family import test_gtap_family_on_core as t
    from tests.family._data import load_params

    from equilibria.family.driver import solve_gtap_family

    load_params(t.DS)
    _env(mp)
    c = next(c for c in t.CASES if c.name == name)
    gdx = t._gdx(c.if_sub) if c.savf == "capFlex" else None
    rr = list(t._params(t.DS).sets.r)[-1]
    cache = tmp / "records"
    shock = {"imptx": {t._shock_cell(cache): 5.0}} if c.user_shock else None
    old = solve_gtap_family(
        t._params(t.DS),
        t._closure(c.if_sub, c.savf),
        rr,
        ref_gdx=gdx,
        shock=shock,
        cache_root=cache,
        jax_cache=False,
        nominal_trade_balance=c.trade_balance or False,
    )
    return {
        "base_skipped": bool(old.results["base"].get("skipped_solve")),
        "codes": {p: int(old.results[p]["code"]) for p in ("check", "shock")},
        "mp": mp_arrays(old.mp),
    }


def case_template_build(mp: Any, tmp: Path) -> dict:
    from tests.family import test_gtap_family_template as t
    from tests.family._data import load_params

    from equilibria.family.driver import (
        calibrate_base_arrays,
        calibrate_base_cached,
        prepare_run,
    )

    load_params(t.DS)
    _env(mp)
    p = t._params(t.DS)
    run, arrays = prepare_run(
        p,
        t._route_closure(),
        list(p.sets.r)[-1],
        cache_root=tmp / "records",
        jax_cache=False,
    )
    settled, _ = calibrate_base_cached(arrays, run)
    direct, _ = calibrate_base_arrays(arrays, run)
    assert direct == settled
    return {
        "mp": mp_arrays(arrays),
        "settled": _nested(settled),
    }


def case_routes(mp: Any) -> dict:
    from tests.templates.gtap import test_routes_helper as t

    mp.setenv("EQUILIBRIA_SEED_CACHE_DISABLE", "1")
    for k in (
        "EQUILIBRIA_GTAP_MODEL_CACHE",
        "EQUILIBRIA_GTAP_SHOCK_CONTINUATION",
        "EQUILIBRIA_GTAP_BACKEND",
        "PATH_CAPI_OPTIONS",
    ):
        mp.delenv(k, raising=False)
    old = t._case("old")
    return {"codes": dict(old.codes), "levels": pyomo_levels(old.model)}


def case_variant(builder: str, mp: Any) -> dict:
    from tests.templates.gtap import test_gtap_template_builders as t

    from equilibria.templates.gtap.gtap_multiperiod_driver import solve_multiperiod
    from equilibria.templates.gtap_loglevels.multiperiod import build_loglevels_model_mp
    from equilibria.templates.gtap_sparse.multiperiod import build_sparse_model_mp

    gdx = None
    if builder == "loglevels":
        gdx = t._loglevels_reference()
        if gdx is None:
            return {
                "skipped": "gtap7_3x3 GAMS MCP reference (fbep=header) not recorded"
            }
        mp.setenv("EQUILIBRIA_GTAP_SOLVE_NLP", "1")
    p = t._params()
    cl = t._closure()
    if gdx is not None:
        p.set_va_subsidy_basis("gempack")
        cl = t._closure(va_subsidy_basis="gempack")
    rr = list(p.sets.r)[-1]
    if builder == "sparse":
        m = build_sparse_model_mp(p, p.sets, cl, rr)[0]
    else:
        m, model = build_loglevels_model_mp(p, p.sets, cl, rr, ref_gdx=gdx)
        model.seed_all_periods(m, gdx)
    res = solve_multiperiod(
        m,
        p,
        cl,
        ref_gdx=gdx,
        skip_base_solve=True,
        mute_welfare=True,
        seed_from_prior=False,
        mode="gtap",
    )
    return {
        "codes": {k: int(res[k]["code"]) for k in ("check", "shock")},
        "levels": {f"{k}": v for k, v in t._levels(m).items()},
    }


def case_burfisher(shock_method: str) -> dict:
    from tests.scripts import test_run_burfisher_routes as t

    from equilibria._local_refs import nus333_dir

    rb = t._rb()
    exp = "TBL45A" if shock_method == "johansen" else "TBL46A"
    m, _, codes = rb.solve_exercise(
        exp, nus333_dir(), shock_method=shock_method, route="old"
    )
    return {"codes": dict(codes), "levels": pyomo_levels(m)}


def case_on_core(name: str, mp: Any) -> dict:
    """The old side of ``_run_both`` in test_gtap_on_core_solve (its env fixture)."""
    from tests.templates.gtap import _period_snapshot as snap
    from tests.templates.gtap import test_gtap_on_core_solve as t
    from tests.templates.gtap._routes import solve

    mp.setenv("EQUILIBRIA_SEED_CACHE_DISABLE", "1")
    for k in (
        "EQUILIBRIA_GTAP_MODEL_CACHE",
        "EQUILIBRIA_GTAP_SHOCK_CONTINUATION",
        "PATH_CAPI_OPTIONS",
    ):
        mp.delenv(k, raising=False)
    kw: dict[str, Any] = {}
    if name == "overwrite_nus333":
        from tests.templates.gtap._nus333 import closure, nus333_params
        from tests.templates.gtap._unemployment import register_unemployment_hooks

        from equilibria.blocks.gtap import overwrite

        p, cl, rr = nus333_params(), closure(), "ROW"
        kw = {
            "shock": {"prdtx_rai": {("USA", "MFG", "MFG"): -10.9414}},
            "solve_check": True,
        }
        register_unemployment_hooks()
        try:
            old = solve("old", p, cl, rr, skip_base_solve=True, **kw)
        finally:
            overwrite.clear()
    else:
        p = snap._load_params()
        if name.startswith("static_"):
            _, mode, ifsub = name.split("_")
            cl, kw = t._closure(ifsub == "ifsub1"), {"mode": mode}
        else:
            cl = t._closure(False)
            kw = (
                {"shock_steps": (0.5, 1.0)}
                if name == "steps"
                else {"shock_method": "johansen"}
            )
        old = solve("old", p, cl, list(p.sets.r)[-1], skip_base_solve=True, **kw)
    return {"codes": dict(old.codes), "levels": pyomo_levels(old.model)}


def child(name: str, out: Path) -> None:
    import pytest

    mp = pytest.MonkeyPatch()
    with tempfile.TemporaryDirectory() as tmp:
        if name.startswith("family_"):
            res = case_family(name[len("family_") :], mp, Path(tmp))
        elif name == "template_build":
            res = case_template_build(mp, Path(tmp))
        elif name == "routes_3x3":
            res = case_routes(mp)
        elif name.startswith("variant_"):
            res = case_variant(name[len("variant_") :], mp)
        elif name.startswith("on_core_"):
            res = case_on_core(name[len("on_core_") :], mp)
        else:
            res = case_burfisher(name[len("burfisher_") :])
    mp.undo()
    out.write_text(json.dumps(res))


def run_case(name: str) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "out.json"
        env = {**os.environ, "PYTHONPATH": f"{ROOT / 'src'}:{ROOT}"}
        p = subprocess.run(
            [sys.executable, __file__, "--child", name, "--out", str(out)],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
        )
        if p.returncode != 0 or not out.exists():
            raise RuntimeError(f"{name}: exit {p.returncode}\n{p.stderr[-3000:]}")
        return json.loads(out.read_text())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--child", default=None, help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    if a.child:
        sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
        child(a.child, a.out)
        return 0
    out = {}
    for name in a.only or CASES:
        print(f"case {name}", flush=True)
        out[name] = run_case(name)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(a.out, "wt", encoding="utf-8") as f:
        json.dump(out, f, sort_keys=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
