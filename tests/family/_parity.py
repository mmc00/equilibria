"""The family (JAX) route measured against the GAMS references, cell by cell.

``family_match`` runs ``_solve_shock(..., backend="family")`` and applies the cell rule
of ``test_gtap7_mcp_parity`` (the same skipped variables, aliases and ``TOL``) to the
family's arrays instead of a Pyomo model. The cases come from the coverage matrix.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from tests.templates.gtap import test_gtap7_mcp_parity as mcp_gate

coverage_matrix = __import__("coverage_matrix")  # on sys.path via the mcp gate


@dataclass(frozen=True)
class _Run:
    """A family run on either route: its arrays and the code of every stage."""

    mp: Any
    codes: dict[str, int]


@dataclass(frozen=True)
class Case:
    dataset: str
    mode: str  # "pure" | "altertax"
    ifsub: int
    floors: dict  # stage -> floor (%)


def gams_cases() -> list[Case]:
    """One case per distinct (dataset, mode, ifsub) of the gams mcp and nlp rows; a
    stage's floor is the larger one when the two rows differ."""
    floors: dict[tuple, dict] = {}
    for r in coverage_matrix.mcp_rows() + coverage_matrix.nlp_rows():
        if r.reference != "gams":
            continue
        got = floors.setdefault((r.dataset, r.mode, r.ifsub), {})
        for stage, floor in r.stage_floors:
            got[stage] = max(floor, got.get(stage, floor))
    return [Case(d, m, i, f) for (d, m, i), f in sorted(floors.items())]


def altertax_cases() -> list[Case]:
    """The altertax rows against GAMS that are not blocked: every phase at gap_min."""
    return [
        Case(r.dataset, "altertax", r.ifsub, dict.fromkeys(r.phases, r.gap_min))
        for r in coverage_matrix.altertax_rows()
        if r.reference == "gams" and r.ci_status != "blocked"
    ]


def gams_reference(dataset: str, ifsub: int, mode: str) -> Path:
    """The GAMS MCP ``out.gdx`` (fbep=header) the case compares against; a missing
    one is an error, not a skip."""
    gdx = mcp_gate._fixture_gdx(dataset, ifsub, mode)
    if gdx is None or not gdx.exists():
        raise FileNotFoundError(
            f"no GAMS MCP fbep=header reference for {dataset} {mode} ifSUB={ifsub}"
        )
    return gdx


def _solve(
    dataset: str, ifsub: int, mode: str, unit_sigma: str = "gams", route: str = "old"
):
    """``(result, code)`` of the family backend on the photos' closure; ``result``
    carries the arrays as ``.mp`` on both routes (``route``: "old" or "core")."""
    import pytest

    from equilibria.family.driver import check_supported
    from tests.family._prep import _newton_mumps
    from tests.templates.gtap.test_gtap7_gempack_parity import _solve_shock

    if mode != "pure":
        # the route's own refusal, before any work
        check_supported(
            if_sub=bool(ifsub),
            savf_flag="capFix",
            mode=mode,
            skip_base_solve=True,
            mute_welfare=True,
            seed_from_prior=False,
            solve_check=True,
            shock_steps=None,
            shock_method="nonlinear",
            env={**os.environ, **_newton_mumps()},
        )
    import importlib.util

    linsolve = "mumps" if importlib.util.find_spec("mumps") else "direct"
    with pytest.MonkeyPatch.context() as mp:
        for k, v in {
            **_newton_mumps(),
            "EQUILIBRIA_GTAP_TR_LINSOLVE": linsolve,
        }.items():
            mp.setenv(k, v)
        if route == "core":
            # _solve_shock's family setup, through the route helper: it keeps the
            # code of every stage (the gate checks each one)
            from equilibria.family.photos import photo_closure
            from equilibria.sources import dataset_dir
            from equilibria.templates.gtap import GTAPParameters
            from tests.templates.gtap._routes import solve
            from tests.templates.gtap.test_gtap7_gempack_parity import _reference_gdx

            mp.setenv("EQUILIBRIA_GTAP_SOLVE_NLP", "1")
            d = dataset_dir(dataset)
            p = GTAPParameters()
            p.load_from_har(
                basedata_path=d / "basedata.har",
                sets_path=d / "sets.har",
                default_path=d / "default.prm",
                baserate_path=d / "baserate.har",
            )
            out = solve(
                "core",
                p,
                photo_closure(ifsub, "capFix", cast("Any", unit_sigma)),
                list(p.sets.r)[-1],
                backend="family",
                ref_gdx=_reference_gdx(dataset, ifsub),
            )
            return _Run(out.model, out.codes), out.codes["shock"]
        res, code = _solve_shock(
            dataset,
            ifsub,
            "capFix",
            solve_nlp=True,
            backend="family",
            unit_sigma=unit_sigma,
        )
        return _Run(res.mp, {k: int(r["code"]) for k, r in res.results.items()}), code


def family_match(
    dataset: str, ifsub: int, mode: str, gdx: Path, route: str = "old"
) -> dict[str, dict]:
    """``{stage: {"match": %, "total": cells, "code": code}}`` of the family run
    against ``gdx``."""
    from _diff_core import gams_levels, list_populated_vars, split_t  # type: ignore

    res, _ = _solve(dataset, ifsub, mode, route=route)
    mp = res.mp
    stages = ("base", "check", "shock")
    tot = dict.fromkeys(stages, 0)
    hit = dict.fromkeys(stages, 0)
    for vn in list_populated_vars(gdx):
        if vn.lower() in mcp_gate.SKIP or vn.lower() in mcp_gate.RF:
            continue
        name = mcp_gate.ALIAS.get(vn, vn)
        if name not in mp.vars:
            name = vn.lower()
        if name not in mp.vars:
            continue
        v = mp.vars[name]
        try:
            g = gams_levels(gdx, vn)
        except Exception:
            continue
        for fk, gval in g.items():
            try:
                body, t = split_t(fk)
            except Exception:
                continue
            if t not in tot:
                continue
            st = tuple(mcp_gate._strip(x) for x in body)
            val = None
            for cand in ((*st, t), (*body, t)):
                try:
                    val = float(v.value[v.pos(cand)])
                    break
                except (KeyError, IndexError):
                    pass
            if val is None:
                continue
            # a non-finite value stays in the denominator and fails the comparison
            # below, as in the Pyomo gate
            tot[t] += 1
            da = abs(val - gval)
            rel = da / abs(gval) if abs(gval) > 1e-12 else (0.0 if da < 1e-6 else 9e9)
            if da <= 1e-6 or rel <= mcp_gate.TOL:
                hit[t] += 1
    return {
        s: {
            "match": 100.0 * hit[s] / max(tot[s], 1),
            "total": tot[s],
            "code": res.codes[s],
        }
        for s in stages
    }


def gempack_cases() -> list[Case]:
    """One case per mcp row with ``reference == "gempack"`` (capFix, mode gtap)."""
    return [
        Case(r.dataset, r.mode, r.ifsub, dict(r.stage_floors))
        for r in coverage_matrix.rows_for("gtap7", "gempack", kind="mcp")
    ]


def gempack_reference(dataset: str) -> Path:
    """The capFix GEMPACK ``sl4dump.har`` the case compares against; a missing one is
    an error, not a skip."""
    from tests.templates.gtap.test_gtap7_gempack_parity import _capfix_fixture_for

    sl4 = _capfix_fixture_for(Case(dataset, "pure", 0, {}))
    if sl4 is None or not sl4.exists():
        raise FileNotFoundError(f"no capFix GEMPACK sl4dump for {dataset}")
    return sl4


def family_gempack_within(dataset: str, ifsub: int, route: str = "old"):
    """``(within_1pp, median_pp, cells, code)`` of the family run against GEMPACK: the
    rule of ``test_gtap7_gempack_parity._measure_pp`` (shock/check %-change of the
    ``Q_TO_VAR`` quantities, check falling back to base) on the family's arrays."""
    import statistics

    from gempack_reference import Q_TO_VAR, gempack_qty_pct  # type: ignore

    from tests.templates.gtap.test_gtap7_gempack_parity import PP_WITHIN

    sl4 = gempack_reference(dataset)
    res, code = _solve(dataset, ifsub, "pure", unit_sigma="gempack", route=route)
    mp = res.mp

    def level(v, key, period):
        try:
            return float(v.value[v.pos((*key, period))])
        except (KeyError, IndexError):
            return None

    diffs = []
    for gvar, spec in Q_TO_VAR.items():
        try:
            gem = gempack_qty_pct(str(sl4), gvar)
        except KeyError:
            continue
        v = mp.vars.get(spec["var"])
        if v is None:
            continue
        for key, gfrac in gem.items():
            s = level(v, key, "shock")
            if s is None:
                continue
            den = None
            for period in ("check", "base"):
                d = level(v, key, period)
                if d is not None and abs(d) > 1e-12:
                    den = d
                    break
            if den is None:
                continue
            d = abs(s / den - 1.0 - gfrac)
            # a non-finite family value is a miss, not a dropped cell
            diffs.append(d if math.isfinite(d) else math.inf)
    if not diffs:
        return 0.0, math.inf, 0, code
    within = sum(1 for x in diffs if x <= PP_WITHIN) / len(diffs)
    return within, statistics.median(diffs), len(diffs), code
