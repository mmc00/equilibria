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

from tests.templates.gtap import test_gtap7_mcp_parity as mcp_gate

coverage_matrix = __import__("coverage_matrix")  # on sys.path via the mcp gate


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


def _solve(dataset: str, ifsub: int, mode: str):
    """``(FamilyResult, code)`` of the family route on the photos' closure."""
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
        return _solve_shock(dataset, ifsub, "capFix", solve_nlp=True, backend="family")


def family_match(dataset: str, ifsub: int, mode: str, gdx: Path) -> dict[str, dict]:
    """``{stage: {"match": %, "total": cells, "code": code}}`` of the family run
    against ``gdx``."""
    from _diff_core import gams_levels, list_populated_vars, split_t  # type: ignore

    res, _ = _solve(dataset, ifsub, mode)
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
            if val is None or not math.isfinite(val):
                continue
            tot[t] += 1
            da = abs(val - gval)
            rel = da / abs(gval) if abs(gval) > 1e-12 else (0.0 if da < 1e-6 else 9e9)
            if da <= 1e-6 or rel <= mcp_gate.TOL:
                hit[t] += 1
    return {
        s: {
            "match": 100.0 * hit[s] / max(tot[s], 1),
            "total": tot[s],
            "code": int(res.results[s]["code"]),
        }
        for s in stages
    }
