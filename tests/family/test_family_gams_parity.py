"""The family (JAX) route against the GAMS references of the coverage matrix.

One case per distinct ``(dataset, mode, ifsub)`` of the ``mcp`` and ``nlp`` rows with
``reference == "gams"`` (the PATH and NLP rows compare the same solution: on the family
route it is one run), and one per ``altertax`` row that is not blocked. Each stage must
reach the row's floor (the larger one when the mcp and nlp rows differ) with code 1, by
the cell rule of ``test_gtap7_mcp_parity``: a cell matches if ``da <= 1e-6 or rel <= 1e-2``.

The family route runs ``mode="gtap"`` only: the altertax cases are strict xfails on its
refusal, so they turn red the day the route accepts them (Task 6 keeps their Pyomo tests).
"""

from __future__ import annotations

import pytest

from tests.family._parity import (
    altertax_cases,
    family_match,
    gams_cases,
    gams_reference,
)

pytest.importorskip("jax")
pytestmark = [pytest.mark.integration, pytest.mark.needs_gdxdump]


def _params(cases):
    out = []
    for c in cases:
        marks = []
        if c.mode != "pure":
            marks.append(
                pytest.mark.xfail(
                    raises=NotImplementedError,
                    strict=True,
                    reason="family route runs mode='gtap' only",
                )
            )
        out.append(
            pytest.param(c, marks=marks, id=f"{c.dataset}-{c.mode}-ifsub{c.ifsub}")
        )
    return out


@pytest.mark.parametrize("case", _params(gams_cases()))
def test_family_matches_gams(case):
    gdx = gams_reference(case.dataset, case.ifsub, case.mode)
    per = family_match(case.dataset, case.ifsub, case.mode, gdx)
    tag = f"{case.dataset}/{case.mode}/ifSUB={case.ifsub}"
    print(
        f"[{tag}] "
        + " ".join(f"{k}={v['match']:.2f}%/{v['total']}" for k, v in per.items())
    )
    bad = {k: v["code"] for k, v in per.items() if v["code"] != 1}
    assert not bad, f"[{tag}] stages did not converge: {bad}"
    for stage, floor in case.floors.items():
        mm = per[stage]
        assert mm["total"] > 0, f"[{tag}/{stage}] no comparable cells"
        assert mm["match"] >= floor, (
            f"[{tag}/{stage}] family match {mm['match']:.2f}% < floor {floor}% "
            f"(over {mm['total']} cells)"
        )


@pytest.mark.parametrize("case", _params(altertax_cases()))
def test_family_altertax_matches_gams(case):
    gdx = gams_reference(case.dataset, case.ifsub, case.mode)
    per = family_match(case.dataset, case.ifsub, case.mode, gdx)
    tag = f"{case.dataset}/altertax/ifSUB={case.ifsub}"
    for stage in case.floors:
        assert per[stage]["code"] == 1, f"[{tag}/{stage}] code {per[stage]['code']}"
        assert per[stage]["match"] >= case.floors[stage], (
            f"[{tag}/{stage}] family match {per[stage]['match']:.2f}% < "
            f"gap_min {case.floors[stage]}%"
        )


def test_cases_cover_every_gams_row():
    """Every gams mcp/nlp row and every unblocked altertax row has its case."""
    from tests.family._parity import coverage_matrix

    want = {
        (r.dataset, r.mode, r.ifsub)
        for r in coverage_matrix.mcp_rows() + coverage_matrix.nlp_rows()
        if r.reference == "gams"
    }
    assert {(c.dataset, c.mode, c.ifsub) for c in gams_cases()} == want
    alt = {
        (r.dataset, r.ifsub)
        for r in coverage_matrix.altertax_rows()
        if r.reference == "gams" and r.ci_status != "blocked"
    }
    assert {(c.dataset, c.ifsub) for c in altertax_cases()} == alt


def test_floor_is_the_larger_of_the_mcp_and_nlp_rows():
    from tests.family._parity import coverage_matrix

    for c in gams_cases():
        for r in coverage_matrix.mcp_rows() + coverage_matrix.nlp_rows():
            if (r.dataset, r.mode, r.ifsub, r.reference) != (
                c.dataset,
                c.mode,
                c.ifsub,
                "gams",
            ):
                continue
            for stage, floor in r.stage_floors:
                assert c.floors[stage] >= floor
