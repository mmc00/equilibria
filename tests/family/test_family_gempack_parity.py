"""The family (JAX) route against the GEMPACK references of the coverage matrix.

One case per ``mcp`` row with ``reference == "gempack"``: the family route solves the
capFix closure with exact sigma = 1 (``unit_sigma="gempack"``), as
``test_gtap7_gempack_parity`` does, and the shock/check quantity %-changes must reach
the row's shock floor as the share of cells within 1 pp of GEMPACK's ``sl4dump.har``.
The Pyomo test runs 15x10 as a slow NLP; the family route is fast there, so no case is
slow here.
"""

from __future__ import annotations

import pytest

from tests.family._parity import family_gempack_within, gempack_cases
from tests.templates.gtap._routes import ROUTES

pytest.importorskip("jax")
pytestmark = pytest.mark.integration


@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize(
    "case", gempack_cases(), ids=lambda c: f"{c.dataset}-ifsub{c.ifsub}"
)
def test_family_matches_gempack(case, route):
    within, med, n, code = family_gempack_within(case.dataset, case.ifsub, route)
    tag = f"{case.dataset}/gempack/ifSUB={case.ifsub}"
    print(f"[{tag}] within 1pp={within * 100:.2f}%/{n} median={med * 100:.3f}pp")
    assert code == 1, f"[{tag}] shock did not converge (code={code})"
    assert n > 0, f"[{tag}] no comparable quantity cells"
    floor = case.floors["shock"]
    assert within * 100 >= floor, (
        f"[{tag}] {within * 100:.2f}% of quantity cells within 1pp < floor {floor}% "
        f"(median |d|={med * 100:.3f}pp)"
    )


def test_cases_cover_every_gempack_row():
    from tests.family._parity import coverage_matrix

    want = {
        (r.dataset, r.ifsub)
        for r in coverage_matrix.rows_for("gtap7", "gempack", kind="mcp")
    }
    assert {(c.dataset, c.ifsub) for c in gempack_cases()} == want
