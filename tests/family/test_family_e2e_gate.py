"""The large datasets end to end against their committed photo.

15x10 and 20x41 keep only the ``gate`` photo in git (``photos.VARIANTS``: the run's
``final.pkl`` and the per-period summaries, not every step), so their per-step tests
stay slow and local. This is their check (20x41 slow, local; see the mark): ``_solve_shock(..., backend="family")``
matches ``final.pkl`` on every Var cell of the three periods within 1e-9, with the
fixed flags and bounds, every period at code 1, as ``test_family_e2e`` does on 3x3.
"""

from __future__ import annotations

import json
import os

import pytest

from tests.family._prep import _newton_mumps, capture_dir
from tests.family.test_mp_build import diff_to_photo

pytest.importorskip("pyomo")


# 20x41 starves a standard GitHub runner (it lost the runner twice, at 51 and 64 min,
# 2026-11-16) and a larger one is not paid for: slow, run on the Mac (6.6 min cold,
# peak RSS 3.9 GB). 15x10 takes 1.4 min cold, 0.3 warm, 2.8 GB on the runner.
@pytest.mark.parametrize(
    "dataset", ["gtap7_15x10", pytest.param("gtap7_20x41", marks=pytest.mark.slow)]
)
def test_family_run_equals_the_gate_photo(dataset, monkeypatch):
    from equilibria.family import photos
    from equilibria.family.prep_capture import load
    from tests.templates.gtap.test_gtap7_gempack_parity import _solve_shock

    # the photos are MUMPS runs: another linear solver raises PhotoError here
    reference = capture_dir(dataset, photos.LINSOLVE)
    for k, v in {
        **_newton_mumps(),
        "EQUILIBRIA_GTAP_TR_LINSOLVE": photos.LINSOLVE,
    }.items():
        monkeypatch.setenv(k, v)
    res, code = _solve_shock(dataset, 1, "capFix", solve_nlp=True, backend="family")

    codes = {p: int(r["code"]) for p, r in res.results.items()}
    assert code == 1 and set(codes.values()) == {1}, codes
    for n, period in ((2, "check"), (4, "check"), (5, "shock")):
        s = json.loads((reference / f"{n:03d}_S_{period}.json").read_text())
        assert s["code"] == 1
    final = load(reference / "final.pkl")
    assert diff_to_photo(res.mp, final, 1e-9, lb_rtol=1e-9) == []
    print(
        dataset, "nit", {p: r.get("nit") for p, r in res.results.items()}, os.getpid()
    )
