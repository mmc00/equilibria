"""3x3 check+shock with newton_tr: family F/J must give Pyomo's solution cell by cell."""

import importlib.util
import re
from pathlib import Path

import pytest

pytest.importorskip("jax")

ROOT = Path(__file__).resolve().parents[2]
# Cross-period Fisher rows whose multiperiod form differs from the SP one. eq_pgdpmp
# is cross-period too, but its check/shock form equals the SP row (verified at three
# points, pattern included), so it is served.
FALLBACK = {"eq_pabs", "eq_pfact", "eq_pwfact", "eq_rgdpmp", "eq_chisave"}


@pytest.mark.slow
def test_family_solve_matches_pyomo_3x3():
    path = ROOT / "scripts" / "family" / "hybrid_solve_3x3.py"
    spec = importlib.util.spec_from_file_location("hybrid_solve_3x3", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    run = mod.run

    ref, fam = run(family=False), run(family=True)
    assert ref["code"] == 1 and fam["code"] == 1
    assert ref["family_eval"] == []
    assert fam["nit"] == ref["nit"] and len(ref["nit"]) >= 2

    periods = [re.findall(r"period=(\w+)", s)[0] for s in fam["family_eval"]]
    assert {"check", "shock"} <= set(periods)
    for s in fam["family_eval"]:
        fallback = set(re.findall(r"'(\w+)'", s.split("fallback=", 1)[1]))
        assert fallback == FALLBACK, s

    bad = []
    for name, b in ref["levels"].items():
        if not (
            name.endswith(",check]")
            or name.endswith(",shock]")
            or name.endswith("[check]")
            or name.endswith("[shock]")
        ):
            continue
        a = fam["levels"][name]
        if b is None or a is None:
            assert a == b, name
            continue
        if abs(a - b) > 1e-9 * max(1.0, abs(b)):
            bad.append((name, a, b))
    assert not bad, f"{len(bad)} cells differ, first: {bad[:5]}"
