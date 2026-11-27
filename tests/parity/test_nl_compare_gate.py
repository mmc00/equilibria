"""nl_compare on the cascade compares the .nl of the case the MCP gate builds,
written by the core driver at the solve of each period (the other periods fixed,
as GAMS holds them), against the data/ GAMS .nl of the same variant (MAR-219).
It used to write the old single-period monolith (978 columns against GAMS's
2338) and look for the GAMS .nl in /tmp."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from equilibria.sources import find_reference

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "gtap"))

import nl_compare as NC  # noqa: E402  # ty: ignore[unresolved-import]

_NL = find_reference(
    "gtap7_3x3", "gams", form="nl", mode="altertax", ifsub=0, fbep="header"
)


def test_written_form_families_are_left_out_and_counted():
    stats = {"eq_pf": {"n_diff": 36}, "eq_pft": {"n_diff": 0}, "eq_x": {"n_diff": 0}}
    cells = [("eq_pf[USA,Land,Food]", "xf[USA,Land,Food]", 0.0, -1.0, 1.0, 1.0)]
    kept, left_out = NC.drop_written_form(stats, cells)
    assert kept == [] and left_out == {"eq_pf": 36}


def test_not_gams_cells_are_dropped_before_the_per_family_cap():
    """More amgm cells than cell_cap must not push a real diff of the same family
    past the cap, and the family count must leave them out (review of #275)."""
    from _nl_parser import NLModel  # ty: ignore[unresolved-import]

    def model(gams: bool) -> NLModel:
        m = NLModel()
        for k in range(60):
            m.con_names.append(f"eq_xmgm[Svces,R{k},Rice,CHN]")
            m.var_names.append(f"xwmg[R{k},Rice,CHN]")
            m.J[k] = {k: -1.00015 if gams else -1.0}
        m.con_names.append("eq_xmgm[Svces,R60,Rice,CHN]")
        m.var_names.append("xmgm[Svces,R60,Rice,CHN]")
        m.J[60] = {60: 2.0 if gams else 1.0}
        m.n_cons = m.n_vars = 61
        return m

    res = NC.diff_nl_models(
        model(False),
        model(True),
        cell_cap=50,
        skip_coefficients=NC.NOT_GAMS_COEFFICIENTS,
    )
    assert [c[1] for c in res["cell_diffs"]] == ["xmgm[Svces,R60,Rice,CHN]"]
    assert res["family_stats"]["eq_xmgm"]["n_diff"] == 1
    assert res["family_stats"]["eq_xmgm"]["n_not_gams"] == 60


def test_the_gate_case_has_its_references_on_gtap7_3x3():
    refs = NC.gate_references("gtap7_3x3", "altertax", 0)
    assert refs is not None and refs["nl"] == _NL


@pytest.mark.skipif(_NL is None, reason="GAMS .nl altertax ref absent")
@pytest.mark.needs_path
@pytest.mark.needs_gdxdump
@pytest.mark.parametrize("phase", ["check", "shock"])
def test_gtap7_3x3_altertax_is_clean(phase, tmp_path):
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/gtap/nl_compare.py",
            "--dataset",
            "gtap7_3x3",
            "--phase",
            phase,
            "--skip-gams",
            "--out-dir",
            str(tmp_path),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=1800,
    )
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    assert out["status"] == "clean", out["headline"]
    per = out["meta"]["per_phase"][phase]
    assert per["n_common_cons"] > 500
    assert per["written_form_left_out"] == {"eq_pf": 36}


def _nl_args(**kw):
    import argparse

    base = {
        "dataset": "gtap7_20x41",
        "mode": "altertax",
        "ifsub": 0,
        "skip_python": False,
        "phase": ["shock"],
        "gate_only": True,
        "gdx": None,
    }
    return argparse.Namespace(**{**base, **kw})


def test_the_cascade_case_never_falls_back_to_the_monolith(monkeypatch):
    """Without the data/ references of the gate case, --gate-only is an error, not a
    silent diff of the old single-period model (review of #275)."""
    monkeypatch.setattr(NC, "gate_references", lambda *a: None)
    out = NC._run_nl(_nl_args())
    assert out["status"] == "error"
    assert out["meta"]["error_kind"] == "gate_case_missing"


def test_the_cascade_case_seeds_from_the_cascade_gdx(monkeypatch, tmp_path):
    gdx = tmp_path / "other.gdx"
    gdx.write_bytes(b"x")
    monkeypatch.setattr(
        NC, "gate_references", lambda *a: {"nl": tmp_path, "gdx": Path("data.gdx")}
    )
    seen = {}
    monkeypatch.setattr(NC, "_run_nl_gate", lambda args, refs: seen.update(refs))
    NC._run_nl(_nl_args(gdx=gdx))
    assert seen["gdx"] == gdx


@pytest.mark.skipif(_NL is None, reason="GAMS .nl altertax ref absent")
@pytest.mark.needs_path
@pytest.mark.needs_gdxdump
def test_the_python_nl_is_the_system_path_solves(tmp_path):
    """The .nl is written after PATH's own fixes, as diff_mcp_pairing reads them:
    the zero-flow wedge rows PATH switches off (eq_dintxeq, eq_mintxeq) are not in
    it (MAR-252)."""
    refs = NC.gate_references("gtap7_3x3", "altertax", 0)
    assert refs is not None
    paths = NC.build_python_nls_gate(
        "gtap7_3x3", ["check", "shock"], tmp_path, "altertax", 0, refs["gdx"]
    )
    for period, nl in paths.items():
        rows = nl.with_suffix(".row").read_text().split()
        off = [r for r in rows if r.startswith(("eq_dintxeq[", "eq_mintxeq["))]
        assert not off, f"{period}: {len(off)} rows PATH switches off, e.g. {off[:2]}"
