"""diff_causal_propagation perturbs bucket A on the case the MCP gate builds, seeded
at the data/ reference, and reads only the requested period's cells (MAR-219). It
used to build the old single-period model and read the June refs."""

import importlib.util
import sys
from pathlib import Path

import pyomo.environ as pyo

from equilibria.sources import find_reference

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "gtap"))
_spec = importlib.util.spec_from_file_location(
    "diff_causal_propagation", ROOT / "scripts/gtap/diff_causal_propagation.py"
)
assert _spec is not None and _spec.loader is not None
CP = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(CP)

_FOLDER = find_reference(
    "gtap7_3x3", "gams", form="mcp", mode="altertax", ifsub=0, fbep="header"
)


def _model():
    """pf enters eq_xda in both periods; only check's cells may be read for check."""
    m = pyo.ConcreteModel()
    m.t = pyo.Set(initialize=["check", "shock"])
    m.pf = pyo.Var(m.t, initialize=2.0)
    m.xda = pyo.Var(m.t, initialize=1.0)
    m.eq_xda = pyo.Constraint(m.t, rule=lambda m, t: m.xda[t] == 0.5 * m.pf[t])
    return m


def test_scan_reads_only_the_requested_period():
    rows, n_o = CP.scan_coupling(_model(), "check", 1e-4)
    assert n_o == 1
    assert [(v, cell) for _, v, cell, _, _ in rows] == [("pf", ())]
    sens, _, _, touched, worst = rows[0]
    assert sens > CP.SYMPTOM_THRESHOLD and touched == 1 and worst == ("eq_xda", ())


def test_the_default_gdx_is_the_data_reference():
    assert _FOLDER is not None
    assert CP._default_gdx("gtap7_3x3", "altertax", 0) == _FOLDER / "out.gdx"


def test_an_empty_scan_is_an_error_not_independent(monkeypatch, tmp_path):
    """No bucket-O row or no free bucket-A cell measured nothing: never "clean"
    (review of #275)."""
    import argparse

    gdx = tmp_path / "out.gdx"
    gdx.write_bytes(b"x")
    monkeypatch.setattr(CP, "run_propagation_scan", lambda *a: ([], 0))
    out = CP._work(
        argparse.Namespace(
            dataset="gtap7_3x3",
            gdx=gdx,
            period="check",
            delta=1e-4,
            mode="altertax",
            ifsub=0,
        )
    )
    assert out["status"] == "error"
    assert out["meta"]["error_kind"] == "nothing_scanned"
