import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "gtap"))

from cascade_classify import LayerResult
from cascade_config import GdxResolution
from cascade_orchestrator import build_report, render_tree


def _lr(name, status, action, kind=None, headline="h"):
    return LayerResult(name, status, kind, headline, action, 0, {})


def test_report_records_first_dirty_layer():
    ref = GdxResolution(Path("/ref/out.gdx"), "durable", "n", True)
    results = {
        "shock": [
            _lr("mcp_pairing", "clean", "continue"),
            _lr("nl_compare", "dirty", "explain_stop"),
        ],
    }
    rep = build_report(
        "gtap7_3x3",
        ["check", "shock"],
        ref=ref,
        period_results=results,
        kkt_reader="pure-python",
    )
    assert rep["periods"]["shock"]["first_dirty_layer"] == "nl_compare"
    assert rep["ref"]["source"] == "durable"
    assert rep["kkt_reader"] == "pure-python"


def test_tree_shows_provenance_line():
    ref = GdxResolution(Path("/ref/out.gdx"), "adapter_output", "n", True)
    results = {"shock": [_lr("mcp_pairing", "clean", "continue")]}
    rep = build_report(
        "gtap7_3x3",
        ["shock"],
        ref=ref,
        period_results=results,
        kkt_reader="gdxdump-text",
    )
    tree = render_tree(rep)
    assert "/ref/out.gdx" in tree
    assert "adapter-fallback" in tree
    assert "gdxdump-text" in tree


def test_vacuous_layer_is_visible_in_report():
    ref = GdxResolution(Path("/ref/out.gdx"), "durable", "n", True)
    results = {
        "shock": [
            _lr(
                "nl_compare",
                "error",
                "vacuous_continue",
                kind="no_common_constraints",
                headline="no common constraints",
            )
        ]
    }
    rep = build_report(
        "gtap7_3x3",
        ["shock"],
        ref=ref,
        period_results=results,
        kkt_reader="pure-python",
    )
    layer = rep["periods"]["shock"]["layers"][0]
    assert layer["action"] == "vacuous_continue"
    assert layer["status"] == "error"  # surfaced, not painted clean
    tree = render_tree(rep)
    assert "did not opine" in tree or "vacuous" in tree
