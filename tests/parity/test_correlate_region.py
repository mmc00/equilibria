"""The correlated-by-region grouper: dirty layers sharing a common first-index token
(region/block) are grouped so the reader doesn't correlate by hand. A locus touched by
only ONE layer is not a correlation. Only EXPLAIN_STOP (dirty) layers participate."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "gtap"))

from cascade_classify import LayerResult, EXPLAIN_STOP, CONTINUE
from cascade_orchestrator import _correlate_by_region


def _layer(name, viols, action=EXPLAIN_STOP):
    return LayerResult(name, "dirty", None, "h", action, 1, {"violations": viols})


def _v(entity, index):
    return {"entity": entity, "index": index, "metric": "m", "value": 1.0}


def test_two_layers_same_region_group():
    results = [
        _layer("tautology", [_v("pd", ["EU_28", "Food"])]),
        _layer("calibration", [_v("ytax[gc]", ["EU_28"])]),
    ]
    groups = _correlate_by_region(results)
    assert len(groups) == 1
    g = groups[0]
    assert g["locus"] == "EU_28"
    assert set(g["layers"]) == {"tautology", "calibration"}


def test_single_layer_locus_is_not_a_group():
    results = [
        _layer("tautology", [_v("pd", ["USA", "Food"])]),
        _layer("calibration", [_v("ytax[gc]", ["EU_28"])]),
    ]
    # USA touched only by tautology, EU_28 only by calibration → no correlation
    assert _correlate_by_region(results) == []


def test_clean_layers_excluded():
    results = [
        _layer("tautology", [_v("pd", ["EU_28"])]),
        _layer("kkt", [_v("x", ["EU_28"])], action=CONTINUE),  # clean → excluded
    ]
    # only one dirty layer touches EU_28 → no group
    assert _correlate_by_region(results) == []


def test_three_layers_one_region_ranked_first():
    results = [
        _layer("tautology", [_v("pd", ["EU_28"])]),
        _layer("calibration", [_v("ytax", ["EU_28"]), _v("ytax", ["USA"])]),
        _layer("drift", [_v("xg", ["EU_28"]), _v("xg", ["USA"])]),
        _layer("probe", [_v("eq", ["USA"])]),
    ]
    groups = _correlate_by_region(results)
    loci = {g["locus"]: set(g["layers"]) for g in groups}
    # EU_28: tautology+calibration+drift (3); USA: calibration+drift+probe (3)
    assert loci["EU_28"] == {"tautology", "calibration", "drift"}
    assert loci["USA"] == {"calibration", "drift", "probe"}


def test_empty_index_does_not_group():
    # holdfixed-style violations carry index=[] (no region) → never correlate
    results = [
        _layer("holdfixed", [_v("pf", [])]),
        _layer("tautology", [_v("pd", ["EU_28"])]),
    ]
    assert _correlate_by_region(results) == []
