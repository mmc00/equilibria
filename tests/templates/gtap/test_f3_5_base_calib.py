"""F3.5 base-calibrated mode — adopt the settled (check) point as base so the
shock response matches GEMPACK instead of the check-contaminated GAMS path."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts" / "gtap"))
_PATH_CAPI = Path("/Users/marmol/proyectos/path-capi-python/src")
if _PATH_CAPI.exists() and str(_PATH_CAPI) not in sys.path:
    sys.path.insert(0, str(_PATH_CAPI))

DATASETS = ROOT / "datasets"
REF_GDX = ROOT / "tests/fixtures/gtap7/gtap7_3x3/out_gtap_shock_ifsub1.gdx"


def _has_solver():
    return importlib.util.find_spec("path_capi_python") is not None


def _ref_slice(period: str) -> dict[tuple[str, tuple], float]:
    from _diff_core import gams_levels, split_t

    out: dict[tuple[str, tuple], float] = {}
    for vn in ("pft", "xf", "xp", "va", "nd", "xda", "xma", "xaa"):
        try:
            g = gams_levels(REF_GDX, vn)
        except Exception:
            continue
        for fk, val in g.items():
            body, t = split_t(fk)
            if t == period:
                out[(vn, tuple(str(x) for x in body))] = val
    return out


def test_settled_base_land_response_matches_gempack():
    """The mechanism, on the reference: shock-vs-SETTLED-base land price is small
    (~-3%, near GEMPACK's -2.68%); shock-vs-RAW-base is the -18% contaminated path."""
    base = _ref_slice("base")
    check = _ref_slice("check")
    shock = _ref_slice("shock")
    k = ("pft", ("EU_28", "Land"))
    raw_resp = 100.0 * (shock[k] - base[k]) / base[k]
    settled_resp = 100.0 * (shock[k] - check[k]) / check[k]
    assert raw_resp < -15.0, (
        f"expected contaminated raw path <-15%, got {raw_resp:.2f}%"
    )
    assert -8.0 < settled_resp < 0.0, (
        f"expected clean settled response, got {settled_resp:.2f}%"
    )
    assert abs(settled_resp - (-2.68)) < abs(raw_resp - (-2.68))
