"""What each FBEP variant of a GAMS result must hold (#187, phase 2b oracle).

Written and committed before any run. The data come from the dataset's own
basedata.har, never from the GDX under test. At ``base``, for every factor
cell the HAR subsidises (FBEP != 0):

    zero    the official getData.gms: fbep = 0 and ftrv = EVFP - EVFB, so
            pfa/pf = EVFP/EVFB and ytax(fs) = 0
    header  FBEP/FTRV injected after fbep = 0, FBEP with its sign flipped (the
            HAR stores the subsidy negative, cal.gms takes fctts = -fbep/evfb):
            pfa/pf = 1 + (FTRV + FBEP)/EVFB = EVFP/EVFB, the data's wedge, in
            every subsidised cell, also where FTRV is zero (IND agriculture in
            10x7/15x10/20x41)

    ytax(fs) is checked for zero; header only has to show it somewhere
    (its value depends on more than these four headers): a header run whose
    injection silently did not happen would otherwise pass as header.

Amended 2026-11-01 (3ba4f88): the first version asked header for
1 + (ftrv - FBEP)/EVFB, the subsidy counted as a tax. That rule came from
cal.gms, not from the data, and encoded the bug the 20x41 header exposed:
the HAR satisfy EVFP = EVFB + FTRV + FBEP, so the base was no equilibrium.
Tightened the same day: the amended rule still took ftrv = EVFP - EVFB where
FTRV is zero, copying the generator's fallback, which counts FBEP twice there.
Tolerance widened the same day by the HAR's float32 rounding (wedge_ok): the
first 20x41 header result is exact in GAMS (1 + fctts + fcttx) but CHE Land
Rice, a 98% subsidy, sits 1.4e-6 from EVFP/EVFB because the data's identity
holds only to float32.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from equilibria.babel.har.reader import read_har
from equilibria.sources import dataset_dir, iter_sources

pytestmark = pytest.mark.needs_gdxdump
REL = 1e-6
RESULT = "gams_mcp_gtap_ifsub1_fbep-header"  # the 3x3 MCP result solved locally
# the HAR are float32: each term is rounded to half an epsilon of itself
HALF_EPS32 = float(np.finfo(np.float32).eps) / 2


def wedge_ok(got: float, evfp: float, evfb: float, ftrv: float, fbep: float) -> bool:
    """``got`` is the data's wedge EVFP/EVFB, to REL plus the HAR's own rounding.

    EVFP = EVFB + FTRV + FBEP holds only to the float32 rounding of its four
    terms; a near-full subsidy (small wedge) magnifies that relative to it.
    """
    want = evfp / evfb
    rounding = HALF_EPS32 * (abs(evfp) + abs(evfb) + abs(ftrv) + abs(fbep)) / abs(evfb)
    return abs(got - want) <= REL * abs(want) + rounding


def _gdx(gdx: Path, symbol: str) -> dict[tuple[str, ...], float]:
    """``symbol`` at base (or without a period), keyed by its labels."""
    gdxdump = shutil.which("gdxdump")
    assert gdxdump is not None
    dump = subprocess.run(
        [gdxdump, str(gdx), f"Symb={symbol}", "Format=csv"],
        capture_output=True, text=True, check=True,
    ).stdout  # fmt: skip
    out = {}
    for line in dump.splitlines()[1:]:
        *labels, value = line.split(",")
        keys = tuple(lbl.strip('"') for lbl in labels)
        if "base" in keys or not {"check", "shock"} & set(keys):
            out[tuple(k for k in keys if k != "base")] = float(value)
    return out


def _har(dataset: str) -> dict[str, Any]:
    har = read_har(
        dataset_dir(dataset) / "basedata.har",
        select_headers=["FBEP", "FTRV", "EVFB", "EVFP"],
    )
    # nus333's data has no FBEP header: no subsidised cell
    return {name: har[name] for name in ("FBEP", "FTRV", "EVFB", "EVFP") if name in har}


def variant_problems(gdx: Path, dataset: str, fbep: str) -> list[str]:
    """Every subsidised cell where ``gdx`` breaks what ``fbep`` must hold."""
    h = _har(dataset)
    pfa, pf = _gdx(gdx, "pfa"), _gdx(gdx, "pf")
    cells = np.argwhere(h["FBEP"].array != 0) if "FBEP" in h else []
    endw, acts, regs = h["FBEP"].set_elements if "FBEP" in h else ([], [], [])
    a_label = {a.split("_", 1)[-1] if a.startswith("a_") else a: a for (_, _, a) in pfa}
    problems = []
    for i, j, k in cells:
        f, a, r = endw[i], acts[j], regs[k]
        key = (r, f, a_label.get(a, a))
        fb, ft = (float(h[n].array[i, j, k]) for n in ("FBEP", "FTRV"))
        evfb, evfp = (float(h[n].array[i, j, k]) for n in ("EVFB", "EVFP"))
        want = evfp / evfb  # both variants carry the data's wedge
        if key not in pfa or key not in pf:
            problems.append(f"{key}: pfa/pf missing at base")
            continue
        got = pfa[key] / pf[key]
        if not wedge_ok(got, evfp, evfb, ft, fb):
            problems.append(f"{key}: pfa/pf {got:.9g}, {fbep} needs {want:.9g}")
    ytax_fs = {key: v for key, v in _gdx(gdx, "ytax").items() if "fs" in key}
    if fbep == "zero":
        problems += [
            f"ytax{key} = {v:.6g}, zero needs 0"
            for key, v in ytax_fs.items()
            if abs(v) > 1e-9
        ]
    elif len(cells) and not any(abs(v) > 1e-9 for v in ytax_fs.values()):
        problems.append("ytax(fs) = 0 everywhere, header needs the subsidy split out")
    return problems


def _recorded() -> list[Any]:
    return [
        r
        for r in iter_sources()
        if r.kind == "result"
        and r.tool == "gams"
        and r.variant.get("form") in ("mcp", "nlp")
        and "fbep" in r.variant
    ]


@pytest.mark.parametrize(
    "rec", _recorded() or [None], ids=lambda r: str(r and r.path.name)
)
def test_result_holds_its_fbep_variant(rec: Any) -> None:
    if rec is None:
        pytest.skip("no GAMS MCP/NLP result recorded with an fbep variant yet")
    assert (
        variant_problems(rec.path / "out.gdx", rec.dataset, rec.variant["fbep"]) == []
    )


def test_oracle_accepts_the_injected_reference() -> None:
    gdx = dataset_dir("gtap7_3x3") / RESULT / "out.gdx"
    assert variant_problems(gdx, "gtap7_3x3", "header") == []


def test_oracle_rejects_the_wrong_variant() -> None:
    # the header reference read as if it were fbep=zero must be red: both carry
    # the data's wedge, but only header splits out the subsidy, ytax(fs) != 0
    gdx = dataset_dir("gtap7_3x3") / RESULT / "out.gdx"
    problems = variant_problems(gdx, "gtap7_3x3", "zero")
    assert any("ytax" in p for p in problems)


def test_wedge_tolerance_is_the_data_precision() -> None:
    # the HAR are float32: EVFP = EVFB + FTRV + FBEP holds to their rounding
    # only, and a near-full subsidy (CHE Land Rice in 20x41, wedge 0.0195)
    # magnifies it to 1.4e-6 of the wedge. Both sides of the identity must
    # pass; a real error (1e-5) or the old sign (ftrv - FBEP) must not.
    h = _har("gtap7_20x41")
    m = h["FBEP"].array != 0
    evfp, evfb, ftrv, fbep = (
        h[n].array[m].astype(float) for n in ("EVFP", "EVFB", "FTRV", "FBEP")
    )
    want = evfp / evfb
    for got in ((evfb + ftrv + fbep) / evfb, want):
        assert all(map(wedge_ok, got, evfp, evfb, ftrv, fbep))
    assert not any(map(wedge_ok, want * (1 + 1e-5), evfp, evfb, ftrv, fbep))
    # the old sign differs by 2 FBEP/EVFB: rejected wherever that shows
    old_sign = 1 + (ftrv - fbep) / evfb
    visible = 2 * np.abs(fbep / evfb) > 1e-6
    assert visible.sum() > 400
    assert not any(
        map(wedge_ok, old_sign[visible], evfp[visible], evfb[visible],
            ftrv[visible], fbep[visible])
    )  # fmt: skip


def test_oracle_rejects_zero_read_as_header() -> None:
    # the mirror: a header run whose injection silently did not happen looks
    # like zero, so the zero reference read as header must be red too
    gdx = dataset_dir("gtap7_3x3") / RESULT.replace("header", "zero") / "out.gdx"
    problems = variant_problems(gdx, "gtap7_3x3", "header")
    assert any("ytax" in p for p in problems)
