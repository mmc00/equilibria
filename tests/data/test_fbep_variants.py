"""What each FBEP variant of a GAMS result must hold (#187, phase 2b oracle).

Written and committed before any run. The data come from the dataset's own
basedata.har, never from the GDX under test. At ``base``, for every factor
cell the HAR subsidises (FBEP != 0):

    zero    the official getData.gms: fbep = 0 and ftrv = EVFP - EVFB, so
            pfa/pf = EVFP/EVFB and ytax(fs) = 0
    header  FBEP/FTRV injected after fbep = 0, FBEP with its sign flipped (the
            HAR stores the subsidy negative, cal.gms takes fctts = -fbep/evfb):
            pfa/pf = 1 + (ftrv + FBEP)/EVFB with ftrv = FTRV where the header
            is nonzero, else EVFP - EVFB (the injection only overrides nonzero
            cells); where FTRV is nonzero that is EVFP/EVFB, the data's wedge

    ytax(fs) is checked for zero only: header's ytax(fs) depends on more
    than these four headers.

Amended 2026-11-01 (3ba4f88): the first version asked header for
1 + (ftrv - FBEP)/EVFB, the subsidy counted as a tax. That rule came from
cal.gms, not from the data, and encoded the bug the 20x41 header exposed:
the HAR satisfy EVFP = EVFB + FTRV + FBEP, so the base was no equilibrium.
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
    return {name: har[name] for name in ("FBEP", "FTRV", "EVFB", "EVFP")}


def variant_problems(gdx: Path, dataset: str, fbep: str) -> list[str]:
    """Every subsidised cell where ``gdx`` breaks what ``fbep`` must hold."""
    h = _har(dataset)
    endw, acts, regs = h["FBEP"].set_elements
    pfa, pf = _gdx(gdx, "pfa"), _gdx(gdx, "pf")
    a_label = {a.split("_", 1)[-1] if a.startswith("a_") else a: a for (_, _, a) in pfa}
    problems = []
    for i, j, k in np.argwhere(h["FBEP"].array != 0):
        f, a, r = endw[i], acts[j], regs[k]
        key = (r, f, a_label.get(a, a))
        fb, ft = (float(h[n].array[i, j, k]) for n in ("FBEP", "FTRV"))
        evfb, evfp = (float(h[n].array[i, j, k]) for n in ("EVFB", "EVFP"))
        if fbep == "zero":
            want = evfp / evfb
        else:
            ftrv = ft if ft != 0 else evfp - evfb
            want = 1 + (ftrv + fb) / evfb
        if key not in pfa or key not in pf:
            problems.append(f"{key}: pfa/pf missing at base")
            continue
        got = pfa[key] / pf[key]
        if abs(got - want) > REL * abs(want):
            problems.append(f"{key}: pfa/pf {got:.9g}, {fbep} needs {want:.9g}")
    if fbep == "zero":
        problems += [
            f"ytax{key} = {v:.6g}, zero needs 0"
            for key, v in _gdx(gdx, "ytax").items()
            if "fs" in key and abs(v) > 1e-9
        ]
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
