"""Each subsidy basis books the factor wedge as its GAMS reference does (#191, #193).

The data satisfy EVFP = EVFB + FTRV + FBEP, with the subsidy FBEP <= 0.

- "gams" is the official getData.gms (``fbep=zero``): fbep = 0 and
  ftrv = EVFP - EVFB, one net tax that holds the subsidy.
- "gempack" is the HAR's headers (``fbep=header``): the tax is the FTRV header
  (0 where the data subsidise with no tax) and the subsidy is +FBEP.

Both carry the data's wedge, 1 + fcttx + fctts = EVFP / EVFB.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from equilibria.blocks.gtap.factor_wedge import factor_wedge_rates, factor_wedge_values

TAXED = ("USA", "Land", "Food")  # FTRV header and FBEP
ONLY_SUBSIDISED = ("IND", "Land", "Rice")  # FBEP, no FTRV header (10x7 IND)


def _bench() -> SimpleNamespace:
    evfb = {TAXED: 100.0, ONLY_SUBSIDISED: 100.0}
    ftrv_har = {TAXED: 30.0}
    fbep = {TAXED: -10.0, ONLY_SUBSIDISED: -5.0}
    vfm = {k: evfb[k] + ftrv_har.get(k, 0.0) + fbep[k] for k in evfb}  # EVFP
    # what the loader keeps in ftrv: EVFP - EVFB, overwritten by the header
    ftrv = {k: vfm[k] - evfb[k] for k in evfb} | ftrv_har
    return SimpleNamespace(vfm=vfm, evfb=evfb, fbep=fbep, ftrv=ftrv, ftrv_har=ftrv_har)


@pytest.mark.parametrize(
    "cell", [TAXED, ONLY_SUBSIDISED], ids=["taxed", "only_subsidised"]
)
def test_gams_basis_is_the_official_net_wedge(cell) -> None:
    bm = _bench()
    assert factor_wedge_values(bm, cell[0], cell[1], cell[2], "gams") == pytest.approx(
        (bm.vfm[cell] - bm.evfb[cell], 0.0)
    )


@pytest.mark.parametrize(
    "cell,tax",
    [(TAXED, 30.0), (ONLY_SUBSIDISED, 0.0)],
    ids=["taxed", "only_subsidised"],
)
def test_gempack_basis_is_the_har_headers(cell, tax) -> None:
    bm = _bench()
    assert factor_wedge_values(
        bm, cell[0], cell[1], cell[2], "gempack"
    ) == pytest.approx((tax, bm.fbep[cell]))


@pytest.mark.parametrize("basis", ["gams", "gempack"])
@pytest.mark.parametrize(
    "cell", [TAXED, ONLY_SUBSIDISED], ids=["taxed", "only_subsidised"]
)
def test_both_bases_carry_the_data_wedge(basis, cell) -> None:
    bm = _bench()
    fcttx, fctts = factor_wedge_rates(bm, cell[0], cell[1], cell[2], basis)
    assert 1 + fcttx + fctts == pytest.approx(bm.vfm[cell] / bm.evfb[cell])


def test_loader_keeps_the_ftrv_header_for_gempack() -> None:
    # gempack's tax is the FTRV header itself, not EVFP - EVFB - FBEP: the
    # identity holds only to the HAR's float32 rounding
    from equilibria.babel.har.reader import read_har
    from equilibria.sources import dataset_dir
    from equilibria.templates.gtap import GTAPParameters

    d = dataset_dir("gtap7_3x3")
    p = GTAPParameters()
    p.load_from_har(
        basedata_path=d / "basedata.har",
        sets_path=d / "sets.har",
        default_path=d / "default.prm",
        baserate_path=d / "baserate.har",
    )
    har = read_har(d / "basedata.har", select_headers=["FTRV"])["FTRV"]
    endw, acts, regs = har.set_elements
    bm = p.benchmark
    checked = 0
    for (i, j, k), v in np.ndenumerate(har.array):
        if float(v) == 0.0:
            continue
        key = (regs[k], endw[i], acts[j])
        if key not in bm.evfb:
            continue
        assert (
            factor_wedge_values(bm, key[0], key[1], key[2], "gempack")[0]
            == bm.ftrv[key]
        )
        checked += 1
    assert checked > 0
