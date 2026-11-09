"""``summary.har`` of ALTERTAX on the family route: GEMPACK's GTAPSUM output.

``gtapv7.tab`` section E (lines 5214-5330) builds the data summary and accounting checks
from the value coefficients of the database (``VDFP``, ``EVFP``, ``MAKEB``, ...). The
``Formula`` statements run pre-simulation, on the input data, and ``postsim`` re-writes
the five percentage checks (``*Q`` headers) after the ``Update`` statements, on the
updated data. So the four level tables (``GDPE COST GDPI SALE``) are computed from the
``check`` period (which reproduces the base) and the five ``*Q`` headers from the ``shock``
period, with the value builders of :mod:`equilibria.family.altertax.updated`. The ten
pre-simulation accounting checks (``VMCK .. VOCS``) are the rounding residuals of GEMPACK's
float32 input data: the model is exact in float64 and returns zero for them, so they are
computed on the input ``basedata.har`` itself, as GEMPACK does (``_PRE_ON_DATA``).

====== =========================================================== ========
header formula (``X(...)`` are the value coefficients)             tab line
====== =========================================================== ========
GDPE   VPP, VIP, VGP sums; sum VFOB; sum VST; -sum VCIF             2965-2973
COST   sum_c VDFP, sum_c VMFP, EVFP, sum_c PTAX                     5235-5238
GDPI   sum_a EVFP; INDTAX - TAXRFU                                  5247-5248
SALE   sum_a VDFB, VDPB, VDIB, VDGB, VXDXSB, VST (0 if not margin)  5255-5261
VMCK   VCB - MAKEBCOM                                               5268
VMCP   100 * VMCK / VCB                                             5273
TRCK   VCIF - (VFOB + VTFSD)                                        5280
TRCP   100 * TRCK / VCIF                                            5284
VTCK   VTMUSE - VTMPROV                                             5291
VTCP   100 * VTCK / VTMUSE                                          5295
MPCK   sum_s VMSB - sum_a VMFB - VMPB - VMGB - VMIB                 5305
MPCP   100 * MPCK / sum_s VMSB                                      5310
VOCK   MAKEB - (MAKES + PTAX)  (PTAX = MAKEB - MAKES)              5317
VOCS   100 * VOCK / MAKEB                                           5321
*Q     the five percentages (VMCP TRCP VTCP MPCP VOCS) post-sim     5325-5330
====== =========================================================== ========

``INDTAX - TAXRFU`` (line 1389) is TAXRPC + TAXRGC + TAXRIC + TAXRIU + TAXROUT + TAXREXP
+ TAXRIMP, each a difference of two value headers (lines 722-770, 734, 800, 1360-1385).
``VTFSD = sum_m VTWR`` (1888), ``VTMUSE = sum VTWR`` (1901), ``VTMPROV = sum_r VST`` (1924).
A zero denominator gives zero (GEMPACK prints 0 for those cells).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from equilibria.babel.har.reader import read_har
from equilibria.babel.har.symbols import HeaderArray
from equilibria.babel.har.writer import write_har
from equilibria.family.altertax.run import AltertaxRun
from equilibria.family.altertax.updated import headers_at

# (header, coefficient, long name, set names) as in ``altax_summary.har``
_META: tuple[tuple[str, str, str, tuple[str, ...]], ...] = (
    ("GDPE", "GDPEXP", " GDP expenditure side summary", ("REG", "GDPEX")),
    ("COST", "COSTSUM", " industry cost summary", ("ACTS", "REG", "COSTS")),
    ("GDPI", "GDPINC", " income-side GDP", ("REG", "GDPIN")),
    ("SALE", "SALESUM", " commodity sales summary", ("COMM", "REG", "SALES")),
    ("VMCK", "VDBCHK", " VCB(c,r) - MAKEBCOM(c,r) should be tiny", ("COMM", "REG")),
    ("VMCP", "VDBCHK", "VDBCHK as %", ("COMM", "REG")),
    ("TRCK", "TRDCHK", " VCIF(c,s,d) - [VFOB(c,s,d) + VTFSD(c,s,d)] should be tiny", ("COMM", "REG", "REG")),
    ("TRCP", "TRDCHK", "TRDCHK as %", ("COMM", "REG", "REG")),
    ("VTCK", "VTMCHK", " VTMUSE(m) - VTMPROV(m) should be tiny", ("MARG",)),
    ("VTCP", "VTMCHK", "VTMCHK as %", ("MARG",)),
    ("MPCK", "CHKMKClIMP", " Checking the accounting on imports supply and demand", ("COMM", "REG")),
    ("MPCP", "CHKMKClIMP", "CHKMKClIMP as %", ("COMM", "REG")),
    ("VOCK", "VOSCHK", " MAKEB - [MAKES + PTAX] should be tiny", ("COMM", "ACTS", "REG")),
    ("VOCS", "VOSCHK", "VOSCHK as %", ("COMM", "ACTS", "REG")),
    ("VMCQ", "VDBCHK", "PostSim VDBCHK as %", ("COMM", "REG")),
    ("TRCQ", "TRDCHK", "PostSim TRDCHK as %", ("COMM", "REG", "REG")),
    ("VTCQ", "VTMCHK", "PostSim VTMCHK as %", ("MARG",)),
    ("MPCQ", "CHKMKClIMP", "PostSim CHKMKClIMP as %", ("COMM", "REG")),
    ("VOCQ", "VOSCHK", "PostSim VOPCHK as %", ("COMM", "ACTS", "REG")),
)  # fmt: skip
_BY_NAME = {m[0]: m for m in _META}
SUMMARY_HEADERS: tuple[str, ...] = tuple(m[0] for m in _META)
_POST = {"VMCQ": "VMCP", "TRCQ": "TRCP", "VTCQ": "VTCP", "MPCQ": "MPCP", "VOCQ": "VOCS"}
_PRE_ON_DATA = [
    "VMCK",
    "VMCP",
    "TRCK",
    "TRCP",
    "VTCK",
    "VTCP",
    "MPCK",
    "MPCP",
    "VOCK",
    "VOCS",
]  # accounting checks of the input data, not of the model
_GDPEX = ["Household", "Investment", "Government", "Exports", "IntnlMargins", "Imports"]
_SALES = [
    "Intermediate",
    "Household",
    "Investment",
    "Government",
    "Exports",
    "IntnlMargins",
]


def _pct(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """``100 * num / den``, zero where ``den`` is zero."""
    safe = np.where(den == 0.0, 1.0, den)
    return np.where(den == 0.0, 0.0, 100.0 * num / safe)


def _tables(
    db: dict[str, HeaderArray],
) -> tuple[dict[str, np.ndarray], dict[str, list]]:
    """Raw tables and the element lists of the set names, from a database of value headers."""
    v = {k: np.asarray(h.array, dtype=np.float64) for k, h in db.items()}
    els = {
        "COMM": list(db["VDFB"].set_elements[0]),
        "ACTS": list(db["VDFB"].set_elements[1]),
        "REG": list(db["VDFB"].set_elements[2]),
        "ENDW": list(db["EVFP"].set_elements[0]),
        "MARG": list(db["VST"].set_elements[0]),
    }
    return v, els


def summary_from_database(db: dict[str, HeaderArray]) -> dict[str, HeaderArray]:
    """The 19 GTAPSUM headers of a database of value headers (``VDFB`` ... ``VTWR``,
    ``MAKB``, ``MAKS``), as GEMPACK evaluates the section-E formulas on it."""
    v, els = _tables(db)
    nc, na, nr = len(els["COMM"]), len(els["ACTS"]), len(els["REG"])
    marg = [els["COMM"].index(m) for m in els["MARG"]]

    def vsum(*names: str) -> np.ndarray:  # sum of (COMM, REG) agent headers
        return np.sum([v[n] for n in names], axis=0)

    # --- GDPE (2965-2973)
    gdpe = np.zeros((nr, 6))
    gdpe[:, 0] = vsum("VDPP", "VMPP").sum(axis=0)
    gdpe[:, 1] = vsum("VDIP", "VMIP").sum(axis=0)
    gdpe[:, 2] = vsum("VDGP", "VMGP").sum(axis=0)
    gdpe[:, 3] = v["VFOB"].sum(axis=(0, 2))
    gdpe[:, 4] = v["VST"].sum(axis=0)
    gdpe[:, 5] = -v["VCIF"].sum(axis=(0, 1))

    # --- tax coefficients (722-770, 800, 1360-1385)
    ptax = v["MAKB"] - v["MAKS"]  # (c, a, r)
    taxrpc = (vsum("VDPP", "VMPP") - vsum("VDPB", "VMPB")).sum(axis=0)
    taxrgc = (vsum("VDGP", "VMGP") - vsum("VDGB", "VMGB")).sum(axis=0)
    taxric = (vsum("VDIP", "VMIP") - vsum("VDIB", "VMIB")).sum(axis=0)
    taxriu = (v["VDFP"] - v["VDFB"] + v["VMFP"] - v["VMFB"]).sum(axis=(0, 1))
    taxrfu = (v["EVFP"] - v["EVFB"]).sum(axis=(0, 1))
    taxrout = ptax.sum(axis=(0, 1))
    taxrexp = (v["VFOB"] - v["VXSB"]).sum(axis=(0, 2))  # by source s
    taxrimp = (v["VMSB"] - v["VCIF"]).sum(axis=(0, 1))  # by destination d
    indtax = taxrpc + taxrgc + taxric + taxriu + taxrfu + taxrout + taxrexp + taxrimp

    # --- COST (5235-5238), GDPI (5247-5248)
    nf = len(els["ENDW"])
    cost = np.zeros((na, nr, 3 + nf))
    cost[:, :, 0] = v["VDFP"].sum(axis=0)
    cost[:, :, 1] = v["VMFP"].sum(axis=0)
    cost[:, :, 2 : 2 + nf] = np.moveaxis(v["EVFP"], 0, -1)
    cost[:, :, 2 + nf] = ptax.sum(axis=0)
    gdpi = np.zeros((nr, nf + 1))
    gdpi[:, :nf] = v["EVFP"].sum(axis=1).T
    gdpi[:, nf] = indtax - taxrfu

    # --- SALE (5255-5261)
    vxdxsb = v["VXSB"].sum(axis=2)  # (c, s)
    sale = np.zeros((nc, nr, 6))
    sale[:, :, 0] = v["VDFB"].sum(axis=1)
    sale[:, :, 1] = v["VDPB"]
    sale[:, :, 2] = v["VDIB"]
    sale[:, :, 3] = v["VDGB"]
    sale[:, :, 4] = vxdxsb
    for k, ci in enumerate(marg):
        sale[ci, :, 5] = v["VST"][k]

    # --- accounting checks (5268-5322)
    vdb = vsum("VDPB", "VDGB", "VDIB") + v["VDFB"].sum(axis=1)
    vcb = vdb + vxdxsb
    for k, ci in enumerate(marg):
        vcb[ci] += v["VST"][k]
    vmck = vcb - v["MAKB"].sum(axis=1)
    vtfsd = v["VTWR"].sum(axis=0)  # (c, s, d)
    trck = v["VCIF"] - (v["VFOB"] + vtfsd)
    vtmuse = v["VTWR"].sum(axis=(1, 2, 3))
    vtck = vtmuse - v["VST"].sum(axis=1)
    vmsb_in = v["VMSB"].sum(axis=1)  # (c, d)
    mpck = vmsb_in - v["VMFB"].sum(axis=1) - vsum("VMPB", "VMGB", "VMIB")
    vock = v["MAKB"] - (v["MAKS"] + ptax)

    pct = {
        "VMCP": _pct(vmck, vcb),
        "TRCP": _pct(trck, v["VCIF"]),
        "VTCP": _pct(vtck, vtmuse),
        "MPCP": _pct(mpck, vmsb_in),
        "VOCS": _pct(vock, v["MAKB"]),
    }
    arrays = {
        "GDPE": gdpe, "COST": cost, "GDPI": gdpi, "SALE": sale, "VMCK": vmck,
        "TRCK": trck, "VTCK": vtck, "MPCK": mpck, "VOCK": vock, **pct,
    }  # fmt: skip
    arrays.update({q: pct[p] for q, p in _POST.items()})

    sets = {
        **els,
        "GDPEX": _GDPEX,
        "SALES": _SALES,
        "COSTS": ["IntDom", "IntImp", *els["ENDW"], "PTAX"],
        "GDPIN": [*els["ENDW"], "IndTaxes"],
    }
    out: dict[str, HeaderArray] = {}
    for name, coeff, long_name, names in _META:
        out[name] = HeaderArray(
            name=name,
            coeff_name=coeff,
            long_name=long_name,
            array=arrays[name],
            set_names=list(names),
            set_elements=[list(sets[x]) for x in names],
        )
    return out


def input_database(
    run: AltertaxRun, basedata: Path | None = None
) -> dict[str, HeaderArray]:
    """Numeric value headers of the input ``basedata.har`` (default of ``run.dataset_dir``)."""
    har = read_har(str(basedata or run.dataset_dir / "basedata.har"))
    return {k: h for k, h in har.items() if np.asarray(h.array).dtype.kind == "f"}


def summary_headers(
    run: AltertaxRun, basedata: Path | None = None
) -> dict[str, HeaderArray]:
    """The 19 headers of ``altax_summary.har``: level tables from the ``check`` period,
    ``*Q`` from the ``shock`` period, accounting checks of the input data from
    ``basedata`` (see the module docstring)."""
    model = summary_from_database(headers_at(run, "check"))
    post = summary_from_database(headers_at(run, "shock"))
    data = summary_from_database(input_database(run, basedata))
    src = {**model, **{n: data[n] for n in _PRE_ON_DATA}, **{n: post[n] for n in _POST}}
    return {n: src[n] for n in SUMMARY_HEADERS}


def write_summary_har(run: AltertaxRun, path: Path) -> None:
    """Write the GTAPSUM summary of ``run`` to ``path`` (GEMPACK ``summary.har`` shape)."""
    write_har(path, summary_headers(run))
