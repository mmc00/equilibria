"""``decomp.har`` of ALTERTAX on the family route: GEMPACK's WELVIEW welfare decomposition.

``gtapv7.tab`` section 11-5 (lines 3799-4291) defines the contributions ``CNT*`` to the
equivalent variation as *linear change variables*: ``CNT = 0.01 * EVSCALFACT * COEF *
[q% - pop%]`` with ``COEF`` a value coefficient (a tax revenue, ``PTAX`` ... ``MTAX``) and
``q%`` the cumulative percent change of a quantity; ``postsim`` (lines 4325-5100) turns
them into the 30 data headers of ``altax_decomp.har``. As in :mod:`volume`, ``q%`` is
``100 * (x_post / x_pre - 1)`` of the live model quantity, ``COEF`` is the value of the
``pre`` period (``check``, which reproduces the base) and ``post`` is ``shock``. The
shock is ~1e-5 so the cumulative path integral of GEMPACK's multistep solution equals this
one step to first order. ``EVSCALFACT`` is 1 at the base (``UTILELASEV = UTILELAS``,
``INCOMEEV = INCOME``, line 3809); it moves by the same ~1e-5 along the path, so it is
taken as 1. Values come from :mod:`equilibria.family.altertax.updated`
(``headers_at``, ``_trade_prices``, ``_Period``).

Quantity and price map (``qes = qfe``, line 2361):

========= ================================== ========= ==================================
GEMPACK   equilibria                         GEMPACK   equilibria
========= ================================== ========= ==================================
qca(c,a)  x[r,a,c]                           qxs       xw[s,c,d]
qfe, qes  xf[r,e,a]                          qo(a,r)   xp[r,a]
qfm, qfd  xma, xda[r,c,a]                    kb(r)     kstock[r]
qpm,qpd   xma, xda[r,c,hhd]                  pfob      pefob of ``_trade_prices``
qgm,qgd   xma, xda[r,c,gov]                  pds(c,r)  pd[r,c]
qim,qid   xma, xda[r,c,inv]                  pt(m)     ptmg[m]
pinv(r)   pi[r]                              psave(r)  psave[r]
========= ================================== ========= ==================================

Formula table (``gtapv7.tab`` lines; every ``CNT`` is ``0.01 * COEF * [q% - pop%]``):

====== ======================================================== ===================
header what                                                      tab line
====== ======================================================== ===================
A      WELFARE(r, 7 columns): sum_t CNTalleffr, CNTendwr, 0      4565-4585
       (tech), CNTpop (0), CNTtotr, CNTpinv, CNTdpar (0)
A1     ENDW: CNTqfeer + CNTqe; a_*: sum_c CNTqca; c_*:           4596-4609
       CNTalleffcr
A2     CNTalleffr(r, tax type)                                   4119-4140, 4617
A21    OTAX = CNTqo(a,r) = sum_c PTAX * q%(qca)                  3903-3910, 4643
A211   CNTqca, BMAKEB*qo/100, BOUTAX, UOUTAX                      4655-4670
A22    ATAX: firm/private/govt/investment x domestic/imported    4688-4702
A221   STAX = CNTqfd / CNTqfm                                    4713-4718
A222   ITAX = CNTqid / CNTqim                                    4728-4733
A22F   FIRM: welcnt, BVDFB*qfd/100 (BVMFB*qfm/100), base and     4743-4762
       updated rate of the tax wedge
A22P/G/I  as A22F for private, government, investment            4773-4849
A23    TTAX: xtax = CNTqxs(c,s,d), mtax = CNTqms(c,s,d)          4864-4870
A231   TRADE: welcnt, BVXSB*qxs/100, BXTAX/BMTAX, UXTAX/UMTAX    4890-4905
A24    FACTAX = sum_a CNTqfe                                     4925-4928
A241   FACTORTAX: CNTqfe, BEVFB*qfe/100, BETAX, UETAX            4943-4952
A25    YTAX = sum_a CNTqea                                       4966-4969
A251   INCOMETAX: CNTqea, BVEB*qes/100, BINCTAX, UINCTAX         4975-4985
B1     CNTendw(e,r) = 0.01 * sum_a EVOS * q%(qes)                4991-4996, 4198-4203
B2     CNTkb = -0.01 * VDEP * [kb% - pop%]                       4286-4291, 5002
C1     CNTtechr (7 types)                                        5013-5018, 4206-4222
C11-17 CNTtech_*: coefficient * technical change                 5020-5039, 4226-4283
E1     WTOT percent: c1, c2, -c3; value: CNTtotr * share of      5058-5086, 3446-3464
       the percent in the sum
F1     ISbal = NETINV - SAVE (initial, line 4432), pinv%, psave%  5095-5104
====== ======================================================== ===================

Explanatory-factor headers hold four columns ``welcnt dvol taxrateb taxrateu``:
``taxrateb`` is the rate at ``pre`` and ``taxrateu`` at ``post`` (``Update`` d coefficients
of ``postsim``).

ALTERTAX shocks one output tax: technical change (``ao afe ava aint af atmfsd ams``),
population and the distribution shifters are exogenous at zero, so C1..C17, CNTpop and
CNTdpar are zeros; ``decomp_headers`` refuses a run where those variables move
(``_require_unchanged``) instead of returning a wrong zero.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np

from equilibria.babel.har.symbols import HeaderArray
from equilibria.babel.har.writer import write_har
from equilibria.family.altertax.run import AltertaxRun
from equilibria.family.altertax.updated import (
    _elements,
    _g,
    _Period,
    _trade_prices,
    headers_at,
)

_COL = ["welcnt", "dvol", "taxrateb", "taxrateu"]
_DIR = ["domestic", "imported"]
_COLUMN = ["alloc_A1", "ENDWB1", "tech_C1", "pop_D1", "tot_E1", "IS_F1", "pref_G1"]
_CTAX = [
    "prodtax", "pfacttax", "inctax", "inputtax", "contax", "invtax", "govtax", "xtax", "mtax",
]  # fmt: skip
_ACTIVITY = ["firm", "private", "investment", "govt"]
_TYPE = ["xtax", "mtax"]
_DIREC = ["import", "export"]
_TECHTYPE = ["output", "primfac", "v_added", "ininput", "invest", "transp", "import"]
_PRICES = ["pworld", "pexport", "pimport"]
_FORM = ["percent", "value"]
_COLM = ["ISbal", "p_cgds", "p_save"]

# (header, coefficient, long name, set names) as in ``altax_decomp.har``
_META: tuple[tuple[str, str, str, tuple[str, ...]], ...] = (
    ("A", "WELFARE", "EV Decomposition: Summary", ("REG", "COLUMN")),
    ("A1", "CNTalleff", "Allocative Efficiency Effect: Summary", ("ALLOCEFF", "REG")),
    ("A2", "CNTalleffkr", "Allocative Efficiency Effect: Tax Type Summary", ("REG", "CTAX")),
    ("A21", "OTAX", "Output Tax Effect", ("ACTS", "REG")),
    ("A211", "OUTPUT", "Output Tax Effect: Explanatory Factors", ("COMM", "ACTS", "REG", "COL")),
    ("A22", "ATAX", "Domestic Tax Effect: Summary", ("COMM", "ACTIVITY", "REG", "DIR")),
    ("A221", "STAX", "Intermed. Input Tax Effect", ("COMM", "ACTS", "REG", "DIR")),
    ("A222", "ITAX", "Investment Intermed. Input Tax Effect", ("COMM", "REG", "DIR")),
    ("A22F", "FIRM", "Intermed. Input Tax Effect: Explanatory Factors", ("COMM", "ACTS", "REG", "DIR", "COL")),
    ("A22P", "PRIVATE", "Private Cons. Tax Effect: Explanatory Factors", ("COMM", "REG", "DIR", "COL")),
    ("A22G", "GOVT", "Government Cons. Tax Effect: Explanatory Factors", ("COMM", "REG", "DIR", "COL")),
    ("A22I", "INVESTMENT", "Investment Tax Effect: Explanatory Factors", ("COMM", "REG", "DIR", "COL")),
    ("A23", "TTAX", "Trade Tax Effect", ("COMM", "REG", "REG", "TYPE")),
    ("A231", "TRADE", "Trade Tax Effect: Explanatory Factors", ("COMM", "REG", "REG", "COL", "DIREC")),
    ("A24", "FACTAX", "Endowment Tax Effect", ("ENDW", "REG")),
    ("A241", "FACTORTAX", "Endowment Tax Effect: Explanatory Factors", ("ENDW", "ACTS", "REG", "COL")),
    ("A25", "YTAX", "Income Tax Effect", ("ENDW", "REG")),
    ("A251", "INCOMETAX", "Income Tax Effect: Explanatory Factors", ("ENDW", "ACTS", "REG", "COL")),
    ("B1", "CNTendowment", "Endowment Effect, Gross of Depreciation", ("REG", "ENDW")),
    ("B2", "CNTkb", "Endowment Effect: Depreciation", ("REG",)),
    ("C1", "CNTtech", "Technical Change Effect: Summary", ("REG", "TECHTYPE")),
    ("C11", "CNTtech_ao", "(ao) Output Augm. Tech. Change Effect", ("ACTS", "REG")),
    ("C12", "CNTtech_afe", "(afe) Primary Factor Augm. Tech. Change Effect", ("ENDW", "ACTS", "REG")),
    ("C13", "CNTtech_ava", "(ava) Value Added Augm. Tech. Change Effect", ("ACTS", "REG")),
    ("C14", "CNTtech_aint", "(aint) Composite Intermed. Input Augm. Tech. Change Effect", ("ACTS", "REG")),
    ("C15", "CNTtech_af", "(af) Intermed. Input Augm. Tech. Change Effect", ("COMM", "ACTS", "REG")),
    ("C16", "CNTtech_atmf", "(atmfsd) Internat. Margin Augm. Tech. Change Effect", ("MARG", "COMM", "REG", "REG")),
    ("C17", "CNTtech_ams", "(ams) Bilateral Import Augm. Tech. Change Effect", ("COMM", "REG", "REG")),
    ("E1", "WTOT", "Terms of Trade Effect", ("COMM", "REG", "PRICES", "FORM")),
    ("F1", "CGDSCOMP", "I-S Effect: Explanatory Factors", ("REG", "COLM")),
)  # fmt: skip
DECOMP_HEADERS: tuple[str, ...] = tuple(m[0] for m in _META)

# variables of ALTERTAX that are exogenous at zero change: see the module docstring
_UNCHANGED = ("axp", "aft", "lambdaf", "lambdava", "lambdam", "lambdamg", "pop")


def _ratio(pre: dict, post: dict, key: tuple[str, ...]) -> float:
    """Cumulative percent change ``100 * (x_post / x_pre - 1)``, zero where ``x_pre = 0``."""
    x0 = _g(pre, key)
    return 0.0 if x0 == 0.0 else 100.0 * (_g(post, key) / x0 - 1.0)


def _div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """``num / den`` with 0 where ``den = 0`` (``Zerodivide default 0``, lines 4326-4327).
    GEMPACK's postsim block resets ``nonzero_by_zero`` to 0.5 (line 4523); only the E1
    value denominator could hit it and it is never zero here, so 0 is used throughout."""
    out = np.zeros(np.broadcast(num, den).shape)
    np.divide(num, den, out=out, where=den != 0.0)
    return out


def _require_unchanged(A: _Period, B: _Period) -> None:
    for var in _UNCHANGED:
        if var not in A._vars:
            raise KeyError(
                f"decomp.har needs the variable {var!r} to check it is unchanged"
            )
        a, b = A[var], B[var]
        for k, x in a.items():
            if abs(b.get(k, x) - x) > 1e-12 * max(1.0, abs(x)):
                raise NotImplementedError(
                    f"decomp.har assumes {var} is unchanged (ALTERTAX); {var}{k} moved"
                )


def _fill(shape: tuple[int, ...], fn: Callable[..., float]) -> np.ndarray:
    out = np.zeros(shape)
    for ix in np.ndindex(*shape):
        out[ix] = fn(*ix)
    return out


# GEMPACK variables of the cumulative changes the welfare terms are built from; the
# shape of each is given by its sets (``E`` ENDW, ``C`` COMM, ``A`` ACTS, ``R`` REG,
# ``M`` MARG).
CHANGE_SHAPES: dict[str, str] = {
    "qca": "CAR", "qfe": "EAR", "qfm": "CAR", "qfd": "CAR", "qpm": "CR", "qpd": "CR",
    "qgm": "CR", "qgd": "CR", "qim": "CR", "qid": "CR", "qxs": "CRR", "qo": "AR",
    "kb": "R", "pop": "R", "pfob": "CRR", "pds": "CR", "pinv": "R", "psave": "R",
    "pt": "M",
}  # fmt: skip


def model_changes(
    run: AltertaxRun, pre: str = "check", post: str = "shock"
) -> dict[str, np.ndarray]:
    """Cumulative percent changes (GEMPACK names, ``CHANGE_SHAPES`` axes) of the model
    from ``pre`` to ``post``: ``100 * (x_post / x_pre - 1)``."""
    return changes_between(_Period(run, pre), _Period(run, post), _elements(run))


def changes_between(
    A: _Period, B: _Period, els: dict[str, list[str]]
) -> dict[str, np.ndarray]:
    """:func:`model_changes` between two periods that may belong to different solves of
    the same model (``A`` the pre, ``B`` the post): the steps of a shock path."""
    _require_unchanged(A, B)
    comm, acts, reg, endw, marg = (
        els[k] for k in ("COMM", "ACTS", "REG", "ENDW", "MARG")
    )
    nc, na, nr, ne, nm = len(comm), len(acts), len(reg), len(endw), len(marg)

    def pct(var: str, key_of: Callable[..., tuple[str, ...]], shape: tuple[int, ...]):
        return _fill(shape, lambda *ix: _ratio(A[var], B[var], key_of(*ix)))

    def final(var: str, agent: str) -> np.ndarray:
        return pct(var, lambda c, r: (reg[r], comm[c], agent), (nc, nr))

    def pfob_at(P: _Period) -> np.ndarray:
        return _fill(
            (nc, nr, nr),
            lambda c, s, d: _trade_prices(P, reg[s], comm[c], reg[d])["pefob"],
        )

    p0, p1 = pfob_at(A), pfob_at(B)
    return {
        "qca": pct("x", lambda c, a, r: (reg[r], acts[a], comm[c]), (nc, na, nr)),
        "qfe": pct("xf", lambda e, a, r: (reg[r], endw[e], acts[a]), (ne, na, nr)),
        "qfm": pct("xma", lambda c, a, r: (reg[r], comm[c], acts[a]), (nc, na, nr)),
        "qfd": pct("xda", lambda c, a, r: (reg[r], comm[c], acts[a]), (nc, na, nr)),
        "qpm": final("xma", "hhd"),
        "qpd": final("xda", "hhd"),
        "qgm": final("xma", "gov"),
        "qgd": final("xda", "gov"),
        "qim": final("xma", "inv"),
        "qid": final("xda", "inv"),
        "qxs": pct("xw", lambda c, s, d: (reg[s], comm[c], reg[d]), (nc, nr, nr)),
        "qo": pct("xp", lambda a, r: (reg[r], acts[a]), (na, nr)),
        "kb": pct("kstock", lambda r: (reg[r],), (nr,)),
        "pop": pct("pop", lambda r: (reg[r],), (nr,)),
        "pfob": 100.0
        * (_div(p1, p0) - (p0 != 0.0)),  # 100 * (p1/p0 - 1), 0 where p0 = 0
        "pds": pct("pd", lambda c, r: (reg[r], comm[c]), (nc, nr)),
        "pinv": pct("pi", lambda r: (reg[r],), (nr,)),
        "psave": pct("psave", lambda r: (reg[r],), (nr,)),
        "pt": pct("ptmg", lambda m: (marg[m],), (nm,)),
    }


def decomp_headers(
    run: AltertaxRun, pre: str = "check", post: str = "shock"
) -> dict[str, HeaderArray]:
    """The 30 data headers of ``altax_decomp.har`` for the change ``pre`` to ``post``
    (``check`` to ``shock``; ``check`` twice gives exact zeros in every welfare cell)."""
    return decomp_from_changes(
        {
            k: np.asarray(h.array, dtype=np.float64)
            for k, h in headers_at(run, pre).items()
        },
        {
            k: np.asarray(h.array, dtype=np.float64)
            for k, h in headers_at(run, post).items()
        },
        model_changes(run, pre, post),
        _elements(run),
    )


def decomp_from_changes(
    v: dict[str, np.ndarray],
    w: dict[str, np.ndarray],
    changes: dict[str, np.ndarray],
    els: dict[str, list[str]],
) -> dict[str, HeaderArray]:
    """The WELVIEW headers from value coefficients before (``v``) and after (``w``) the
    change, the cumulative percent changes (``changes``, see ``CHANGE_SHAPES``) and the
    set elements. The formulas of the module docstring; the changes may come from the
    family model (:func:`model_changes`) or from any other solution of the same model.

    Valid for SMALL shocks only: each CNT is one step ``COEF(pre) * q%`` of what GEMPACK
    integrates along the multistep path with the coefficient updated at every step. For
    a large shock, :mod:`equilibria.family.altertax.path` sums it over sub-steps."""
    comm, acts, reg, endw, marg = (
        els[k] for k in ("COMM", "ACTS", "REG", "ENDW", "MARG")
    )
    nc, na, nr, ne, nm = len(comm), len(acts), len(reg), len(endw), len(marg)
    mi = [comm.index(m) for m in marg]
    q_ca, q_fe, q_fm, q_fd = (changes[k] for k in ("qca", "qfe", "qfm", "qfd"))
    q_pm, q_pd, q_gm, q_gd = (changes[k] for k in ("qpm", "qpd", "qgm", "qgd"))
    q_im, q_id, q_xs, q_o = (changes[k] for k in ("qim", "qid", "qxs", "qo"))
    kb, pop, pfob, pds = (changes[k] for k in ("kb", "pop", "pfob", "pds"))
    pinv, psave, ptmg = changes["pinv"], changes["psave"], changes["pt"]

    # ---- tax revenues at ``pre`` (gtapv7.tab 728-826) ----------------------------
    ptax = v["MAKB"] - v["MAKS"]  # (c,a,r)
    etax = v["EVFP"] - v["EVFB"]  # (e,a,r)
    inctax = v["EVFB"] - v["EVOS"]
    mftax, dftax = v["VMFP"] - v["VMFB"], v["VDFP"] - v["VDFB"]
    mptax, dptax = v["VMPP"] - v["VMPB"], v["VDPP"] - v["VDPB"]
    mgtax, dgtax = v["VMGP"] - v["VMGB"], v["VDGP"] - v["VDGB"]
    mitax, ditax = v["VMIP"] - v["VMIB"], v["VDIP"] - v["VDIB"]
    xtaxd = v["VFOB"] - v["VXSB"]  # (c,s,d)
    mtax = v["VMSB"] - v["VCIF"]

    # ---- CNT contributions: 0.01 * COEF * [q% - pop%] (EVSCALFACT = 1) ------------
    pr = pop  # (r,)
    cnt_qca = 0.01 * ptax * (q_ca - pr)
    cnt_qfe = 0.01 * etax * (q_fe - pr)
    cnt_qea = 0.01 * inctax * (q_fe - pr)  # qes = qfe
    cnt_qe = 0.01 * (inctax * (q_fe - pr)).sum(axis=1)  # (e,r)
    cnt_qfm = 0.01 * mftax * (q_fm - pr)
    cnt_qfd = 0.01 * dftax * (q_fd - pr)
    cnt_qpm, cnt_qpd = 0.01 * mptax * (q_pm - pr), 0.01 * dptax * (q_pd - pr)
    cnt_qgm, cnt_qgd = 0.01 * mgtax * (q_gm - pr), 0.01 * dgtax * (q_gd - pr)
    cnt_qim, cnt_qid = 0.01 * mitax * (q_im - pr), 0.01 * ditax * (q_id - pr)
    # CNTqxs(c,s,d) uses pop(s), CNTqms(c,s,d) uses pop(d) (lines 4098, 4116)
    cnt_qxs = 0.01 * xtaxd * (q_xs - pr[None, :, None])
    cnt_qms = 0.01 * mtax * (q_xs - pr[None, None, :])

    alleff = np.zeros((nr, len(_CTAX)))  # CNTalleffr(r, t), lines 4119-4140
    t = {k: i for i, k in enumerate(_CTAX)}
    alleff[:, t["prodtax"]] = cnt_qca.sum(axis=(0, 1))
    alleff[:, t["inctax"]] = cnt_qea.sum(axis=(0, 1))
    alleff[:, t["pfacttax"]] = cnt_qfe.sum(axis=(0, 1))
    alleff[:, t["inputtax"]] = cnt_qfm.sum(axis=(0, 1)) + cnt_qfd.sum(axis=(0, 1))
    alleff[:, t["contax"]] = cnt_qpm.sum(axis=0) + cnt_qpd.sum(axis=0)
    alleff[:, t["govtax"]] = cnt_qgm.sum(axis=0) + cnt_qgd.sum(axis=0)
    alleff[:, t["invtax"]] = cnt_qim.sum(axis=0) + cnt_qid.sum(axis=0)
    alleff[:, t["xtax"]] = cnt_qxs.sum(axis=(0, 2))
    alleff[:, t["mtax"]] = cnt_qms.sum(axis=(0, 1))
    # CNTalleffcr(c,r), lines 4147-4163
    alleff_cr = (
        cnt_qfm.sum(axis=1) + cnt_qfd.sum(axis=1) + cnt_qpm + cnt_qpd + cnt_qgm + cnt_qgd
        + cnt_qim + cnt_qid + cnt_qxs.sum(axis=2) + cnt_qms.sum(axis=1)
    )  # fmt: skip

    # ---- endowments and depreciation (4188-4203, 4286-4291) ----------------------
    cnt_endw = 0.01 * (v["EVOS"] * (q_fe - pr)).sum(axis=1)  # (e,r)
    cnt_kb = -0.01 * v["VDEP"] * (kb - pr)  # (r,)

    # ---- terms of trade (3389-3464, 4166-4176) -----------------------------------
    vfob, vst, vtw = v["VFOB"], v["VST"], v["VTWR"]  # (c,s,d) (m,r) (m,c,s,d)
    vtmcomm = vtw.sum(axis=(1, 2))  # VTMCOMM(m,s) = VTMD(m,s)
    vxw = vfob.sum(axis=2)  # VXW(c,r) = VXDFOB + VST for margins
    for k, ci in enumerate(mi):
        vxw[ci] += vst[k]
    vxwreg = vxw.sum(axis=0)
    vxwcom = vxw.sum(axis=1)
    vxwld = vxwreg.sum()
    num = (vfob * pfob).sum(axis=2)
    for k, ci in enumerate(mi):
        num[ci] += vst[k] * pds[ci]
    pxw = _div(num, vxw)
    pxwreg = _div((vxw * pxw).sum(axis=0), vxwreg)
    pxwcom = _div((vxw * pxw).sum(axis=1), vxwcom)
    pxwwld = float((vxwreg * pxwreg).sum() / vxwld)
    # APPROXIMATION: GEMPACK's pt(m) is sum_r VTSUPPSHR * pds(m,r) (E_pt, line 1979); the
    # family model's margin price ptmg is used. Measured: pt differs by -14% from GEMPACK's
    # and moves the terms-of-trade total by about -0.0005 (of EV ~ 1).
    pt = ptmg
    vmwdcr = vfob.sum(axis=1).copy()  # VMWDCRALL(c,s)
    for k, ci in enumerate(mi):
        vmwdcr[ci] += vtmcomm[k]
    sm_cr = _div(vmwdcr, vmwdcr.sum(axis=0)[None, :])
    sx_cr = _div(vxw, vxwreg[None, :])  # SX_CR(c,r) = VXW(c,r) / VXWREGION(r)
    sm_crs = _div(vfob, vmwdcr[:, None, :])  # (c,r,s)
    pm_cr = (sm_crs * pfob).sum(axis=1)
    for k, ci in enumerate(mi):
        pm_cr[ci] += _div(vtmcomm[k], vmwdcr[ci]) * pt[k]
    c1 = (sx_cr - sm_cr) * (pxwcom - pxwwld)[:, None]
    c2 = sx_cr * (pxw - pxwcom[:, None])
    c3 = sm_cr * (pm_cr - pxwcom[:, None])
    cnt_tot = 0.01 * (
        (vfob * (pfob - pxwwld)).sum(axis=(0, 2))
        - (vfob * (pfob - pxwwld)).sum(axis=(0, 1))
        - (vtmcomm * (pt[:, None] - pxwwld)).sum(axis=0)
    )
    for k, ci in enumerate(mi):
        cnt_tot += 0.01 * vst[k] * (pds[ci] - pxwwld)

    # ---- investment-saving (4179-4185, 4432) -------------------------------------
    reginv = (v["VDIP"] + v["VMIP"]).sum(axis=0)
    netinv = reginv - v["VDEP"]
    cnt_pinv = 0.01 * (netinv * (pinv - pxwwld) - v["SAVE"] * (psave - pxwwld))

    out: dict[str, np.ndarray] = {}
    # -- C: technical change is exogenous at zero in ALTERTAX (checked above) ---
    out["C1"] = np.zeros((nr, len(_TECHTYPE)))
    out["C11"] = np.zeros((na, nr))
    out["C12"] = np.zeros((ne, na, nr))
    out["C13"] = np.zeros((na, nr))
    out["C14"] = np.zeros((na, nr))
    out["C15"] = np.zeros((nc, na, nr))
    out["C16"] = np.zeros((nm, nc, nr, nr))
    out["C17"] = np.zeros((nc, nr, nr))

    out["A"] = np.stack(
        [alleff.sum(axis=1), cnt_endw.sum(axis=0) + cnt_kb, out["C1"].sum(axis=1),
         0.0 * pop, cnt_tot, cnt_pinv, 0.0 * pop],
        axis=1,
    )  # fmt: skip
    cnt_alleff = np.concatenate(
        [cnt_qfe.sum(axis=1) + cnt_qe, cnt_qca.sum(axis=0), alleff_cr], axis=0
    )  # ALLOCEFF = ENDW + UACT + UCOM, (e+a+c, r)
    out["A1"] = cnt_alleff
    out["A2"] = alleff
    out["A21"] = cnt_qca.sum(axis=0)  # OTAX(a,r) = CNTqo
    out["B1"] = cnt_endw.T
    out["B2"] = cnt_kb

    # ---- A22 family: domestic/imported by agent ---------------------------------
    dom = {
        "firm": cnt_qfd.sum(axis=1),
        "private": cnt_qpd,
        "govt": cnt_qgd,
        "investment": cnt_qid,
    }
    imp = {
        "firm": cnt_qfm.sum(axis=1),
        "private": cnt_qpm,
        "govt": cnt_qgm,
        "investment": cnt_qim,
    }
    out["A22"] = np.stack(
        [np.stack([dom[a], imp[a]], axis=-1) for a in _ACTIVITY], axis=1
    )  # (c, ACTIVITY, r, DIR)
    out["A221"] = np.stack([cnt_qfd, cnt_qfm], axis=-1)
    out["A222"] = np.stack([cnt_qid, cnt_qim], axis=-1)

    def rate(num: np.ndarray, den: np.ndarray) -> np.ndarray:
        return 100.0 * _div(num, den)

    def cols(welcnt, dvol, rateb, rateu) -> np.ndarray:
        return np.stack([welcnt, dvol, rateb, rateu], axis=-1)

    # A211: OUTPUT(c,a,r,co)
    out["A211"] = cols(
        cnt_qca,
        v["MAKB"] * q_o[None, :, :] / 100.0,
        rate(v["MAKB"] - v["MAKS"], v["MAKS"]),
        rate(w["MAKB"] - w["MAKS"], w["MAKS"]),
    )
    # A22F: FIRM(c,a,r,d,co)
    firm = []
    for dn, qn, cnt in (("D", q_fd, cnt_qfd), ("M", q_fm, cnt_qfm)):
        b, p = v[f"V{dn}FB"], v[f"V{dn}FP"]
        firm.append(
            cols(
                cnt,
                b * qn / 100.0,
                rate(p - b, b),
                rate(w[f"V{dn}FP"] - w[f"V{dn}FB"], w[f"V{dn}FB"]),
            )
        )
    out["A22F"] = np.stack(firm, axis=3)
    for name, ag, qd, qm, cd, cm in (
        ("A22P", "P", q_pd, q_pm, cnt_qpd, cnt_qpm),
        ("A22G", "G", q_gd, q_gm, cnt_qgd, cnt_qgm),
        ("A22I", "I", q_id, q_im, cnt_qid, cnt_qim),
    ):
        parts = []
        for dn, qn, cnt in (("D", qd, cd), ("M", qm, cm)):
            b, p = v[f"V{dn}{ag}B"], v[f"V{dn}{ag}P"]
            bu, pu = w[f"V{dn}{ag}B"], w[f"V{dn}{ag}P"]
            parts.append(cols(cnt, b * qn / 100.0, rate(p - b, b), rate(pu - bu, bu)))
        out[name] = np.stack(parts, axis=2)  # (c, r, DIR, COL)

    out["A23"] = np.stack([cnt_qxs, cnt_qms], axis=-1)  # (c,r,s,TYPE)
    dvol_t = v["VXSB"] * q_xs / 100.0
    exp_cols = cols(
        cnt_qxs,
        dvol_t,
        rate(v["VFOB"] - v["VXSB"], v["VXSB"]),
        rate(w["VFOB"] - w["VXSB"], w["VXSB"]),
    )
    imp_cols = cols(
        cnt_qms,
        dvol_t,
        rate(v["VMSB"] - v["VCIF"], v["VCIF"]),
        rate(w["VMSB"] - w["VCIF"], w["VCIF"]),
    )
    out["A231"] = np.stack([imp_cols, exp_cols], axis=-1)  # DIREC = (import, export)
    out["A24"] = cnt_qfe.sum(axis=1)
    out["A241"] = cols(
        cnt_qfe,
        v["EVFB"] * q_fe / 100.0,
        rate(v["EVFP"] - v["EVFB"], v["EVFB"]),
        rate(w["EVFP"] - w["EVFB"], w["EVFB"]),
    )
    out["A25"] = cnt_qea.sum(axis=1)
    out["A251"] = cols(
        cnt_qea,
        v["EVFB"] * q_fe / 100.0,
        rate(v["EVFB"] - v["EVOS"], v["EVOS"]),
        rate(w["EVFB"] - w["EVOS"], w["EVOS"]),
    )

    # ---- E1, F1 -------------------------------------------------------------------
    pct_t = np.stack([c1, c2, -c3], axis=2)  # (c, r, PRICES)
    den = pct_t.sum(axis=(0, 2))  # sum over commodities and price types, per region
    val_t = _div(cnt_tot[None, :, None] * pct_t, den[None, :, None])
    out["E1"] = np.stack([pct_t, val_t], axis=-1)
    out["F1"] = np.stack([netinv - v["SAVE"], pinv, psave], axis=1)

    sets = {
        "COMM": comm, "ACTS": acts, "REG": reg, "ENDW": endw, "MARG": marg,
        "COLUMN": _COLUMN, "CTAX": _CTAX, "COL": _COL, "DIR": _DIR, "ACTIVITY": _ACTIVITY,
        "TYPE": _TYPE, "DIREC": _DIREC, "TECHTYPE": _TECHTYPE, "PRICES": _PRICES,
        "FORM": _FORM, "COLM": _COLM,
        "ALLOCEFF": [*endw, *[f"a_{a}" for a in acts], *[f"c_{c}" for c in comm]],
    }  # fmt: skip
    res: dict[str, HeaderArray] = {}
    for name, coeff, long_name, names in _META:
        arr = np.asarray(out[name], dtype=np.float64)
        expect = tuple(len(sets[x]) for x in names)
        if arr.shape != expect:
            raise ValueError(f"{name}: shape {arr.shape}, expected {expect}")
        res[name] = HeaderArray(
            name=name,
            coeff_name=coeff,
            long_name=long_name,
            array=arr,
            set_names=list(names),
            set_elements=[list(sets[x]) for x in names],
        )
    return res


def write_decomp_har(run: AltertaxRun, path: Path) -> None:
    """Write the WELVIEW welfare decomposition of ``run`` to ``path`` (GEMPACK ``decomp.har`` shape)."""
    write_har(path, decomp_headers(run))
