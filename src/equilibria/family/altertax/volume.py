"""``volume.har`` of ALTERTAX on the family route: GEMPACK's GTAPVOL output.

``gtapv7.tab`` section D (lines 5105-5214, ``postsim``) turns every cumulative quantity
change ``q%`` of the solution into a volume change at the initial basic prices,
``DQ = q% / 100 * BV``, with ``BV`` the pre-simulation value coefficient. Here ``q%`` is
``x(post) / x(pre) - 1`` of the live model quantity and ``BV`` the value of the ``check``
period (which reproduces the base), so ``DQ = BV * (x_post / x_pre - 1)`` (a cell with
``x_pre = 0`` has no change). ``pre`` is ``check`` and ``post`` ``shock``. Quantity
scalings cancel in the ratio. Values come from :mod:`equilibria.family.altertax.updated`.

====== ============================================ =============================== ========
header BV (initial value)                           quantity ratio                  tab line
====== ============================================ =============================== ========
DQXS   VXSB(c,s,d)                                  xw[s,c,d]                       5118
DQDS   VDB(c,r) (BVDB, line 4380)                   sum of the domestic agents      5124
DQFE   EVFB(e,a,r)                                  xf[r,e,a]                       5130
DQFM   VMFB(c,a,r)                                  xma[r,c,a]                      5136
DQFD   VDFB(c,a,r)                                  xda[r,c,a]                      5142
DQPM   VMPB(c,r)                                    xma[r,c,hhd]                    5148
DQPD   VDPB(c,r)                                    xda[r,c,hhd]                    5154
DQGM   VMGB(c,r)                                    xma[r,c,gov]                    5160
DQGD   VDGB(c,r)                                    xda[r,c,gov]                    5166
DQO    BVOS(a,r) = sum EVFP + VMFP + VDFP (4347)    xp[r,a]                         5172
DQC    BVCB(c,r) = VDB + sum VXSB + VST (4393)      x[r,a,c] by MAKEB share         5178
DTOT   -                                            sum_c DQXS(c,s,d)               5184
====== ============================================ =============================== ========

``qds`` (``E_qds``, line 2128) is the share-weighted sum of the domestic purchases
``qfd, qpd, qgd, qid`` with the shares ``VD?B / VDB`` of the initial data, so
``DQDS = sum_agents BV_agent * (x_post / x_pre - 1)`` (the investment agent is ``inv``).
``qc`` (``E_qc``, line 1358) is ``sum_a MAKEBCOMSHR * qca`` with ``qca`` the make
quantity ``x[r,a,c]`` and ``MAKEBCOMSHR = MAKEB / MAKEBCOM``. ``qo`` is the activity output
``xp``. The headers are volumes at initial prices, so they hold no price effect.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from equilibria.babel.har.symbols import HeaderArray
from equilibria.babel.har.writer import write_har
from equilibria.family.altertax.run import AltertaxRun
from equilibria.family.altertax.updated import _elements, _g, _Period, headers_at

# (header, coefficient, long name, set names) as in ``altax_volume.har``
_META: tuple[tuple[str, str, str, tuple[str, ...]], ...] = (
    ("DQXS", "DQXS", " ch. in vol. of exports of c from region s to d at initial basic price", ("COMM", "REG", "REG")),
    ("DQDS", "DQDS", " ch. in vol. of domestic sales of com. c in r at initial basic prices", ("COMM", "REG")),
    ("DQFE", "DQFE", " ch. in vol. of endowment e by activity a in r at initial basic prices", ("ENDW", "ACTS", "REG")),
    ("DQFM", "DQFM", " ch. in vol. of imp. intermed. c by act. a in r at initial basic price", ("COMM", "ACTS", "REG")),
    ("DQFD", "DQFD", " ch. in vol. of dom. intermed. c by act. a in r at initial basic price", ("COMM", "ACTS", "REG")),
    ("DQPM", "DQPM", " ch. in vol. of priv. hhld cons. for imp. c in r at initial basic pric", ("COMM", "REG")),
    ("DQPD", "DQPD", " ch. in vol. of priv. hhld cons. for dom. c in r at initial basic pric", ("COMM", "REG")),
    ("DTOT", "DTOT", " ch. in vol. of total exports from region s to d at initial basic pric", ("REG", "REG")),
    ("DQO", "DQO", " ch. in vol. of output by activity a in r at initial basic prices", ("ACTS", "REG")),
    ("DQC", "DQC", " ch. in vol. of supply of com. c in r at initial basic prices", ("COMM", "REG")),
    ("DQGD", "DQGD", " ch. in vol. of government cons. for dom. c in r at initial basic pric", ("COMM", "REG")),
    ("DQGM", "DQGM", " ch. in vol. of government cons. for imp. c in r at initial basic pric", ("COMM", "REG")),
)  # fmt: skip
VOLUME_HEADERS: tuple[str, ...] = tuple(m[0] for m in _META)


def _ratio(pre: dict, post: dict, key: tuple[str, ...]) -> float:
    """``x_post / x_pre - 1`` of one cell, zero where the initial quantity is zero."""
    x0 = _g(pre, key)
    return 0.0 if x0 == 0.0 else _g(post, key) / x0 - 1.0


def volume_bases(
    run: AltertaxRun,
    period: str = "check",
    db: dict[str, HeaderArray] | None = None,
) -> dict[str, np.ndarray]:
    """``BV`` of every ``DQ`` header: the value coefficients at ``period`` (``check``)."""
    v = {
        k: np.asarray(h.array, dtype=np.float64)
        for k, h in (headers_at(run, period) if db is None else db).items()
    }
    vdb = v["VDPB"] + v["VDGB"] + v["VDIB"] + v["VDFB"].sum(axis=1)
    marg = [list(run.params.sets.i).index(m) for m in run.params.sets.marg]
    vcb = vdb + v["VXSB"].sum(axis=2)
    for k, ci in enumerate(marg):
        vcb[ci] += v["VST"][k]
    return {
        "DQXS": v["VXSB"],
        "DQDS": vdb,
        "DQFE": v["EVFB"],
        "DQFM": v["VMFB"],
        "DQFD": v["VDFB"],
        "DQPM": v["VMPB"],
        "DQPD": v["VDPB"],
        "DTOT": v["VXSB"].sum(axis=0),
        "DQO": (v["EVFP"].sum(axis=0) + v["VMFP"].sum(axis=0) + v["VDFP"].sum(axis=0)),
        "DQC": vcb,
        "DQGD": v["VDGB"],
        "DQGM": v["VMGB"],
    }


def volume_headers(
    run: AltertaxRun, pre: str = "check", post: str = "shock"
) -> dict[str, HeaderArray]:
    """The 12 headers of ``altax_volume.har``: volume changes from ``pre`` to ``post``
    (``check`` to ``shock``; any other pair of periods is accepted, e.g. ``check`` twice
    gives exact zeros)."""
    A, B = _Period(run, pre), _Period(run, post)
    els = _elements(run)
    comm, acts, reg, endw = els["COMM"], els["ACTS"], els["REG"], els["ENDW"]
    vd = headers_at(run, pre)
    bv = volume_bases(run, pre, vd)

    def rt(var: str, key: tuple[str, ...]) -> float:
        return _ratio(A[var], B[var], key)

    arr: dict[str, np.ndarray] = {}
    arr["DQXS"] = np.array(
        [[[bv["DQXS"][ci, si, di] * rt("xw", (sr, c, dr)) for di, dr in enumerate(reg)]
          for si, sr in enumerate(reg)] for ci, c in enumerate(comm)]
    )  # fmt: skip
    arr["DTOT"] = arr["DQXS"].sum(axis=0)
    arr["DQFE"] = np.array(
        [[[bv["DQFE"][ei, ai, ri] * rt("xf", (r, e, a)) for ri, r in enumerate(reg)]
          for ai, a in enumerate(acts)] for ei, e in enumerate(endw)]
    )  # fmt: skip
    for h, var in (("DQFM", "xma"), ("DQFD", "xda")):
        arr[h] = np.array(
            [[[bv[h][ci, ai, ri] * rt(var, (r, c, a)) for ri, r in enumerate(reg)]
              for ai, a in enumerate(acts)] for ci, c in enumerate(comm)]
        )  # fmt: skip
    for h, var, agent in (
        ("DQPM", "xma", "hhd"), ("DQPD", "xda", "hhd"),
        ("DQGM", "xma", "gov"), ("DQGD", "xda", "gov"),
    ):  # fmt: skip
        arr[h] = np.array(
            [[bv[h][ci, ri] * rt(var, (r, c, agent)) for ri, r in enumerate(reg)]
             for ci, c in enumerate(comm)]
        )  # fmt: skip
    # qds (E_qds, 2128): share-weighted sum of the domestic purchases by agent
    dqid = np.array(
        [[np.asarray(vd["VDIB"].array)[ci, ri] * rt("xda", (r, c, "inv"))
          for ri, r in enumerate(reg)] for ci, c in enumerate(comm)]
    )  # fmt: skip
    arr["DQDS"] = arr["DQFD"].sum(axis=1) + arr["DQPD"] + arr["DQGD"] + dqid
    # qo: activity output xp
    arr["DQO"] = np.array(
        [[bv["DQO"][ai, ri] * rt("xp", (r, a)) for ri, r in enumerate(reg)]
         for ai, a in enumerate(acts)]
    )  # fmt: skip
    # qc (E_qc, 1358): sum_a MAKEBCOMSHR * qca with qca the make quantity x[r,a,c]
    makeb = np.asarray(vd["MAKB"].array, dtype=np.float64)  # (c, a, r)
    dqc = np.zeros((len(comm), len(reg)))
    for ci, c in enumerate(comm):
        for ri, r in enumerate(reg):
            den = makeb[ci, :, ri].sum()
            if den == 0.0:
                continue
            q = sum(makeb[ci, ai, ri] * rt("x", (r, a, c)) for ai, a in enumerate(acts))
            dqc[ci, ri] = bv["DQC"][ci, ri] * q / den
    arr["DQC"] = dqc

    sets = {"COMM": comm, "ACTS": acts, "REG": reg, "ENDW": endw}
    out: dict[str, HeaderArray] = {}
    for name, coeff, long_name, names in _META:
        out[name] = HeaderArray(
            name=name,
            coeff_name=coeff,
            long_name=long_name,
            array=np.asarray(arr[name], dtype=np.float64),
            set_names=list(names),
            set_elements=[list(sets[x]) for x in names],
        )
    return out


def write_volume_har(run: AltertaxRun, path: Path) -> None:
    """Write the GTAPVOL volume changes of ``run`` to ``path`` (GEMPACK ``volume.har`` shape)."""
    write_har(path, volume_headers(run))
