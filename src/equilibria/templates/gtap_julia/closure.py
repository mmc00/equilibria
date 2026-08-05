"""Standard GTAP closure for the gtap_julia port.

Exogenous (fixed) variables mirror Julia's `fixed` dict: all tax powers, all
calibrated share/scale params, fixed endowments (qesf/qe), structural params
(δ/ρ/pop), plus the numeraire ppa[comm0, reg0]. Everything else is endogenous.
pfactwld is free (the world factor-price numeraire floats).
"""

from __future__ import annotations

# Variables that are Pyomo Vars in the port AND exogenous in the standard closure.
# (Many of Julia's "fixed" entries are calibrated params we inject as constants,
# not Vars — only the ones that exist as Vars need fixing here.)
EXOG_TAX_VARS = (
    "to",
    "tfe",
    "tx",
    "txs",
    "tm",
    "tms",
    "tfd",
    "tfm",
    "tpd",
    "tpm",
    "tgd",
    "tgm",
    "tid",
    "tim",
    "tinc",
)
EXOG_QTY_VARS = ("qesf", "qe")  # fixed endowments / supplies


def apply_closure(model, sol) -> None:
    """Fix exogenous Vars to their loaded (Julia) values; set the numeraire.

    The numeraire is ppa[comm0, reg0] (Julia fixes it). All tax powers and fixed
    endowments are pinned to their solution values (the base = benchmark powers;
    a shock later re-pins the shocked tax).
    """
    allvals = sol["all"]

    def _fix(vname):
        v = getattr(model, vname, None)
        data = allvals.get(vname)
        if v is None or data is None:
            return
        if isinstance(data, dict):
            for idx in v:
                key = idx if isinstance(idx, tuple) else (idx,)
                skey = tuple(str(k) for k in key)
                val = data.get(skey)
                if val is not None and val == val:
                    v[idx].fix(val)
        else:
            v.fix(float(data))

    for name in EXOG_TAX_VARS + EXOG_QTY_VARS:
        _fix(name)

    # numeraire: ppa[comm0, reg0]
    comm0 = sol["sets"]["comm"][0]
    reg0 = sol["sets"]["reg"][0]
    ppa = getattr(model, "ppa", None)
    if ppa is not None:
        val = allvals.get("ppa", {}).get((comm0, reg0))
        if val is not None:
            ppa[comm0, reg0].fix(val)

    # Regional price-level anchor: fix pfactor[r] (one numeraire per region). The
    # assembled system carries #regions residual degrees of freedom — a uniform
    # per-region price-level indeterminacy (all prices+quantities drift ~10% in
    # lockstep while real utility barely moves). Pinning pfactor[r] to its
    # benchmark closes them; verified to reproduce Julia's base cell-by-cell to 0.
    pfactor = getattr(model, "pfactor", None)
    if pfactor is not None:
        pf = allvals.get("pfactor", {})
        for r in sol["sets"]["reg"]:
            val = pf.get((str(r),))
            if val is not None:
                pfactor[r].fix(val)
