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

    # (No pfactor anchor needed: the CDE-closure equation e_up determines private
    # utility per region, closing the #regions DOF natively — same as Julia.)
