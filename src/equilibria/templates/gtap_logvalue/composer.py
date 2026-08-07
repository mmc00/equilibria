"""Log-value GTAPv7 composer.

Assembles the 7 log-value blocks (equilibria.blocks.gtap_logvalue) through the neutral
PyomoBackend, applies the log-value closure (fix exogenous tax powers + numeraire is a
block equation), and solves the whole system in one IPOPT call (CNS / feasibility,
Objective(expr=0)). Reuses only Block/SymbolicEquation/PyomoBackend — not the
levels-specific build_block_single_period. Mirrors gtap_julia/model.py's semantics
(seed at the calibrated point; shock scales tms multiplicatively).
"""

from __future__ import annotations

from typing import Any

import pyomo.environ as pyo

from equilibria.backends.pyomo_backend import PyomoBackend
from equilibria.blocks.gtap_logvalue import GTAP_LOGVALUE_BLOCK_ORDER
from equilibria.core.sets import Set as ESet
from equilibria.model import Model

# port set key -> block-model set names built from the calibrated point's sets
_SETS = {
    "r": "reg",
    "i": "comm",
    "a": "acts",
    "f": "endw",
    "fm": "endwm",
    "fs": "endws",
    "ff": "endwf",
    "fms": "endwms",
    "fc": "endwc",
    "marg": "marg",
    "rp": "reg",
}

# exogenous multiplicative tax powers fixed by the closure (all but tms in shock)
_TAX_VARS = [
    "to",
    "tfe",
    "tinc",
    "tfd",
    "tfm",
    "tpd",
    "tpm",
    "tgd",
    "tgm",
    "tid",
    "tim",
    "tx",
    "tm",
    "txs",
    "tms",
]
# fixed endowments / supplies (Julia EXOG_QTY_VARS)
_QTY_FIX = ["qesf", "qe"]


def _setmap(sol: dict[str, Any]) -> dict[str, list[str]]:
    s = sol["sets"]
    return {k: list(s.get(v, [])) for k, v in _SETS.items()}


def build_logvalue_model(sol: dict[str, Any], rordelta: int = 1):
    """Compose the 7 blocks → Pyomo ConcreteModel with closure + feasibility objective."""
    setmap = _setmap(sol)
    model = Model(name="gtap_logvalue")
    for name, elems in setmap.items():
        model.add_set(ESet(name=name, elements=tuple(elems)))
    for cls in GTAP_LOGVALUE_BLOCK_ORDER:
        model.add_block(cls(sol=sol))
    backend = PyomoBackend()
    backend.build(model)
    pm = backend.pyomo_model
    _strip_con_suffix(pm)
    _apply_closure(pm, sol)
    pm.add_component("_obj", pyo.Objective(expr=0.0))
    return pm


def _strip_con_suffix(pm) -> None:
    """Rename {eq}_con → {eq} to match the port's equation names."""
    for c in list(pm.component_objects(pyo.Constraint)):
        if c.name.endswith("_con"):
            base = c.name[:-4]
            if not hasattr(pm, base):
                pm.del_component(c)
                pm.add_component(base, c)


def _apply_closure(pm, sol: dict[str, Any]) -> None:
    """Replicate the port's closure exactly (gtap_julia/closure.py):
    fix exogenous tax powers + fixed endowments (qesf/qe) to their calibrated levels,
    pin the numeraire ppa[comm0, reg0], and fix every δ-masked cell (calibrated seed is
    NaN → the cell does not exist) so the system is square, as the port's
    apply_closure + _fix_orphan_vars do. The e_numeraire block equation is deactivated
    (the .fix pins it instead) to avoid an extra constraint."""
    allv = sol
    for tname in _TAX_VARS + _QTY_FIX:
        comp = pm.component(tname)
        d = allv.get(tname)
        if comp is None or not isinstance(d, dict):
            continue
        for idx in comp:
            key = idx if isinstance(idx, tuple) else (idx,)
            v = d.get(key)
            if v is not None and v == v:
                comp[idx].fix(float(v))

    # numeraire via .fix (drop the e_numeraire equation — replaced by the fix)
    comm0 = sol["sets"]["comm"][0]
    reg0 = sol["sets"]["reg"][0]
    ppa = pm.component("ppa")
    ppa_val = (
        allv.get("ppa", {}).get((comm0, reg0))
        if isinstance(allv.get("ppa"), dict)
        else None
    )
    if ppa is not None and ppa_val is not None:
        ppa[comm0, reg0].fix(float(ppa_val))
    e_num = pm.component("e_numeraire")
    if e_num is not None:
        e_num.deactivate()

    # fix every δ-masked cell: a cell whose calibrated seed is NaN does not exist in
    # the port's model (its equation is skipped), so pin it at 1.0 to keep the system
    # square instead of leaving it as an under-determined free var.
    for v in pm.component_data_objects(pyo.Var):
        if v.fixed:
            continue
        pc = v.parent_component()
        d = allv.get(pc.name)
        if not isinstance(d, dict):
            continue
        idx = v.index()
        key = idx if isinstance(idx, tuple) else (idx,)
        val = d.get(key)
        if val is None or val != val:  # missing or NaN → masked cell
            v.fix(1.0)


def _fix_orphans(pm) -> None:
    from pyomo.core.expr.visitor import identify_variables

    used = set()
    for c in pm.component_data_objects(pyo.Constraint):
        for v in identify_variables(c.body):
            used.add(id(v))
    for v in pm.component_data_objects(pyo.Var):
        if not v.fixed and id(v) not in used:
            val = v.value
            v.fix(1.0 if val is None or val != val else val)


def _ipopt():
    opt = pyo.SolverFactory("ipopt")
    opt.options["tol"] = 1e-8
    opt.options["constr_viol_tol"] = 1e-8
    opt.options["bound_push"] = 1e-15
    return opt


def solve(sol: dict[str, Any], rordelta: int = 1, tee: bool = False) -> dict[str, Any]:
    """Build + solve the base model. Returns {status, ok, model}."""
    pm = build_logvalue_model(sol, rordelta=rordelta)
    _fix_orphans(pm)
    res = _ipopt().solve(pm, tee=tee, load_solutions=True)
    tc = str(res.solver.termination_condition)
    return {"status": tc, "ok": tc in ("optimal", "locallyOptimal"), "model": pm}


def solve_shock(
    sol: dict[str, Any],
    tariff_power: float = 1.10,
    rordelta: int = 1,
    tee: bool = False,
) -> dict[str, Any]:
    """Solve base, then scale every bilateral tms power by tariff_power (× base) and
    re-solve — the faithful multiplicative import-tariff shock."""
    pm = build_logvalue_model(sol, rordelta=rordelta)
    _fix_orphans(pm)
    _ipopt().solve(pm, tee=False, load_solutions=True)  # base warm start
    tms = pm.component("tms")
    for idx in tms:
        base_power = pyo.value(tms[idx])
        tms[idx].fix(base_power * tariff_power)
    res = _ipopt().solve(pm, tee=tee, load_solutions=True)
    tc = str(res.solver.termination_condition)
    return {"status": tc, "ok": tc in ("optimal", "locallyOptimal"), "model": pm}
