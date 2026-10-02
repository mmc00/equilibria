"""Herramientas de depuracion justo antes de resolver un periodo del driver.

Todas se activan con una variable de entorno que lleva el periodo en MAYUSCULAS
(``P`` = BASE / CHECK / SHOCK) y son no-op si no esta puesta:

  EQUILIBRIA_DEBUG_PROBE_PF_<P>=r,f       imprime la semilla de pf/pft/xf/xft de
                                          (r,f) y los residuos de sus ecuaciones
  EQUILIBRIA_DEBUG_EXPORT_NL_<P>=ruta     escribe el modelo preparado a .nl y para
  EQUILIBRIA_DEBUG_EXPORT_GMS_<P>=ruta    escribe el modelo preparado a .gms (como
                                          NLP maximize walras) y para
  EQUILIBRIA_DEBUG_EXPORT_NL_<P>_NLP=ruta .nl reformulado como NLP maximize walras
                                          (la rama ifMCP=0 de GAMS) y para
  EQUILIBRIA_DEBUG_SOLVE_NLP_<P>_REPORT=ruta  resuelve ese NLP con IPOPT en proceso
                                          y escribe los peores residuos a JSON

Las exportaciones lanzan RuntimeError a proposito: paran la corrida justo antes
del solve.  Los nombres para el check son los de siempre
(EQUILIBRIA_DEBUG_EXPORT_NL_CHECK, ...).

El driver llama a ``debug_before_solve`` antes de resolver el base y el check.
El shock todavia tiene su propio hook (EQUILIBRIA_DEBUG_EXPORT_GMS_SHOCK) hasta
que su preparacion pase a period_prep.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
from pathlib import Path


def _in_period(idx, period: str) -> bool:
    if isinstance(idx, tuple):
        return bool(idx) and idx[-1] == period
    return idx == period


def _walras_as_objective(m, period: str, attr: str):
    """Reformula el periodo como NLP maximize walras (GAMS ifMCP=0): apaga
    eq_walras[period] y expone walras[period], libre, como objetivo."""
    from pyomo.environ import Objective, maximize

    eq_walras = getattr(m, "eq_walras", None)
    if eq_walras is not None:
        for idx in list(eq_walras):
            if _in_period(idx, period) and eq_walras[idx].active:
                eq_walras[idx].deactivate()
    walras = getattr(m, "walras", None)
    if walras is None:
        raise RuntimeError(f"{attr}: model has no `walras` Var")
    try:
        wvd = walras[period]
    except Exception:
        wvd = walras
    # _mute_welfare_tail (gtap) FIJA walras=0; como objetivo debe quedar libre o
    # el objetivo es una constante sin gradiente.
    if wvd.fixed:
        wvd.unfix()
    setattr(m, attr, Objective(expr=wvd, sense=maximize))
    return wvd


def _row_residuals(m, period: str, keep=lambda name, idx: True):
    from pyomo.environ import Constraint, value

    rows = []
    for c in m.component_objects(Constraint, active=True):
        for idx in c:
            cd = c[idx]
            if not cd.active or not _in_period(idx, period) or not keep(c.name, idx):
                continue
            try:
                b = value(cd.body)
                lo = value(cd.lower) if cd.lower is not None else None
                up = value(cd.upper) if cd.upper is not None else None
                r = 0.0
                if lo is not None:
                    r = max(r, abs(b - lo))
                if up is not None:
                    r = max(r, abs(b - up))
                rows.append((r, c.name, str(idx)))
            except Exception:
                pass
    rows.sort(reverse=True)
    return rows


def _probe_pf(m, period: str, spec: str) -> None:
    from pyomo.environ import value

    rr, ff = spec.split(",")[0], spec.split(",")[1]
    err = sys.stderr
    print(f"[probe-pf] --- {rr},{ff} {period} seed state ---", file=err)
    for idx in getattr(m, "pf", []):
        if idx[0] == rr and idx[1] == ff and idx[-1] == period:
            vd = m.pf[idx]
            print(f"[probe-pf]   pf{idx} = {value(vd):.6f}  fixed={vd.fixed}", file=err)
    for nm in ("pft", "xf", "xft"):
        comp = getattr(m, nm, None)
        if comp is None:
            continue
        for idx in comp:
            if (
                idx[0] == rr
                and rr
                and len(idx) >= 2
                and idx[1] == ff
                and idx[-1] == period
            ):
                with contextlib.suppress(Exception):
                    print(
                        f"[probe-pf]   {nm}{idx} = {value(comp[idx]):.6f}  "
                        f"fixed={comp[idx].fixed}",
                        file=err,
                    )

    def _touches_factor(name, idx):
        nml = name.lower()
        si = str(idx)
        return (
            any(k in nml for k in ("pf", "xf", "fnm", "endw", "fact"))
            and rr in si
            and ff in si
        )

    for res, n, i in _row_residuals(m, period, _touches_factor)[:12]:
        print(f"[probe-pf]   resid {res:.6e}  {n}{i}", file=err)
    for eqn in ("eq_pfteq", "eq_xfteq", "eq_fnm", "eq_pfeq"):
        eqc = getattr(m, eqn, None)
        if eqc is None:
            continue
        for idx in eqc:
            if rr in str(idx) and ff in str(idx) and period in str(idx):
                print(f"[probe-pf]   {eqn}{idx} active={eqc[idx].active}", file=err)


def _solve_nlp_report(m, period: str, path: str, var: str) -> None:
    from pyomo.environ import SolverFactory, value

    wvd = _walras_as_objective(m, period, "_nlp_walras_objective")
    opt = SolverFactory("ipopt")
    opt.options["max_iter"] = 1000
    # Escalado por gradiente para celdas SAM casi nulas (xd~1e-6 con sigma 3.87):
    # la escala unitaria trata una fila 1e-6 como una O(1).
    opt.options["nlp_scaling_method"] = "gradient-based"
    opt.options["bound_relax_factor"] = 0
    opt.options["mu_strategy"] = "adaptive"
    opt.options["tol"] = 1e-7
    # EQUILIBRIA_IPOPT_OPTS='{"mu_strategy":"monotone",...}' para A/B de opciones.
    override = os.environ.get("EQUILIBRIA_IPOPT_OPTS")
    if override:
        for k, v in json.loads(override).items():
            opt.options[k] = v
    res = opt.solve(m, tee=True)
    report = {
        "solver_status": str(res.solver.status),
        "termination_condition": str(res.solver.termination_condition),
        "walras_value": value(wvd),
        "top_residuals": [
            {"resid": r, "eq": n, "idx": i}
            for r, n, i in _row_residuals(m, period)[:40]
        ],
    }
    Path(path).write_text(json.dumps(report, indent=2))
    print(f"[report] wrote {path}", file=sys.stderr)
    raise RuntimeError(f"{var}: stopping after in-process NLP solve+report")


def debug_before_solve(m, period: str) -> None:
    """Corre las herramientas pedidas por variable de entorno para ``period``."""
    P = period.upper()
    env = os.environ.get

    spec = env(f"EQUILIBRIA_DEBUG_PROBE_PF_{P}")
    if spec:
        _probe_pf(m, period, spec)

    var = f"EQUILIBRIA_DEBUG_EXPORT_NL_{P}"
    path = env(var)
    if path:
        m.write(path, format="nl", io_options={"symbolic_solver_labels": True})
        print(f"[export] wrote {period}-period .nl to {path}", file=sys.stderr)
        raise RuntimeError(f"{var}: stopping right before PATH solves {period}")

    var = f"EQUILIBRIA_DEBUG_EXPORT_GMS_{P}"
    path = env(var)
    if path:
        # El escritor GAMS necesita un objetivo; el CNS/MCP no tiene.
        _walras_as_objective(m, period, "_nlp_walras_objective_gms")
        m.write(path, format="gams", io_options={"symbolic_solver_labels": True})
        print(f"[export] wrote {period}-period .gms to {path}", file=sys.stderr)
        raise RuntimeError(f"{var}: stopping right before PATH solves {period}")

    var = f"EQUILIBRIA_DEBUG_EXPORT_NL_{P}_NLP"
    path = env(var)
    if path:
        _walras_as_objective(m, period, "_nlp_walras_objective")
        m.write(path, format="nl", io_options={"symbolic_solver_labels": True})
        print(
            f"[export] wrote {period}-period NLP(maximize walras) .nl to {path}",
            file=sys.stderr,
        )
        raise RuntimeError(f"{var}: stopping right before PATH solves {period}")

    var = f"EQUILIBRIA_DEBUG_SOLVE_NLP_{P}_REPORT"
    path = env(var)
    if path:
        _solve_nlp_report(m, period, path, var)
