"""Foto del modelo justo antes de cada solve del driver multiperiodo.

El driver (base -> check -> shock) prepara cada periodo con muchos pasos
(congelar, recalibrar, fijar, cotas, holdfix...).  Esta foto registra, por cada
llamada al solver y al final de la corrida, el estado de cada componente:

  * Var: por celda, fixed / lb / ub / dominio / valor
  * Param: por celda, su valor (el cuerpo de una ecuacion solo nombra el Param)
  * Constraint / Expression / Objective: si esta activa y su expresion, con los
    numeros redondeados (un float recalibrado no debe depender de la plataforma)
  * los atributos que el driver lee de ``m`` (_gtap_mode, _residual_region...)
  * el objeto ``params`` que recibe el solver (p.ej. calibration.phip)

Se reemplaza el solver por uno falso que solo toma la foto y devuelve code=1, asi
que no hace falta PATH (unos 0,5 s por caso en gtap7_3x3).  La foto se guarda
como un hash POR COMPONENTE: cuando cambia, el test dice que Var o ecuacion
cambio, en que llamada.

Regenerar (solo si el cambio de la preparacion es INTENCIONAL):

    EQUILIBRIA_UPDATE_PERIOD_SNAPSHOT=1 uv run pytest tests/templates/gtap/test_period_prep_snapshot.py
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from pyomo.environ import Constraint, Expression, Objective, Param, Var

ROOT = Path(__file__).resolve().parents[3]
DATASET = ROOT / "datasets" / "gtap7_3x3"
FIXTURE = ROOT / "tests" / "fixtures" / "period_prep_snapshot.json"

# Cada caso: (nombre, mode, if_sub, opciones de build, opciones del driver). La
# opcion "shock" no va al driver: se escribe en el ShockBlock con apply_shock.
# Cubren cada receta: base, check normal, check copiado del base (F3.5), shock,
# en los dos modos, y las ramas de las opciones (holdfix_cd, seed_from_prior,
# mute_welfare, skip_base_solve, solve_check, shock de instrumento).
CASES: dict[str, tuple[str, bool, dict[str, Any], dict[str, Any]]] = {
    "gtap_ifsub0": ("gtap", False, {}, {}),
    "gtap_ifsub1": ("gtap", True, {}, {}),
    "altertax_ifsub0": ("altertax", False, {}, {}),
    "altertax_ifsub1": ("altertax", True, {}, {}),
    "altertax_options_off": (
        "altertax",
        False,
        {},
        {
            "holdfix_cd": False,
            "seed_from_prior": True,
            "mute_welfare": False,
            "skip_base_solve": True,
        },
    ),
    "gtap_f35": ("gtap", False, {"base_calibrated": True}, {}),
    "gtap_f35_solve_check": (
        "gtap",
        False,
        {"base_calibrated": True},
        {"solve_check": True},
    ),
    "gtap_lambdava": (
        "gtap",
        False,
        {},
        {"shock": {"lambdava": {("USA", "Mnfcs"): 10.0}}},
    ),
}


def _num(x):
    return None if x is None else repr(round(float(x), 10))


_FLOAT = re.compile(r"-?\d+\.\d+(?:[eE][-+]?\d+)?|-?\d+[eE][-+]?\d+")

# Atributos que el driver y el solver leen del modelo.
_M_ATTRS = ("_gtap_mode", "_residual_region", "_base_calibrated", "_sp_source")


def _expr(e) -> str:
    """Texto de una expresion con cada numero redondeado a 10 cifras."""
    return _FLOAT.sub(lambda mt: f"{float(mt.group()):.10g}", str(e))


def _component_hashes(m) -> dict[str, str]:
    out: dict[str, str] = {}
    for v in m.component_objects(Var, descend_into=True):
        h = hashlib.sha256()
        for idx in sorted(v, key=repr):
            vd = v[idx]
            h.update(
                f"{idx!r}|{vd.fixed}|{_num(vd.lb)}|{_num(vd.ub)}|{vd.domain}|"
                f"{_num(vd.value)}\n".encode()
            )
        out[f"Var:{v.name}"] = h.hexdigest()[:16]
    for prm in m.component_objects(Param, descend_into=True):
        h = hashlib.sha256()
        for idx in sorted(prm, key=repr):
            try:
                h.update(f"{idx!r}|{_num(prm[idx])}\n".encode())
            except (TypeError, ValueError):
                h.update(f"{idx!r}|{prm[idx]!r}\n".encode())
        out[f"Param:{prm.name}"] = h.hexdigest()[:16]
    for kind, ctype in (("Con", Constraint), ("Expr", Expression), ("Obj", Objective)):
        for c in m.component_objects(ctype, descend_into=True):
            h = hashlib.sha256()
            for idx in sorted(c, key=repr):
                cd = c[idx]
                active = getattr(cd, "active", True)
                h.update(f"{idx!r}|{active}".encode())
                if active:
                    if ctype is Constraint:
                        h.update(
                            f"|{_expr(cd.body)}|{_num(cd.lower)}|{_num(cd.upper)}".encode()
                        )
                    else:
                        h.update(f"|{_expr(cd.expr)}".encode())
                h.update(b"\n")
            out[f"{kind}:{c.name}"] = h.hexdigest()[:16]
    for a in _M_ATTRS:
        out[f"Attr:{a}"] = repr(getattr(m, a, None))
    return out


def _params_hash(params) -> str:
    from equilibria.blocks.gtap.fingerprint import _feed, _Uncoverable

    h = hashlib.sha256()
    try:
        for name, val in sorted(vars(params).items()):
            h.update(name.encode() + b"=")
            _feed(h, val, numeric=False)
    except _Uncoverable as exc:
        return f"uncoverable: {exc}"
    return h.hexdigest()[:16]


def _load_params():
    from equilibria.templates.gtap import GTAPParameters

    p = GTAPParameters()
    p.load_from_har(
        basedata_path=DATASET / "basedata.har",
        sets_path=DATASET / "sets.har",
        default_path=DATASET / "default.prm",
        baserate_path=DATASET / "baserate.har",
    )
    return p


def take(case: str, monkeypatch) -> list[dict]:
    """Corre el driver del caso con un solver falso; devuelve las fotos en orden:
    una por llamada al solver (con el nombre del cierre) y una final."""
    from equilibria.templates.gtap import gtap_multiperiod_driver as drv
    from equilibria.templates.gtap.gtap_block_model import build_block_model
    from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig
    from equilibria.templates.gtap.shocks import apply_shock

    mode, if_sub, build_kw, drive_kw = CASES[case]
    shots: list[dict] = []

    class _FakeSolver:
        @staticmethod
        def _run_path_capi_nonlinear_full(m, params, **kw):
            comps = _component_hashes(m)
            comps["params"] = _params_hash(params)
            shots.append(
                {"call": f"solve:{kw['closure_config'].name}", "components": comps}
            )
            return {"termination_code": 1, "residual": 0.0}

    monkeypatch.setattr(drv, "_load_run_gtap", lambda: _FakeSolver)
    monkeypatch.setenv("EQUILIBRIA_SEED_CACHE_DISABLE", "1")
    monkeypatch.delenv("EQUILIBRIA_GTAP_MODEL_CACHE", raising=False)
    monkeypatch.delenv("EQUILIBRIA_GTAP_SHOCK_CONTINUATION", raising=False)

    p = _load_params()
    closure = GTAPClosureConfig(
        name="base",
        closure_type="MCP",
        capital_mobility="sluggish",
        fix_endowments=False,
        fix_taxes=False,
        fix_technology=False,
        if_sub=if_sub,
        numeraire="pnum",
    )
    m, _ = build_block_model(p, p.sets, closure, list(p.sets.r)[-1], **build_kw)
    drive_kw = dict(drive_kw)
    shock = drive_kw.pop("shock", None)
    if shock is not None:
        apply_shock(m, shock)
    drv.solve_multiperiod(m, p, closure, mode=mode, **drive_kw)
    shots.append({"call": "final", "components": _component_hashes(m)})
    return shots


def load_fixture() -> dict:
    if not FIXTURE.exists():
        return {}
    return json.loads(FIXTURE.read_text())


def save_fixture(data: dict) -> None:
    FIXTURE.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")


def diff(expected: list[dict], got: list[dict]) -> list[str]:
    """Lista legible de diferencias: llamadas distintas y componentes cambiados."""
    out = []
    if [s["call"] for s in expected] != [s["call"] for s in got]:
        out.append(
            f"secuencia de llamadas: {[s['call'] for s in expected]} -> "
            f"{[s['call'] for s in got]}"
        )
    for i, (e, g) in enumerate(zip(expected, got, strict=False)):
        ec, gc = e["components"], g["components"]
        for name in sorted(set(ec) | set(gc)):
            if ec.get(name) != gc.get(name):
                out.append(f"#{i} {g['call']}: {name}")
    return out
