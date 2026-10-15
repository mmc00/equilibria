"""Reparto del tiempo de una corrida GTAP por fase, con cronometros y no con cProfile.

cProfile infla las llamadas chicas de Pyomo (el 20x41 pasa de 7,4 a 22 min
perfilado), asi que su reparto sirve para orientar pero no para citar. Esto envuelve
una docena de funciones con un cronometro de pared y reparte el tiempo EXCLUSIVO:
cada segundo cuenta en la fase mas interna que estaba corriendo.

    <python con pymumps> scripts/gtap/phase_times.py [dataset] [--out out.json]

Misma ruta que perf_budget_20x41.py (Newton-TR + MUMPS, base calibrada, en frio).
Sale con 2 (medicion invalida) si falta pymumps, si no converge, si alguna fase no
se llamo nunca (un envoltorio que no engancha porque la funcion se importo por
nombre antes de envolverla, no una fase gratis) o si no hay una linea de Newton por
cada solve.

El lazo de Newton-TR esta escrito en linea dentro de _run_path_capi_nonlinear_full
(no pasa por scipy.optimize.root), asi que no se puede envolver: se toma el "wall="
que el propio solver imprime al terminar ("scipy.root(newton_tr) done: ... wall=Xs")
y se descuenta de "preparar solve (resto)".
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [
    str(ROOT / "src"),
    str(ROOT / "scripts" / "gtap"),
    str(ROOT / "tests"),
    str(ROOT),
]

from perf_budget_20x41 import NEWTON_MUMPS, _falta_mumps  # noqa: E402

PREPARAR_SOLVE = "preparar solve (resto)"
NEWTON = "Newton-TR + MUMPS"

_pila: list[list] = []  # [fase, inicio, segundos de hijos]
excl: dict[str, float] = {}
llamadas: dict[str, int] = {}


def _cronometrar(fase: str, fn):
    def envuelta(*a, **k):
        _pila.append([fase, time.perf_counter(), 0.0])
        try:
            return fn(*a, **k)
        finally:
            f, t0, hijos = _pila.pop()
            dt = time.perf_counter() - t0
            excl[f] = excl.get(f, 0.0) + dt - hijos
            llamadas[f] = llamadas.get(f, 0) + 1
            if _pila:
                _pila[-1][2] += dt

    return envuelta


class _Tee:
    """stderr que ademas suma el wall= de cada solve de Newton que el solver imprime."""

    _RE = re.compile(r"scipy\.root\([^)]*\) done: .*?wall=([0-9.]+)s")

    def __init__(self, base):
        self.base = base
        self.newton: list[float] = []

    def write(self, texto):
        self.newton += [float(w) for w in self._RE.findall(texto)]
        return self.base.write(texto)

    def __getattr__(self, nombre):
        return getattr(self.base, nombre)


def _envolver(owner, nombre: str, fase: str) -> None:
    setattr(owner, nombre, _cronometrar(fase, getattr(owner, nombre)))


def _fases() -> list[tuple[object, str, str]]:
    """(dueno, funcion, fase) de cada funcion envuelta, de afuera hacia adentro.

    El tiempo exclusivo de cada una es lo que no cae en otra fase envuelta. La
    misma lista sirve para envolver y para comprobar que todas se llamaron.
    """
    from pyomo.contrib.pynumero.interfaces import pyomo_nlp

    import equilibria.solver._closure_patches as cp
    import equilibria.solver.path_capi as pc
    import equilibria.templates.gtap.gtap_block_model as gbm
    import equilibria.templates.gtap.gtap_multiperiod_driver as drv
    import equilibria.templates.gtap.period_prep as pp
    from equilibria.blocks.gtap.factor import FactorBlock

    mp = gbm.GTAPBlockMultiPeriodModel
    return [
        (gbm, "build_block_single_period", "construir SP de bloques"),
        (mp, "build_vars", "reflejar Var a 3 periodos"),
        (mp, "build_equations_all_periods", "reflejar filas a 3 periodos"),
        (FactorBlock, "calibrate_base", "calibracion (resto)"),
        (pp.PeriodPreparer, "prepare", "preparar periodo"),
        (drv, "freeze_period", "congelar periodo"),
        (pc, "_run_path_capi_nonlinear_full", PREPARAR_SOLVE),
        (cp, "structural_matching", "emparejamiento estructural"),
        (pyomo_nlp.PyomoNLP, "__init__", "escribir .nl (PyomoNLP)"),
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dataset", nargs="?", default="gtap7_20x41")
    ap.add_argument("--out", type=Path, help="donde escribir el JSON")
    args = ap.parse_args()

    if msg := _falta_mumps():
        print(msg, file=sys.stderr)
        return 2
    os.environ.update(NEWTON_MUMPS)
    os.environ.pop("EQUILIBRIA_GTAP_MODEL_CACHE", None)

    from templates.gtap.test_gtap7_gempack_parity import _solve_shock

    fases = _fases()
    for owner, nombre, fase in fases:
        _envolver(owner, nombre, fase)

    tee = _Tee(sys.stderr)
    sys.stderr = tee
    t0 = time.perf_counter()
    try:
        _, code = _solve_shock(args.dataset, 1, "capFix", solve_nlp=True)
    finally:
        sys.stderr = tee.base
    total = time.perf_counter() - t0

    # El lazo de Newton corre dentro de "preparar solve": se lo separa.
    newton = sum(tee.newton)
    excl[PREPARAR_SOLVE] = excl.get(PREPARAR_SOLVE, 0.0) - newton
    excl[NEWTON] = newton
    llamadas[NEWTON] = len(tee.newton)
    excl["fuera de las fases"] = total - sum(excl.values())
    llamadas["fuera de las fases"] = 1

    faltan = [fase for *_, fase in fases if not llamadas.get(fase)]
    solves = llamadas.get(PREPARAR_SOLVE, 0)
    out = {
        "dataset": args.dataset,
        "code": code,
        "total_s": round(total, 1),
        "fases": {
            f: {
                "s": round(s, 1),
                "pct": round(100 * s / total, 1),
                "llamadas": llamadas.get(f, 0),
            }
            for f, s in sorted(excl.items(), key=lambda x: -x[1])
        },
        "sin_enganchar": faltan,
    }
    texto = json.dumps(out, indent=2, ensure_ascii=False)
    print(texto)
    if args.out:
        args.out.write_text(texto + "\n")

    problemas = []
    if code != 1:
        problemas.append(f"no convergio (code={code})")
    if faltan:
        problemas.append(f"fases sin enganchar: {', '.join(faltan)}")
    if len(tee.newton) != solves:
        problemas.append(f"{len(tee.newton)} lineas de Newton para {solves} solves")
    if excl[PREPARAR_SOLVE] < 0:
        problemas.append(
            "Newton supera a 'preparar solve': el wall= salio de otra fase"
        )
    for p in problemas:
        print(f"medicion invalida: {p}", file=sys.stderr)
    return 2 if problemas else 0


if __name__ == "__main__":
    raise SystemExit(main())
