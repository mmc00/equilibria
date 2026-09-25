#!/usr/bin/env python3
"""Johansen vs Gragg 2-4-6 contra la tabla 4.5 del libro, celda por celda.

LA PREGUNTA: con Johansen 1 paso, GEMPACK reproduce 4 de las 6 celdas de la
tabla 4.5 al cuarto decimal, pero MFG no:

                   GEMPACK/Johansen   libro    equilibria
    ppa[MFG,USA]        -0,1972       -0,79      -0,9871
    qpa[MFG,USA]         5,4572        4,44       4,2154

La hipotesis es que el libro uso un metodo multi-paso y la brecha en MFG es
error de linealizacion. Este script la MIDE: si al pasar a Gragg 2-4-6 MFG se
acerca al libro, la hipotesis se sostiene; si no se mueve, queda refutada.

Uso:
    python cmp_gragg.py                # TBL45A
    python cmp_gragg.py TBL45A         # idem, explicito
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))

from read_sl4 import SL4  # noqa: E402

# Tabla 4.5 del libro (Burfisher 3e, pag. 126), caso CDE = TBL45A.
# Solo la region shockeada: el libro no imprime ROW para esta tabla.
# 2 decimales es TODA la precision que da el libro -- una diferencia por
# debajo de 0,005 no es medible contra esta fuente.
LIBRO = {
    "TBL45A": {
        "ppa": {("AGR", "USA"): 0.59, ("MFG", "USA"): -0.79, ("SER", "USA"): -5.07},
        "qpa": {("AGR", "USA"): 1.96, ("MFG", "USA"): 4.44, ("SER", "USA"): 9.66},
    },
}


def celdas(sl4: SL4, var: str) -> dict[tuple[str, ...], float]:
    labs, pct = sl4.labels(var), sl4.pct(var)
    # strict=True a proposito: un .sl4 truncado o mal escrito da menos
    # valores que etiquetas, y con strict=False eso se convierte en un
    # dict corto -- luego en un KeyError opaco a 40 lineas de aqui, o
    # peor, en una comparacion de celdas que no existen. Que falle aca.
    if len(labs) != len(pct):
        raise ValueError(
            f"{var}: {len(labs)} etiquetas pero {len(pct)} valores en CUMS. "
            f"El .sl4 esta truncado o no es el que se cree."
        )
    return dict(
        zip((tuple(l.split("|")) for l in labs), (float(v) for v in pct), strict=True)
    )


def main() -> int:
    name = sys.argv[1] if len(sys.argv) > 1 else "TBL45A"

    joh_p = HERE / "out" / name / f"{name}.sl4"
    gra_p = HERE / "out-gragg" / name / f"{name}.sl4"
    for etiqueta, p, como in (
        ("Johansen", joh_p, "01-run-all.bat"),
        ("Gragg", gra_p, f"03-gragg.bat {name}"),
    ):
        if not p.exists():
            print(f"falta el .sl4 de {etiqueta}: {p}\n  correlo con: {como}")
            return 1

    joh, gra = SL4(joh_p), SL4(gra_p)
    libro = LIBRO.get(name, {})
    if not libro:
        print(
            f"nota: no tengo los valores del libro para {name}; "
            f"solo se compara Johansen vs Gragg."
        )

    print(f"=== {name}: Johansen 1 paso  vs  Gragg 2-4-6 ===\n")
    algo_se_movio = False
    veredicto: list[str] = []

    for var in ("ppa", "qpa"):
        j, g = celdas(joh, var), celdas(gra, var)
        ref = libro.get(var, {})
        print(f"{var}")
        print(
            f"  {'celda':<12}{'Johansen':>11}{'Gragg':>11}{'libro':>9}"
            f"{'|J-lib|':>10}{'|G-lib|':>10}  veredicto"
        )
        for k in sorted(j, key=lambda t: (t[1] != "USA", t)):
            b = ref.get(k)
            fila = f"  {'|'.join(k):<12}{j[k]:>11.4f}{g[k]:>11.4f}"
            if b is None:
                print(f"{fila}{'-':>9}{'-':>10}{'-':>10}")
                continue
            dj, dg = abs(j[k] - b), abs(g[k] - b)
            # El libro trae 2 decimales: por debajo de 0,005 la diferencia
            # no es distinguible del redondeo de la fuente.
            if dg < 0.005:
                v = "Gragg coincide con el libro"
            elif dg < dj - 0.005:
                v = f"Gragg ACERCA ({dj:.4f} -> {dg:.4f})"
            elif dg > dj + 0.005:
                v = f"Gragg ALEJA ({dj:.4f} -> {dg:.4f})"
            else:
                v = "sin cambio medible"
            if abs(g[k] - j[k]) > 0.005:
                algo_se_movio = True
            print(f"{fila}{b:>9.2f}{dj:>10.4f}{dg:>10.4f}  {v}")
            veredicto.append(f"{var}[{'|'.join(k)}]: {v}")
        print()

    print("--- que contesta esto ---")
    if not algo_se_movio:
        print("Gragg NO movio ninguna celda mas de 0,005. El metodo de solucion")
        print("NO explica la brecha de MFG: la hipotesis queda REFUTADA y hay")
        print("que buscar la causa en otro lado (datos, parametros, o cierre).")
    else:
        mfg = [v for v in veredicto if "MFG" in v]
        print("Gragg SI movio resultados. Sobre MFG, que es la celda en disputa:")
        for v in mfg:
            print(f"  {v}")
        print()
        print("Si MFG coincide o acerca: el error de linealizacion de Johansen")
        print("explica la brecha. Si aleja o no se mueve: no la explica, aunque")
        print("otras celdas si se hayan movido.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
