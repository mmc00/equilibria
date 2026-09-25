#!/usr/bin/env python3
"""Johansen vs Gragg 2-4-6 contra la tabla 4.5 del libro, celda por celda.

MEDIDO (2026-09-25): con Johansen 1 paso GEMPACK reproduce las SEIS celdas de
la tabla 4.5/CDE exactamente -- diferencia 0,00 en las seis al redondear a los
2 decimales que imprime el libro. Y la tercera columna del libro (qpa+ppa)
tambien sale de sumar las dos primeras. El libro dice al pie: "We use the
Johansen solution method". Coincide.

Correr Gragg 2-4-6 lo CONFIRMA por contraste: aleja las 6 celdas (qpa[AGR]
0,0005 -> 0,1524, ppa[SER] 0,0020 -> 0,1696). Si el libro fuera multi-paso,
Gragg tendria que acercar, no alejar.

Este script queda como el gate que lo verifica. El caso de uso vivo es
comprobar que una corrida nueva sigue reproduciendo el libro, y medir el costo
de cambiar de metodo.

ADVERTENCIA sobre la fuente: los valores del libro de abajo salen de la tabla
4.5 (pag. 127), leida del PDF. Un par "-0,79 / 4,44" circulo en notas previas
como si fuera de esta tabla: NO lo es. El 4,44 es qc("MFG","USA") de la Tabla
ME 3.1 (pag. 318 y clave de respuestas pag. 411), OTRO experimento -- un
subsidio del 10% a la produccion de manufacturas de USA, no el shock de TFP en
servicios. Y "-0,79" no aparece en el libro. Verificar contra el PDF antes de
agregar una fila aca.

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
    # Tabla 4.5, bloque CDE (pag. 127). Leido del PDF, no de notas.
    "TBL45A": {
        "ppa": {("AGR", "USA"): 0.59, ("MFG", "USA"): -0.20, ("SER", "USA"): -5.07},
        "qpa": {("AGR", "USA"): 1.96, ("MFG", "USA"): 5.46, ("SER", "USA"): 9.66},
    },
    # Bloque CES (3x3ces.prm).
    "TBL45B": {
        "ppa": {("AGR", "USA"): 0.91, ("MFG", "USA"): -0.24, ("SER", "USA"): -5.19},
        "qpa": {("AGR", "USA"): 6.22, ("MFG", "USA"): 6.80, ("SER", "USA"): 9.27},
    },
    # Bloque Cobb-Douglas (3x3CobbDouglas.prm).
    "TBL45C": {
        "ppa": {("AGR", "USA"): 0.64, ("MFG", "USA"): -0.18, ("SER", "USA"): -5.05},
        "qpa": {("AGR", "USA"): 4.10, ("MFG", "USA"): 4.92, ("SER", "USA"): 9.78},
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
            if dj < 0.005 and dg >= 0.005:
                v = "JOHANSEN coincide"
            elif dg < 0.005:
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
    n_joh = sum(1 for v in veredicto if v.endswith("JOHANSEN coincide"))
    n_gra = sum(1 for v in veredicto if "Gragg coincide" in v)
    n_tot = len(veredicto)
    print(
        f"contra el libro: Johansen coincide en {n_joh}/{n_tot} celdas, "
        f"Gragg en {n_gra}/{n_tot}."
    )
    if not algo_se_movio:
        print("Gragg no movio ninguna celda mas de 0,005: el .cmf probablemente")
        print("corrio Johansen igual. Revisa que tenga UNA linea Method = Gragg.")
    elif n_joh > n_gra:
        print("Johansen reproduce el libro mejor que Gragg, y el libro lo dice al")
        print("pie de la tabla 4.5: 'We use the Johansen solution method'. El")
        print("metodo multi-paso ALEJA, asi que no hay brecha que explicar por")
        print("linealizacion. Si una celda no cierra, la causa esta en los datos,")
        print("los parametros o el cierre -- no en el metodo.")
    else:
        print("Gragg reproduce el libro mejor que Johansen, lo que CONTRADICE la")
        print("nota al pie de la tabla 4.5. Antes de concluir nada, verifica que")
        print("ambas corridas usen el mismo .prm y el mismo shock.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
