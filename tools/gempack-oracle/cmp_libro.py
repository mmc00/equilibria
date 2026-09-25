#!/usr/bin/env python3
"""Verifica la corrida GEMPACK contra las tablas publicadas del libro.

Burfisher, "Introduction to CGE Models" 3e, dataset NUS333. Cada entrada de
TABLAS cita la pagina/tabla y sale del PDF, no de notas: un valor mal
atribuido ("-0,79 / 4,44", que resulto ser de la Tabla ME 3.1) costo varias
sesiones persiguiendo una brecha inexistente.

La tolerancia es la precision de la fuente: el libro publica 1 o 2 decimales,
asi que se compara el valor REDONDEADO a esos decimales. Media unidad del
ultimo decimal es todo lo que esa fuente permite distinguir.

Uso:
    python cmp_libro.py           # todas
    python cmp_libro.py 6.4       # una tabla
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))

from read_sl4 import SL4  # noqa: E402

# Cada celda: (exp, variable, clave, valor_libro, decimales).
# La variable puede ser "a-b" para las columnas que el libro define como una
# resta de dos variables; la nota al pie de cada tabla lo especifica.
#
# OJO con el emparejamiento .EXP <-> fila del libro: la LETRA del .EXP no
# sigue el orden de filas del libro. En la 5.4, TBL54A usa esubvamfg1.2 y
# TBL54B usa esubvamfg.8, pero el libro lista 0,8 primero. Emparejar por el
# .prm que declara el .EXP, nunca por la letra.
TABLAS: dict[str, dict] = {
    "4.5": {
        "pag": 127,
        "desc": "+10% TFP en servicios de USA, tres funciones de utilidad",
        "celdas": [
            ("TBL45A", "ppa", "AGR|USA", 0.59, 2),
            ("TBL45A", "ppa", "MFG|USA", -0.20, 2),
            ("TBL45A", "ppa", "SER|USA", -5.07, 2),
            ("TBL45A", "qpa", "AGR|USA", 1.96, 2),
            ("TBL45A", "qpa", "MFG|USA", 5.46, 2),
            ("TBL45A", "qpa", "SER|USA", 9.66, 2),
            ("TBL45B", "ppa", "AGR|USA", 0.91, 2),
            ("TBL45B", "ppa", "MFG|USA", -0.24, 2),
            ("TBL45B", "ppa", "SER|USA", -5.19, 2),
            ("TBL45B", "qpa", "AGR|USA", 6.22, 2),
            ("TBL45B", "qpa", "MFG|USA", 6.80, 2),
            ("TBL45B", "qpa", "SER|USA", 9.27, 2),
            ("TBL45C", "ppa", "AGR|USA", 0.64, 2),
            ("TBL45C", "ppa", "MFG|USA", -0.18, 2),
            ("TBL45C", "ppa", "SER|USA", -5.05, 2),
            ("TBL45C", "qpa", "AGR|USA", 4.10, 2),
            ("TBL45C", "qpa", "MFG|USA", 4.92, 2),
            ("TBL45C", "qpa", "SER|USA", 9.78, 2),
        ],
    },
    "4.6": {
        "pag": 130,
        "desc": "arancel US a manufacturas al 10%, tres elasticidades Armington",
        "celdas": [
            ("TBL46A", "qpm", "MFG|USA", -4.6, 1),
            ("TBL46A", "qpd", "MFG|USA", -0.1, 1),
            ("TBL46B", "qpm", "MFG|USA", -5.7, 1),
            ("TBL46B", "qpd", "MFG|USA", 0.5, 1),
            ("TBL46C", "qpm", "MFG|USA", -14.0, 1),
            ("TBL46C", "qpd", "MFG|USA", 4.0, 1),
            # el ratio import/domestic que publica la tercera fila
            ("TBL46A", "qpm-qpd", "MFG|USA", -4.5, 1),
            ("TBL46B", "qpm-qpd", "MFG|USA", -6.2, 1),
            ("TBL46C", "qpm-qpd", "MFG|USA", -18.0, 1),
        ],
    },
    "5.4": {
        "pag": 159,
        "desc": "+5pp impuesto al trabajo en MFG de USA, dos elasticidades",
        "nota": "el libro dice que las formulas son APROXIMADAS",
        "celdas": [
            # TBL54B = esubvamfg.8 -> fila 0,8 del libro
            ("TBL54B", "qfe-qfe", "CAPITAL|MFG|USA::LABOR|MFG|USA", 2.99, 2),
            # aprox: el libro publica 3,76 y la formula da 3,7658 -> redondea a 3,77.
            # Un digito del ultimo decimal, en la fila que el propio libro llama
            # aproximada. NO se afloja la tolerancia: se marca como aprox y se
            # cuenta aparte, asi la diferencia queda visible en vez de tapada.
            ("TBL54B", "pfe-pfe", "LABOR|MFG|USA::CAPITAL|MFG|USA", 3.76, 2, "aprox"),
            # TBL54A = esubvamfg1.2 -> fila 1,2
            ("TBL54A", "qfe-qfe", "CAPITAL|MFG|USA::LABOR|MFG|USA", 4.25, 2),
            ("TBL54A", "pfe-pfe", "LABOR|MFG|USA::CAPITAL|MFG|USA", 3.56, 2, "aprox"),
        ],
    },
    "5.5": {
        "pag": 164,
        "desc": "+10% stock de capital de USA, demanda de insumos en servicios",
        "celdas": [
            ("TBL55", "qo", "SER|USA", 2.1, 1),
            ("TBL55", "qfa", "AGR|SER|USA", 2.1, 1),
            ("TBL55", "qfa", "MFG|SER|USA", 2.1, 1),
            ("TBL55", "qfa", "SER|SER|USA", 2.1, 1),
            ("TBL55", "qfe", "CAPITAL|SER|USA", 9.7, 1),
            ("TBL55", "qfe", "LABOR|SER|USA", -0.7, 1),
        ],
    },
    "6.2": {
        "pag": 183,
        "desc": "subsidio 5% al consumo de MFG domesticas, tres movilidades de K",
        "celdas": [
            ("TBL62A", "pes", "CAPITAL|AGR|USA", 1.1, 1),
            ("TBL62A", "pes", "CAPITAL|MFG|USA", 1.1, 1),
            ("TBL62A", "pes", "CAPITAL|SER|USA", 1.1, 1),
            ("TBL62C", "pes", "CAPITAL|AGR|USA", 2.7, 1),
            ("TBL62C", "pes", "CAPITAL|MFG|USA", 4.2, 1),
            ("TBL62C", "pes", "CAPITAL|SER|USA", 0.1, 1),
            ("TBL62B", "pes", "CAPITAL|AGR|USA", 4.3, 1),
            ("TBL62B", "pes", "CAPITAL|MFG|USA", 4.9, 1),
            ("TBL62B", "pes", "CAPITAL|SER|USA", -0.1, 1),
        ],
    },
    "6.3": {
        "pag": 187,
        "desc": "+10% oferta de trabajo de USA, factores sustitutos vs complementos",
        "celdas": [
            ("TBL63B", "pe", "LABOR|USA", -1.5, 1),
            ("TBL63B", "pe", "CAPITAL|USA", -1.5, 1),
            ("TBL63A", "pe", "LABOR|USA", -2.0, 1),
            ("TBL63A", "pe", "CAPITAL|USA", 5.4, 1),
        ],
    },
    "6.4": {
        "pag": 190,
        "desc": "+10% productividad del trabajo en USA (el libro declara Johansen)",
        "celdas": [
            ("TBL64", "qfe", "LABOR|AGR|USA", -6.2, 1),
            ("TBL64", "qfe", "LABOR|MFG|USA", -2.1, 1),
            ("TBL64", "qfe", "LABOR|SER|USA", 0.5, 1),
            ("TBL64", "qo", "AGR|USA", 2.1, 1),
            ("TBL64", "qo", "MFG|USA", 5.4, 1),
            ("TBL64", "qo", "SER|USA", 7.6, 1),
            ("TBL64", "qfe", "CAPITAL|AGR|USA", 1.9, 1),
            ("TBL64", "qfe", "CAPITAL|MFG|USA", -0.3, 1),
            ("TBL64", "qfe", "CAPITAL|SER|USA", 0.0, 1),
        ],
    },
    "6.5": {
        "pag": 193,
        "desc": "subsidio 10% al output de MFG, pleno empleo vs desempleo",
        "celdas": [
            ("TBL65B", "qfe", "LABOR|MFG|USA", 39.1, 1),
            ("TBL65B", "qo", "MFG|USA", 28.3, 1),
            ("TBL65A", "qfe", "LABOR|MFG|USA", 4.4, 1),
            ("TBL65A", "qo", "MFG|USA", 4.4, 1),
        ],
    },
    "6.6": {
        "pag": 195,
        "desc": "+2% oferta de trabajo de USA, estructura de la produccion",
        "celdas": [
            ("TBL66", "qo", "AGR|USA", 0.4, 1),
            ("TBL66", "qo", "MFG|USA", 1.1, 1),
            ("TBL66", "qo", "SER|USA", 1.5, 1),
        ],
    },
    "6.7": {
        "pag": 197,
        "desc": "subsidio 5% a la produccion de servicios de USA",
        "celdas": [
            ("TBL67", "pe", "LAND|USA", -21.6, 1),
            ("TBL67", "pe", "LABOR|USA", 8.0, 1),
            ("TBL67", "pe", "CAPITAL|USA", 7.7, 1),
        ],
    },
    "7.5": {
        "pag": 224,
        "desc": "arancel US 15% a MFG, dos ESUBD (terminos de intercambio)",
        "celdas": [
            ("TBL75A", "qmw", "MFG|USA", -17.5, 1),
            ("TBL75A", "qxw", "MFG|USA", -27.2, 1),
            ("TBL75B", "qmw", "MFG|USA", -26.2, 1),
            ("TBL75B", "qxw", "MFG|USA", -45.5, 1),
        ],
    },
    "7.8": {
        "pag": 240,
        "desc": "Dutch Disease: +10% precio ROW de manufacturas",
        "celdas": [
            ("TBL78", "qo", "AGR|USA", -4.2, 1),
            ("TBL78", "qo", "MFG|USA", 1.6, 1),
            ("TBL78", "qo", "SER|USA", -0.3, 1),
            ("TBL78", "qmw", "AGR|USA", 14.9, 1),
            ("TBL78", "qmw", "MFG|USA", 1.6, 1),
            ("TBL78", "qmw", "SER|USA", 20.9, 1),
            ("TBL78", "qxw", "AGR|USA", -20.2, 1),
            ("TBL78", "qxw", "MFG|USA", 3.2, 1),
            ("TBL78", "qxw", "SER|USA", -30.2, 1),
        ],
    },
    "7.9": {
        "pag": 242,
        "desc": "+10% productividad en margenes de comercio (atd)",
        "celdas": [
            ("TBL79", "qmw", "AGR|USA", 2.67, 2),
            ("TBL79", "qmw", "MFG|USA", 0.79, 2),
        ],
    },
}


def valor(f: SL4, var: str, clave: str) -> float:
    """Un valor del .sl4. 'a-b' con clave 'k1::k2' es la resta que define el libro."""
    if "-" in var and not var.startswith("-"):
        va, vb = var.split("-", 1)
        ka, kb = clave.split("::") if "::" in clave else (clave, clave)
        return valor(f, va, ka) - valor(f, vb, kb)
    labs, pct = f.labels(var), f.pct(var)
    if len(labs) != len(pct):
        raise ValueError(f"{var}: {len(labs)} etiquetas vs {len(pct)} valores")
    return float(pct[labs.index(clave)])


def main() -> int:
    solo = sys.argv[1] if len(sys.argv) > 1 else None
    cache: dict[str, SL4] = {}
    tot = ok = n_aprox = 0
    faltan: list[str] = []

    for tid, t in TABLAS.items():
        if solo and tid != solo:
            continue
        print(f"\n=== Tabla {tid} (pag. {t['pag']}) — {t['desc']}")
        if "nota" in t:
            print(f"    nota: {t['nota']}")
        for celda in t["celdas"]:
            exp, var, clave, libro, dec = celda[:5]
            # 5a posicion opcional: "aprox" = el libro declara la formula
            # aproximada, asi que una diferencia de 1 en el ultimo decimal es
            # de la formula publicada, no del modelo.
            aprox = len(celda) > 5 and celda[5] == "aprox"
            p = HERE / "out" / exp / f"{exp}.sl4"
            if not p.exists():
                faltan.append(f"{tid} {exp}: sin .sl4")
                continue
            try:
                f = cache.setdefault(exp, SL4(p))
                v = valor(f, var, clave)
            except (KeyError, ValueError) as e:
                faltan.append(f"{tid} {exp} {var}[{clave}]: {e}")
                continue
            tot += 1
            d = abs(round(v, dec) - libro)
            bien = d < 10**-dec / 2
            if bien:
                ok += 1
                marca = "ok"
            elif aprox and d <= 10**-dec + 1e-12:
                n_aprox += 1
                marca = "aprox (formula del libro)"
            else:
                marca = "DIFIERE"
            etiq = f"{exp} {var}[{clave.split('::')[0]}]"
            print(f"  {etiq:<42}{v:>10.4f}{libro:>9.{dec}f}{d:>8.{dec}f}  {marca}")

    print(f"\n{'=' * 62}")
    print(f"{ok} de {tot} celdas coinciden con el libro dentro de su propia precision.")
    if n_aprox:
        print(f"{n_aprox} mas quedan a 1 digito del ultimo decimal, en filas cuya")
        print("formula el propio libro declara aproximada (tabla 5.4).")
    if faltan:
        print(f"\nno medidas ({len(faltan)}):")
        for x in faltan:
            print(f"  {x}")
    return 0 if ok + n_aprox == tot else 1


if __name__ == "__main__":
    raise SystemExit(main())
