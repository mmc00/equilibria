"""Traduce la sintaxis de shock de la GUI de RunGTAP a valores directos.

Los 3 `.EXP` que no corren usan una sintaxis que resuelve la GUI, no el
ejecutable de GEMPACK:

    Shock tfd("AGR",ACTS,"USA") = target% 5 from file tfd.shk;   <- TBL53
    Shock tpdall("MFG","USA")   = rate% 1 from file tpdall.shk;  <- TBL813
    Shock tfe(ENDW,ACTS,"USA")  = rate% 1 from file tfe.shk;     <- ME8 (x12)

Los dos casos NO son lo mismo, y la diferencia importa:

* `rate% 1` escala el vector del `.shk` por 1, o sea que **los valores del
  `.shk` SON los shocks**. Se puede traducir celda por celda sin mas.

* `target% N` pide "mover el instrumento hasta que el objetivo cambie N%".
  Eso es un calculo que hace la GUI, no un dato que este en el `.shk`. El
  `.shk` trae el vector de referencia, pero el factor de escala sale de
  resolver el modelo. **No se puede traducir a literal sin correr el modelo.**

Asi que este script solo emite los `rate%` (TBL813 y ME8 = 3 experimentos de
los 3 archivos). Para TBL53 se explica en el README por que queda afuera.

Salida: un `.EXP` nuevo por cada uno, con sufijo `-DIR` (directo), al lado del
original. No sobreescribe nada.

Uso (en Windows o Mac, no necesita GEMPACK):
    python mk_exp_directos.py nus333
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# El .shk trae el header de dimensiones (`3 3 2  real row_order;`) y despues
# las filas, con los indices en comentarios `! %1=AGR`. Se parsea eso.
_DIM = re.compile(r"^\s*([\d\s]+?)\s+real\s+row_order\s*;", re.I)


def leer_shk(path: Path) -> tuple[list[int], list[float]]:
    """Devuelve (dims, valores en orden de fila) de un .shk de SHOCKSv7."""
    dims: list[int] = []
    vals: list[float] = []
    for linea in path.read_text().splitlines():
        m = _DIM.match(linea)
        if m and not dims:
            dims = [int(x) for x in m.group(1).split()]
            continue
        if not dims:
            continue
        # Se corta el comentario de la derecha (`! %1=AGR`) y se leen los numeros.
        datos = linea.split("!", 1)[0].strip()
        if not datos:
            continue
        for tok in datos.replace(",", " ").split():
            try:
                vals.append(float(tok))
            except ValueError:
                # Una linea que no es de datos (texto suelto): se ignora.
                pass
    if not dims:
        raise ValueError(f"{path.name}: no se encontro el header `real row_order`")
    esperados = 1
    for d in dims:
        esperados *= d
    if len(vals) != esperados:
        raise ValueError(
            f"{path.name}: dims {dims} piden {esperados} valores, se leyeron "
            f"{len(vals)}. Parseo incorrecto — NO se emite el .EXP."
        )
    return dims, vals


def _celdas_literales(
    idx: str, dims: list[int], vals: list[float], escala: float
) -> list[tuple[str, float]] | None:
    """El valor directo de un shock, solo cuando TODOS los indices son literales.

    Devuelve [(indices, valor)] o None si hay que desplegar un set.

    Por que solo el caso escalar: con un set (COMM, ACTS, ENDW) el orden de las
    celdas del .shk lo fija su `row_order`, y mapear eso a nombres exige conocer
    el orden de los elementos del set en el .har — que no es el orden
    alfabetico ni necesariamente el del .EXP. Inventarlo produciria shocks
    aplicados al sector equivocado, que es peor que no traducir: el modelo
    correria y daria numeros plausibles y falsos.
    """
    partes = [p.strip() for p in idx.split(",")]
    if not all(p.startswith('"') and p.endswith('"') for p in partes):
        return None  # hay un set: no se puede resolver sin el orden del .har

    # Escalar: el .shk tiene que traer exactamente una celda por combinacion, y
    # sin el orden de los sets no se puede ubicar CUAL. Se resuelve solo si el
    # .shk tiene tantas dimensiones como indices y se conoce el mapeo por
    # posicion — que es el caso de tpdall(COMM,REG) con COMM/REG en row_order.
    if len(partes) != len(dims):
        return None
    # Los indices del .EXP son literales, asi que la celda esta identificada por
    # NOMBRE; lo que falta es su POSICION en el .shk, y eso pide el orden de cada
    # set. Para COMM y REG el orden esta en el propio .shk, en los comentarios
    # `! %1=AGR` / `! %2=USA`, que es de donde salen estos nombres.
    orden = _orden_de_sets(dims)
    if orden is None:
        return None
    pos = 0
    paso = 1
    for eje in reversed(range(len(dims))):
        nombre = partes[eje].strip('"')
        elementos = orden[eje]
        if nombre not in elementos:
            return None  # un set que no conocemos: mejor no adivinar
        pos += elementos.index(nombre) * paso
        paso *= dims[eje]
    return [(", ".join(partes), vals[pos] * escala)]


# El orden de COMM y REG en los .shk de nus333, leido de sus comentarios
# (`! %1=AGR` ... / `! %2=USA  ROW`). Solo se usa para ubicar una celda cuyos
# indices ya son literales — nunca para desplegar un set.
_SETS_NUS333 = {3: ["AGR", "MFG", "SER"], 2: ["USA", "ROW"]}


def _orden_de_sets(dims: list[int]) -> list[list[str]] | None:
    """El orden de elementos de cada dimension, por su tamano. None si no consta."""
    orden: list[list[str]] = []
    for d in dims:
        if d not in _SETS_NUS333:
            return None
        orden.append(_SETS_NUS333[d])
    return orden


def main(carpeta: str) -> int:
    base = Path(carpeta)
    if not base.is_dir():
        print(f"ERROR: {base} no es una carpeta")
        return 1

    # `rate% N`: el .shk trae los shocks, escalados por N.
    # re.M es obligatorio: sin el, `^` ancla al inicio del ARCHIVO y ningun
    # `Shock` de la linea 29 matchea. Combinado con el CRLF de los .EXP, el
    # script emitia 0 shocks sin decir por que.
    rate = re.compile(
        r"^(?P<pre>[ \t]*Shock\s+(?P<var>\w+)\((?P<idx>[^)]*)\)\s*=\s*)"
        r"rate%\s+(?P<n>[\d.]+)\s+from\s+file\s+(?P<shk>\S+?)\s*;",
        re.I | re.M,
    )
    target = re.compile(r"target%\s+[\d.]+\s+from\s+file", re.I)

    total = 0
    for exp in sorted(base.glob("*.EXP")):
        texto = exp.read_text()
        if not rate.search(texto) and not target.search(texto):
            continue

        if target.search(texto):
            print(f"{exp.name}: usa `target%` — NO se traduce.")
            print("   `target%` es un objetivo que la GUI resuelve corriendo el")
            print("   modelo; el factor de escala no esta en el .shk. Hay que")
            print("   calcularlo, no transcribirlo.")
            if not rate.search(texto):
                continue

        salida: list[str] = []
        emitidos = 0
        for linea in texto.splitlines(keepends=True):
            # Los .EXP vienen con CRLF de Windows: hay que sacar el \r ademas del
            # \n, o el `;\s*$` del regex no matchea nunca y el script emite 0
            # shocks en silencio (me paso).
            m = rate.match(linea.rstrip("\r\n"))
            if not m:
                salida.append(linea)
                continue
            shk = base / m.group("shk")
            if not shk.exists():
                print(f"   {exp.name}: falta {m.group('shk')} — se deja la linea.")
                salida.append(linea)
                continue
            try:
                dims, vals = leer_shk(shk)
            except ValueError as e:
                print(f"   {e}")
                salida.append(linea)
                continue

            escala = float(m.group("n"))
            var, idx = m.group("var"), m.group("idx")
            # `rate% 1` = el .shk tal cual. Con otra N, se escala.
            # Se emite UNA linea por celda para no depender de que el
            # ejecutable acepte leer el .shk directo.
            salida.append(
                f"! Traducido de `rate% {m.group('n')} from file "
                f"{m.group('shk')}` (dims {dims}); valores del .shk x{escala:g}.\n"
            )
            salida.append(f"! Original: {linea.strip()}\n")

            # Dos variantes, porque `= file X.shk;` es sintaxis de GEMPACK que NO
            # se pudo verificar sin GEMPACK (se desarrollo en Mac). La A es la
            # limpia si el ejecutable la acepta; la B no depende de eso, pero solo
            # se puede emitir cuando los indices son literales (un escalar).
            # En Windows: probar A y, si corta, comentarla y descomentar B.
            literales = _celdas_literales(idx, dims, vals, escala)
            if literales is not None:
                salida.append("! --- variante A (si GEMPACK lee el .shk) ---\n")
                salida.append(f"Shock {var}({idx}) = file {m.group('shk')};\n")
                salida.append(
                    "! --- variante B (valor directo, sin depender de A) ---\n"
                )
                for k, val in literales:
                    salida.append(f"! Shock {var}({k}) = {val!r};\n")
            else:
                salida.append(
                    "! Indices con sets (COMM/ACTS/ENDW): la variante de valor\n"
                    "! directo necesitaria una linea por celda con los nombres\n"
                    "! desplegados, y el orden depende del row_order del .shk.\n"
                )
                salida.append(f"Shock {var}({idx}) = file {m.group('shk')};\n")
            emitidos += 1

        # Cuantos `rate%` habia que traducir, contados aparte del loop: si el
        # regex falla (CRLF, un formato distinto), `emitidos` queda en 0 y sin
        # esta guarda el script escribiria un .EXP identico al original y diria
        # que todo bien. Un .EXP "traducido" que no tradujo nada es peor que un
        # error: en Windows falla igual y el mensaje no dice por que.
        esperados_rate = len(rate.findall(texto.replace("\r\n", "\n")))
        if esperados_rate and emitidos != esperados_rate:
            print(
                f"{exp.name}: habia {esperados_rate} shocks `rate%` y se "
                f"tradujeron {emitidos}. NO se emite el .EXP — revisar el parseo."
            )
            continue

        if emitidos:
            destino = exp.with_name(exp.stem + "-DIR.EXP")
            # NO se pisa un -DIR.EXP que ya existe. El flujo real es "correr el
            # .bat, y si la variante A corta, editar a mano y volver a correr";
            # regenerar en cada corrida borraba esa edicion. Paso en Windows: el
            # .bat volvio a poner la variante A, fallo, y dejo un .upd de 0 B y
            # un .sl4 truncado ENCIMA del resultado bueno. Para regenerar de
            # cero, borrar el -DIR.EXP a mano.
            if destino.exists():
                print(
                    f"{exp.name}: {destino.name} ya existe, NO se toca "
                    "(puede tener ediciones a mano). Borralo para regenerarlo."
                )
                continue
            destino.write_text("".join(salida))
            print(f"{exp.name} -> {destino.name}  ({emitidos} shocks traducidos)")
            total += 1

    print(f"\n{total} .EXP emitidos.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "nus333"))
