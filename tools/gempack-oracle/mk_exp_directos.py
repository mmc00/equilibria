"""Traduce la sintaxis de shock de la GUI de RunGTAP a valores directos.

Los 3 `.EXP` que no corren usan una sintaxis que resuelve la GUI, no el
ejecutable de GEMPACK:

    Shock tfd("AGR",ACTS,"USA") = target% 5 from file tfd.shk;   <- TBL53
    Shock tpdall("MFG","USA")   = rate% 1 from file tpdall.shk;  <- TBL813
    Shock tfe(ENDW,ACTS,"USA")  = rate% 1 from file tfe.shk;     <- ME8 (x12)

Los dos casos NO son lo mismo, y la diferencia importa:

* `rate% N` sube N% la TASA del impuesto. El `.shk` NO trae ese shock: trae el
  que ELIMINA cada impuesto, en % de la potencia (medido en tpdall.shk: 6/6
  celdas = 100*(1/(1+t0)-1) a 4 decimales, t0 del basedata). Subir la tasa N%
  mueve la potencia -N/100 x ese valor (exacto: (1+t0(1+N/100))/(1+t0)-1 =
  N/100 * t0/(1+t0)). Se traduce celda por celda con escala -N/100.

  CORREGIDO 2026-09-30: antes se escalaba por N ("los valores del .shk SON los
  shocks"). TBL813-DIR corrio asi y dio EV USA +13.202 (eliminar el impuesto),
  cuando el libro (Tabla 8.12, 2a ed.) da -236,3 para +1%.

* `target% N` pide "mover el instrumento hasta que el objetivo cambie N%".
  Eso es un calculo que hace la GUI, no un dato que este en el `.shk`. El
  `.shk` trae el vector de referencia, pero el factor de escala sale de
  resolver el modelo. **No se puede traducir a literal sin correr el modelo.**

Asi que este script solo emite los `rate%` (TBL813 y ME8 = 3 experimentos de
los 3 archivos). Para TBL53 se explica en el README por que queda afuera.

Los indices con sets (`tfe(ENDW,ACTS,"USA")`) se despliegan en una linea por
celda. Los nombres de cada celda salen del propio .shk (`! %1=LAND`, `! %2=AGR
MFG SER`, `(%1,%2,"USA")`), no de un orden de sets supuesto, y se contrastan con
la lectura sin etiquetas (mismos valores, mismo orden).

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


def leer_shk_etiquetado(path: Path) -> dict[tuple[str, ...], float]:
    """{(nombre eje 1, eje 2[, eje 3]): valor} de un .shk de SHOCKSv7.

    Los NOMBRES salen del propio .shk, no de un orden supuesto: cada fila termina
    en `! %1=LAND`, la cabecera de columnas es `! %2=AGR  MFG  SER`, y en 3
    dimensiones cada matriz se abre con `(%1,%2,"USA")`. Se contrasta contra
    ``leer_shk``: mismos valores, mismo orden, y una celda por combinacion.
    """
    dims, vals = leer_shk(path)
    if len(dims) not in (2, 3):
        raise ValueError(f"{path.name}: {len(dims)} dimensiones, se esperaban 2 o 3")
    celdas: dict[tuple[str, ...], float] = {}
    orden: list[float] = []
    tercero: str | None = None
    cols: list[str] | None = None
    for linea in path.read_text().splitlines():
        m = re.search(r'\(%1,%2,"(\w+)"\)', linea)
        if m:
            tercero = m.group(1)
            continue
        m = re.match(r"^\s*!\s*%2=(.*)$", linea)
        if m:
            cols = m.group(1).split()
            continue
        m = re.search(r"!\s*%1=(\w+)", linea)
        if not (m and cols):
            continue
        datos = [float(t) for t in linea.split("!", 1)[0].split()]
        if len(datos) != len(cols):
            raise ValueError(f"{path.name}: fila con {len(datos)} valores y {len(cols)} columnas")
        if len(dims) == 3 and tercero is None:
            raise ValueError(f"{path.name}: fila antes de la cabecera (%1,%2,...)")
        for c, v in zip(cols, datos):
            clave = (m.group(1), c) + ((tercero,) if len(dims) == 3 else ())
            if clave in celdas:
                raise ValueError(f"{path.name}: celda {clave} repetida")
            celdas[clave] = v
            orden.append(v)
    if orden != vals:
        raise ValueError(
            f"{path.name}: las etiquetas no cubren los {len(vals)} valores en orden "
            f"(se etiquetaron {len(orden)}). NO se emite el .EXP."
        )
    return celdas


def _celdas(
    idx: str, celdas: dict[tuple[str, ...], float], escala: float
) -> list[tuple[str, float]] | None:
    """[(indices literales, valor)] de cada celda del .shk que cubre ``idx``.

    Un indice entre comillas (`"USA"`) fija ese eje; uno sin comillas (`COMM`,
    `ENDW`) es un set y se despliega a todas las celdas del .shk en ese eje. Los
    nombres son los del .shk, asi que no hay un orden de sets supuesto. None si
    el numero de indices no coincide con las dimensiones del .shk o si un literal
    no aparece en el .shk.
    """
    partes = [p.strip() for p in idx.split(",")]
    n = len(next(iter(celdas)))
    if len(partes) != n:
        return None
    fijo = [p.strip('"') if p.startswith('"') else None for p in partes]
    for eje, nombre in enumerate(fijo):
        if nombre is not None and nombre not in {k[eje] for k in celdas}:
            return None
    out = []
    for clave, v in celdas.items():
        if all(f is None or f == c for f, c in zip(fijo, clave)):
            # + 0.0: una celda en 0 escalada por -N/100 da -0.0; se emite 0.0.
            out.append((", ".join(f'"{c}"' for c in clave), v * escala + 0.0))
    return out


def main(carpeta: str) -> int:
    base = Path(carpeta)
    if not base.is_dir():
        print(f"ERROR: {base} no es una carpeta")
        return 1

    # `rate% N`: +N% a la tasa; el .shk trae el shock de ELIMINAR el impuesto.
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
                celdas = leer_shk_etiquetado(shk)
            except ValueError as e:
                print(f"   {e}")
                salida.append(linea)
                continue

            escala = -float(m.group("n")) / 100.0
            var, idx = m.group("var"), m.group("idx")
            # `rate% N` = -N/100 x el .shk (ver el docstring del modulo).
            # Se emite UNA linea por celda para no depender de que el
            # ejecutable acepte leer el .shk directo.
            salida.append(
                f"! Traducido de `rate% {m.group('n')} from file "
                f"{m.group('shk')}` (dims {dims}); valores del .shk x{escala:g}.\n"
            )
            salida.append(f"! Original: {linea.strip()}\n")

            # Solo valor directo. La forma `= file X.shk;` lee el .shk TAL CUAL, o
            # sea que aplica el shock de ELIMINAR el impuesto, no el `rate% N`: es
            # justo lo que dio TBL813-DIR (EV USA +13.202). Ya no se emite.
            literales = _celdas(idx, celdas, escala)
            if literales is None:
                # El .shk no tiene las dimensiones o los nombres del shock: NO se
                # traduce. Un shock mal ubicado corre y da numeros plausibles y
                # falsos.
                print(
                    f"   {exp.name}: {var}({idx}) no calza con {m.group('shk')}; "
                    "no se traduce."
                )
                salida.append(linea)
                continue
            for k, val in literales:
                salida.append(f"Shock {var}({k}) = {val!r};\n")
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
