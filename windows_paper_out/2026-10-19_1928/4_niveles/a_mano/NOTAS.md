# Paso 4 a mano (2026-10-19, Windows)

`corridas_paper.ps1 -Pasos 4` (commit fbf46eb) dejo bien `make_levels_tab` y `tablo`
(`../tab/`), pero los dos gemsim fallaron. Aqui estan las corridas rehechas a mano con
el mismo `gtapv7_levels.gss/gst` de `../tab/`. Ver `../../log.txt` y `../../resumen.txt`.

## Falla 1: gtap7_3x3 — `no esta runs\gempack_matrix\gtap7_3x3\tm10.cmf`

`tm10.cmf` ya no existe. Ademas era el cierre viejo con swap dpsave=del_tbalry (ver
`tm10.log` del 23-JUL). El fixture `sl4dump_gtap7_3x3_tm10.har` es identico (252/252
cabeceras) a `sl4dump_tm10_s8-16-32.har`: cierre estandar, `default.prm` original
(06dd878). Corrida con tag `tm10_s8-16-32` -> `gtap7_3x3_tm10_s8-16-32/`.

**Arreglo en el script:** en el paso 4, el job del 3x3 deberia ser `tm10_s8-16-32`, no `tm10`.

## Falla 2: gtap7_20x41 capfix — gemsim no lee `default_tm10_s8-16-32_capfix.prm`

`C:\GP` es GEMPACK **11.3** (GEMSIM 4.3 / TABLO 6.4, 2015). Las corridas originales eran
con `C:\runGTAP375\gtapv7.exe` (GEMPACK 12.1), que es mas tolerante. El `.prm` capfix
lo reescribe `_force_rdlt` (run_gempack_matrix.py) con `equilibria.babel.har.write_har`,
y GEMPACK 11.3 rechaza ese archivo:

1. `XXCR`/`XXCD`: 1CFULL de ancho 70 en el original; `writer.py:_write_1cfull` lo
   recalcula como max(12, len) = 41. Ademas `XXCP` es 6 en el original y queda en 12.
   -> `E-UNEXPECTED LENGTH OF STRINGS` (GETCVC).
2. Con los anchos arreglados (`fix_prm.py`) sigue fallando en `ENDOWFLAG`:
   `E-NREC not as expected` (RD7RBSS). El writer parte los registros distinto de
   como lo espera 11.3.

Workaround usado: `patch_rdlt.py` copia el `default.prm` original byte a byte y cambia
solo el int32 de RDLT (1 -> 0). Un byte distinto; su contenido es identico
(read_har) al `_capfix.prm` de la corrida 12.1. Con eso gemsim corre bien: 19 min,
max residual ratio 1.05e-7 -> `gtap7_20x41_tm10_s8-16-32_capfix/`.

**Arreglo de fondo:** `write_har` deberia conservar el ancho de los 1CFULL leidos y el
troceo de registros. O bien `_force_rdlt` deberia parchear RDLT in situ.

## Resultados (`check.py` / `check2.py` = test_levels_six_vars.py + quien falla el xcheck)

| dataset | seis en NOLV | y / qinv / rore vs oraculo (max rel) | pinv / pva / pe | xcheck-mismatch |
|---|---|---|---|---|
| gtap7_3x3   | ninguna | 3.0e-8 / 4.2e-8 / 2.2e-7 | = 1 exacto | CNTendwr, CNTendw |
| gtap7_20x41 | ninguna | 6.7e-8 / 7.0e-8 / 4.2e-7 | = 1 exacto | CNTendwr, CNTendw |

Las dos variables del xcheck son contribuciones (decomposicion) de orden 1e-3 con signo
cambiado en 1-2 celdas: ruido numerico gemsim 11.3 vs gtapv7.exe 12.1, no las seis.
`export_sl4_levels` las descarta en silencio (no van ni a LEVB ni a NOLV).

Los `.har` para la aceptacion en la Mac (copiar a `tests/fixtures/gtap7_gempack/`):
- `gtap7_3x3_tm10_s8-16-32/niveles/sl4levels6_gtap7_3x3_tm10.har`
- `gtap7_20x41_tm10_s8-16-32_capfix/niveles/sl4levels6_tm10_s8-16-32_capfix.har`
  (renombrar a `sl4levels6_gtap7_20x41_tm10_s8-16-32_capfix.har`)

Se borraron los `.slc` y `tm10_s8-16-32_capfix_levels.txt` del 20x41 (63 MB). Se
regenera con sltoht desde el `.sl4`.
