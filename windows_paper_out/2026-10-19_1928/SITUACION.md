# Paso 4 en Windows — situacion (2026-10-19)

**Estado: la aceptacion de las seis variables pasa en los dos datasets**, pero con
corridas hechas a mano. El script `corridas_paper.ps1 -Pasos 4` (fbf46eb) falla tal
como esta. Hay que tocar dos cosas antes de que corra solo.

## Que corrio

| paso | resultado |
|---|---|
| `make_levels_tab.py` | OK -> `4_niveles/tab/` |
| `tablo -sti gtapv7_levels.sti` | OK (gss/gst). El orden de respuestas del `.sti` funciono |
| gemsim gtap7_3x3 (script) | FALLA: no existe `tm10.cmf` |
| gemsim gtap7_20x41 capfix (script) | FALLA: GEMSIM 11.3 no lee el `.prm` capfix |
| gemsim 3x3 `tm10_s8-16-32` (a mano) | OK, 11 s |
| gemsim 20x41 capfix con `.prm` parchado (a mano) | OK, 19 min, max residual ratio 1.05e-7 |
| `export_sl4_levels.py` (ambos) | OK; los % de cambio coinciden con el sl4dump sin parche, salvo `CNTendwr`/`CNTendw` |

## Aceptacion (`test_levels_six_vars.py`, replicado en `4_niveles/a_mano/check*.py`)

| dataset | seis en NOLV | y / qinv / rore vs oraculo (max rel) | pinv / pva / pe |
|---|---|---|---|
| gtap7_3x3   | ninguna | 3.0e-8 / 4.2e-8 / 2.2e-7 | = 1 exacto |
| gtap7_20x41 | ninguna | 6.7e-8 / 7.0e-8 / 4.2e-7 | = 1 exacto |

## Que falta (Mac)

1. **Fixtures:** copiar a `tests/fixtures/gtap7_gempack/` y correr `test_levels_six_vars.py`:
   - `4_niveles/a_mano/gtap7_3x3_tm10_s8-16-32/niveles/sl4levels6_gtap7_3x3_tm10.har`
   - `4_niveles/a_mano/gtap7_20x41_tm10_s8-16-32_capfix/niveles/sl4levels6_tm10_s8-16-32_capfix.har`
     -> renombrar a `sl4levels6_gtap7_20x41_tm10_s8-16-32_capfix.har`
2. **Script, job del 3x3:** usar el tag `tm10_s8-16-32`, no `tm10`. El fixture
   `sl4dump_gtap7_3x3_tm10.har` es identico (252/252 cabeceras) a esa corrida.
   `tm10.cmf` era el cierre viejo con swap dpsave (antes de 06dd878) y ya no existe.
3. **`.prm` capfix vs GEMPACK 11.3** (`C:\GP` = GEMPACK 11.3, GEMSIM 4.3; las corridas
   originales eran con `gtapv7.exe` de GEMPACK 12.1). `write_har` no conserva el layout:
   - `writer.py:_write_1cfull` recalcula el ancho: XXCR/XXCD 70 -> 41, XXCP 6 -> 12
     -> `E-UNEXPECTED LENGTH OF STRINGS`.
   - Con los anchos arreglados falla igual en ENDOWFLAG: `E-NREC not as expected`
     (troceo de registros).

   Workaround usado: `a_mano/patch_rdlt.py` (copia byte a byte de `default.prm` con
   RDLT 1 -> 0). El arreglo de fondo: que `_force_rdlt` parchee in situ y/o que
   `write_har` respete ancho y registros. Si no, el paso 4 del script tiene que usar
   el parche byte a byte.
4. **Opcional:** `export_sl4_levels` descarta en silencio las variables que fallan el
   xcheck. Hoy son `CNTendwr` y `CNTendw`: contribuciones ~1e-3, ruido 11.3 vs 12.1.
   No van ni a LEVB ni a NOLV. Habria que avisar o listarlas.

Detalle tecnico: `4_niveles/a_mano/NOTAS.md`. Log y resumen del script: `log.txt` y `resumen.txt`.
