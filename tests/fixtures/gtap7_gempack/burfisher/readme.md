# Soluciones GEMPACK de los ejercicios de Burfisher (nus333)

Una corrida por ejercicio de la Tabla 3, sobre `datasets/nus333`, el paquete del libro. Cada `.exp` se anexó tal cual al command file, con su `.prm`, su cierre y su método.

Corrió en Windows el 2026-09-24 con `tools/gempack-oracle/01-run-all.bat`, de la rama `tools/gempack-oracle-nus333`:
- GEMPACK Release 12.1.003;
- el modelo `gtapuv7.for` de RunGTAP, sin el parche de niveles.

`me8` y `tbl813` son de la corrida del 2026-09-30, con sus `-dir.exp` (ver `scripts/gtap/mk_exp_directos.py`).

| Archivo | Qué es |
|---|---|
| `<exp>.sl4.gz` | la solución (gzip). Trae los niveles pre (`LEVB`) y post (`LEVA`) de 44 variables |
| `<exp>.upd` | los datos actualizados; de ahí sale `VDFB` post |
| `<exp>.log.gz` | el log de GEMPACK. Repite el command file entero, con el método (`Method`/`Steps`) y los shocks |
| `gtapv7.sl4.gz` | `gtapv7.exp` corrido con `gtapv7.for`, el modelo condensado. Ese modelo **omite 11 variables**, así que prueba que los argumentos se leen de la lista de la solución (`VCNA`/`VCAR`) y no de la del modelo (`VCSP`/`VCSN`) (#148) |
| `gtapuv7.sl4.gz` | `gtapuv7.exp`: el mismo shock al numerario (`pfactwld` +10 %, Johansen de un paso), con el modelo sin condensar como el resto. Su cierre deja exógenos los impuestos que sólo existen en ese modelo; en equilibria ya son parámetros, así que se compara con la misma corrida que GTAPV7 (homogeneidad, §6.4 del paper) |
| `sl4dump_tbl45a.har`, `sl4dump_me9c.har`, `sl4dump_gtapv7.har` | el volcado de cambios % de esos `.sl4`, hecho por `sltoht` (SLTOHT 5.53, GEMPACK 11.3.004) el mismo día. Es el oráculo del cambio de #148: lo escribió `sltoht`, no equilibria |

**Método.** Las 36 corridas usan Gragg, salvo 6 que usan Johansen de un paso porque así las resolvió el libro: TBL45A, TBL45B, TBL45C, TBL64, ME4A y ME4B.

**Variables sin nivel.** Sin el parche de `scripts/gtap/make_levels_tab.py`, `rore`, `y`, `pinv`, `qinv`, `pva` y `pe` no tienen nivel en el `.sl4`. El paso 5 de `scripts/windows/corridas_paper.ps1` las vuelve a correr con ese parche y con Gragg.

**Ejercicio que falta.** TBL53 no está, porque su `.exp` usa `target%`, que sólo resuelve la GUI de RunGTAP.

**Lectura.** Los `.sl4` van comprimidos. Para leerlos, `tests/compare/_oracle.py` (`burfisher_sl4`) y `scripts/gtap/compare_burfisher_gempack.py` los descomprimen en un directorio temporal.
