# Soluciones GEMPACK de los ejercicios de Burfisher (nus333)

Una corrida por ejercicio de la Tabla 3, sobre el dataset nus333 de Burfisher 3e. Cada `.EXP` del libro se anexó tal cual al command file, con su `.prm`, su cierre y su método. Corrió en Windows el 2026-09-24 con `tools/gempack-oracle/01-run-all.bat`, con GEMPACK 12.1 y el `GTAPV7.TAB` de RunGTAP sin parche. `ME8` y `TBL813` son la corrida del 2026-09-30 con `04-exp-directos.bat`: sus `.EXP` usan `rate% N from file X.shk`, sintaxis de la GUI de RunGTAP que el ejecutable no lee, y se tradujo al valor directo del shock.

| Archivo | Qué es |
|---|---|
| `<exp>.sl4.gz` | la solución (gzip). Trae los niveles pre (`LEVB`) y post (`LEVA`) de 44 variables |
| `<exp>.upd` | los datos actualizados; de ahí sale `VDFB` post |
| `<exp>.log.gz` | el log de GEMPACK: repite el command file entero, con el método (`Method`/`Steps`) y los shocks |

Las 36 corridas usan Gragg salvo 6, que usan Johansen de un paso porque así las resolvió el libro: TBL45A, TBL45B, TBL45C, TBL64, ME4A y ME4B.

Sin el parche de `scripts/gtap/make_levels_tab.py`, `rore`, `y`, `pinv`, `qinv`, `pva` y `pe` no tienen nivel en el `.sl4`.

TBL53 no está: su `.EXP` usa `target%`, un objetivo que sólo resuelve la GUI de RunGTAP.
