# Soluciones GEMPACK de Burfisher con Gragg y niveles de las 6 variables

Las mismas corridas que `../burfisher/`, rehechas por el paso 5 de
`scripts/windows/corridas_paper.ps1` (Windows, 2026-10-23, rama
`feat/windows-paso6-sltoht`), con dos cambios:

- **Gragg en todos los ejercicios**, también en los seis que el libro resuelve con
  Johansen de un paso (TBL45A/B/C, TBL64, ME4A/B). Un modelo en niveles no tiene por
  qué coincidir con una extrapolación lineal de un paso; con Gragg sí es comparable.
- **El TAB con el parche de niveles** (`gtapuv7_levels.tab`, de
  `scripts/gtap/make_levels_tab.py`): `rore`, `y`, `pinv`, `qinv`, `pva` y `pe` tienen
  nivel en el `.sl4`.

GEMPACK Release 11.3.004 (lo dice cada log). 43 corridas: las 41 de la Tabla 3 salvo
TBL53 (`target%`, sólo la GUI de RunGTAP) y GTAPV7 (el paso 5 no la corrió), más `gtapuv7`, `altax` y `closures`.

| Archivo | Qué es |
|---|---|
| `<exp>.sl4.gz` | la solución (gzip); **sin `PLEV`**: los niveles no se leen en la Mac (llegan con el paso 7, vía sltoht) |
| `<exp>.upd` | los datos actualizados; de ahí sale `VDFB` post |
| `<exp>.log.gz` | el log de GEMPACK (UTF-16, lo escribe gemsim en Windows), con el command file, el método y los pasos |

`me8` y `tbl813` vienen de sus `-dir.exp` (como en `../burfisher/`); aquí van con el
nombre del ejercicio.

**ME4A y ME4B no tienen solución.** GEMPACK termina con error fatal ("accuracy too
low") porque Gragg no converge en el shock del libro. Su `.sl4` es el de una corrida
fallida: no es un oráculo.

Se leen con `scripts/gtap/compare_burfisher_gempack.py --gempack-dir
tests/fixtures/gtap7_gempack/burfisher_gragg`.
