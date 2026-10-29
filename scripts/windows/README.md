# Corridas del paper en Windows

`corridas_paper.ps1` corre de una vez todo lo que solo se puede hacer en Windows (GEMPACK y `sltoht`).

| Paso | Qué | Issue |
|---|---|---|
| 1 | texto real de `sltoht -SHL -SIC` y `.har` de niveles validado para TBL45A, ME9C y GTAPV7 | #148 (oráculo 3) |
| 2 | lo mismo para gtap7_3x3 tm10 y gtap7_20x41 capFix; copia del `.sl4` del 20x41 | #133, #135 |
| 3 | SIMPLE de Horridge: GEMPACK (tablo, gemsim fixcap/mobcap) y GAMS (NLP, MCP, MPSGE) | #134 |
| 4 | niveles de `y`, `qinv`, `rore`, `pinv`, `pva`, `pe`: `make_levels_tab.py` parcha `gtapv7.tab` (`orig_level`), `tablo` lo compila para GEMSIM y `gemsim` re-corre 3x3 tm10 y 20x41 capFix; `export_sl4_levels.py` exporta los niveles y comprueba que los % de cambio son los de la corrida sin parche | #135 |
| 5 | los ejercicios de Burfisher (`data\gtap7\nus333`) con esos niveles: `make_levels_tab.py --uncondensed` arma `gtapuv7_levels` (sin condensar, como el GTAPUV7 de RunGTAP), `tablo` lo compila y `gemsim` corre cada `.exp`; los de Johansen de 1 paso pasan a Gragg 2-4-6 | #135 |
| 6 | los cambios (% y ordinarios) de cada ejercicio de Burfisher, con `EV` y `u`: `sltoht` vuelca cada `.sl4` a `sl4dump_<exp>.har`; (a) las soluciones del libro de `tests\fixtures\gtap7_gempack\burfisher` y (b) las del paso 5 de la misma corrida (Gragg en todos). Es el oráculo GEMPACK del bienestar (el lector de la Mac solo lee niveles) | paper §5.4 |

## Requisitos
- GEMPACK en `C:\GP` (o pasar `-GP`), con `sltoht.exe`, `tablo` y `gemsim`.
- `gams.exe` en el PATH (o pasar `-Gams`).
- `uv` en el PATH.
- Las corridas de `runs\gempack_matrix` (3x3 tm10 y 20x41 capFix) en esa carpeta.
- Pasos 4 y 5: `gtapv7.tab` y `gtapv7.sti` en `C:\runGTAP375` (o pasar `-GtapDir`).

## Uso
Desde la raíz del repo:

```
git fetch && git checkout main   # o la rama que traiga el paso que se quiere correr
powershell -ExecutionPolicy Bypass -File scripts\windows\corridas_paper.ps1
```

Solo algunos pasos (p. ej. solo el 4):

```
powershell -ExecutionPolicy Bypass -File scripts\windows\corridas_paper.ps1 -Pasos 4
```

Para indicar otras rutas:

```
powershell -ExecutionPolicy Bypass -File scripts\windows\corridas_paper.ps1 -GP C:\GP -Gams "C:\GAMS\53\gams.exe"
```

El script deja todo en `windows_paper_out\<fecha>\`:
- `log.txt` y `resumen.txt`, con OK/FALLA por paso;
- `manifest.txt`, con el SHA-256 de cada archivo.

Un paso que falla no corta los demás. Al final imprime los comandos para subir la carpeta a una **rama de datos** `windows/corridas-<fecha>`, que no se mergea: los archivos que necesitan los tests se copian después a `tests/fixtures/`. Al commitear solo se exime el hook de archivos grandes (`SKIP=check-added-large-files`), nunca `--no-verify`.

**Para el paper:** `-Pasos 5,6` en una sola corrida, así el paso 6 vuelca también las soluciones Gragg del paso 5. Corrió el 2026-10-23 desde `f173370` (salidas enviadas aparte, no en GitHub). Hallazgos de esa corrida, ya corregidos en el script:
- `-Pasos 5,6` con `-File` llegaba como el texto "5,6" y `[int[]]` lo volvía 56: no corría nada. Ahora se parte en la coma (y un paso fuera de 1-6 corta). Con `-Command "& .\scripts\windows\corridas_paper.ps1 -Pasos 5,6"` también funciona.
- ME4A y ME4B (Gragg 2-4-6) terminan en "Accuracy too low": el shock de tierra de −81 % pide más pasos o subintervalos. `gemsim` escribe el `.sl4` igual, y el resumen los marcaba OK; ahora OK exige `rc=0` y el cierre normal del log. Sus `.sl4`/`sl4dump` de esa corrida **no** sirven de oráculo.

**Estado:** el paso 4 corrió el 2026-10-19 (rama de datos `windows/corridas-paper-2026-10-19_1928`): `make_levels_tab` y `tablo` bien; los dos `gemsim` fallaron y se rehicieron a mano. Este script ya trae esos arreglos (tag `tm10_s8-16-32` en el 3x3, `.prm` capFix con `_force_rdlt` byte a byte), pero no volvió a correr solo de punta a punta.
