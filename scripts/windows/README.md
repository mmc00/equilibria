# Corridas del paper en Windows

`corridas_paper.ps1` corre de una vez todo lo que solo se puede hacer en Windows (GEMPACK y `sltoht`).

| Paso | Qué | Issue |
|---|---|---|
| 1 | texto real de `sltoht -SHL -SIC` y `.har` de niveles validado para TBL45A, ME9C y GTAPV7 | #148 (oráculo 3) |
| 2 | lo mismo para gtap7_3x3 tm10 y gtap7_20x41 capFix; copia del `.sl4` del 20x41 | #133, #135 |
| 3 | SIMPLE de Horridge: GEMPACK (tablo, gemsim fixcap/mobcap) y GAMS (NLP, MCP, MPSGE) | #134 |
| 4 | niveles de `y`, `qinv`, `rore`, `pinv`, `pva`, `pe`: `make_levels_tab.py` parcha `gtapv7.tab` (`orig_level`), `tablo` lo compila para GEMSIM y `gemsim` re-corre 3x3 tm10 y 20x41 capFix; `export_sl4_levels.py` exporta los niveles y comprueba que los % de cambio son los de la corrida sin parche | #135 |
| 5 | los ejercicios de Burfisher (`runs\burfisher_nus333`) con esos niveles: `make_levels_tab.py --uncondensed` arma `gtapuv7_levels` (sin condensar, como el GTAPUV7 de RunGTAP), `tablo` lo compila y `gemsim` corre cada `.exp`; los de Johansen de 1 paso pasan a Gragg 2-4-6 | #135 |

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

**Estado:** el paso 4 corrió el 2026-10-19 (rama de datos `windows/corridas-paper-2026-10-19_1928`): `make_levels_tab` y `tablo` bien; los dos `gemsim` fallaron y se rehicieron a mano. Este script ya trae esos arreglos (tag `tm10_s8-16-32` en el 3x3, `.prm` capFix con `_force_rdlt` byte a byte), pero no volvió a correr solo de punta a punta.
