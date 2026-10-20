# Corridas del paper en Windows

`corridas_paper.ps1` corre de una vez todo lo que solo se puede hacer en Windows (GEMPACK y `sltoht`).

| Paso | Qué | Issue |
|---|---|---|
| 1 | texto real de `sltoht -SHL -SIC` y `.har` de niveles validado para TBL45A, ME9C y GTAPV7 | #148 (oráculo 3) |
| 2 | lo mismo para gtap7_3x3 tm10 y gtap7_20x41 capFix; copia del `.sl4` del 20x41 | #133, #135 |
| 3 | SIMPLE de Horridge: GEMPACK (tablo, gemsim fixcap/mobcap) y GAMS (NLP, MCP, MPSGE) | #134 |
| 4 | niveles de `y`, `qinv`, `rore`, `pinv`, `pva`, `pe`: `make_levels_tab.py` parcha `gtapv7.tab` (`orig_level`), `tablo` lo compila para GEMSIM y `gemsim` re-corre 3x3 tm10 y 20x41 capFix; `export_sl4_levels.py` exporta los niveles y comprueba que los % de cambio son los de la corrida sin parche | #135 |

## Requisitos
- GEMPACK en `C:\GP` (o pasar `-GP`), con `sltoht.exe`, `tablo` y `gemsim`.
- `gams.exe` en el PATH (o pasar `-Gams`).
- `uv` en el PATH.
- Las corridas de `runs\gempack_matrix` (3x3 tm10 y 20x41 capFix) en esa carpeta.
- Paso 4: `gtapv7.tab` y `gtapv7.sti` en `C:\runGTAP375` (o pasar `-GtapDir`).

## Uso
Desde la raíz del repo:

```
git fetch && git checkout feat/gempack-levels-6vars
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

Un paso que falla no corta los demás. Al final imprime los comandos `git add -f / commit / push` para traer la carpeta a la Mac.

**Estado:** escrito en macOS **sin PowerShell para probarlo**. La primera corrida es también su prueba: si un paso falla, el `*_console.txt` de su carpeta dice por qué.
