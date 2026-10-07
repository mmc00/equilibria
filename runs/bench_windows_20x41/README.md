# Benchmark GTAP 20x41 en una sola máquina (Windows)

**Para qué:** medir **equilibria, GEMPACK y GAMS en la misma máquina y con el mismo
experimento**. Hasta hoy los tiempos citables no se pueden comparar: los ~7,0 min de
equilibria son de una Mac M3 Pro, y los 2 min 33 s de GEMPACK (agosto) son de este
Windows pero con otro cierre (capFlex). GAMS nunca resolvió el 20x41.

**Experimento (igual en las tres):** `gtap7_20x41`, aranceles `tm` +10% uniforme,
cierre capFix (RORDELTA=0, capital sluggish, residual = ROW), ifSUB=1.

## 1. Preparar la máquina

Desde la raíz del repo, en PowerShell:

```powershell
git fetch; git checkout bench/windows-20x41
uv sync --dev
uv pip install psutil          # RAM y núcleos en el resultado (opcional)
uv run pyomo download-extensions   # PyNumero ASL
```

**PATH (librería para Windows):**

```powershell
curl.exe -L -o $env:TEMP\path_win.zip https://pages.cs.wisc.edu/~ferris/path/path_5.0.05_Win64.zip
# sha256 esperado: 1c665e2455603eee4caf8a87b9ff85e381cd31b6a7971ab11f0c03f0a2d363ae
Get-FileHash $env:TEMP\path_win.zip -Algorithm SHA256
Expand-Archive $env:TEMP\path_win.zip $env:TEMP\path_win -Force
mkdir .cache\path_capi -Force
copy $env:TEMP\path_win\path_5\pathlib\lib\path50.dll .cache\path_capi\
copy $env:TEMP\path_win\path_5\pathlib\lib\lusol.dll  .cache\path_capi\
```

**path-capi-python** (el binding de PATH, mismo SHA que la CI):

```powershell
git clone https://github.com/mmc00/path-capi-python ..\path-capi-python
git -C ..\path-capi-python checkout 786110fdb4ed274dff5b747cd10f19ce5cb0295c
```

**Variables** (en la misma sesión de PowerShell donde se corre todo):

```powershell
$env:EQUILIBRIA_PATH_CAPI_LIB_DIR = "$PWD\.cache\path_capi"
$env:EQUILIBRIA_PATH_CAPI_SRC     = (Resolve-Path ..\path-capi-python\src).Path
$env:PATH_LICENSE_STRING = "1259252040&Courtesy&&&USR&GEN2035&5_1_2026&1000&PATH&GEN&31_12_2035&0_0_0&6000&0_0"
# Si RunGTAP no está en C:\runGTAP375:
# $env:GTAPV7 = "C:\ruta\gtapv7.exe"
```

## 2. Revisar qué hay

```powershell
uv run python runs/bench_windows_20x41/check_env.py
```

Dice qué encontró y qué falta. **Mirar la línea de la licencia de GAMS:** con licencia
demo/community el 20x41 no entra; en ese caso GAMS se registra como "licencia
insuficiente" y se siguen midiendo las otras dos.

## 3. Medir

Con la máquina **sin otras cargas** (cerrar navegador, etc.). equilibria necesita
~5 GB de RAM libre; si la máquina empieza a usar swap, los tiempos no sirven. En
este orden:

```powershell
uv run python runs/bench_windows_20x41/bench_gempack.py --reps 3
uv run python runs/bench_windows_20x41/bench_equilibria.py --mode cold --reps 1
uv run python runs/bench_windows_20x41/bench_equilibria.py --mode warm --reps 3
uv run python runs/bench_windows_20x41/bench_gams.py --reps 1
uv run python runs/bench_windows_20x41/collect.py
```

- **Tiempos:** GEMPACK tarda unos minutos por corrida, equilibria warm ~7-20 min y
  equilibria cold bastante más.
- **Si equilibria da `code=5`:** es el límite de tiempo de PATH (1 h por solve),
  no el modelo. Pasa cuando la máquina está lenta o sin RAM. En una prueba en la
  Mac, con 12 GB de swap y otras sesiones corriendo, tardó 87 min y dio
  `code=5`. El log completo queda en `results/equilibria_<modo>_repN.log`.
- **GAMS:** puede tardar horas o no converger. `path.opt` le pone un tope de 1 h por
  solve.

## 4. Devolver los resultados

```powershell
git add runs/bench_windows_20x41/results
git commit -m "bench(20x41): resultados en <máquina>"
git push
```

`results/RESULTADOS.md` trae la tabla y, abajo, **lo que no es igual entre las tres**
(método, períodos, arranque, hilos). Leerlo antes de citar un número.

## Archivos

| archivo | qué hace |
|---|---|
| `check_env.py` | revisa Python/equilibria, PATH, ASL, gdxdump, GEMPACK y GAMS (+ licencia) |
| `bench_equilibria.py` | `warm` (como la medición de la Mac) y `cold` (desde el benchmark); un proceso nuevo por corrida |
| `bench_gempack.py` | `.cmf` capFix vía `scripts/gtap/run_gempack_matrix.prepare`; con hilos por defecto y con 1 hilo |
| `bench_gams.py` | corre `gams/…ifsub1.gms.gz` (bundle generado, `iterlim` subido) y lee el `.lst` |
| `collect.py` | arma `results/RESULTADOS.md` |
