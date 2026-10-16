# GTAP 20x41 en Windows — GEMPACK vs equilibria (phase_times)

Medido el 2026-10-15/16 sobre `main` en `6a7081c`, cierre capFix.

## Máquina

- 13th Gen Intel Core i7-1365U
- 32 GB instalados (31,4 GB utilizables)
- Windows 11 Pro 25H2, build 26200

Durante las mediciones finales seguían abiertos VS Code, WSL (~1,9 GB), Edge y
McAfee. Chrome y Outlook estaban cerrados.

## GEMPACK (`gtapv7.exe`, RunGTAP 3.75)

```
uv run python scripts\gtap\run_gempack_matrix.py --datasets gtap7_20x41 --rordelta 0 --gtapv7 C:\runGTAP375\gtapv7.exe
```

Líneas textuales de `solve_console_tm10_s8-16-32_capfix.txt`. El log trae una
sola línea "Total CPU":

| corrida | Total CPU | Total elapsed | 32 pasos: CPU / elapsed |
|---|---|---|---|
| 2.ª, Chrome y Outlook cerrados | `[Total CPU is 507.45 seconds.]` | `Total elapsed time is:  4 minutes, 19 seconds.` | 265.08 s / 2 min 4 s |
| 1.ª, Chrome y Outlook abiertos | `[Total CPU is 548.27 seconds.]` | `Total elapsed time is:  5 minutes, 14 seconds.` | 289.50 s / 2 min 44 s |

Residuo máximo: 1.0868212E-07 en las dos. Los `.har` del fixture
(`tests/fixtures/gtap7_gempack/*_gtap7_20x41_tm10_s8-16-32_capfix.har`) salen
iguales byte a byte salvo la fecha de creación que escribe GEMPACK. No se
commitearon.

## equilibria (Newton-TR + MUMPS)

```
python scripts\gtap\phase_times.py gtap7_10x7                          # exit 0, code 1, sin_enganchar []
python scripts\gtap\phase_times.py gtap7_20x41 --out fases_win.json    # exit 0, code 1
```

Total **2096,6 s (~35 min)**, unas cuatro veces los ~8 min estimados. El
reparto por fase está en [fases_win.json](fases_win.json).

### Entorno `eq` (conda-forge, Miniforge)

| paquete | versión |
|---|---|
| python | 3.13.16 |
| pyomo | 6.9.5 (igual que la Mac; `pip install -e .` había traído 6.10.1) |
| pymumps / mumps-seq | 0.4.0 / 5.8.2 |
| numpy / scipy | 2.5.4 / 1.18.1 |
| equilibria | 0.5.1 (editable) |

### Requisitos en Windows que el protocolo no traía

1. **pytest**: `phase_times.py` importa
   `tests/templates/gtap/test_gtap7_gempack_parity.py`, que hace `import pytest`.
   Hay que instalar con `pip install -e ".[dev]"`.
2. **pynumero_ASL**: `pynumero_libraries` de conda-forge solo existe para
   Python ≤ 3.8, y `pyomo download-extensions` ya no lo trae. Se tomó de IDAES,
   en un env aparte (`idaes-pse` no se instala en `eq`):
   ```
   conda create -n idaesbin -c conda-forge python=3.11
   pip install idaes-pse && idaes get-extensions      # IDAES ext v3.4.2 windows-x86_64
   copy %LOCALAPPDATA%\idaes\bin\lib*.dll %LOCALAPPDATA%\Pyomo\lib\
   ```
   El archivo se llama `libpynumero_ASL.dll`. Hay que copiar también las demás
   `lib*.dll`: sin ellas `AmplInterface.available()` da `True`, pero
   `ctypes.CDLL` falla con "Could not find module ... (or one of its
   dependencies)".
3. El log de GEMPACK se llama `solve_console_tm10_s8-16-32_capfix.txt`, no
   `solve_console_capfix.txt`.
