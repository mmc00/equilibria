# 20x41 — como se obtuvo la fila de equilibria

**El metodo NO es PATH**, aunque la nota de `RESULTADOS.md` (generada por
`collect.py`) lo diga. Con PATH el 20x41 no cierra: dos corridas warm 1 rep dieron
`shock code=5` (tope de 1 h de PATH), residual 1.914, 112-121 min en total. Liberar
7 GB de swap no cambio nada.

La fila sale del camino de produccion: **Newton-TR + MUMPS + continuacion del shock**,
la misma configuracion que `scripts/gtap/bench_symbolic_reuse.py`:

```
EQUILIBRIA_GTAP_SOLVE_NLP=1
EQUILIBRIA_GTAP_SOLVER=scipy_newton_tr
EQUILIBRIA_GTAP_TR_LINSOLVE=mumps
EQUILIBRIA_GTAP_NLP_NO_JACSCALE=1
EQUILIBRIA_GTAP_TR_GATE=1
EQUILIBRIA_GTAP_TR_FTOL=1e-7
EQUILIBRIA_GTAP_SCIPY_MAXITER=300
EQUILIBRIA_GTAP_TR_DELTA0=10.0
EQUILIBRIA_GTAP_SHOCK_CONTINUATION=0.125,0.25,0.375,0.5,0.625,0.75,0.875,1.0
EQUILIBRIA_GTAP_GMIN=1e-9
EQUILIBRIA_GTAP_TR_RELTOL=1e-6
GTAP_GATES_SKIP=1
EQUILIBRIA_SEED_CACHE_DISABLE=1
```

Resultado: `codes {base:1, check:1, shock:1}`, residuales check 5.7e-09 / shock 4.9e-07,
build 29.5 min + solve 30.6 min. Sin la continuacion (shock en un salto) el shock se
estanca en ||F||_inf = 0.179 tras 295 iteraciones.

## Requisitos en Windows (no estan en el README)

1. **MUMPS por conda.** `pymumps` y `python-mumps` de PyPI son solo codigo fuente y no
   compilan sin un compilador C. Se uso Miniforge + `conda install -c conda-forge
   pymumps mumps-seq` (Python 3.12) y equilibria instalado en ese env.
2. **Orden de DLLs.** `<env>\Library\bin` tiene que ir ANTES de
   `%LOCALAPPDATA%\idaes\bin` en el PATH. Al reves, las BLAS de IDAES tapan las de
   conda y la factorizacion numerica de MUMPS muere con 0xC06D007F (procedure not
   found).
3. **`%LOCALAPPDATA%\idaes\bin` en el PATH** para `libpynumero_ASL.dll`:
   `pyomo download-extensions` no trae el ASL en Windows.
4. Requiere `10bdc60` (contador `_SYMBOLIC_FACT_COUNT`); sin el, MUMPS "falla" con
   KeyError y el trust-region cae a otro backend.

## Pendiente

- `equilibria (cold)` del 20x41 no se corrio (otra hora). El README del bench dice que
  es la comparacion justa contra GEMPACK.
- El "~7,0 min en la Mac" del docstring de `bench_equilibria.py` no se reproduce con
  esta configuracion; el numero documentado del mismo solver en la Mac es ~67 min.
