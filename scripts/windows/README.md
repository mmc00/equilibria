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

## Phase 3a-1: the GEMPACK matrix into `data/`

`data/gtap7/gempack/tools/regenerate_gempack.py` re-solves the linearization-study matrix (6 datasets ×
10 grid configurations in capFlex, plus `tm10_s8-16-32` in capFix: 66 jobs) and
leaves each one in `data/gtap7/<ds>/gempack_<tag>/` with its `source.yaml`
(command, GEMPACK version, machine, duration, input and output hashes). The two
scripts that produce a result (`regenerate_gempack.py`, `run_gempack_matrix.py`) live
in `data/gtap7/gempack/tools/`, and every result records the hash of that folder:
editing either script marks the results it produced as stale. It
deletes nothing: the old files in `tests/fixtures/gtap7_gempack/` stay until the
Mac compares them (phase 3b).

From the repo root, in PowerShell:

```powershell
git fetch origin
git switch data/phase-3-gempack
uv sync --frozen

# 1. the GEMPACK version gtapv7.exe prints in its banner ("GEMPACK Release 12.1.003 ...");
#    if it is wrong, the smoke test stops and says which version gtapv7 reports
$v = "12.1.003"

# 2. what would run (66 jobs), nothing is solved
uv run python data/gtap7/gempack/tools/regenerate_gempack.py --tool-version $v --dry-run

# 3. one job as a smoke test
uv run python data/gtap7/gempack/tools/regenerate_gempack.py --tool-version $v `
    --datasets gtap7_3x3 --only tm10_s8-16-32_capflex
Get-Content data\gtap7\gtap7_3x3\gempack_tm10_s8-16-32_capflex\source.yaml
#    expect: status: verified, tool: gempack, tool_version: 12.1.003,
#    run.machine with your CPU name, and sl4dump.har among the outputs

# 4. everything (hours); rerunning skips the jobs already recorded
uv run python data/gtap7/gempack/tools/regenerate_gempack.py --tool-version $v

# 5. send the results back
git add data/gtap7
git commit -m "data(gempack): regenerate the GEMPACK matrix on Windows"
git push
```

Paths default to `C:\runGTAP375\gtapv7.exe` and `C:\GP\sltoht.exe`; pass
`--gtapv7 PATH` / `--sltoht PATH` if yours differ (the driver checks both exist
before the first job; the `.cmf` points GEMPACK's auxiliary files at the same
folder). Work folders go to `runs\gempack_matrix\job_*` and are deleted after each
job. A failed job prints the end
of the GEMPACK console and leaves no folder; the final summary lists it. If
`gtap7_20x41` fails in every configuration, say so: its old fixtures came from
an older setup and phase 3b has to explain the difference.

### Repair: commit the results with their real line endings

The first push (016bec41) went through `core.autocrlf`: git stored
`run.cmf`, the two `*_console.txt` and the tools with LF, while every
`source.yaml` had hashed the CRLF bytes on disk. All 66 records fail
`check_tree()` on any other machine. `data/.gitattributes` (`* -text`) now keeps
git away from those bytes. Nothing is rerun: the files on the Windows disk are
still the ones that were hashed.

Re-add only the GEMPACK folders: other files under `data/` may sit converted on
this disk, and committing them would break their records everywhere else.

```powershell
cd C:\Documentos\proyectos\equilibria
git switch data/phase-3-gempack
git pull                      # brings data/.gitattributes; do not run GEMPACK again

# re-add the GEMPACK results and tools as they are on disk, CRLF included
git add --renormalize data/gtap7/gempack/tools (Get-ChildItem -Directory data\gtap7\*\gempack_*).FullName
git diff --cached --stat      # expect 66 x 3 text outputs and the tools; nothing else

# must print [] before committing
uv run python -c "from equilibria.sources import check_tree; print([e for e in check_tree() if 'gempack' in e])"

git commit -m "data(gempack): keep the Windows line endings the records hashed"
git push
git restore data              # the rest of data/ back to the exact repository bytes
```

## Phase 3a-2: the `corridas_paper.ps1` runs and `runs/` into `data/`

`data/gtap7/gempack/tools/regenerate_paper.py` does what `corridas_paper.ps1`
(steps 1-7) and the tracked `runs/` folders did, straight into `data/`: 203 jobs
in ten groups, each recorded by `write_source.py` like the 3a-1 matrix. Plan:
`specs/data/changes/one-data-folder/plan-phase-3a2.md`. `corridas_paper.ps1`
is superseded; it stays until phase 3b.

| group | what | jobs | tool |
|---|---|---|---|
| `book` | Burfisher's book experiments, `gtapuv7.exe` (`gtapv7.exe` for `gtapv7.exp`) | 39 | gempack |
| `book-sltoht` | steps 1 and 6a: `sltoht` on them | 39 | sltoht |
| `levels` | step 2: levels of matrix solutions, re-solved | 6 | sltoht |
| `horridge` | step 3: Horridge's SIMPLE, GEMPACK side | 3 | gemsim |
| `levels6` | step 4: six more levels, RunGTAP's 7.0 TAB patched | 12 | gemsim |
| `gragg` | step 5: Burfisher with Gragg 2-4-6, `gtapv7` and `tbl53` included | 45 | gemsim |
| `gragg-sltoht` | steps 6b and 7: `sltoht` on them | 45 | sltoht |
| `swap` | `runs/{nus333,9x10}_compare`, swap closure (nus333 tm 1-10) | 11 | gempack |
| `swap-sltoht` | levels of the two tm10 swap runs | 2 | sltoht |
| `validation` | `runs/gempack_20x41_validation`: the recorded abort | 1 | gempack |

Two releases are given, because they differ on this machine: `gtapv7.exe` and
`gtapuv7.exe` (RunGTAP) print 12.1.003; `tablo`, `gemsim` and `sltoht` in `C:\GP`
print 11.3.004. Every job checks the banner of each program it runs against them.

Before running:

- RunGTAP in `C:\runGTAP375` and GEMPACK in `C:\GP` (else `--gtapdir` / `--gp`).
- Programs run in `runs/gempack_work/` (git-ignored), never in `data/` or
  `%TEMP%`: GEMPACK writes its current folder into consoles and `.sl4` files,
  and `%TEMP%` names the Windows user.
- Every record hashes every file of `data/gtap7/gempack/tools/`: a change there
  makes every recorded result stale. Change `tools/` in one batch, then run
  everything once.

From the repo root:

```powershell
git switch data/phase-3a2-gempack
git pull
uv sync --frozen
$r = @("--gempack-release", "12.1.003", "--gp-release", "11.3.004")

# 1. what would run (203 jobs), nothing is solved
uv run python data/gtap7/gempack/tools/regenerate_paper.py @r --dry-run

# 2. one small job per group, in this order (a group that reads another refuses
#    a job whose prerequisite is not recorded yet)
uv run python data/gtap7/gempack/tools/regenerate_paper.py @r --group book --only me9c
uv run python data/gtap7/gempack/tools/regenerate_paper.py @r --group book-sltoht --only me9c
uv run python data/gtap7/gempack/tools/regenerate_paper.py @r --group levels --datasets gtap7_3x3 --only sltoht_tm10_s8-16-32_capflex
uv run python data/gtap7/gempack/tools/regenerate_paper.py @r --group horridge --only gemsim_fixcap
uv run python data/gtap7/gempack/tools/regenerate_paper.py @r --group levels6 --datasets gtap7_3x3 --only gemsim_levels6_tm10_s8-16-32_capflex
uv run python data/gtap7/gempack/tools/regenerate_paper.py @r --group gragg --only me9c
uv run python data/gtap7/gempack/tools/regenerate_paper.py @r --group gragg-sltoht --only me9c
uv run python data/gtap7/gempack/tools/regenerate_paper.py @r --group swap --datasets nus333 --only gempack_tm10_s8-16-32_dpsaveswap
uv run python data/gtap7/gempack/tools/regenerate_paper.py @r --group swap-sltoht --datasets nus333 --only sltoht_tm10_s8-16-32_dpsaveswap
uv run python data/gtap7/gempack/tools/regenerate_paper.py @r --group validation

# 3. everything (about 3 hours); rerun it until only ME4A and ME4B fail
uv run python data/gtap7/gempack/tools/regenerate_paper.py @r

# 4. commit only with a clean tree
uv run python -c "from equilibria.sources import check_tree; print(check_tree())"   # must print []
git add data
git commit -m "data(gempack): phase 3a-2 on Windows"
git push
```

What to expect:

- **ME4A and ME4B never record in `gragg`** (and so not in `gragg-sltoht`):
  GEMSIM stops with "Accuracy too low" under Gragg 2-4-6, as in October. These
  are the four failures that stay.
- **Windows sometimes does not start a program** (exit 0xC0000142,
  `STATUS_DLL_INIT_FAILED`; a GEMPACK console with no banner). `write_source.py`
  and the jobs wait 15 s and retry once, and the record's `notes` says so. If
  the retry fails too, the job leaves no folder; rerunning step 3 fills it.
- Run step 3 in a console as it is, or from Git Bash with `> file`. With
  PowerShell's `*>` redirection, Python sometimes could not write its last
  line (`OSError: [Errno 9] Bad file descriptor`, exit 120) and the job counted
  as failed.
- TBL53 records only if the updated data has the target tax rates (1.05, 1.10,
  1.02 in its nine cells): its `target%` shocks are written out as plain shocks.
- `levels6` names the variables the cross-check drops in `export_console.txt`
  (`CNTendwr`, `CNTendw` in October); the run still counts. `levels` and
  `swap-sltoht` must match the recorded `sl4dump.har` exactly, or nothing is
  recorded.
