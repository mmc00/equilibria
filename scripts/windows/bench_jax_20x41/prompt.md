# Task (Windows laptop): GEMPACK vs equilibria JAX on the 20x41, same machine

You work on the user's Windows laptop (i7-1365U, Windows 11, RunGTAP 3.75 / GEMPACK
12.1.003), in the repo `C:\Documentos\proyectos\equilibria`. Talk to the user in short,
simple Spanish, with no em dashes. When they say "no entendí", give the real technical
detail in a table, not a simpler story.

## Why

We only have GEMPACK and equilibria timed on different machines:

| Tool | Machine | Minutes | Source |
|---|---|---|---|
| GEMPACK, AlterTax 20x41 | this laptop | 1.93 | `data/gtap7/gtap7_20x41/gempack_altertax-rungtap_tms-rate10_s8-16-32/solve_console.txt` (2026-11-09) |
| equilibria JAX, same exercise, cold cache | Windows Server VM, i7-7700HQ | 11.22 | homelab, 2026-11-25, commit 5ddde12d |
| equilibria JAX, same exercise, warm cache | same VM | 4.70 | same |
| equilibria JAX, same, MUMPS `ifx` build, cold / warm | same VM | 10.12 / 3.44 | same day |
| equilibria JAX, same, MUMPS `ifx` build, cold / warm | same VM | 10.12 / 3.44 | same day |
| equilibria JAX, warm cache | Mac M3 Pro | ~0.8 | MAR-21 hunt runs, 2026-11-23 |

The goal is one table with both tools on **this** laptop, same exercise, same session.

## Rules

- Write only under `scripts\windows\bench_jax_20x41\`. Never write to `data\`, and do not
  change any script outside this folder.
- Ask the user before every commit and before the push. Commits are in English and end
  with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Do not touch the repo's `.venv`. The JAX run uses a new conda env (step 2).
- If something fails twice the same way, stop and tell the user. Do not work around it.

## Steps

### 0. Prepare the machine

Ask the user to plug the laptop in, set the power mode to "Best performance", close the
browser, Teams and anything heavy, and wait until the machine is idle. McAfee cannot be
closed (corporate). Write down in the results whether it was on.

### 1. Get the branch

```powershell
cd C:\Documentos\proyectos\equilibria
git status            # must be clean; if not, stop and ask
git fetch origin
git switch bench/jax-windows-laptop
git log --oneline -3
```

### 2. Build the JAX env (conda-forge MUMPS + pip JAX)

Use Miniforge (install it to `C:\miniforge3` if it is not there). This is the recipe
that worked on the Windows VM. **Pick the Intel Fortran (`ifx`) build of MUMPS**: the
default `flang` build is about 4.5x slower (see the table below).

```powershell
C:\miniforge3\Scripts\conda.exe create -y -p C:\eq-jax\env -c conda-forge `
    python=3.12 "mumps-seq=5.8.2=ifx*" intel-fortran-rt pymumps mpi4py psutil
$env:PATH = "C:\eq-jax\env;C:\eq-jax\env\Library\bin;C:\eq-jax\env\Scripts;$env:PATH"
$env:CONDA_PREFIX = "C:\eq-jax\env"
$env:MKL_NUM_THREADS = "1"
python -m pip install -e ".[dev]" "jax==0.12.1" "jaxlib==0.12.1"
```

Pitfalls already met:

- The script `scripts\family\altertax_rungtap_measure.py` uses SuperLU unless
  `EQUILIBRIA_GTAP_TR_LINSOLVE=mumps` is set, and SuperLU runs out of memory on the
  20x41 (MAR-235). `time_cold_warm.py` sets it for you.
- pymumps needs `mpi4py` (conda-forge brings `msmpi`). Without it every factorization
  fails with `cannot load MPI library`.
- The `ifx` build of `mumps-seq` does not pull the Intel Fortran runtime. Without
  `intel-fortran-rt`, `import mumps` fails with `DLL load failed while importing _dmumps`.
- Why `ifx`: one real 20x41 Jacobian (n = 210,134, 1.3 M nonzeros), MUMPS alone with
  equilibria's settings (`ICNTL(7)=0`, AMD ordering on every platform, so METIS plays
  no role), on the Windows VM (i7-7700HQ):

  | MUMPS build | first factorization | refactorization |
  |---|---|---|
  | `flang` (conda-forge default), OpenBLAS or MKL | 20.3 s | 4.1 s |
  | `ifx` + MKL | 4.5 s | 1.3 s |
  | for reference: Linux Xeon E5-1680 v2, gfortran | 5.4 s | 1.7 s |
  | for reference: Mac M3 Pro, gfortran | 2.6 s | 0.7 s |

  If you have time after step 6, repeat step 5 once with the `flang` build (an env
  without `=ifx*` and without `intel-fortran-rt`) and add it as a second equilibria row.

Check the env before timing:

```powershell
python -c "import jax, jax.numpy as jnp, mumps, mpi4py, equilibria; print(jax.__version__, float(jnp.arange(3.0).sum()), equilibria.__version__)"
```

### 3. Smoke test (about 1 min)

```powershell
python scripts\windows\bench_jax_20x41\time_cold_warm.py --dataset gtap7_10x7 --warm 1
```

Both runs must end with `"overall_pass": true`. Then delete `scripts\windows\bench_jax_20x41\out\`.

### 4. GEMPACK, twice (about 2 min each)

Same command that produced the committed reference, but the output goes under this
folder, never under `data\`:

```powershell
foreach ($i in 1, 2) {
  & .venv\Scripts\python.exe data\gtap7\gempack\tools\altertax_job.py `
      --dataset gtap7_20x41 --parameters rungtap `
      --out scripts\windows\bench_jax_20x41\gempack_run$i `
      --gtapv7 C:\runGTAP375\gtapv7.exe --altparv7 C:\runGTAP375\altparv7.exe `
      --sltoht C:\GP\sltoht.exe --expect-release 12.1.003 --release altparv7.exe=12.1.003
}
```

For each run, take the line `Total elapsed time is: ...` from `solve_console.txt` and
the `duration_s` of its `source.yaml`.

### 5. equilibria JAX: one cold run and two warm runs (about 30 min)

Same env as step 2, nothing else running:

```powershell
python scripts\windows\bench_jax_20x41\time_cold_warm.py --dataset gtap7_20x41 --warm 2
```

It writes `scripts\windows\bench_jax_20x41\out\results.json` with wall minutes, peak memory
(working set), solve codes, the MUMPS factorization times, versions and the machine.
Every run must end with `"overall_pass": true`. If one does not, stop and show the log.

### 6. Write the results

Create `scripts\windows\bench_jax_20x41\results.md` with:

1. The machine (CPU, RAM, Windows version, power mode, McAfee on or off).
2. One table, in minutes: GEMPACK run 1, GEMPACK run 2, equilibria cold, warm 1, warm 2.
3. The MUMPS times (symbolic and numeric, first values of each run) next to the VM's
   with the `ifx` build (symbolic 3.5-4.0 s, numeric 1.3 s; whole run 10.12 min cold,
   3.44 min warm) and the Mac's (symbolic 2.2 s, numeric 0.9 s).
4. The versions block from `results.json`.

Commit only `results.md` and `out\results.json` (the `.gitignore` in this folder keeps
the logs, the cache and the GEMPACK outputs out). Ask the user, then push the branch.

## What to send the user

The table from step 6, in Spanish, plus one line: which tool was faster on this laptop
and by how much, with the warm run as equilibria's number.
