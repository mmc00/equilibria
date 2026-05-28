# GTAP v6.2 on macOS — setup and reproduction guide

End-to-end plan for running the Phase 3.36-3.38 GTAP v6.2 work on
macOS, including the path forward for the `gtap6_20x41` dataset that
the Windows-IDAES IPOPT/MUMPS stack cannot solve (32-bit integer
workspace overflow).

The Phase 3.36-3.38 model fixes themselves (`sav` as residual Var,
`eq_qxs` diagonal redundancy detection, bake tolerance) are 100%
portable — they live in pure Pyomo code under
`src/equilibria/templates/gtap_v62/` and `scripts/gtap_v62/`. Only
the solver infrastructure differs between Windows and macOS.

## 1. Initial setup (Apple Silicon and Intel — both supported)

### 1a. Toolchain

```bash
# Xcode CLI tools (if not already installed)
xcode-select --install

# Homebrew (skip if installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Fortran + parallel build tools (needed if building MUMPS / SuiteSparse from source)
brew install gcc       # provides gfortran
brew install cmake make ninja pkg-config
```

### 1b. equilibria itself

```bash
git clone https://github.com/mmc00/equilibria.git
cd equilibria

# Install Python via uv (matches Windows workflow)
uv sync                                # core
uv sync --group v62-explore            # numerical stack for v6.2 work
```

This installs:

* equilibria + core deps (numpy, pyomo, pandas, …)
* `cmake`, `idaes-pse`, `scipy-openblas32`, `cvxopt` for the solver
  investigation flow

## 2. Get IPOPT working

The Phase 3.38 IPOPT NLP parity tests (`_test_gtap6_all_nlp.py`,
`_test_3x3_ipopt_nlp.py`, etc.) need an `ipopt` binary on PATH or
at `.idaes-bin/ipopt`.

### 2a. Easy path — IDAES extensions (same as Windows)

```bash
uv run idaes get-extensions --to .idaes-bin
```

On macOS this downloads precompiled IPOPT 3.13.x + HSL libraries
under `.idaes-bin/`. Sufficient for gtap6_3x3 through 15x10.

### 2b. Pro path — conda-forge IPOPT with int64 MUMPS

For gtap6_20x41 the IDAES IPOPT crashes inside MUMPS at numeric
factorization (32-bit integer overflow on the ~290 K-variable
problem). conda-forge offers an IPOPT build linked against int64
MUMPS that should clear this limit:

```bash
# Use micromamba or miniforge (no licensing issues on macOS)
brew install --cask miniforge
mamba create -n equilibria-solver python=3.12
mamba activate equilibria-solver
mamba install -c conda-forge ipopt=3.14 mumps-seq metis

# Get the absolute path to the conda IPOPT
which ipopt    # e.g. /opt/homebrew/Caskroom/miniforge/base/envs/equilibria-solver/bin/ipopt
```

Test scripts that take an `executable=` keyword (the `_test_*.py`
files in `scripts/gtap_v62/`) need to be pointed at this path.
Recommended: set an env var and update the test scripts to read it:

```python
ipopt_executable = os.environ.get(
    "EQUILIBRIA_IPOPT",
    str(ROOT / ".idaes-bin" / "ipopt"),  # default falls back to IDAES bundle
)
solver = SolverFactory("ipopt", executable=ipopt_executable)
```

Then:

```bash
EQUILIBRIA_IPOPT=$(which ipopt) uv run python scripts/gtap_v62/_test_20x41_ipopt_nlp.py
```

### 2c. Hardcore path — HSL MA57 / MA86 (recommended for serious work)

HSL (the original collection — Harwell Subroutine Library) provides
MA57 (symmetric indefinite sparse LU) and MA86 (multi-frontal symmetric)
which outperform MUMPS on CGE-shape Jacobians.

Free academic license available at <https://www.hsl.rl.ac.uk/ipopt/>.

```bash
# After obtaining the HSL distribution (coinhsl-VERSION.tar.gz):
brew install coin-or-tools/coinor/coinbrew
coinbrew fetch Ipopt --no-prompt
# Drop the HSL tarball into ThirdParty/HSL/
coinbrew build Ipopt --with-hsl --no-prompt

# Resulting ipopt + libipopt in dist/bin/ipopt
```

Configure your test scripts to use this build via the
`linear_solver=ma57` IPOPT option.

## 3. Reproducing the Phase 3.38 parity sweep

```bash
PYTHONIOENCODING=utf-8 uv run python scripts/gtap_v62/_test_gtap6_all_nlp.py
```

Expected output on macOS (same as the Windows run committed in
`docs/findings/gtap_v62_phase338_sav_var_budget_identity.md`):

```
gtap6_3x3     VIWS = +62.40%   gap = +0.04 pp   (0.07% rel)
gtap6_5x5     VIWS = +64.59%   gap = +0.04 pp   (0.06% rel)
gtap6_10x7    VIWS = +64.43%   gap = +0.04 pp   (0.07% rel)
gtap6_15x10   VIWS = +66.78%   gap = +0.42 pp   (0.64% rel)
```

`walras` should be < 2e-8 at the shocked solve in every case
(Phase 3.38 budget-identity fix verification).

## 4. Tackling gtap6_20x41 on Mac

### 4a. Diagnosis to confirm the int32 issue

```bash
# Use the IDAES IPOPT first to reproduce the Windows failure
EQUILIBRIA_IPOPT=$(pwd)/.idaes-bin/ipopt \
  uv run python scripts/gtap_v62/_test_20x41_ipopt_nlp.py
```

Expected: same `Problem with integer stack size 1 1 6` / segfault
that Windows showed. Confirms infrastructure parity (the bug is in
the linear solver, not OS-specific).

### 4b. The conda-forge fix

```bash
# Within the conda env from 2b:
EQUILIBRIA_IPOPT=$(which ipopt) \
  uv run python scripts/gtap_v62/_test_20x41_ipopt_nlp.py
```

The conda-forge IPOPT-3.14 + MUMPS-int64 combination should clear
the integer stack overflow. Expected:

* MUMPS symbolic phase: ~30-90 seconds
* MUMPS first numeric factorization: ~3-10 minutes
* IPOPT iterations: 30-100 Newton steps
* Total time: 30 minutes - 2 hours on a modern Mac (M2/M3 or i9)
* Result: VIWS within ~1-3% of GEMPACK's +51.43% reference for
  `tms[FoodProd, USA, EU_28]`
  
  (Phase 3.25 doc reported "IPOPT no-conv." for 20x41 — that pre-
  dated Phase 3.38, and was probably also limited by the MUMPS
  int32 issue. With Phase 3.38 + int64 MUMPS, convergence is
  expected.)

### 4c. If conda-forge IPOPT still struggles

* Switch to HSL MA86 — better than MUMPS for CGE Jacobians at this
  size. `linear_solver=ma86 ma86_print_level=2` in IPOPT options.
* Increase `mumps_mem_percent` (option name varies by IPOPT version
  — IPOPT 3.14 accepts `mumps_mem_percent 5000` for 50× the default
  workspace).
* Try the PEDRO Python prototype from `docs/PEDRO.md` — matrix-free
  Newton-Krylov sidesteps the sparse-LU scaling wall entirely.

## 5. PATH on macOS (optional, only for MCP mode)

PATH is closed-source. The IDAES bundle includes a copy of `path52`
but the licensing is restrictive. For MCP work on macOS:

* Obtain a PATH license + library set from <http://pages.cs.wisc.edu/~ferris/path.html>
* Set `PATH_CAPI_LIBPATH=/path/to/libpath52.dylib`
* Set `PATH_LICENSE_STRING=<your license>`
* The `scripts/gtap_v62/_test_15x10_shocked.py` + related scripts
  will then run identically to Windows (Phase 3.36 bake_tolerance,
  Phase 3.37 qxs diagonal fix all apply).

PATH MCP mode on gtap6_15x10 still hits a matching lottery (Phase
3.37 finding); IPOPT NLP from Section 3 is the production path.

## 6. What is NOT portable yet

* The locally-built UMFPACK + BLAS shim DLL (`vendor/umfpack/`) is
  Windows-only. On macOS, install via `brew install suite-sparse`
  instead. PATH should accept `factorization_method umfpack` if
  the symbols are visible, but our investigation found PATH falls
  back to LUSOL silently when UMFPACK is misconfigured (see Phase
  3.37 notes in the diagnostic scripts).
* The Output_SetInterface hook in `path-capi-python` (commit
  `7c6e1ca` in that repo) works on macOS but requires that the PATH
  shared library exports `Output_SetInterface` (verified for the
  GAMS 53 Windows build; needs verification per macOS build).

## 7. Recommended workflow for the daily case

If you primarily want IPOPT NLP parity on gtap6_3x3 through 15x10:

```bash
# One-time setup
uv sync --group v62-explore
uv run idaes get-extensions --to .idaes-bin

# Sweep parity
PYTHONIOENCODING=utf-8 uv run python scripts/gtap_v62/_test_gtap6_all_nlp.py

# GEMPACK oracle (verify reference values — requires GEMPACK install)
uv run python scripts/gtap_v62/run_gempack_generic.py \
  --workdir runs/gempack_3x3_food_usa_eu \
  --dataset-dir datasets/gtap6_3x3 \
  --shock-comm Food --shock-src USA --shock-dst EU_28
```

For the gtap6_20x41 case, switch to conda-forge IPOPT (Section 2b)
and follow Section 4.

## 8. Open items

* Verify on Apple Silicon that the `MA27` and `MA57` HSL solvers
  perform identically to Linux/Intel (some HSL releases have AVX2
  optimizations that may degrade on ARM64).
* Benchmark conda-forge IPOPT 3.14 + int64 MUMPS against IDAES
  IPOPT 3.13.2 on gtap6_15x10 to confirm no parity regression.
* Document the actual gtap6_20x41 VIWS reference value once we
  have a converged solve (the +51.43% in Phase 3.25 is GEMPACK
  Gragg-multi, but no Python solver had matched it yet).
