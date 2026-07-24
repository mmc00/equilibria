# Against-GEMPACK Linearization Study — Design

**Date:** 2026-07-24
**Status:** Design (approved for planning)
**Track:** F5 (against-GEMPACK validation), equilibria-1.0 roadmap

## Goal

Close, with quantitative evidence across all five matrix datasets, the question:
**is the against-GEMPACK residual (~52% within 1pp at a +10% global bilateral tariff)
the Gragg-linearized↔levels method gap, or a model defect?**

The prior finding (`docs/findings/gempack_residual_is_linearization_2026-07-24.md`)
concluded "linearization" from the Horridge SIMPLE run (GAMS≡GEMPACK = 100% @1pp on a
small localized shock, PR #40). This study makes that conclusion **quantitative on the
real GTAP7 matrix**: it shows the match% climbing toward 100% as the shock shrinks and
as GEMPACK's solution method is refined, and it removes model-condensation (`ifSUB`) as
a confound.

## Background — what the source actually says

Read from van der Mensbrugghe, *The Standard GTAP Model in GAMS, Version 7*
(`~/Downloads/GTAP7Gams.pdf`, Table D.1; the same table is C.1 in the 2018 JGEA
edition the user photographed). Two corrections to earlier assumptions are baked into
this design:

1. **`ifSUB` = "if SUBstitution", a model-condensation switch — NOT import subsidies.**
   Table D.1: *"Setting this parameter to 1 will implement the model with substitution.
   This reduces significantly the size of the model as many variables are substituted
   out by simple linear expressions of other model variables. The variables are updated
   at the end of each simulation."* The GAMS model echoes GEMPACK's own practice of
   substituting equilibrium conditions out of the specification (doc notes 20, 24, 29:
   "The equilibrium condition is substituted out of the model specification"; U^S "is
   essentially substituted out"; government volume "replaced by the expression
   yg − pgov"). `ifSUB` changes which variables are solved explicitly vs. recovered at
   the end — it does **not** change the economics.

2. **GEMPACK has no `ifSUB` switch.** GEMPACK is a Johansen (percent-change) solver and
   is condensed by nature. The GAMS `ifSUB=1` mode is precisely the attempt to *replicate*
   GEMPACK's native condensation inside a levels solver. Therefore there is no "ifSUB=0
   GEMPACK closure" to build; the only ifSUB question is internal to our Python model.

3. **Capital-account closures are controlled by `savfFlag`** (CapFlex / CapFixShr /
   CapFix / CapSFix), not by ad-hoc swaps. Our matrix `.cmf` uses manual
   `swap dpsave(r)=del_tbalry(r)` for non-residual regions; this maps to
   `savfFlag=CapFix`. Out of scope to switch to the native flag here, but noted so the
   closure is not mistaken for something exotic.

## The four evidences

| # | Evidence | What it proves | Where it runs |
|---|---|---|---|
| 1 | **Shock-size sweep** — `tm = 10 / 3 / 1 / 0.3 / 0.1 %`, all 5 datasets | match% → 100% as shock → 0 ⟹ the residual **is** shock-size (linearization) | GEMPACK (Windows) |
| 2 | **Gragg refinement** — `Steps = 4 / 8 / 16 / 32 / 64` @ 10%, all 5 datasets | match% rises as the SAME shock is solved with a finer method ⟹ it is GEMPACK's Gragg approximation, not the model | GEMPACK (Windows) |
| 3 | **ifSUB fidelity** — Python ifSUB=1 ≡ ifSUB=0 in levels | condensation is faithful and does **not** explain the gap | **Python (mac) — gate, runs first** |
| 4 | **Welfare** — read `decomp.har` (WELVIEW), EV$ + 3-branch decomposition vs Python EV | documents *why* welfare is out of the quantity matrix (per prior finding) | comparator (any OS) |

Evidences 1 and 2 are the scientific core (two independent angles on the same claim).
Evidence 3 removes a confound *before* any Windows run. Evidence 4 is a diagnostic
report, not a floor-gate (welfare `u` is a sign-flipping second-order quantity — see
`docs/findings/gempack_welfare_not_cellwise_2026-07-23.md`).

## Datasets

All five matrix datasets: `gtap7_3x3`, `gtap7_3x4`, `gtap7_5x5`, `gtap7_10x7`,
`gtap7_15x10`. (nus333/9x10 remain out of the matrix scope, unchanged.)

## Execution order — fidelity gates first

1. **Gate mac #1 — ifSUB equivalence (evidence 3).** In Python, solve each dataset's
   shock at ifSUB=1 and ifSUB=0 and assert the post-shock **levels** agree cell-by-cell
   (tol ~1e-4 rel). If they diverge, condensation has a bug — STOP and fix before any
   Windows work. This is the hardest fidelity gate and it runs entirely on mac.
2. **Gate mac #2 — extended tooling passes local tests.** The runner (`--steps`,
   `--shock-pct`) and the comparator (welfare from `decomp.har`, per-(dataset×config)
   table) pass their unit tests against the *existing* fixtures before generating new
   ones.
3. **Windows phase.** A single `.bat` drives the grid: shock×dataset (evidence 1) +
   Gragg×dataset (evidence 2) ≈ 25 + 25 = **50 GEMPACK runs**, each producing an
   `updated.har` + `sl4` solution. The user runs it and returns the `.har` files.
4. **Consolidation.** Ingest the returned `.har` as fixtures, run the comparator, and
   generate the docs page + finding with the convergence curves.

## Architecture — two phases

**Phase A (mac, this repo):** parameterize existing infrastructure. This is extension,
not greenfield:
- `scripts/gtap/run_gempack_matrix.py` already has `--shock-pct`, `Method = Gragg`,
  `Steps = 8 16 32`, `Subintervals = 1`, and the capFix swap closure. Add a `--steps`
  flag (the Gragg-refinement axis) and confirm `--shock-pct` drives the sweep. A small
  batch generator emits one `.cmf` per (dataset × shock) and per (dataset × steps).
- `scripts/gtap/gempack_reference.py` already has `Q_TO_VAR` (15-var quantity map) and
  the pp-measure. Extend the comparator to (a) read welfare from `decomp.har`, (b) emit
  a table keyed by (dataset, config) so the sweep and Gragg curves fall out.
- New: `scripts/gtap/verify_ifsub_equivalence.py` — the mac gate #1.
- New docs page + finding.

**Phase B (Windows, user):** run the generated `.bat`, return the `.har` files.

## Metrics

- **Quantity match:** the existing measure — fraction of `Q_TO_VAR` cells within 1pp,
  in absolute percentage points (NOT relative 1% tol — GEMPACK output is %-change, so
  the comparison is |Δ(%change)| ≤ 1pp). Reported per (dataset × config).
- **Non-linearity, measured:** `|match(10%) − match(0.1%)|` per dataset — quantifies how
  much of the residual is shock-size.
- **Gragg convergence:** match% as a function of Steps at fixed 10% shock, per dataset —
  a monotone climb is the numerical-method signature.
- **Welfare:** EV$ per region and its 3-branch decomposition (allocative / terms-of-trade
  / investment-savings) from `decomp.har` next to Python's EV; reported as a diagnostic
  table with the sign-flip caveat, no floor.

## Outputs

- **Raw:** returned post-shock `.har` files → `tests/fixtures/gtap7_gempack/` (following
  the existing `updated_*` / `sl4dump_*` / `sl4levels_*` naming).
- **Docs page:** `docs/site/guide/gtap7_gempack_linearization_study.md` — per-dataset
  match%→100% curve (shock sweep), Gragg-convergence table, measured non-linearity
  column, welfare diagnostic section. Generated by a `gen_*` script mirroring
  `gen_gempack_doc.py`.
- **Finding:** consolidated `docs/findings/gempack_linearization_study_2026-07-24.md`
  that supersedes/extends the single-shock `gempack_residual_is_linearization` finding
  with the five-dataset quantitative curves.

## Acceptance gates (parity floors)

Per the roadmap convention "write acceptance gates in specs as parity floors":

- **Gate 3 (ifSUB, mac):** Python if1 ≡ if0 post-shock levels agree ≥ 99.9% of cells at
  rel 1e-4, all 5 datasets. Hard gate — blocks the Windows phase.
- **Gate 1 (shock sweep):** for every dataset, match(0.1%) > match(10%), and
  match(0.1%) ≥ 95% within 1pp. (If a small shock does NOT approach ~100%, the residual
  is NOT purely linearization — that would reopen the model question, which is the whole
  point of measuring.)
- **Gate 2 (Gragg):** for every dataset, match% is non-decreasing in Steps at fixed 10%
  shock, and match(Steps=64) ≥ match(Steps=4).
- **Welfare:** no numeric floor (diagnostic only); the page must state the sign-flip
  caveat explicitly.

## Non-goals / YAGNI

- No new capital-account closure work (savfFlag native flag) — the manual capFix swap
  stays.
- No full shadow-integrator EV parity — welfare is the lightweight `decomp.har` read
  only (per the user's decision).
- No nus333/9x10 (out of matrix scope).
- No change to the Python model equations — this study measures, it does not modify the
  model (unless gate 3 exposes a condensation bug, which would be its own fix).

## Risks

- **Gate 3 fails (ifSUB divergence):** would mean our condensation is not faithful — a
  real bug, and the study pauses to fix it (route via `equilibria-parity-debug`
  cascade: this is a "same-model, different-representation" class → drift test / closure
  diff). This is a *feature* of the design: we surface it on mac before spending Windows
  runs.
- **Windows run cost:** 50 GEMPACK solves. Mitigated by the `.bat` automating the grid
  and by the small datasets (3x3..15x10 all solve fast in GEMPACK; PATH's 1000-row cap
  is irrelevant — GEMPACK has no such limit).
- **`decomp.har` structure varies by RunGTAP version:** the welfare reader must be
  defensive (locate the EV header by name, not position), mirroring the existing
  `read_har` name-based access.
