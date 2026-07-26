# F3 — GTAP to Symbolic Blocks (with framework repair) — Design Spec

**Date:** 2026-07-25 (revised same day after an Explore pass on the symbolic framework)
**Status:** Design (approved for planning)
**Roadmap:** F3 (modularization to blocks), prerequisite of F3.5 (no-check variant), F5 (GEMPACK), F7 (GTAP6), F9 (PEP blocks).

## Goal

Move the GTAP model from the 8076-line imperative monolith
(`gtap_model_equations.py`) onto the repo's **symbolic Block framework**
(`equilibria.blocks` / `SymbolicEquation`), so the model is composable and
multi-backend, with the success gate being **GTAP 3x3 solving through the symbolic
framework and matching GAMS** (NLP + MCP). This is the ambitious path — chosen over
the pragmatic "copy the PEP imperative-Pyomo pattern (C)" — because the user
prioritizes the clean multi-backend architecture for the long run.

## Critical background — the symbolic framework is a never-solved prototype

An Explore pass (2026-07-25) over `core/symbolic_equations.py`,
`blocks/base.py`, `backends/pyomo_backend.py`, `model.py`, and the only user
(`templates/simple_open.py`) found the framework (option "B") is **half-built and
has never solved a Pyomo model**:

1. **Two incompatible `build_expression` APIs.** The ABC declares
   `build_expression(set_manager, variables, parameters, indices) -> residual
   callable`; the real blocks override it as `build_expression(pyomo_model,
   indices) -> Pyomo expr`. The `ResidualEquation` + `var/param/power` DSL is
   **dead code** — nothing consumes it.
2. **The Pyomo bridge silently drops equations.** `pyomo_backend._build_constraints`
   wraps each build in `except (ValueError, KeyError, AttributeError, TypeError):
   … continue` and, if zero constraints survive, injects
   `dummy_constraint = Constraint(expr=1==1)`. **A model that fails to translate
   still "builds" as a trivial feasible problem** — fatal for GAMS parity (a
   dropped equation is invisible).
3. **No solve, ever.** No test calls `.solve()` and asserts a feasible/optimal
   status for a symbolic-block model. `simple_open`'s "solution" is analytic
   benchmark residuals, not a Pyomo solve; its block equations are **log-linear
   Cobb-Douglas approximations valid only at σ=1**, so they would not match GAMS
   even if solved; and it omits its own trade blocks to avoid name conflicts.

**Consequence:** the multi-backend / introspection advantages of B are, today,
aspirational — the DSL that would provide them is dead code, and the solve path
does not work. The user accepted this and chose to **repair the framework fully**
and prove it on real GTAP, rather than adopt the working imperative pattern.

## Design decisions (all user-approved 2026-07-25)

### Target: the symbolic Block framework (B), repaired
GTAP blocks become `Block` subclasses producing symbolic equations that the
(repaired) Pyomo bridge emits to real `Constraint`s, solved by PATH/IPOPT.

### Success gate (north star): GTAP 3x3 via B matches GAMS
The framework is "repaired" when **gtap7_3x3 solves through the symbolic framework
and matches GAMS** on both NLP-vs-NLP and MCP-vs-MCP, under the 4-layer gate below.
Not a toy model, not analytic simple_open — the real parity target.

### Approach: fused, repair guided by real GTAP
Do NOT repair B in the abstract against hypothetical cases. Bring real GTAP blocks
onto B and fix the framework by exactly what that path demands (bridge, DSL, solve,
true CES/CET equations), until 3x3-via-B is green. Lower risk of fixing the wrong
things.

### Repair scope (complete)
1. **Pyomo bridge** (`backends/pyomo_backend.py`, 420 lines): remove the
   equation-swallowing `except` and the `dummy_constraint` stub — translation
   failures must raise loudly. This is what makes parity trustworthy.
2. **Symbolic API** (`core/symbolic_equations.py`, 241 lines): resolve the two
   `build_expression` contracts — either revive the DSL as the single contract or
   retire the dead `ResidualEquation`/combinator code and standardize on the
   `(pyomo_model, indices) -> expr` form the real blocks use.
3. **The GTAP blocks on B** — the 7 extraction units (see order below) as `Block`
   subclasses with **true CES/CET equations** (not log-linear), variables
   distributed to their owning block. The existing ~4500 lines of generic blocks
   (production/trade/demand/institutions/equilibrium) are saned/replaced as needed.
4. **A real solve+parity test suite** — the framework's first tests that actually
   `.solve()` and assert feasible status + GAMS parity (none exist today).

### Order of work: all blocks at once, then solve (user's explicit choice)
Migrate all 7 GTAP blocks onto B (fixing bridge/DSL along the way), THEN attempt the
first 3x3 solve. The user accepted the higher bisect risk of this over an
incremental one-block-at-a-time cutover.

**Mandatory risk mitigation (because B has never solved anything):** before the
first solve, the plan must stand up **diagnostic tooling** so a failed solve is
debugged directed, not blind:
- per-equation residual report at a seeded point (which equation is violated),
- per-block equation-form diff vs the monolith (`diff_equation_form.py`, tool 5),
- per-variable domain+bounds check vs the monolith,
- the bridge must, by then, raise on any un-translated equation (no silent drop).

### Dependency structure (from the earlier dependency map)
~10 banner blocks collapse to **7 units** due to 5 circular dependencies (extracted
as bundles): TRADE_CET (leaf) · PRODUCTION+SUPPLY · FACTOR · ARMINGTON+BILATERAL ·
DEMAND+UTILITY · INCOME · CLOSURE (last). All 105 vars currently live in one central
block (3509–5106); they get distributed to their owning block, and the domain/bounds
gate guards against spurious-corner regressions (xw/xet → Reals).

## Per-solve fidelity gate (4 layers, cheap → expensive; ALL green to accept)
1. **Equation-form diff** — expanded Pyomo expr of each B-block == the monolith's,
   cell-by-cell. Catches form drift before solving.
2. **Var domain+bounds diff** — each var keeps EXACTLY its domain/bounds.
3. **Canary (gtap7_3x3)** — solves first; seconds; no full sweep if it breaks.
4. **Full NLP+MCP sweep** (`run_parity_gates.py`) vs native GAMS; refreshes stamp.

## Architecture
- `src/equilibria/blocks/gtap/` — the GTAP blocks as `Block` subclasses.
- A composer assembles them via `BlockRegistry` into a model the (repaired) bridge
  emits to Pyomo, solved by the existing PATH/IPOPT path.
- The monolith `gtap_model_equations.py` stays intact as the **parity oracle**
  throughout F3 (B is compared against it AND against GAMS). It is not removed until
  3x3-via-B is fully green.

## Acceptance gates (parity floors)
- **F3a (framework repaired):** the bridge raises on untranslated equations (no
  silent drop / no dummy_constraint); a real test solves a symbolic-block model and
  asserts feasible status.
- **F3 done (north star):** gtap7_3x3 solves via B and matches GAMS on NLP + MCP
  under all 4 gate layers; then the remaining datasets (3x4…15x10) pass the full
  sweep via B; the coverage matrix is unchanged (0-diff) vs the monolith.
- **Non-regression:** no intermediate commit drops any matrix cell below its current
  measured value; the monolith remains the oracle until the end.

## Non-goals / YAGNI
- No F3.5 work here (calibrated-base variant is a later phase; F3 only ensures the
  block form exposes the calibration hooks — `CalibrationMixin` is the one sound
  part of B already).
- No new solver/closure logic — the closure block is translated last, as-is.
- No GTAP6/PEP block work (F7/F9 reuse these blocks later).
- No behavior change in the model's economics — any result change is a bug.

## Risks (and mitigations)
- **B never solved → first 3x3 solve likely fails** (user accepted the all-at-once
  bisect risk). Mitigation: diagnostic tooling stood up BEFORE the first solve;
  bridge raises loudly; monolith as oracle for form/residual diffs.
- **Silent-drop bridge** hides missing equations → **removed first** (repair item 1).
- **Log-linear block equations** would pass a solve but fail GAMS parity → replaced
  with true CES/CET; the form-diff gate catches any residual approximation.
- **Var domain/bounds drift** → spurious corners → domain+bounds gate per var.
- **Circular-dependency bundles** can't be split → migrated as merged modules.
- **Scope is large** (~5000 lines saned before/while GTAP lands) → the fused,
  GTAP-guided approach avoids repairing framework code GTAP doesn't exercise.
