---
name: gams-to-equilibria
description: Convert static CGE models from GAMS into Equilibria template implementations with equation-level parity checks, calibration alignment, and solver initialization consistency. Use when migrating or mirroring GAMS SAM/VAL_PAR/calibration/equation blocks into `src/equilibria/templates/*`, diagnosing residual mismatches against GAMS, or building reproducible parity workflows (including pre-solve comparison and IPOPT-ready initialization).
---

# GAMS to Equilibria

Implement and validate a migration from GAMS to Equilibria with strict parity and reproducible checks.

## Use this workflow

1. Identify source model and baseline assets.
2. Mirror GAMS sets, symbols, and orientations in Equilibria dataclasses and equations.
3. Port calibration blocks in GAMS order.
4. Validate parity before running solver iterations.
5. Fix residual hotspots by block, not by ad-hoc tweaks.

## Input contract

Require these inputs before porting:
- GAMS model file (`.gms`) with equations and initialization.
- Baseline data (`SAM*.gdx` and, if present, `VAL_PAR.gdx` / `.xlsx`).
- Target Equilibria template paths under `src/equilibria/templates/`.

Prefer the `pep2` reference structure when available:
- `src/equilibria/templates/reference/pep2/data`
- `src/equilibria/templates/reference/pep2/scripts`

## Mapping rules

Apply these invariants:
- Keep original set semantics (`I,J,H,F,AG,AGNG,AGD,K,L`) exactly.
- Preserve transfer orientation exactly as in GAMS.
: For PEP, use `TR(recipient, source)` conventions consistently.
- Preserve conditional equation domains (`$condition`) via explicit guards.
- Preserve quantity/nominal conversions.
: If GAMS divides by `PCO`, replicate in the same calibration stage.
- Keep equation IDs (`EQxx`) stable for debugging and parity reports.

## Porting sequence

1. Data loading and symbol normalization
- Implement deterministic loaders for SAM and VAL_PAR.
- Normalize aliases and naming (`val*`, `*_O`, scalar vs indexed records).

2. Calibration blocks
- Port in the same logical order as GAMS.
- Recompute derived coefficients only after required upstream variables exist.
- Store all calibrated levels needed by equations and solver init.

3. Equation blocks
- Port equations by family (`EQ1..EQ97` style).
- Keep price/tax/margin terms in the original order.
- Handle corner cases (`beta` in `{0,1}`, zero denominators) without changing equation count.

4. Initialization
- Build strict benchmark initialization from calibrated state.
- Build optional equation-consistent initialization for solver stability.
- Never overwrite calibrated identities unintentionally during init.

5. Solver integration
- Verify residual vector length is fixed across function evaluations.
- Apply stable residual ordering for finite-difference optimizers.
- Add early exit when benchmark already satisfies tolerance.

## Residual debugging protocol

When residuals are non-zero at calibrated benchmark:

1. Run benchmark residual audit first.
2. Sort by absolute residual.
3. Inspect top equations in GAMS and Equilibria side by side.
4. Patch only the smallest block that explains the mismatch.
5. Re-run full audit after each patch.

Prioritize these common fault zones:
- LES and tax function intercept/slope consistency (`EQ52` family, `EQ35/36`).
- Production and commodity tax totals (`EQ29/EQ40` families).
- ROW transfer orientation and external closure (`EQ44/45/46`).
- Walras closure and savings-investment (`EQ87`).
- GDP identity synchronization (`EQ90..EQ93`).

## Project commands (Equilibria)

Use these commands in this project:

```bash
uv run python scripts/qa/verify_calibration.py
uv run python scripts/parity/verify_pep2_full_parity.py --tol 1e-9
uv run python scripts/parity/verify_pep2_full_parity.py --tol 1e-9 --presolve-gdx src/equilibria/templates/reference/pep2/scripts/PreSolveLevels.gdx
uv run python scripts/cli/run_solver.py --sam-file src/equilibria/templates/reference/pep2/data/SAM-V2_0.gdx --method ipopt --init-mode equation_consistent --tolerance 1e-8
```

## Definition of done

Declare migration done only when all are true:
- Full parity report has zero mismatches for compared symbols.
- `verify_calibration.py` reports near-zero benchmark residuals.
- Solver run is stable and reproducible.
- AGENTS.md status is updated with exact commands and metrics.

## Safety checks

- Do not change economic meaning to force convergence.
- Do not hide residuals by dropping equation keys dynamically.
- Do not mix benchmark solved levels with pre-solve levels without explicit labeling.
