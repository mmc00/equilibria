# GTAP GAMS Parity Plan (2026-04-07)

## Objective
Make the Python GTAP model in equilibria an exact mirror of the GAMS reference and validate baseline and tariff shock results against GAMS.

## Reference GAMS Model and Submission Path
- GAMS reference model: [src/equilibria/templates/reference/gtap/scripts/model.gms](../../src/equilibria/templates/reference/gtap/scripts/model.gms)
- GAMS submission script (NEOS): [scripts/gtap/submit_gams_to_neos.py](../../scripts/gtap/submit_gams_to_neos.py)
- Reference data (GTAP 9x10): [src/equilibria/templates/reference/gtap/data](../../src/equilibria/templates/reference/gtap/data)

## Scope
- Baseline: calibrate and solve Python and GAMS models on the same GTAP dataset.
- Tariff shock: apply the same tariff change in both models and compare results.

## Plan
1) **Baseline parity inventory**
   - Inventory active constraints vs. variables in Python.
   - Record skipped equations and identify variables without governing equations.
   - Verify sets and parameter loading are identical to GAMS inputs (9x10 data and elasticities).

2) **Close equation/variable gaps**
   - Implement missing GAMS equations in Python (ensure one-to-one mapping).
   - Eliminate redundant variables or add equations to close DOF.
   - Confirm all bilateral trade, margins, and price linkage equations are active.

3) **Align closure and numeraire**
   - Confirm closure selection and numeraire match GAMS.
   - Ensure the same exogenous/endogenous assignments and market clearing equations.

4) **Run PATH-CAPI and compare (Baseline)**
   - Solve Python with PATH-CAPI using the same dataset.
   - Run GAMS on NEOS and export baseline results.
   - Compare variable levels and key aggregates (GDP, prices, trade, factor returns).

5) **Run PATH-CAPI and compare (Tariff shock)**
   - Apply the same tariff shock in both models.
   - Run Python PATH-CAPI and GAMS on NEOS.
   - Compare deltas from baseline and validate tolerance.

## Task Mapping
- **Baseline parity inventory**
   - Build a table of Python constraints vs. GAMS equations by block (production, trade, margins, closure).
   - List skipped constraints and count variables without equations.
   - Freeze the baseline dataset: 9x10 sets + basedata + default elasticities.

- **Close equation/variable gaps**
   - Implement missing blocks: utility, government, investment/savings, tax totals, transport margins, FOB/CIF links.
   - Remove or constrain unused variables (e.g., inactive routes or sectors).
   - Recompute DOF and confirm square system.

- **Align closure and numeraire**
   - Match exogenous/endogenous lists to GAMS closure.
   - Ensure walras check and price index equations are identical.

- **Baseline run comparison**
   - Run Python PATH-CAPI; export solution snapshot.
   - Submit GAMS model to NEOS; export GDX results.
   - Compare levels with tolerance and report mismatches.

- **Tariff shock comparison**
   - Apply identical tariff shock in Python and GAMS.
   - Compare deltas vs. baseline and document deviations.

## Command Checklist
### Baseline parity inventory
- Python DOF and constraint audit:
   - `uv run python notes/tmp/count_dof.py`
- Constraint residual scan (if available):
   - `uv run python notes/tmp/check_residuals.py`

### Close equation/variable gaps
- Re-run DOF after changes:
   - `uv run python notes/tmp/count_dof.py`

### Align closure and numeraire
- Confirm closure config (inspect contract):
   - `uv run python -c "from equilibria.templates.gtap import build_gtap_contract; print(build_gtap_contract('gtap_standard7_9x10').closure)"`

### Baseline run comparison
- Python PATH-CAPI solve:
   - `python scripts/gtap/run_gtap.py solve --gdx-file src/equilibria/templates/reference/gtap/data/9x10Dat.gdx --solver path-capi --path-capi-mode nonlinear --tee`
- GAMS baseline via NEOS:
   - `python scripts/gtap/submit_gams_to_neos.py --gms-file src/equilibria/templates/reference/gtap/scripts/model.gms --solver PATH`

### Tariff shock comparison
- Python tariff shock (example):
   - `python scripts/gtap/run_tariff_shock_python.py --gdx-file src/equilibria/templates/reference/gtap/data/9x10Dat.gdx --solver path-capi --path-capi-mode nonlinear --shock-file notes/tmp/tariff_shock.json`
- GAMS tariff shock via NEOS:
   - `python scripts/gtap/submit_gams_to_neos.py --gms-file src/equilibria/templates/reference/gtap/scripts/model.gms --solver PATH --shock-file notes/tmp/tariff_shock.gms`

### Example shock files
- Python shock file: [notes/tmp/tariff_shock.json](../../notes/tmp/tariff_shock.json)
- GAMS shock file: [notes/tmp/tariff_shock.gms](../../notes/tmp/tariff_shock.gms)

## Success Criteria
- Python and GAMS baseline levels match within tolerance.
- Python and GAMS tariff-shock deltas match within tolerance.
- No DOF gap (square system) in Python.

## Notes
- Any non-matching values must be fixed in equations or initialization, not by changing data.
