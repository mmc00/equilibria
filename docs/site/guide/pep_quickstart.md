# PEP — from zero to a solved shock

The PEP (Politique Économique et Pauvreté) template implements the
PEP-1-1 v2.1 single-country CGE model. Its engine is `pep_pyomo`, a Pyomo
port that reproduces the GAMS reference cell by cell (see the
{doc}`PEP coverage matrix <pep_coverage_matrix>`).

This guide covers:

1. Calibrating from the reference SAM.
2. Solving the baseline.
3. Running the reference shock (SIM1).
4. Checking the result against GAMS.

## Prerequisites

```bash
pip install -e ".[pyomo,ipopt,excel]"
```

The baseline solves as an NLP with IPOPT. Shocks solve as an MCP with PATH
through `path-capi-python`: install it in the same environment or point
`EQUILIBRIA_PATH_CAPI_SRC` at its `src` directory, as described in
{doc}`path_capi`.

```{warning}
`load_bundled("pep")` does not work today: the Excel SAM it reads uses other
sector labels than the model, and the calibration comes out all zeros
(measured on 2026-10-28). Use the reference SAM as below.
```

## Step 1 — Calibrate

The calibrator reads the SAM and the elasticities (`VAL_PAR.xlsx`) and
returns a `PEPModelState` with every calibrated parameter and benchmark level.
The reference data ships inside the package:

```python
from pathlib import Path

import equilibria.templates
from equilibria.templates.pep_calibration_unified import PEPModelCalibrator

data = Path(equilibria.templates.__file__).parent / "reference" / "pep2" / "data"
state = PEPModelCalibrator(
    sam_file=data / "SAM-V2_0.gdx", val_par_file=data / "VAL_PAR.xlsx"
).calibrate()

print("Sectors:   ", state.sets["J"])
print("Households:", state.sets["H"])
# Sectors:    ['agr', 'ind', 'ser', 'adm']
# Households: ['hrp', 'hup', 'hrr', 'hur']
```

To use your own SAM, pass its paths to `PEPModelCalibrator` the same way.

## Step 2 — Solve the baseline

`build_pep_model` builds the Pyomo model from the calibrated state;
`solve_pep` solves it. `form="nlp"` is the GAMS CNS form, solved with IPOPT.

```python
from pyomo.environ import value

from equilibria.templates.pep_pyomo.pep_pyomo_equations import build_pep_model
from equilibria.templates.pep_pyomo.pep_pyomo_solver import solve_pep

m = build_pep_model(state, variant="base", form="nlp")
res = solve_pep(m)
print(res.code, res.message)
print("GDP_BP", round(value(m.GDP_BP), 1))
# 1 feasible-at-seed (no re-solve)
# GDP_BP 46707.0
```

The benchmark reproduces the SAM, so the calibrated point already solves the
system and `solve_pep` stops there. `res.code == 1` means solved;
`res.max_residual` is the largest equation residual.

## Step 3 — Run the reference shock (SIM1)

SIM1 is the shock of the GAMS reference: a 25 % cut of the export tax rate,
`ttix.fx(i) = ttixO(i)*0.75`. `apply_sim1_export_tax_cut` scales that rate in
the calibrated state, and the model built from it carries the new rate. It
**changes the state in place**, so shock a copy and keep the original for
other runs.

Solve counterfactuals as an MCP (`form="mcp"`), the form the parity gate
checks against GAMS's own SIM1:

```python
import copy

from equilibria.templates.pep_pyomo.pep_pyomo_scenarios import apply_sim1_export_tax_cut

shocked = apply_sim1_export_tax_cut(copy.deepcopy(state), factor=0.75)
m1 = build_pep_model(shocked, variant="base", form="mcp")
res1 = solve_pep(m1)
print(res1.code)
print("GDP_BP", round(value(m1.GDP_BP), 1))
print("LEON ~ 0:", abs(value(m1.LEON)) < 1e-4)
# 1
# GDP_BP 46748.2
# LEON ~ 0: True
```

`46748.2` is GAMS's `GDP_BP` for SIM1. `LEON` is the Walras slack: zero
means the shocked point is an equilibrium. Measured on 2026-10-28, the NLP
form with the same shock converges to a point with `LEON = 2663.9`, which is
not an equilibrium; use the MCP form for shocks.

## Step 4 — Check against GAMS

The parity gates are pytest tests:

```bash
uv run pytest tests/templates/pep_pyomo/test_pep_pyomo_solve.py
```

They check that the benchmark holds in 95 of the 96 equation families (the
one exception is a documented imbalance of the reference data), that the NLP
and MCP forms land on the same point, and that the MCP matches the GAMS-native
MCP for the base and SIM1. The two MCP-vs-GAMS tests read
`Results_mcp.gdx` and `Results_mcp_sim1.gdx`, which are not in the repository
yet, so they skip unless you produce them with GAMS. The measured results are
in the {doc}`PEP coverage matrix <pep_coverage_matrix>`.
