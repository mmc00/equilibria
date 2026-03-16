# PEP core scenarios gate

This guide describes the public gate for the main PEP scenario pack:

- `base`
- `export_tax`
- `import_price_agr`
- `import_shock`
- `government_spending`

The gate runs through the public `PepSimulator` API with:

- `init_mode='excel'`
- `method='ipopt'`
- the same SAM route used by normal users

## What this gate checks

There are two useful modes.

### 1. Convergence gate

This checks that each scenario converges from the public Excel path.

### 2. Parity gate

This additionally compares each scenario against a GAMS reference provided in a JSON manifest.

Important:

- scenarios are run independently from the same cached calibrated base
- the gate does not chain warm starts across shocks
- this keeps the result deterministic and easier to compare against GAMS

## Command

### Default pep2

```bash
uv run python scripts/cli/run_pep_core_scenarios_gate.py \
  --sam-file src/equilibria/templates/reference/pep2/data/SAM-V2_0.gdx \
  --val-par-file src/equilibria/templates/reference/pep2/data/VAL_PAR.xlsx \
  --method ipopt \
  --init-mode excel \
  --save-report output/pep_core_scenarios_default.json
```

### CRI public route

```bash
uv run python scripts/cli/run_pep_core_scenarios_gate.py \
  --sam-file src/equilibria/templates/reference/pep2/data/SAM-CRI-gams.xlsx \
  --val-par-file src/equilibria/templates/reference/pep2/data/VAL_PAR-CRI-gams.xlsx \
  --method ipopt \
  --init-mode excel \
  --sam-qa-mode hard_fail \
  --cri-fix-mode auto \
  --save-report output/pep_core_scenarios_cri.json
```

## Optional GAMS reference manifest

If you want parity against GAMS, pass `--reference-manifest`.

Example shape:

```json
{
  "base": {
    "results_gdx": "/abs/path/base_results.gdx",
    "slice": "base"
  },
  "export_tax": {
    "results_gdx": "/abs/path/export_tax_results.gdx",
    "slice": "sim1"
  },
  "import_price_agr": {
    "results_gdx": "/abs/path/import_price_agr_results.gdx",
    "slice": "sim1"
  },
  "import_shock": {
    "results_gdx": "/abs/path/import_shock_results.gdx",
    "slice": "sim1"
  },
  "government_spending": {
    "results_gdx": "/abs/path/government_spending_results.gdx",
    "slice": "sim1"
  }
}
```

Then run:

```bash
uv run python scripts/cli/run_pep_core_scenarios_gate.py \
  --sam-file src/equilibria/templates/reference/pep2/data/SAM-CRI-gams.xlsx \
  --val-par-file src/equilibria/templates/reference/pep2/data/VAL_PAR-CRI-gams.xlsx \
  --method ipopt \
  --init-mode excel \
  --sam-qa-mode hard_fail \
  --cri-fix-mode auto \
  --reference-manifest /abs/path/pep_core_scenarios_manifest.json \
  --require-reference-manifest \
  --save-report output/pep_core_scenarios_cri_parity.json
```

## Exit code contract

- `0`: every scenario converged, and every provided parity comparison passed
- `2`: any scenario failed to converge, any required reference is missing, or any parity comparison failed

## Current verified state

Using the public Excel path:

- `pep2` default: `base`, `export_tax`, `import_price_agr`, `import_shock`, `government_spending` converge
- `CRI`: the same scenario pack also converges

For the CRI public base run, parity against the fresh GAMS `SIM1` reference was already validated separately in the solver branch work.
