# Templates overview

A *template* is a complete, validated CGE model implementation: sets,
parameters, equations, calibration, and closure, ready to load a dataset
and solve. Status below is the parity status against the reference GAMS
implementation.

| Template | Model | Solve forms | Parity status |
|---|---|---|---|
| `equilibria.templates.gtap` | GTAP Standard 7 | MCP (PATH) · NLP (IPOPT), single- and multi-period | See the [GTAP 7 coverage matrix](gtap7_coverage_matrix.md) |
| `equilibria.templates.pep_pyomo` | PEP-1-1 v2.1 | NLP (IPOPT) · MCP (PATH) | 100% vs GAMS — see the [PEP coverage matrix](pep_coverage_matrix.md) |
| `equilibria.templates.simple_open` | Hosoe's standard CGE (gamslib `stdcge`) | square system (IPOPT, PATH or Newton) | 100% vs GAMS at base, check and tariff shock |
| `equilibria.templates` (legacy PEP) | PEP-1-1 (cyipopt) | NLP | Superseded by `pep_pyomo`; to be removed |

## GTAP Standard 7 (`gtap`)

The flagship template, built from blocks (see {doc}`architecture`): GTAP
Standard 7 with the altertax variant,
single-period and multi-period (base → check → shock), both `ifSUB` modes,
solved as MCP (PATH) or NLP (IPOPT) across 6 datasets (3×3 … 20×41).
Start with the {doc}`GTAP quickstart <gtap_quickstart>`; welfare analysis
in {doc}`welfare_decomposition`.

## PEP-1-1 v2.1 (`pep_pyomo`)

The PEP-1-1 v2.1 single-country CGE ported from GAMS to Pyomo — six
modules (`pep_pyomo_sets`, `pep_pyomo_parameters`, `pep_pyomo_equations`,
`pep_pyomo_blocks`, `pep_pyomo_scenarios`, `pep_pyomo_solver`). Both the
NLP form (vs GAMS CNS) and the MCP form (vs GAMS-native MCP) reproduce the
GAMS reference at 100% cell parity, including the SIM1 export-tax
counterfactual. Start with the {doc}`PEP quickstart <pep_quickstart>`.

The original cyipopt-based PEP template (`pep_*` modules under
`equilibria.templates`, the engine `PepSimulator` runs) is superseded by
`pep_pyomo` and is scheduled for removal. Its calibrator,
`PEPModelCalibrator`, is shared: `pep_pyomo` calibrates with it.

## simple_open (`simple_open`)

The standard CGE model of Hosoe, Gasawa and Hashimoto (chapter 6 of their
handbook), shipped with GAMS as `stdcge`: Cobb-Douglas value added, fixed
intermediates, CET exports, Armington imports with tariffs, one household,
a government and investment. It is built from the generic level-form blocks
(`blocks/production`, `trade`, `demand`, `institutions`, `equilibrium`) and
reads its goods and factors from the SAM, so any SAM with the same accounts
works; the book's SAM (bread and milk) is the default.

It is the first template that runs on the core driver (ADR-0011): one model
with the periods `base`, `check` and `shock`, flown by its default recipe.

```python
from equilibria.core import Exercise
from equilibria.templates import SimpleOpenEconomy

ex = Exercise(SimpleOpenEconomy(), shock={"tau_m": 0.0})  # abolish every tariff
results = ex.run()  # {period: {"code", "residual"}}
ex.model.value("M", ("BRD",), "shock")  # one cell of one period
```

The shock gives new levels of instruments: the tax rates `tau_d`, `tau_z` and
`tau_m` are fixed variables, changed only in the shock period. The three periods
match GAMS in every cell (`equilibria.compare` with the `stdcge` map, reference
in `data/stdcge/`). Example: `examples/cge/example_07_templates.py`.
