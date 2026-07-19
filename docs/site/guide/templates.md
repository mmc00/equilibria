# Templates overview

A *template* is a complete, validated CGE model implementation: sets,
parameters, equations, calibration, and closure, ready to load a dataset
and solve. Status below is the parity status against the reference GAMS
implementation.

| Template | Model | Solve forms | Parity status |
|---|---|---|---|
| `equilibria.templates.gtap` | GTAP Standard 7 | MCP (PATH) · NLP (IPOPT), single- and multi-period | See the [GTAP 7 coverage matrix](gtap7_coverage_matrix.md) |
| `equilibria.templates.pep_pyomo` | PEP-1-1 v2.1 | NLP (IPOPT) · MCP (PATH) | 100% vs GAMS — see the [PEP coverage matrix](pep_coverage_matrix.md) |
| `equilibria.templates.simple_open` | Didactic open economy | NLP | GAMS-parity contract (`simple_open_contract.py`) |
| `equilibria.templates` (legacy PEP) | PEP-1-1 (cyipopt) | NLP | Superseded by `pep_pyomo` |

## GTAP Standard 7 (`gtap`)

The flagship template: GTAP Standard 7 with the altertax variant,
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
`equilibria.templates`) remains for reference but is superseded by
`pep_pyomo`.

## simple_open (`simple_open`)

A small open-economy model used to exercise the framework end-to-end and
as a didactic entry point — one SAM, a handful of sectors, an explicit
GAMS-parity contract (`simple_open_contract.py`,
`simple_open_parity_pipeline.py`). It has no dedicated quickstart; its
parity pipeline doubles as the usage example.
