<div align="center">

<img src="docs/assets/logo.svg" alt="equilibria logo" width="200"/>

# equilibria

**A Modern Python Framework for Computable General Equilibrium Modeling**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://img.shields.io/pypi/v/equilibria.svg)](https://pypi.org/project/equilibria/)
[![Documentation](https://img.shields.io/badge/docs-mkdocs-blue.svg)](https://equilibria.readthedocs.io/)

*Build, calibrate, and solve CGE models with modular blocks — like LEGO for economists.*

**Created by [Marlon Molina](https://github.com/mmc00)**

[Installation](#installation) •
[Quick Start](#quick-start) •
[Documentation](https://equilibria.readthedocs.io/) •
[Examples](#examples) •
[Contributing](#contributing)

</div>

---

## Why equilibria?

Building Computable General Equilibrium (CGE) models traditionally requires deep expertise in GAMS, GEMPACK, or similar domain-specific languages. **equilibria** brings CGE modeling to the Python ecosystem with:

| Feature | Description |
|---------|-------------|
| 🧱 **Modular Blocks** | Snap together production, trade, and demand components like building blocks |
| 🌐 **Universal Data I/O** | Native support for GDX (GAMS), HAR (GEMPACK), Excel, and GTAP databases |
| ⚡ **Multiple Backends** | Solve with Pyomo, PyOptInterface, or export to GAMS code |
| 📐 **Auto-Calibration** | Automatically calibrate CES/CET parameters from SAM data |
| 📋 **Ready Workflows** | Pre-built workflows for PEP pep2 calibration/solve plus reusable model templates |
| 📊 **Built-in Analysis** | Model statistics, result comparison, sensitivity analysis |

---

## Installation

```bash
pip install equilibria
```

**With optional dependencies:**

```bash
# For GDX file support (GAMS)
pip install equilibria[gdx]

# For HAR file support (GEMPACK)
pip install equilibria[har]

# For visualization
pip install equilibria[viz]

# Everything
pip install equilibria[all]
```

---

## Quick Start

### Option 1: Run the PEP pep2 Calibration + Solver API

```python
from pathlib import Path

from equilibria.templates.pep_calibration_unified import PEPModelCalibrator
from equilibria.templates.pep_model_solver import PEPModelSolver

sam_file = Path("src/equilibria/templates/reference/pep2/data/SAM-V2_0.gdx")
val_par_file = Path("src/equilibria/templates/reference/pep2/data/VAL_PAR.xlsx")

# Calibrate
calibrator = PEPModelCalibrator(sam_file=sam_file, val_par_file=val_par_file)
state = calibrator.calibrate()

# Solve
solver = PEPModelSolver(calibrated_state=state, init_mode="excel")
solution = solver.solve(method="auto")
print(solution.summary())
```

### Option 2: Build a Custom Model with Blocks

```python
from equilibria import Model
from equilibria.babel import SAM
from equilibria.blocks import (
    CESValueAdded,
    LeontiefIntermediate,
    ArmingtonCES,
    CETExports,
    LinearExpenditureSystem,
    HouseholdIncome,
    GovernmentBudget,
    MarketClearing,
)

# Load data
sam = SAM.from_gdx("data/country_data.gdx")

# Create model
model = Model(name="CustomCGE")
model.sets.from_sam(sam)

# Add equation blocks (like LEGO pieces)
model.add_blocks([
    CESValueAdded(sigma=0.8),           # Production: CES value-added
    LeontiefIntermediate(),              # Production: Fixed intermediate coefficients
    ArmingtonCES(sigma_m=1.5),          # Trade: Import substitution
    CETExports(sigma_e=2.0),            # Trade: Export transformation
    LinearExpenditureSystem(),           # Demand: LES consumption
    HouseholdIncome(),                   # Institutions: Household accounts
    GovernmentBudget(),                  # Institutions: Government accounts
    MarketClearing(),                    # Equilibrium: Market clearing conditions
])

# Calibrate from SAM
model.calibrate(sam)

# Check model statistics
print(model.statistics)
# ModelStatistics(
#     variables=358,
#     equations=357,
#     degrees_of_freedom=1,
#     blocks=8,
#     sparsity=0.73
# )

# Solve
solution = model.solve()
```

---

## Core Concepts

### 📊 SAM (Social Accounting Matrix)

The SAM is the data foundation of any CGE model. **equilibria** provides a powerful `SAM` class:

```python
from equilibria.babel import SAM

# Read from various formats
sam = SAM.from_excel("data.xlsx")
sam = SAM.from_gdx("data.gdx")       # GAMS format
sam = SAM.from_har("data.har")       # GEMPACK format
sam = SAM.from_gtap("gtapdata/")     # GTAP database

# Validate and explore
sam.validate()                        # Check row/column balance
sam.summary()                         # Print summary statistics
sam.plot()                            # Visualize flows

# Export
sam.to_excel("balanced_sam.xlsx")
sam.to_gdx("for_gams.gdx")
```

### 🧱 Equation Blocks

Blocks are self-contained equation modules that define economic behavior:

```python
from equilibria.blocks import CESValueAdded

# Each block declares its required sets, parameters, variables, and equations
block = CESValueAdded(sigma=0.8)

print(block.info())
# CESValueAdded:
#   Sets required: J (industries)
#   Parameters: sigma_VA, beta_VA, B_VA
#   Variables: VA, LDC, KDC, PVA, WC, RC
#   Equations: 3 (CES aggregation, FOC labor, FOC capital)
```

**Available Blocks:**

| Category | Blocks |
|----------|--------|
| **Production** | `CESValueAdded`, `CETTransformation`, `LeontiefIntermediate`, `NestedCES` |
| **Trade** | `ArmingtonCES`, `CETExports`, `SmallOpenEconomy` |
| **Demand** | `LinearExpenditureSystem`, `CobbDouglasDemand`, `AIDS` |
| **Institutions** | `HouseholdIncome`, `GovernmentBudget`, `FirmAccounts`, `RestOfWorld` |
| **Factors** | `FactorMarkets`, `FactorMobility`, `SectorSpecificCapital` |
| **Equilibrium** | `MarketClearing`, `WalrasLaw`, `PriceNormalization` |

### 📋 Model Templates

Pre-configured models ready to use:

```python
from equilibria.templates import (
    SimpleOpenEconomy,  # Didactic 3-sector model
)

# Templates come with sensible defaults
model = SimpleOpenEconomy().create_model()
```

For PEP pep2 production workflows, use:
- `equilibria.templates.pep_calibration_unified.PEPModelCalibrator`
- `equilibria.templates.pep_model_solver.PEPModelSolver`
- CLI scripts in `scripts/cli/` (for example `run_all_calibration.py`, `run_solver.py`)

---

## Data Interoperability (babel)

**equilibria.babel** reads and writes all major CGE data formats:

```python
from equilibria.babel import read_gdx, read_har, write_gdx

# Read GAMS GDX files
data = read_gdx("results.gdx")
sam = data["SAM"]
parameters = data["parameters"]

# Read GEMPACK HAR files
data = read_har("database.har")

# Write back to GDX (for GAMS users)
write_gdx("python_results.gdx", solution)

# Generate GAMS code from your model
from equilibria.babel.writers import to_gams
to_gams(model, "generated_model.gms")
```

---

## Analysis Tools

```python
# Model diagnostics
stats = model.statistics
print(f"Variables: {stats.variables}")
print(f"Equations: {stats.equations}")
print(f"Degrees of freedom: {stats.dof}")
print(f"Sparsity: {stats.sparsity:.1%}")

# Compare solutions
from equilibria.analysis import compare

diff = compare(baseline, counterfactual)
diff.summary()                          # Print key changes
diff.to_excel("comparison.xlsx")        # Export detailed comparison
diff.plot_bars(variables=["GDP", "C"])  # Visualize changes

# Sensitivity analysis
from equilibria.analysis import sensitivity

results = sensitivity(
    model,
    parameter="sigma_M",
    values=[0.5, 1.0, 1.5, 2.0, 2.5],
    output_vars=["GDP", "imports", "exports"]
)
results.plot()
```

---

## Implementation Status

### GTAP Standard 7 — GAMS reference parity (2026-05-06)

The Python implementation of the **GTAP Standard 7** model template (`equilibria.templates.gtap`) is validated against the canonical GAMS reference (`COMP.gdx` from `tariff_sim.gms`, 9 sectors × 10 regions, 10% uniform import-tariff shock). Solver: PATH C API in nonlinear-full mode (16,166 vars × 16,166 eqs), residuals 2.2e-11 (baseline) / 6.0e-13 (shock).

**Cell-by-cell coverage** — full variable-by-variable parity reached:

| Phase | Vars exact ✅ | Cells diverging | Cells matching | Vars not modeled in Python |
|-------|--------------:|----------------:|---------------:|---------------------------:|
| baseline | 98 / 139 | **0** | 41,477 / 41,477 (100%) | 40 |
| shock    | 98 / 139 | **0** | 41,477 / 41,477 (100%) | 40 |

`validate_gams_parity.py` reports **0.0% error** on EastAsia `regy_delta` vs the GAMS reference.

**Three fixes that closed the residual ~2,000-cell gap** (on top of the earlier rate-scaling shock formula):

1. `diff_9x10_full.py` overrides `closure.if_sub=False` to match the GAMS NEOS reference. With `if_sub=True` the bilateral price/margin equations (`eq_pmeq`, `eq_pmcifeq`, `eq_pefobeq`, `eq_xwmg`, `eq_xmgm`, `eq_pwmg`, `eq_pp_rai`, `eq_pfaeq`, `eq_pfyeq`) are deactivated and their Vars fixed at baseline → shock can't propagate through bilateral trade prices.
2. `rorflex` default raised from 1.0 → **10.0** (matches GAMS `rorFlex0(r)=10`). The 9x10 GDX does not ship `RFLX`, so Python falls back to the default; the wrong value caused `rore` ~115% / `rorg` ~49% relative error.
3. **Land** is now classified as **mobile** (`mf`), not sluggish (`sf`). GAMS `getData.gms:111-117` builds `fnm` only from `endwf={NatRes}`; `endws={Land}` is NOT unioned into `fnm`. `gtap_sets.py` previously routed any factor name containing `lnd` to sluggish — fixed by removing it from `sluggish_defaults`.

**Welfare equations activated:** `eq_cv` and `eq_ev` (CDE-style identities) are now live in the MCP. They were previously deactivated with `cv`/`ev` fixed at 1.0; activating them adds 1 var ↔ 1 eq per region without disturbing anything else and lands `cv`/`ev` exactly on GAMS.

**Reproduce:**

```bash
# Macro deltas vs GAMS (10 regions × 3 macros, exact reference values)
.venv/bin/python scripts/gtap/validate_gams_parity.py --shock-factor 0.10

# Per-variable 1-D/2-D diff (21 macros)
.venv/bin/python scripts/gtap/diff_9x10_vs_gams.py --phase both

# Full coverage report (139 GAMS Vars, ~60k cells, both phases)
.venv/bin/python scripts/gtap/diff_9x10_full.py --phase both --show-worst

# NUS333 — power-scaled shock (matches comp_nus333.gms:149)
.venv/bin/python scripts/gtap/compare_nus333_vs_neos.py
```

Notes:
- 9x10 uses **rate scaling** (`tm = tm_base * (1 + tm_shock)`); NUS333 uses **power scaling** (`(1+imptx) * 1.10 - 1`). The two datasets follow different upstream conventions.
- Counterfactual builds must pass `t0_snapshot=base_model` so CES weights (alphad/alpham/betap/betag/betas) calibrate against the converged baseline rather than the perturbed state.
- The 40 "not in Python" Vars are exogenous shock parameters and pure diagnostics in GAMS — none are endogenous. Categories: technical shifters (`afe*`, `aio*`, `ava*`, `axp*`, `lambda*`), tax instruments treated as parameters in Python (`imptx`, `exptx`, `kappaf`, `fctts`, `fcttx`, `prdtx`, `mtxshft`, `dtxshft`, `rtxshft`), elasticity diagnostics (`ape`, `ced`, `incelas`, `ued`), and summary identities (`chiInv`, `pmuv`, `p`, `ytaxInd`). To shock them in Python, mutate `params.taxes` / `params.elasticities` (Pydantic dicts) before `build_gtap_model`, or use `_apply_shock_to_params(...)` in `scripts/gtap/run_gtap.py` (modes `set` / `pct` / `mult` / `tm_pct`).
- See `CLAUDE.md` and `GTAP_VALIDATION_STATUS.md` for the full audit trail.

---

## Examples

### Tax Policy Analysis

```python
from pathlib import Path

from equilibria.templates.pep_calibration_unified_excel import PEPModelCalibratorExcel
from equilibria.templates.pep_model_solver import PEPModelSolver

sam_file = Path("src/equilibria/templates/reference/pep2/data/SAM-V2_0.xls")
val_par_file = Path("src/equilibria/templates/reference/pep2/data/VAL_PAR.xlsx")

calibrator = PEPModelCalibratorExcel(sam_file=sam_file, val_par_file=val_par_file)
state = calibrator.calibrate()

solver = PEPModelSolver(calibrated_state=state, init_mode="excel")
baseline = solver.solve(method="auto")
print(baseline.summary())
```

### Multi-Region Analysis

Use the block-based `Model` API and `equilibria.babel` data loaders to assemble custom multi-region specifications.

### PEP GDP Comparison (Original SAM vs CRI SAM)

```bash
uv run python examples/pep/example_11_pep_gdp_sam_comparison.py
```

More details: `docs/guides/pep_gdp_example.md`

### Scenario Simulation API (Recommended)

```python
from equilibria.simulations import Scenario, Shock, Simulator

sim = Simulator(
    model="pep",
    sam_file="src/equilibria/templates/reference/pep2/data/SAM-V2_0.gdx",
    val_par_file="src/equilibria/templates/reference/pep2/data/VAL_PAR.xlsx",
    method="ipopt",
).fit()

report = sim.run_scenarios(
    scenarios=[
        Scenario(
            name="export_tax",
            shocks=[Shock(var="ttix", op="scale", values={"*": 0.75})],
        ),
        Scenario(
            name="import_shock",
            shocks=[Shock(var="PWM", op="scale", values={"*": 1.25})],
        ),
        Scenario(
            name="government_spending",
            shocks=[Shock(var="G", op="scale", values=1.2)],
        ),
    ],
    reference_results_gdx="src/equilibria/templates/reference/pep2/scripts/Results.gdx",
)

from equilibria.simulations import available_models
print(available_models())  # ('gtap', 'icio', 'ieem', 'pep')
```

Convenience wrappers for other models:

```python
from equilibria.simulations import GTAPSimulator, ICIOSimulator, IEEMSimulator

ieem = IEEMSimulator(base_state={"x": 1.0}).fit()
gtap = GTAPSimulator(base_state={"x": 1.0}).fit()
icio = ICIOSimulator(base_state={"x": 1.0}).fit()
```

Simpler PEP wrapper (no manual `Scenario(...)`):

```python
from equilibria.simulations import PepSimulator

sim = PepSimulator(
    sam_file="src/equilibria/templates/reference/pep2/data/SAM-V2_0.gdx",
    val_par_file="src/equilibria/templates/reference/pep2/data/VAL_PAR.xlsx",
    method="ipopt",
).fit()

report = sim.run_export_tax(multiplier=0.75)

# Discover shockable variables
print(sim.available_shocks())

# Build a single-shock scenario without writing Scenario(...) manually
scenario = sim.shock(var="PWM", index="agr", multiplier=1.25)
report = sim.run_shock(var="G", multiplier=1.2, name="government_spending")
```

More details:
- `docs/guides/simulations_api.md`
- `docs/guides/pep_user_flow_examples.md`

### PEP BASE vs EXPORT_TAX Parity (Python vs GAMS)

```bash
uv run python scripts/cli/run_pep_base_export_tax_parity.py \
  --save-report output/pep_base_export_tax_parity.json
```

### CRI Structural Finding (IEEM vs PEP SAM Architecture)

For the documented root cause of CRI instability during IEEM -> PEP conversion (trade-account architecture mismatch and production-tax routing drift), see:

- `docs/findings/finding_sam_ieem_vs_sam_pep.md`

---

## Solver Backends

```python
from equilibria import Model
from equilibria.backends import PyomoBackend, PyOptInterfaceBackend

# Default: Pyomo with IPOPT
model = Model(backend=PyomoBackend(solver="ipopt"))

# Alternative: PyOptInterface (faster for large models)
model = Model(backend=PyOptInterfaceBackend(solver="highs"))

# Check available solvers
from equilibria.backends import list_solvers
print(list_solvers())
# ['ipopt', 'conopt', 'path', 'highs', 'cplex', 'gurobi']
```

---

## Comparison with Other Tools

| Feature | equilibria | GAMS | GEMPACK | CGE.jl |
|---------|------------|------|---------|--------|
| Language | Python | GAMS | Tablo | Julia |
| Learning curve | Low | Medium | High | Medium |
| Modular blocks | ✅ | ❌ | ❌ | Partial |
| Read GDX | ✅ | Native | ❌ | ✅ |
| Read HAR | ✅ | ❌ | Native | ❌ |
| Open source | ✅ | ❌ | ❌ | ✅ |
| IDE integration | Any | GAMS Studio | RunGTAP | VS Code |
| Visualization | Built-in | Manual | ViewHAR | Plots.jl |

---

## Citation

If you use **equilibria** in your research, please cite:

```bibtex
@software{equilibria2026,
  author = {Molina, Marlon},
  title = {equilibria: A Modern Python Framework for CGE Modeling},
  year = {2026},
  url = {https://github.com/equilibria-cge/equilibria}
}
```

---

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

**Areas where we need help:**

- 📚 Documentation and tutorials
- 🧱 New equation blocks (AIDADS, CRESH, etc.)
- 🌐 Additional data format support
- 🧪 Test coverage
- 🌍 Translations

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Acknowledgments

**equilibria** builds on the shoulders of giants:

- [PEP Network](https://www.pep-net.org/) - Standard CGE methodology
- [GTAP](https://www.gtap.agecon.purdue.edu/) - Global trade analysis
- [IFPRI](https://www.ifpri.org/) - CGE modeling resources
- [Pyomo](https://www.pyomo.org/) - Optimization modeling
- The broader CGE modeling community

---

<div align="center">

**[Documentation](https://equilibria.readthedocs.io/)** •
**[PyPI](https://pypi.org/project/equilibria/)** •
**[GitHub](https://github.com/equilibria-cge/equilibria)**

*Created by [Marlon Molina](https://github.com/mmc00) • Made with ❤️ for economists who code*

</div>
