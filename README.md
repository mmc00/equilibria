<div align="center">

<img src="docs/assets/logo.svg" alt="equilibria logo" width="120"/>

# equilibria

**A Python framework for Computable General Equilibrium modeling**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Documentation](https://img.shields.io/badge/docs-sphinx-blue.svg)](https://equilibria.readthedocs.io/)
[![DOI](https://zenodo.org/badge/1143895755.svg)](https://doi.org/10.5281/zenodo.23675339)

*Build CGE models from modular blocks — like LEGO for economists — and check them against GAMS and GEMPACK.*

**Created by [Marlon Molina](https://github.com/mmc00)**

[Installation](#installation) •
[Quick Start](#quick-start) •
[Documentation](https://equilibria.readthedocs.io/) •
[Roadmap](ROADMAP.md) •
[Contributing](#contributing)

</div>

---

## Why equilibria?

CGE models are usually written in GAMS or GEMPACK. **equilibria** brings them
to Python without giving up the reference implementations: every template is
checked against them cell by cell.

| | |
|---|---|
| 🧱 **Models built from blocks** | A model is a stack of blocks, each declaring its parameters, variables and equations. GTAP Standard 7 is eight of them. |
| ✅ **Checked against the originals** | GAMS and GEMPACK are the source of truth. Parity gates compare every solved cell, same engine on both sides. |
| 🔁 **One model, two solve forms** | The same model solves as an MCP (PATH, through its C API) or as an NLP (IPOPT), like GAMS's `ifMCP`. |
| 🌐 **Native data I/O** | A pure-Python reader and writer for GEMPACK HAR files, a GDX reader, and SAM tools (audit, balancing, MIP → SAM). |

Templates today: **GTAP Standard 7** (multi-period, both `ifSUB` modes),
**PEP-1-1 v2.1** (`pep_pyomo`) and a didactic **simple open economy**.

---

## Installation

equilibria is not on PyPI yet; install it from source:

```bash
git clone https://github.com/mmc00/equilibria.git
cd equilibria
pip install -e ".[pyomo,ipopt,excel]"
```

Optional extras: `viz` (matplotlib, plotly), `poi` (PyOptInterface),
`modelcache` (cloudpickle), `docs`, `all`.

MCP solves use PATH through [`path-capi-python`](https://github.com/mmc00/path-capi-python);
see [Solving with PATH](https://equilibria.readthedocs.io/en/latest/guide/path_capi.html)
for where equilibria looks for it and for the PATH libraries.

---

## Quick Start

### GTAP: a tariff shock

```python
from equilibria import load_bundled
from equilibria.templates.gtap import apply_shock
from equilibria.templates.gtap.gtap_block_model import build_block_model
from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig
from equilibria.templates.gtap.gtap_multiperiod_driver import solve_multiperiod
from equilibria.templates.gtap.welfare_regional import regional_ev

params = load_bundled("gtap", "nus333")  # ships in the package
closure = GTAPClosureConfig(
    name="base", closure_type="MCP", capital_mobility="sluggish",
    fix_endowments=False, fix_taxes=False, fix_technology=False,
    if_sub=False, numeraire="pnum",
)
m, _ = build_block_model(params, params.sets, closure, "ROW")

# +10% on the tariff power of US manufacturing exports to ROW, as in a GEMPACK .EXP
apply_shock(m, {"imptx": {("USA", "MFG", "ROW"): 10.0}})
results = solve_multiperiod(m, params, closure, mode="gtap", skip_base_solve=True, solve_check=True)

print(results["shock"]["code"])  # 1 = solved
print(regional_ev(m))            # {'USA': -0.1079..., 'ROW': 0.0885...}
```

The model solves `base → check → shock` in one go, as GAMS's `loop(tsim)`.
The full walk-through, including the comparison against GAMS, is the
[GTAP quickstart](https://equilibria.readthedocs.io/en/latest/guide/gtap_quickstart.html).

### PEP: the reference shock

```python
import copy
from pathlib import Path

from pyomo.environ import value

import equilibria.templates
from equilibria.templates.pep_calibration_unified import PEPModelCalibrator
from equilibria.templates.pep_pyomo.pep_pyomo_equations import build_pep_model
from equilibria.templates.pep_pyomo.pep_pyomo_scenarios import apply_sim1_export_tax_cut
from equilibria.templates.pep_pyomo.pep_pyomo_solver import solve_pep

data = Path(equilibria.templates.__file__).parent / "reference" / "pep2" / "data"
state = PEPModelCalibrator(
    sam_file=data / "SAM-V2_0.gdx", val_par_file=data / "VAL_PAR.xlsx"
).calibrate()

# SIM1 of the GAMS reference: export tax rates × 0.75, solved as an MCP
shocked = apply_sim1_export_tax_cut(copy.deepcopy(state), factor=0.75)
m = build_pep_model(shocked, variant="base", form="mcp")
print(solve_pep(m).code, round(value(m.GDP_BP), 1))  # 1 46748.2 (GAMS: 46748.2)
```

See the [PEP quickstart](https://equilibria.readthedocs.io/en/latest/guide/pep_quickstart.html).

### Data: read and write GEMPACK HAR files

```python
import numpy as np

from equilibria.babel.har import HarWriter, read_har

data = read_har("datasets/gtap7_3x3/basedata.har")  # dict[name -> HeaderArray]
vdpp = data["VDPP"]
print(vdpp.long_name, vdpp.array.shape, vdpp.set_names)

with HarWriter("synthetic.har") as w:
    w.add_set("REG", ["USA", "ROW"])
    w.add_set("COMM", ["AGR", "MFG"])
    w.add_array("VDPP", np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
                set_names=["COMM", "REG"], long_name="domestic private purchases")
```

More in [HAR file I/O](https://equilibria.readthedocs.io/en/latest/guide/har_io.html)
and [MIP → SAM](https://equilibria.readthedocs.io/en/latest/guide/mip_to_sam.html).

---

## How it is built

A model is a stack of blocks, written for one period and copied once per
period; a shock is one instrument cell that differs in the `shock` period;
the driver follows a recipe and holds no model logic. The
[architecture overview](https://equilibria.readthedocs.io/en/latest/guide/architecture.html)
explains it with pictures, and each rule is an Architecture Decision Record
in [`docs/adr/`](docs/adr/).

## Status

What matches what, measured: the
[GTAP coverage matrix against GAMS](https://equilibria.readthedocs.io/en/latest/guide/gtap7_coverage_matrix.html),
the [one against GEMPACK](https://equilibria.readthedocs.io/en/latest/guide/gtap7_coverage_matrix_gempack.html),
and the [PEP coverage matrix](https://equilibria.readthedocs.io/en/latest/guide/pep_coverage_matrix.html).
What comes next: [ROADMAP.md](ROADMAP.md).

## Examples

`examples/cge/` walks through the framework step by step, from sets and
parameters to a model on the Pyomo backend:

```bash
python examples/cge/example_01_basic_setup.py
```

---

## Citation

If you use **equilibria** in your research, please cite it. GitHub's
"Cite this repository" button reads [`CITATION.cff`](CITATION.cff); the
archived releases are on Zenodo:

```bibtex
@software{equilibria,
  author = {Molina, Marlon},
  title  = {equilibria: A Python Framework for CGE Modeling},
  year   = {2026},
  doi    = {10.5281/zenodo.23675339},
  url    = {https://github.com/mmc00/equilibria}
}
```

---

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Dev quickstart

```bash
uv sync --dev
uv tool install prek && prek install   # pre-commit hooks (ruff + ty + hygiene)
prek run --all-files
```

Hooks are fast hygiene only (format / lint / type, no solver). Use `prek`, not
`pre-commit` — same config, faster runner. See [CONTRIBUTING.md](CONTRIBUTING.md)
for the `ty` ratchet and the separate parity gates.

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
**[GitHub](https://github.com/mmc00/equilibria)**

*Created by [Marlon Molina](https://github.com/mmc00) • Made with ❤️ for economists who code*

</div>
