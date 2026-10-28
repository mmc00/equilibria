# Installation

`equilibria` requires Python 3.10 or newer.

## From PyPI (when published)

```bash
pip install equilibria
```

## From source

```bash
git clone https://github.com/mmc00/equilibria.git
cd equilibria
pip install -e ".[pyomo,ipopt,excel]"
```

## Optional extras

| Extra | Pulls in | Use when |
|-------|----------|----------|
| `pyomo` | Pyomo + a backend solver wiring | Solving full CGE models |
| `ipopt` | `cyipopt` | NLP solves with IPOPT |
| `excel` | `openpyxl`, `xlrd` | Reading SAM/MIP from Excel |
| `viz`  | `matplotlib`, `plotly` | Plotting results |
| `poi` | `pyoptinterface` | The PyOptInterface backend |
| `modelcache` | `cloudpickle` | Caching built GTAP models on disk |
| `docs` | Sphinx + MyST + sphinx-gallery | Building these docs |

## PATH for MCP solves

GTAP and PEP solve their MCP form with PATH through
[`path-capi-python`](https://github.com/mmc00/path-capi-python), a separate
project that equilibria imports but does not vendor. {doc}`path_capi` explains
how to install it and where equilibria looks for the wrapper and for the PATH
libraries. NLP solves (IPOPT) do not need it.

## Logging

`equilibria` follows the standard library convention: every module uses
`logging.getLogger(__name__)`, and the top-level `equilibria` logger has a
`NullHandler` attached so logs are silent by default.

For scripts and notebooks, opt in with:

```python
import equilibria
equilibria.setup_logging(level="INFO")
```
