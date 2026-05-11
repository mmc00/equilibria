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
pip install -e ".[pyomo,excel]"
```

## Optional extras

| Extra | Pulls in | Use when |
|-------|----------|----------|
| `pyomo` | Pyomo + a backend solver wiring | Solving full CGE models |
| `ipopt` | `cyipopt` | NLP solves with IPOPT |
| `excel` | `openpyxl`, `xlrd` | Reading SAM/MIP from Excel |
| `viz`  | `matplotlib`, `plotly` | Plotting results |
| `docs` | Sphinx + MyST + sphinx-gallery | Building these docs |

## PATH C API solver (required for GTAP Standard 7)

The GTAP Standard 7 quickstart and the benchmarks page rely on
[`path-capi-python`](https://github.com/mmc00/path-capi-python), a thin
Python wrapper around the official PATH C API. It is a **separate
project** — `equilibria` imports it but does not vendor it — and you
need it installed for the full 10,296-equation MCP solve and for any
script in `scripts/gtap/` that ends in `_full` or `bench_`.

Clone it next to `equilibria` and install in the same environment:

```bash
git clone https://github.com/mmc00/path-capi-python.git
cd path-capi-python
pip install -e .
```

Point the wrapper at the PATH shared library (it cannot solve without
one):

```bash
export PATH_CAPI_LIBPATH=/path/to/libpath50.dylib   # .so on Linux, .dll on Windows
export PATH_CAPI_LIBLUSOL=/path/to/liblusol.dylib   # macOS only
export PATH_LICENSE_STRING='<your PATH license string>'   # if you have one
```

PATH itself is third-party software with its own license. The wrapper
is MIT-licensed but you still need a valid PATH license (full or
restricted) to solve. See {doc}`path_capi` for the full setup,
troubleshooting, and a minimal solve example.

## Logging

`equilibria` follows the standard library convention: every module uses
`logging.getLogger(__name__)`, and the top-level `equilibria` logger has a
`NullHandler` attached so logs are silent by default.

For scripts and notebooks, opt in with:

```python
import equilibria
equilibria.setup_logging(level="INFO")
```
