# Solving with PATH via the C API

GTAP is solved as a mixed complementarity problem (MCP) with PATH. equilibria
does not write GAMS or `.nl` files for that: it drives the PATH C API
directly through [`path-capi-python`](https://github.com/mmc00/path-capi-python),
a thin Python wrapper that hands PATH the residuals and the Jacobian of the
Pyomo model as callbacks.

You do not call the wrapper yourself. `solve_multiperiod` (see the
{doc}`GTAP quickstart <gtap_quickstart>`) uses it for every MCP solve, through
`equilibria.solver.path_capi`. What you need is to make the wrapper and the
PATH libraries findable.

```{important}
PATH itself is third-party software with its own license. The wrapper is
MIT-licensed for the bridge code only — you still need a valid PATH license
(full or restricted) to solve anything but small problems.
```

## Installation

`path-capi-python` is a separate project. Either install it in the same
environment:

```bash
git clone https://github.com/mmc00/path-capi-python.git
pip install -e path-capi-python
```

or leave the checkout where it is and tell equilibria where its `src`
directory lives:

```bash
export EQUILIBRIA_PATH_CAPI_SRC=/path/to/path-capi-python/src
```

## The PATH libraries

PATH ships as a shared library (`libpath`) plus LUSOL (`liblusol`). equilibria
looks for them, in this order:

1. the directory in `EQUILIBRIA_PATH_CAPI_LIB_DIR`;
2. `.cache/path_capi/` under the current directory.

| Platform | File names it looks for |
|---|---|
| macOS | `libpath50.silicon.dylib`, `libpath50.dylib` |
| Linux | `libpath50.so`, `libpath50.silicon.so` |
| Windows | `path50.dll`, `libpath50.dll` |

If you have a license string, set it before solving:

```bash
export PATH_LICENSE_STRING='<your license string>'
```

## Solving with IPOPT instead

A GTAP model can also be solved as an NLP with IPOPT (the GAMS `ifMCP=0`
form): set `EQUILIBRIA_GTAP_SOLVE_NLP=1` before solving. In that mode the
PATH bindings are never loaded, so `path-capi-python` is not needed. The
parity gates measure both forms against GAMS, same engine on both sides
(MCP vs MCP, NLP vs NLP).

## Troubleshooting

| Symptom | Likely cause / fix |
|---------|--------------------|
| `Unable to import path_capi_python. Set EQUILIBRIA_PATH_CAPI_SRC ...` | The wrapper is neither installed nor at `EQUILIBRIA_PATH_CAPI_SRC`. |
| PATH cannot find its library | No `libpath` in `EQUILIBRIA_PATH_CAPI_LIB_DIR` or `./.cache/path_capi/`. |
| `PATH license has expired` | Set `PATH_LICENSE_STRING` to a current license. |
