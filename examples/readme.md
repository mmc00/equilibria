# Examples

Examples grouped by domain.

## Layout

- `examples/cge/`: general examples of the CGE framework.
- `examples/pep/`: examples of the PEP template and GDP/SAM parity.
- `examples/gdx/`: examples and utilities to read, write and diagnose GDX.
- `examples/sam/`: examples that transform a SAM with the manual pipeline (`scripts/sam_tools/run_manual_sam_pipeline.py`).
- `examples/notebooks/gtap7/burfisher/`: one notebook per Burfisher (3rd ed.) exercise that replicates it from scratch on nus333 and checks every level against GEMPACK (within 0.5%). Written by `make_notebooks.py` in the same folder; `tests/examples/test_burfisher_notebooks.py` runs them.

## Running the examples

From the repository root:

```bash
uv run python examples/cge/example_01_basic_setup.py
uv run python examples/pep/example_11_pep_gdp_sam_comparison.py
uv run python examples/gdx/multidim_examples.py
uv run python scripts/sam_tools/run_manual_sam_pipeline.py
```

## Note

Several scripts in `examples/gdx/` are technical diagnostic tools.
To use the model, start with `examples/cge/` and `examples/pep/`.
