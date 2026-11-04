# examples

Colección de ejemplos organizados por dominio.

## Estructura

- `examples/cge/`: ejemplos generales del framework CGE.
- `examples/pep/`: ejemplos del template PEP y paridad GDP/SAM.
- `examples/gdx/`: ejemplos y utilidades de lectura/escritura/diagnóstico GDX.
- `examples/sam/`: ejemplos para transformar SAM con el pipeline manual (`scripts/sam_tools/run_manual_sam_pipeline.py`).
- `examples/notebooks/gtap7/burfisher/`: one notebook per Burfisher (3rd ed.) exercise that replicates it from scratch on nus333 and checks every level against GEMPACK (within 0.5%). Written by `make_notebooks.py` in the same folder; `tests/examples/test_burfisher_notebooks.py` runs them.

## Ejecutar ejemplos

Desde la raíz del repo:

```bash
uv run python examples/cge/example_01_basic_setup.py
uv run python examples/pep/example_11_pep_gdp_sam_comparison.py
uv run python examples/gdx/multidim_examples.py
uv run python scripts/sam_tools/run_manual_sam_pipeline.py
```

## Nota

Varios scripts en `examples/gdx/` son herramientas de diagnóstico técnico.
Para uso funcional del modelo, prioriza `examples/cge/` y `examples/pep/`.
