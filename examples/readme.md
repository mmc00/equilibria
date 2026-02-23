# examples

Colección de ejemplos organizados por dominio.

## Estructura

- `examples/cge/`: ejemplos generales del framework CGE.
- `examples/pep/`: ejemplos del template PEP y paridad GDP/SAM.
- `examples/gdx/`: ejemplos y utilidades de lectura/escritura/diagnóstico GDX.

## Ejecutar ejemplos

Desde la raíz del repo:

```bash
uv run python examples/cge/example_01_basic_setup.py
uv run python examples/pep/example_11_pep_gdp_sam_comparison.py
uv run python examples/gdx/multidim_examples.py
```

## Nota

Varios scripts en `examples/gdx/` son herramientas de diagnóstico técnico.
Para uso funcional del modelo, prioriza `examples/cge/` y `examples/pep/`.
