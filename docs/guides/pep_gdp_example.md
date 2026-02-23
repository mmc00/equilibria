# PEP GDP Example: Original SAM vs CRI SAM

Este ejemplo corre el modelo PEP en Python para dos bases:

1. SAM original de PEP2 (`SAM-V2_0.xls`)
2. SAM de CRI (`SAM-CRI-gams-fixed.xlsx`)

En ambos casos usa inicialización `strict_gams` y extrae:

- `GDP_BP`, `GDP_MP`, `GDP_IB`, `GDP_FD`
- `GDP_BP_REAL`, `GDP_MP_REAL`
- Descomposición por demanda:
  - consumo privado
  - consumo de gobierno
  - formación bruta de capital fijo
  - variación de inventarios
  - exportaciones FOB
  - importaciones CIF
  - reconstrucción de `GDP_FD` y brecha numérica

## Script

`examples/pep/example_11_pep_gdp_sam_comparison.py`

## Ejecutar

```bash
uv run python examples/pep/example_11_pep_gdp_sam_comparison.py
```

Guardar además un JSON:

```bash
uv run python examples/pep/example_11_pep_gdp_sam_comparison.py --save-json output/pep_gdp_comparison.json
```

## Notas

- El script intenta usar `strict_gams` (`BASE` para original, `SIM1` para CRI).
- Si el `Results.gdx` no tiene ese slice o la escala de PIB no coincide con la SAM activa, cae automáticamente a `equation_consistent`.
- El reporte impreso incluye: modo de inicialización usado, razón de selección, convergencia y mensaje del solver.
