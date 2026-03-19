# PEP GAMS NLP Reference

Esta guia describe el flujo oficial para generar y consumir referencias `GAMS + IPOPT + NLP` para PEP.

## Objetivo

La referencia oficial sirve para comparar la ruta publica de `equilibria` contra una corrida GAMS del mismo problema `NLP`, escenario por escenario.

El generador oficial es:

```bash
uv run python scripts/parity/generate_pep_gams_nlp_reference.py
```

## Salida esperada

El generador escribe:

- un `manifest.json`
- un workspace por escenario en `output/.../scenarios/<scenario>/scripts/`
- dentro de cada escenario:
  - `Results.gdx`
  - `Parameters.gdx` si existe
  - `PreSolveLevels.gdx` si existe
  - logs de GAMS

Los escenarios core oficiales son:

- `base`
- `export_tax`
- `import_price_agr`
- `import_shock`
- `government_spending`

## Uso recomendado

### Base `pep2`

```bash
uv run python scripts/parity/generate_pep_gams_nlp_reference.py \
  --core-scenarios \
  --gams-bin /Library/Frameworks/GAMS.framework/Versions/48/Resources/gams \
  --output-dir output/gams_nlp_reference/core_default_real
```

### CRI fixed

```bash
uv run python scripts/parity/generate_pep_gams_nlp_reference.py \
  --core-scenarios \
  --gams-bin /Library/Frameworks/GAMS.framework/Versions/48/Resources/gams \
  --gms-script src/equilibria/templates/reference/pep2/scripts/PEP-1-1_v2_1_ipopt_excel_dynamic_sam_cri.gms \
  --sam-file src/equilibria/templates/reference/pep2/data/SAM-CRI-gams-fixed.xlsx \
  --val-par-file output/gams_public_compare/VAL_PAR-CRI-gams-from-excel.gdx \
  --output-dir output/gams_nlp_reference/core_cri_fixed_real
```

## Contrato del manifest

El manifest oficial usa el schema:

- `schema_version = pep_gams_nlp_reference/v1`
- `scenario_references`
- `scenario_slices`

Cada entrada de `scenario_references` apunta al `Results.gdx` de ese escenario y al slice que debe usarse en la comparacion.

## Regla importante: `base -> sim1`

En `RESULTS PEP 1-1.GMS`:

- `BASE` = valores iniciales
- `SIM1` = valores despues del solve

Para paridad del baseline resuelto, el escenario `base` debe apuntar a `sim1`, no a `base`.

Si el manifest usa `base -> base`, la comparacion mezcla:

- benchmark pre-solve de GAMS
- solucion post-solve de Python

Eso puede producir falsos desajustes grandes, especialmente en CRI.

## Consumo desde el gate

El gate oficial ya acepta el manifest completo:

```bash
uv run python scripts/cli/run_pep_core_scenarios_gate.py \
  --sam-file src/equilibria/templates/reference/pep2/data/SAM-V2_0_connect.xlsx \
  --val-par-file src/equilibria/templates/reference/pep2/data/VAL_PAR.xlsx \
  --method ipopt \
  --init-mode excel \
  --reference-manifest output/gams_nlp_reference/core_default_real/manifest.json \
  --require-reference-manifest \
  --save-report output/pep_core_scenarios_default_gams_nlp_smoke.json
```

## Estado validado

Corridas reales validadas:

- `pep2` default: paquete core completo en verde
- `CRI fixed`: paquete core completo en verde

Caso no validado como referencia oficial:

- `SAM-CRI-gams.xlsx` cruda

Ese caso todavia puede fallar en GAMS por problemas de datos del modelo, por ejemplo `division by zero`, y no debe usarse como baseline oficial de paridad.

## Notas operativas

- En esta maquina, algunos scripts no pueden depender de `GDXXRW.EXE`.
- El generador y los `.gms` de referencia soportan reutilizar un `VAL_PAR.gdx` ya existente.
- Cuando se trabaje con CRI, preferir `SAM-CRI-gams-fixed.xlsx` para paridad oficial.
