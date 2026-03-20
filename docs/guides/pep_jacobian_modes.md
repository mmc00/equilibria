# pep jacobian modes

Esta guia resume como controlar y auditar el Jacobiano del solver PEP.

Si lo que quieres es entender la capa comun reutilizable del harness, ver tambien:

- `docs/guides/model_jacobian_harness.md`

## idea simple

Hoy PEP soporta dos modos:

- `analytic`
- `numeric`

`analytic` es el default recomendado.

`numeric` queda como modo de diagnostico y regresion.

## donde se configura

Se configura en `config`, no en `contract`.

Ejemplo:

```python
from equilibria.simulations import PepSimulator

sim = PepSimulator(
    sam_file="src/equilibria/templates/reference/pep2/data/SAM-V2_0.gdx",
    val_par_file="src/equilibria/templates/reference/pep2/data/VAL_PAR.xlsx",
    method="ipopt",
    init_mode="excel",
    contract="pep_nlp_v1",
    config={
        "solver": "ipopt",
        "problem_type": "nlp",
        "jacobian_mode": "analytic",
        "tolerance": 1e-8,
        "max_iterations": 300,
    },
).fit()
```

Para forzar el modo numerico:

```python
sim = PepSimulator(
    sam_file="src/equilibria/templates/reference/pep2/data/SAM-V2_0.gdx",
    val_par_file="src/equilibria/templates/reference/pep2/data/VAL_PAR.xlsx",
    method="ipopt",
    init_mode="excel",
    contract="pep_nlp_v1",
    config={"jacobian_mode": "numeric"},
).fit()
```

## que expone el solver

El resultado ahora incluye `solver_stats`.

Campos utiles:

- `jacobian_mode`
- `wall_time_seconds`
- `constraint_eval_count`
- `jacobian_eval_count`
- `structure_eval_count`
- `finite_difference_eval_count`
- `jacobian_nonzero_count`

Interpretacion practica:

- si `jacobian_mode="analytic"` y `finite_difference_eval_count=0`, el solve no uso fallback por diferencias finitas
- si `jacobian_mode="numeric"`, ese contador debe ser positivo

## benchmark reproducible

Script:

- `scripts/parity/measure_pep_jacobian_modes.py`

Ejemplo:

```bash
uv run python scripts/parity/measure_pep_jacobian_modes.py \
  --sam-file src/equilibria/templates/reference/pep2/data/SAM-V2_0.gdx \
  --val-par-file src/equilibria/templates/reference/pep2/data/VAL_PAR.xlsx \
  --reference-manifest output/gams_nlp_reference/core_default_real_v2/manifest.json \
  --save-report output/pep_jacobian_modes_default.json
```

El reporte compara:

- tiempo
- iteraciones
- residual final
- paridad contra la referencia oficial `GAMS + IPOPT + NLP`

## gate estructural

El mismo script soporta `--gate`.

La idea del gate no es fijar un umbral de tiempo absoluto, porque eso es fragil.

El gate falla si:

- `analytic` no converge
- `numeric` no converge
- `analytic` usa mas diferencias finitas que el maximo permitido
- la paridad de `analytic` es peor que la de `numeric`

Ejemplo:

```bash
uv run python scripts/parity/measure_pep_jacobian_modes.py \
  --sam-file src/equilibria/templates/reference/pep2/data/SAM-CRI-gams-fixed.xlsx \
  --val-par-file src/equilibria/templates/reference/pep2/data/VAL_PAR-CRI-gams.xlsx \
  --reference-manifest output/gams_nlp_reference/core_cri_fixed_real_v1/manifest.json \
  --sam-qa-mode hard_fail \
  --cri-fix-mode off \
  --gate \
  --save-report output/pep_jacobian_modes_cri_fixed.json
```

## criterio recomendado

- usar `analytic` como default
- usar `numeric` solo para:
  - diagnostico
  - regresion
  - comparacion contra cambios del harness

## relacion con la capa comun

PEP ya no carga solo con una implementacion ad hoc.

Hoy PEP consume:

- el harness base comun
- la capa comun de `solver_stats`
- la capa comun de reporting/gate

Por eso, cuando el cambio toca la infraestructura comun, conviene correr tambien:

```bash
uv run python scripts/parity/measure_simple_open_jacobian_modes.py --gate
```
