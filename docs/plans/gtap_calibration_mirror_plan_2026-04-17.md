# Plan: Paridad Espejo GTAP Python ↔ GAMS por Comparación de Calibración

Fecha: 2026-04-17
Autor: dracomarmol@gmail.com
Estado: propuesta
Rama: feature/gtap-cgebox-template

## 1. Motivación

El patrón observado hasta ahora es **whack-a-mole**: arreglamos una ecuación, el residuo baja y reaparece en otra. Ese ciclo ocurre porque tenemos **dos fuentes simultáneas de divergencia** entre Python y GAMS:

1. **Forma de ecuación** (estructura MCP, índices, sustituciones `ifSUB`, dominios activos).
2. **Calibración** (parámetros derivados: `gf`, `gw`, `ge`, `gd`, `af`, `and`, `axp`, `amw`, `aft`, `alphaa`, `tmarg`, `chipm`, `kappaf`, etc.).

Mientras las dos fuentes estén acopladas, un residuo distinto de cero en el benchmark no identifica la causa: puede ser una ecuación mal escrita o un coeficiente mal calibrado. Cada fix resuelve una ecuación pero deja la otra fuente viva.

El enfoque propuesto **separa las dos fuentes** agregando una capa de comparación de calibración contra GAMS, manteniendo la calibración en Python como el camino por defecto.

## 2. Principios de diseño

1. **No reemplazo**: la calibración en Python sigue siendo la fuente primaria. El volcado GAMS es un **segundo set de valores** para comparar y, opcionalmente, ejecutar.
2. **Detección en bloque, no por bloque**: un solo pase produce el diff completo de todos los parámetros derivados y todos los niveles benchmark. Cero ciclos.
3. **Aislar la dimensión que falla**: si el residuo persiste con calibración GAMS inyectada, el problema es de ecuación. Si desaparece, el problema era de calibración.
4. **Reproducible desde el repo**: el dump GAMS se regenera con un script versionado, no es un artefacto opaco.

## 3. Entregables

### 3.1 `scripts/cal_dump.gms`

Script GAMS que:

1. Ejecuta la cadena estándar (`getData.gms` → `cal.gms`) con el mismo dataset/elasticidades que Python (`9x10Dat.gdx`, `default-9x10.gdx`, cierre `gtap_standard`).
2. Al final, declara un `execute_unload` con TODOS los parámetros derivados y niveles benchmark relevantes. Lista mínima:

   - Parámetros CES/CET: `gf`, `gw`, `ge`, `gd`, `amw`, `and`, `af`, `axp`, `aft`, `aa`, `alphaa`.
   - Coeficientes de márgenes/impuestos: `tmarg`, `chipm`, `chipd`, `kappaf`.
   - Elasticidades efectivas: `omegaf`, `etaff`, `esubt`, `esubc`, `esubm`, `esubva`, `sigmas`, `omegaw`.
   - Niveles benchmark `.l` de: `xf`, `xft`, `xc`, `xa`, `xaa`, `xda`, `xma`, `xd`, `xmt`, `xw`, `xe`, `xet`, `xp`, `va`, `nd`, `pf`, `pfa`, `pfy`, `pft`, `pa`, `pd`, `pm`, `pe`, `pet`, `px`, `pva`, `pnd`, `ps`, `pmcif`, `pefob`, `pwmg`, `ptmg`, `yi`, `yc`, `yg`, `regy`, `kstock`, `arent`, `facty`, `ytax`, `etax`, `mtax`, `dintx`, `mintx`.
   - Impuestos `rt*` y escudos sustituidos.

3. Salida: `src/equilibria/templates/reference/gtap/data/gams_cal_dump_9x10.gdx`.

Este script se ejecuta una vez (o cuando cambia el dataset). No se toca durante la iteración de modelo.

### 3.2 Cargador opcional en Python

En `src/equilibria/templates/gtap/gtap_parameters.py` (o módulo nuevo `gtap_gams_calibration.py`):

```python
@dataclass
class GAMSCalibrationDump:
    source_gdx: Path
    derived_params: Dict[str, Dict[Tuple, float]]
    benchmark_levels: Dict[str, Dict[Tuple, float]]

    @classmethod
    def from_gdx(cls, path: Path) -> "GAMSCalibrationDump": ...
```

Es un contenedor pasivo. **No** sustituye a `GTAPParameters`.

### 3.3 Tool de comparación

Módulo nuevo: `src/equilibria/templates/gtap/calibration_compare.py`.

Función principal:

```python
def compare_calibration(
    python_params: GTAPParameters,
    gams_dump: GAMSCalibrationDump,
    tol_abs: float = 1e-8,
    tol_rel: float = 1e-6,
) -> CalibrationDiff
```

`CalibrationDiff` produce:

- Tabla por parámetro: `name, n_indices, n_mismatch, max_abs_diff, max_rel_diff, top5_offenders`.
- Tabla por nivel `.l`: idem para benchmark.
- Export CSV/Parquet a `docs/findings/calibration_diff_<fecha>.csv`.
- Un `summary.json` con conteo total de mismatches por categoría (CES, CET, impuestos, márgenes, niveles).

Pase único → reporte completo → priorización por `max_rel_diff`.

### 3.4 Modo de inyección (opt-in)

Flag en `gtap_contract.py` o en el runner:

```
calibration_source: "python" | "gams" | "mixed:<param1>,<param2>,..."
```

- `python` (default): comportamiento actual.
- `gams`: la construcción del modelo Pyomo lee parámetros derivados y niveles iniciales del `GAMSCalibrationDump` en lugar de `GTAPParameters`.
- `mixed`: permite sobreescribir sólo un subconjunto (útil para A/B por bloque una vez identificado el culpable).

Este modo permite **ejecutar el MCP de Python con calibración GAMS**. Es el test definitivo: si con calibración GAMS el residuo benchmark cae a ~0, el problema restante es calibración Python. Si sigue alto, el problema es de ecuación.

### 3.5 Residual scan por ecuación

Independiente pero complementario: un utilitario que, dado un punto (benchmark Python, benchmark GAMS, o punto intermedio), evalúa **todas** las ecuaciones del modelo Pyomo y reporta `(eq_name, index, residual)` ordenado. Ya existe parcialmente en `gtap_solver` como diagnóstico post-solve; se extrae a un helper reutilizable `scripts/gtap/scan_residuals.py`.

## 4. Flujo de trabajo propuesto

1. **Generar dump GAMS** (una vez): `gams scripts/cal_dump.gms` → `gams_cal_dump_9x10.gdx`.
2. **Comparar calibraciones**: `python scripts/gtap/compare_calibration.py` → CSV + summary.json.
3. **Corregir divergencias calibración** ordenadas por `max_rel_diff` hasta que el summary reporte 0 mismatches (módulo tolerancia).
4. **Scan residual con calibración Python corregida**: si residuo ≈ 0 → paridad estructural + calibración alcanzada. Si residuo > 0 → son errores de ecuación puros.
5. **Scan residual con calibración GAMS inyectada** (modo `gams`): confirma que con calibración idéntica las ecuaciones convergen. Si no, la divergencia está en forma de ecuación y se corrige con diff localizado vs `model.gms`.
6. Una vez paridad alcanzada, remover la dependencia al dump para corridas de producción (se mantiene solo para regression).

## 5. ¿Garantiza el espejo entre ambas versiones?

**Parcialmente, y de forma demostrable.** Hay que ser preciso:

### Lo que SÍ garantiza

1. **Detección exhaustiva de divergencias de calibración** en una sola pasada: cualquier parámetro derivado o nivel benchmark que difiera entre Python y GAMS queda registrado, sin omisión.
2. **Aislamiento causal**: corriendo el modelo Pyomo con calibración GAMS (modo `gams`), el residuo resultante mide **exclusivamente** discrepancias de forma de ecuación, porque los coeficientes son idénticos a GAMS por construcción.
3. **Condición necesaria de espejo**: si la comparación de calibración reporta cero diff Y el residuo con calibración GAMS es cero en el benchmark, entonces `F_python(x_bench) = 0 = F_gams(x_bench)` en ese punto. El modelo Python reproduce exactamente el equilibrio benchmark que GAMS reproduce.

### Lo que NO garantiza por sí solo

1. **Equivalencia fuera del benchmark**: dos modelos con mismo equilibrio benchmark pueden diferir en simulaciones contrafactuales si tienen formas funcionales distintas que coinciden sólo en ese punto. Esto se cubre con un **test de shock** adicional (ej. arancel uniforme +10%) comparando soluciones Python vs GAMS.
2. **Robustez numérica del solver**: paridad estructural no garantiza que PATH converja con la misma ruta desde cualquier warm-start. Ese es un tema separado de tuning.
3. **Correcciones de ecuación**: el plan detecta qué ecuaciones difieren, pero reescribirlas sigue requiriendo edición manual contra `model.gms` línea a línea. La diferencia es que ahora sabemos **cuáles** y **cuánto**, en vez de adivinar por residuos.

### Condiciones suficientes para el espejo

El espejo queda garantizado cuando se cumplen los tres simultáneamente:

- **(C1)** `compare_calibration` reporta 0 mismatches (dentro de tolerancia).
- **(C2)** `scan_residuals` en el benchmark con calibración Python reporta `max |F_i| < ε`.
- **(C3)** Un shock de referencia (counterfactual definido) produce en Python la misma solución que GAMS (ej. `max |x_python - x_gams| / |x_gams| < 1e-4`).

C1 + C2 ⇒ benchmark idéntico. C3 ⇒ comportamiento fuera del benchmark idéntico. Juntos ⇒ espejo.

## 6. Riesgos y mitigaciones

| Riesgo | Mitigación |
| --- | --- |
| Versiones de GAMS/GTAP distintas entre dumps y datos base | Fijar versión GAMS en `cal_dump.gms` y checksums del GDX en repo |
| Nombres de parámetros GAMS que no tienen 1-a-1 Python | Mapa explícito `gams_name → python_field` en compare tool, con lista de conocidos-no-mapeables |
| Tolerancias demasiado estrictas generan ruido | Dos tolerancias (abs/rel) y clasificación por severidad |
| El dump queda desactualizado frente a cambios de dataset | Hash del dataset GDX en el dump; warning si no coinciden |

## 7. Orden de trabajo concreto

1. (0.5 d) Escribir `cal_dump.gms` con el `execute_unload` completo y ejecutarlo una vez.
2. (0.5 d) Loader `GAMSCalibrationDump.from_gdx`.
3. (1 d) `calibration_compare.py` + export CSV + summary.
4. (0.5 d) Primer reporte de diff Python vs GAMS → triage.
5. (iterativo) Corregir calibración Python en orden de severidad hasta diff = 0.
6. (0.5 d) Modo `calibration_source="gams"` en el constructor del modelo.
7. (iterativo) Si residuo con calibración GAMS > 0, diff localizado de ecuaciones vs `model.gms`.
8. (0.5 d) Test de shock para C3.

Estimado total: 3-4 días hombre hasta primer reporte ejecutable; el cierre completo depende del volumen real de divergencias que revele el primer diff.

## 8. Referencias

- `scripts/model.gms` — ecuaciones GAMS
- `scripts/cal.gms` — calibración GAMS (fuente de verdad para el dump)
- `docs/findings/gtap_baseline_inventory_2026-04-07.md` — DOF actual
- `docs/findings/gtap_mcp_structural_gap_2026-04-10.md` — gap estructural `eq_pfeq` (ya cerrado)
- `/Users/marmol/proyectos/path-capi-python/docs/gtap_path_capi_status_2026-04-17.md` — estado del runtime PATH-CAPI
