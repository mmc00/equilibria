# Plan de Implementación: Conversión MIP a SAM

## Contexto

El proyecto equilibria necesita funcionalidad para convertir Matrices Insumo-Producto (MIP) estándar de cuentas nacionales en Matrices de Contabilidad Social (SAM) compatibles con los modelos CGE PEP.

**Problema**: Las MIP tradicionales tienen Valor Agregado (VA) como una fila agregada sin desagregar en factores (trabajo/capital) ni instituciones (hogares/gobierno/empresas). Los modelos CGE PEP requieren SAMs completas con factores e instituciones explícitas.

**Solución**: Crear un pipeline de transformación `run_mip_to_sam()` que sigue el patrón comprobado de `run_ieem_to_pep()`, pero con transformaciones específicas para:
1. Desagregar VA en factores L (trabajo) y K (capital)
2. Crear distribución de ingresos de factores a instituciones
3. Establecer flujos fiscales (impuestos)
4. Crear cuentas de hogares, gobierno, y resto del mundo

**Alcance inicial**: Soporte para 1 hogar agregado (simplificación), con arquitectura extensible para múltiples tipos de hogares en el futuro.

## Arquitectura de la Solución

### Nuevos Módulos

Siguiendo el patrón IEEM→PEP existente:

```
src/equilibria/sam_tools/
├── mip_raw_excel.py         # Nuevo: Parser de MIP estándar
├── mip_to_sam_transforms.py # Nuevo: Transformaciones MIP→SAM
└── api.py                    # Modificar: Agregar run_mip_to_sam()
```

### Flujo de Transformación

```
MIP Excel (VA agregado)
    ↓ [parse]
Sam con ("RAW", label)
    ↓ [normalize_mip_accounts]
Sam con (J, I, VA, FD)
    ↓ [disaggregate_va_to_factors]
Sam con (J, I, L, K, FD)
    ↓ [create_factor_income_distribution]
Sam con (J, I, L, K, AG)
    ↓ [create_household_expenditure]
Sam con (J, I, L, K, AG.hh completo)
    ↓ [create_government_flows]
Sam con fiscal completo (AG.gvt, AG.ti, AG.tm)
    ↓ [create_export_block]
Sam con (J, I, X, L, K, AG, OTH)
    ↓ [balance_ras]
SAM PEP balanceada
```

## Datos Requeridos y Fuentes

### Datos DENTRO de la MIP (input_path Excel)
- ✓ Flujos intermedios I×J
- ✓ Valor Agregado por sector (fila VA)
- ✓ Demanda final (Hogares, Gobierno, Inversión, Exportaciones)
- ✓ Importaciones por commodity

### Datos EXTERNOS Requeridos

#### 1. **Shares de Factores** (`va_factor_shares`)
**Qué es**: Proporción del Valor Agregado que va a trabajo (L) vs capital (K)

**Fuente sugerida**:
- Cuentas Nacionales detalladas (tabla de remuneración de factores)
- Matriz de insumo-producto ampliada si existe
- Literatura: Típicamente L=60-70%, K=30-40% (varía por país/sector)

**Default si no disponible**: `{"L": 0.65, "K": 0.35}` (IFPRI, Lofgren et al. 2002)

**Formato**:
```python
va_factor_shares = {
    "L": 0.65,  # 65% del VA es remuneración al trabajo
    "K": 0.35   # 35% del VA es remuneración al capital
}
```

#### 2. **Distribución de Ingreso de Factores** (`factor_to_household_shares`)
**Qué es**: Qué proporción del ingreso de cada factor va a hogares vs empresas vs gobierno

**Fuente sugerida**:
- Encuestas de hogares (ENIGH, EPH, etc.)
- Cuentas de ingreso y gasto de hogares del banco central
- Para capital: balances de empresas, distribución de dividendos

**Default si no disponible** (1 hogar agregado):
```python
factor_to_household_shares = {
    "L": {"hh": 0.95, "gvt": 0.05},   # 95% trabajo a hogares, 5% gob (impuestos directos)
    "K": {"hh": 0.50, "firm": 0.45, "gvt": 0.05}  # 50% capital a hogares, 45% empresas, 5% impuestos
}
```

#### 3. **Tasas de Impuestos** (`tax_rates`)
**Qué es**: Tasas efectivas de impuestos indirectos, aranceles, impuestos directos

**Fuente sugerida**:
- Administración tributaria (SAT, AFIP, SII, etc.)
- Informes de recaudación fiscal
- Estadísticas de comercio exterior para aranceles

**Default si no disponible**:
```python
tax_rates = {
    "production_tax": 0.10,  # IVA/impuestos indirectos típicos 8-15%
    "import_tariff": 0.05,   # Aranceles promedio 3-8%
    "direct_tax": 0.15       # Impuestos sobre la renta 10-20%
}
```

**Nota**: Si la MIP ya incluye flujos de impuestos (fila "Impuestos sobre productos"), usarlos directamente en vez de aplicar tasas.

### Resumen: Configuración Mínima vs Completa

**Configuración MÍNIMA** (solo MIP, todo con defaults):
```python
run_mip_to_sam(
    input_path="mip_2020.xlsx"
    # Usa todos los defaults de literatura
)
```

**Configuración RECOMENDADA** (con datos nacionales si disponibles):
```python
run_mip_to_sam(
    input_path="mip_2020.xlsx",
    va_factor_shares={"L": 0.68, "K": 0.32},  # De cuentas nacionales
    tax_rates={
        "production_tax": 0.12,  # IVA efectivo del país
        "import_tariff": 0.06,   # Arancel promedio
        "direct_tax": 0.18       # Impuesto renta efectivo
    }
    # factor_to_household_shares usa default (1 hogar)
)
```

## Implementación Detallada

### 1. Parser MIP (`mip_raw_excel.py`)

**Nuevo archivo**: `/Users/marmol/proyectos/equilibria/src/equilibria/sam_tools/mip_raw_excel.py`

Crear clase `MIPRawSAM` siguiendo patrón de `IEEMRawSAM`:

```python
class MIPRawSAM(Sam):
    """SAM helper for loading raw MIP matrices."""

    @classmethod
    def from_mip_excel(
        cls,
        path: Path,
        sheet_name: str = "MIP",
        *,
        va_row_label: str = "Valor Agregado",
        import_row_label: str = "Importaciones",
    ) -> MIPRawSAM:
        """Load standard MIP from Excel.

        Expected structure:
        - Rows: Commodities + VA row + Import row
        - Columns: Sectors + Final demand (HH, GOV, INV, EXP)
        """
```

**Características**:
- Detectar automáticamente inicio de datos (similar a IEEM parser)
- Validar que existe fila de VA
- Normalizar labels a `("RAW", element)` tuples
- Manejo de importaciones como fila separada

### 2. Transformaciones MIP→SAM (`mip_to_sam_transforms.py`)

**Nuevo archivo**: `/Users/marmol/proyectos/equilibria/src/equilibria/sam_tools/mip_to_sam_transforms.py`

#### Transformación 1: `normalize_mip_accounts()`
**Propósito**: Convertir RAW labels a estructura inicial PEP (J, I, VA, FD)

**Antes**:
```
("RAW", "agr") → ("RAW", "agr") = 10
("RAW", "VA") → ("RAW", "agr") = 50
```

**Después**:
```
("I", "agr") → ("J", "agr") = 10     # Commodity → Sector
("VA", "aggregate") → ("J", "agr") = 50  # VA temporal
```

#### Transformación 2: `disaggregate_va_to_factors()`
**Propósito**: Dividir VA agregado en L y K usando shares

**Parámetros**:
- `va_factor_shares`: dict con {"L": float, "K": float} (deben sumar 1.0)

**Antes**:
```
("VA", "aggregate") → ("J", "agr") = 100
```

**Después**:
```
("L", "labor") → ("J", "agr") = 65   # 100 * 0.65
("K", "capital") → ("J", "agr") = 35 # 100 * 0.35
```

**Implementación**:
```python
def disaggregate_va_to_factors(sam: Sam, op: dict[str, Any]) -> dict[str, Any]:
    """Split VA row into L and K factors."""
    shares = op.get("va_factor_shares", {"L": 0.65, "K": 0.35})

    # Validar shares suman 1.0
    assert abs(sum(shares.values()) - 1.0) < 1e-6

    # Para cada sector J, obtener VA total
    # Distribuir según shares a L y K
    # Eliminar fila VA original

    return {"factors_created": ["L", "K"], "total_va": total}
```

#### Transformación 3: `create_factor_income_distribution()`
**Propósito**: Rutear ingreso de factores a instituciones (hogares, empresas, gobierno)

**Parámetros**:
- `factor_to_household_shares`: dict anidado de distribución

**Antes**:
```
("L", "labor") → ("J", "agr") = 65  # Solo flujo productivo
```

**Después**:
```
("L", "labor") → ("J", "agr") = 65           # Mantener costo productivo
("AG", "hh") → ("L", "labor") = 61.75        # 65 * 0.95 ingreso a hogares
("AG", "gvt") → ("L", "labor") = 3.25        # 65 * 0.05 impuestos directos
```

**Nota económica**: Los factores ahora tienen doble rol:
- **Columna J**: Costo de factores para producción
- **Fila AG**: Ingreso de factores a instituciones

#### Transformación 4: `create_household_expenditure()`
**Propósito**: Convertir demanda final "HH" de MIP en flujos AG.hh → I

**Antes**:
```
("I", "agr") → ("FD", "HH") = 20  # Demanda final temporal
```

**Después**:
```
("AG", "hh") → ("I", "agr") = 20  # Consumo de hogares
```

**Validación**: Verificar constraint presupuestario:
```
Ingreso_hogares = Consumo_hogares + Ahorro_hogares + Impuestos_directos
```

#### Transformación 5: `create_government_flows()`
**Propósito**: Consolidar flujos de gobierno (ingresos fiscales + gasto)

**Ingresos**:
- Impuestos indirectos sobre producción
- Aranceles sobre importaciones
- Impuestos directos (ya creados en transformación 3)

**Gastos**:
- Consumo de gobierno (de demanda final MIP)
- Transferencias (si aplicable)

**Después**:
```
("AG", "ti") → ("AG", "gvt") = X    # Impuestos indirectos
("AG", "tm") → ("AG", "gvt") = Y    # Aranceles
("AG", "gvt") → ("I", "ser") = Z    # Consumo gobierno
```

#### Transformación 6: `create_row_account()`
**Propósito**: Crear cuenta resto del mundo (ROW) para comercio exterior

**Antes**:
```
("RAW", "IMP") → ("I", "agr") = 15  # Importaciones
("I", "agr") → ("FD", "EXP") = 10   # Exportaciones
```

**Después**:
```
("AG", "row") → ("I", "agr") = 15   # Oferta de importaciones
("I", "agr") → ("AG", "row") = 10   # Demanda de exportaciones (luego va a X)
```

#### Transformación 7: `create_investment_account()`
**Propósito**: Crear cierre ahorro-inversión

**Después**:
```
("AG", "hh") → ("OTH", "inv") = savings_hh
("AG", "firm") → ("OTH", "inv") = savings_firm
("AG", "gvt") → ("OTH", "inv") = savings_gvt  # Si hay superávit
("I", "*") → ("OTH", "inv") = investment_demand
```

### 3. API Principal (`api.py`)

**Modificar archivo existente**: `/Users/marmol/proyectos/equilibria/src/equilibria/sam_tools/api.py`

Agregar:

```python
class MIPToSAMResult(NamedTuple):
    """Result object returned by run_mip_to_sam."""
    sam: Sam
    steps: list[dict[str, Any]]
    output_path: Path | None
    report_path: Path | None

def run_mip_to_sam(
    input_path: Path | str,
    *,
    # Parámetros de desagregación
    va_factor_shares: dict[str, float] | None = None,
    factor_to_household_shares: dict[str, dict[str, float]] | None = None,
    tax_rates: dict[str, float] | None = None,

    # Opciones de formato
    sheet_name: str = "MIP",
    va_row_label: str = "Valor Agregado",

    # Balanceo
    ras_type: str = "arithmetic",
    ras_tol: float = 1e-9,
    ras_max_iter: int = 200,

    # Output
    output_path: Path | str | None = None,
    report_path: Path | str | None = None,
) -> MIPToSAMResult:
    """
    Convert MIP to SAM compatible with PEP CGE models.

    Args:
        input_path: Path to MIP Excel file
        va_factor_shares: Factor shares of VA. Default: {"L": 0.65, "K": 0.35}
        factor_to_household_shares: Income distribution. Default: simple 1-household
        tax_rates: Tax rates. Default: {"production_tax": 0.10, ...}

    Returns:
        MIPToSAMResult with transformed SAM and transformation history

    Example:
        >>> result = run_mip_to_sam(
        ...     "data/mip_ecuador_2020.xlsx",
        ...     va_factor_shares={"L": 0.68, "K": 0.32},
        ...     output_path="output/sam_ecuador_2020.xlsx"
        ... )
    """
```

**Implementación** (siguiendo patrón de `run_ieem_to_pep()`):

```python
def run_mip_to_sam(...) -> MIPToSAMResult:
    # 1. Cargar MIP
    sam = MIPRawSAM.from_mip_excel(input_path, sheet_name=sheet_name)
    steps: list[dict[str, Any]] = []

    # 2. Helper para registrar pasos
    def record(step: str, details: dict[str, Any] | None = None) -> None:
        steps.append({
            "step": step,
            "details": details or {},
            "balance": _sam_balance_stats(sam),
        })

    # 3. Secuencia de transformaciones
    record("normalize_mip", normalize_mip_accounts(sam, {}))
    record("disaggregate_va", disaggregate_va_to_factors(sam, {"va_factor_shares": ...}))
    record("factor_income", create_factor_income_distribution(sam, {...}))
    record("household_expenditure", create_household_expenditure(sam, {}))
    record("government", create_government_flows(sam, {"tax_rates": ...}))
    record("row_account", create_row_account(sam, {}))
    record("investment", create_investment_account(sam, {}))

    # 4. Reutilizar transformaciones PEP existentes
    record("create_x_block", create_x_block_on_sam(sam))
    record("convert_exports", convert_exports_to_x_on_sam(sam))

    # 5. Balanceo final
    record("balance_ras", balance_sam_ras(sam, {...}))

    # 6. Exportar
    if output_path:
        export_sam(sam, output_path, output_format="excel")
    if report_path:
        # JSON con steps y configuración

    return MIPToSAMResult(sam, steps, output_path, report_path)
```

## Archivos Críticos a Modificar/Crear

### Nuevos archivos:
1. `/Users/marmol/proyectos/equilibria/src/equilibria/sam_tools/mip_raw_excel.py` - Parser MIP
2. `/Users/marmol/proyectos/equilibria/src/equilibria/sam_tools/mip_to_sam_transforms.py` - Transformaciones
3. `/Users/marmol/proyectos/equilibria/tests/sam_tools/test_mip_to_sam.py` - Tests
4. `/Users/marmol/proyectos/equilibria/docs/guides/mip_to_sam_guide_en.md` - Documentación

### Archivos a modificar:
1. `/Users/marmol/proyectos/equilibria/src/equilibria/sam_tools/api.py` - Agregar `run_mip_to_sam()`
2. `/Users/marmol/proyectos/equilibria/src/equilibria/sam_tools/__init__.py` - Exportar nueva API

### Archivos de referencia (no modificar, solo estudiar):
1. `/Users/marmol/proyectos/equilibria/src/equilibria/sam_tools/ieem_to_pep_transformations.py` - Patrón
2. `/Users/marmol/proyectos/equilibria/src/equilibria/sam_tools/ieem_raw_excel.py` - Patrón parser
3. `/Users/marmol/proyectos/equilibria/src/equilibria/sam_tools/sam_transforms.py` - Utilidades reutilizables
4. `/Users/marmol/proyectos/equilibria/src/equilibria/sam_tools/balancing.py` - RAS balancer

## Testing

### Test fixtures:
Crear MIP sintético de prueba (3 sectores):
```
tests/sam_tools/fixtures/simple_mip.xlsx
- 3 sectores: agr, ind, ser
- 3 commodities: agr, ind, ser
- Fila VA agregada
- Columnas demanda final: HH, GOV, INV, EXP
- Fila importaciones
```

### Unit tests (`test_mip_to_sam.py`):
```python
def test_normalize_mip_accounts():
    """Test RAW → (J, I, VA, FD) conversion"""

def test_disaggregate_va_to_factors():
    """Test VA → L + K with custom shares"""

def test_factor_shares_must_sum_to_one():
    """Validation test"""

def test_household_budget_constraint():
    """Income = Consumption + Savings + Taxes"""

def test_sam_row_col_balance():
    """Final SAM is balanced"""
```

### Integration test:
```python
def test_run_mip_to_sam_full_pipeline():
    """Test complete MIP→SAM conversion with defaults"""
    result = run_mip_to_sam(
        "tests/sam_tools/fixtures/simple_mip.xlsx",
        output_path=tmp_path / "sam_output.xlsx"
    )
    assert result.sam is not None
    assert len(result.steps) > 0
    assert result.output_path.exists()
```

## Verificación End-to-End

### Paso 1: Preparar MIP de prueba
```python
# Usar MIP sintético o real de cuentas nacionales
mip_path = "data/mip_test_3x3.xlsx"
```

### Paso 2: Ejecutar conversión
```python
from equilibria.sam_tools import run_mip_to_sam

result = run_mip_to_sam(
    mip_path,
    va_factor_shares={"L": 0.65, "K": 0.35},
    output_path="output/sam_test.xlsx",
    report_path="output/conversion_report.json"
)
```

### Paso 3: Validar resultado
```python
# 1. Verificar balance
sam = result.sam
assert sam.is_balanced(tol=1e-9)

# 2. Verificar cuentas creadas
assert any(cat == "L" for cat, _ in sam.row_keys)  # Labor exists
assert any(cat == "K" for cat, _ in sam.row_keys)  # Capital exists
assert any(elem == "hh" for _, elem in sam.row_keys)  # Household exists
assert any(elem == "gvt" for _, elem in sam.row_keys)  # Government exists

# 3. Verificar identidad contable
# GDP = C + I + G + (X-M)
# Debe ser consistente

# 4. Revisar reporte JSON
import json
report = json.loads(Path("output/conversion_report.json").read_text())
print(f"Steps executed: {len(report['steps'])}")
for step in report['steps']:
    print(f"  - {step['step']}: balance={step['balance']['max_row_col_abs_diff']}")
```

### Paso 4: Usar SAM en modelo PEP
```python
from equilibria.templates import PEP

# Cargar SAM generada
model = PEP.from_sam("output/sam_test.xlsx")

# Verificar calibración
model.calibrate()
assert model.is_calibrated()

# Simular shock de prueba
model.simulate(shocks={"tfp": {"agr": 1.10}})
```

## Extensibilidad Futura

La arquitectura está diseñada para soportar extensiones:

### Fase 2 (futura): Múltiples hogares
- Modificar `factor_to_household_shares` para soportar 4+ tipos de hogares
- Agregar parámetro `household_types: list[str]`
- Distribuir consumo según propensiones marginales

### Fase 3 (futura): Impuestos diferenciados
- Soportar `tax_rates` como dict por sector/commodity
- Ejemplo: `{"production_tax": {"agr": 0.05, "ind": 0.15}}`

### Fase 4 (futura): Márgenes comerciales
- Desagregar servicios de distribución
- Crear cuenta MARG explícita

## Referencias Técnicas

### Literatura para defaults:
- **Lofgren, Harris & Robinson (2002)**: "A Standard CGE Model in GAMS" - Factor shares típicos
- **SNA 2008 Chapter 26**: Metodología SAM oficial
- **Pyatt & Round (1985)**: "Social Accounting Matrices: A Basis for Planning"

### Código de referencia:
- **IEEM→PEP pipeline**: `/src/equilibria/sam_tools/ieem_to_pep_transformations.py`
- **Sam class**: `/src/equilibria/sam_tools/models.py`
- **Transformation utils**: `/src/equilibria/sam_tools/sam_transforms.py`

## Criterios de Éxito

La implementación será exitosa cuando:

1. ✓ Puede cargar MIP Excel estándar con VA agregado
2. ✓ Desagrega VA en factores L y K con shares configurables
3. ✓ Crea cuentas institucionales (hh, gvt, firm, row)
4. ✓ SAM resultante está balanceada (max_diff < 1e-9)
5. ✓ SAM es compatible con PEP (model.calibrate() funciona)
6. ✓ Documentación clara de datos requeridos vs opcionales
7. ✓ Tests cubren >85% del código
8. ✓ Guía de usuario con ejemplos completos
