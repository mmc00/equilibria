# Análisis MIP Bolivia: Balance y Conversión a SAM

**Fecha**: 2026-04-09
**Archivo**: `mip_bol_unbalanced.xlsx`
**Dimensiones**: 70 sectores × 76 cuentas

---

## Resumen Ejecutivo

La Matriz Insumo-Producto de Bolivia analizada presenta **desbalances significativos** que deben corregirse antes de su uso en modelos CGE. Sin embargo, tiene la **ventaja importante** de que el Valor Agregado ya está desagregado en sus componentes (trabajo y capital), lo que simplifica la conversión a SAM.

### Estado Actual

| Aspecto | Estado | Comentario |
|---------|--------|------------|
| **Balance contable** | ❌ Desbalanceada | Diferencia PIB: 1,513 (3.1%) |
| **Estructura básica** | ✅ Completa | 70 sectores + demanda final |
| **Valor Agregado** | ✅ Desagregado | Ya tiene L y K separados |
| **Importaciones** | ✅ Presente | Por sector individual |
| **Demanda Final** | ✅ Completa | C, G, I, ΔS, X |
| **Para SAM** | ⚠️ Requiere extensión | Falta estructura institucional |

---

## 1. Estructura de la MIP

### 1.1. Composición

**Sectores (70 sectores industriales)**:
- Etiquetados como `ind-01` a `ind-70`
- Matriz de flujos intermedios: 70×70

**Valor Agregado (3 componentes - VENTAJA)**:
```
Fila 141: Remuneraciones (trabajadores asalariados)      = 19,092.54
Fila 142: Excedente Bruto de Explotación                 = 29,275.35
Fila 143: Otros impuestos menos subsidios                =    246.21
          ────────────────────────────────────────────────────────
          TOTAL VA (≈ PIB)                                = 48,614.10
```

**Importaciones (filas 71-140)**:
- Una fila por cada sector (70 filas)
- Total importaciones: 7,635.71

**Demanda Final (columnas 71-75)**:
```
Col 71: Consumo Final de los Hogares                     = 32,346.51
Col 72: Consumo Final del Gobierno                       =  8,530.58
Col 73: Formación Bruta de Capital                       =  5,989.57
Col 74: Variación de Stock y Existencias                 =    396.57
Col 75: Exportaciones FOB                                = 10,500.07
        ────────────────────────────────────────────────────────
        TOTAL Demanda Final                               = 57,763.29
```

### 1.2. Ventajas de esta MIP

✅ **Valor Agregado ya desagregado**: A diferencia de MIPs estándar, ésta ya tiene:
- **L (Labor)**: Remuneraciones = 39.3% del VA
- **K (Capital)**: Excedente Bruto = 60.2% del VA
- **Impuestos netos**: 0.5% del VA

Esto elimina la necesidad de estimar factor shares y simplifica la conversión a SAM.

✅ **Demanda final completa**: Todas las categorías estándar presentes (C, G, I, ΔS, X)

✅ **Importaciones detalladas**: Por sector, permitiendo análisis de dependencia externa

---

## 2. Problemas de Balance (Por qué está desbalanceada)

### 2.1. Identidad Macroeconómica Fundamental

**Problema principal**: El PIB calculado por dos métodos no coincide:

```
PIB (enfoque producción) = Valor Agregado
                         = 48,614.10

PIB (enfoque gasto)      = C + G + I + ΔS + (X - M)
                         = 32,346.51 + 8,530.58 + 5,989.57 + 396.57 + (10,500.07 - 7,635.71)
                         = 50,127.58

DIFERENCIA = 1,513.48 (3.1% del PIB)
```

**Interpretación**: Esta diferencia indica errores de medición o datos faltantes. En una MIP balanceada, ambos métodos deben dar el **mismo resultado**.

### 2.2. Balance Oferta-Demanda por Producto

Para cada producto debe cumplirse:
```
Oferta Total = Demanda Total
(Producción doméstica + Importaciones) = (Consumo intermedio + Demanda final)
```

**Productos con mayor desbalance**:

| Producto | Oferta | Demanda | Desbalance | % Error |
|----------|--------|---------|------------|---------|
| ind-01 | 9,068.87 | 5,746.29 | +3,322.58 | +57.8% |
| ind-28 | 3,194.58 | 5,791.01 | -2,596.42 | -44.8% |
| ind-08 | 5,238.79 | 2,645.83 | +2,592.96 | +98.0% |
| ind-10 | 5,957.45 | 8,262.34 | -2,304.89 | -27.9% |
| ind-49 | 6,111.36 | 4,342.77 | +1,768.59 | +40.7% |

**Estadísticas de desbalance**:
- **Máximo desbalance**: 3,322.58 (producto ind-01)
- **Promedio desbalance**: 386.26
- **Productos afectados**: 70 (100%)

### 2.3. Posibles Causas del Desbalance

1. **Márgenes comerciales no contabilizados**: Diferencia entre precio básico y precio consumidor
2. **Impuestos sobre productos mal asignados**: Pueden estar en agregados pero no distribuidos por producto
3. **Errores de medición**: Encuestas diferentes para producción vs consumo
4. **Datos faltantes**: Algunos flujos no registrados
5. **Inconsistencia temporal**: Datos de diferentes años

---

## 3. Cómo Corregir la MIP (Hacerla Balanceada)

### 3.1. Método RAS (Recomendado)

**Qué es**: Algoritmo iterativo que ajusta proporcionalmente las celdas para que las sumas de filas y columnas coincidan, minimizando la distorsión de la estructura original.

**Implementación**:
```python
from equilibria.sam_tools import MIPRawSAM
from equilibria.sam_tools.balancing import RASBalancer

# Cargar MIP
mip = MIPRawSAM.from_mip_excel(
    "mip_bol_unbalanced.xlsx",
    sheet_name="mip"
)

# Aplicar RAS
balancer = RASBalancer()
mip_balanced = mip.balance_ras(
    ras_type="arithmetic",
    tolerance=1e-6,
    max_iterations=500
)

# Verificar balance
assert mip_balanced.is_balanced(tol=1e-6)

# Exportar
mip_balanced.to_excel("mip_bol_balanced.xlsx")
```

**Ventajas**:
- ✅ Preserva la estructura de la matriz
- ✅ Cambios mínimos a los datos originales
- ✅ Método estándar en la literatura

**Desventajas**:
- ⚠️ No corrige errores sistemáticos de medición
- ⚠️ Distribuye el error proporcionalmente (puede no ser realista)

### 3.2. Ajuste Manual (Más Trabajo, Más Preciso)

**Paso 1**: Identificar la fuente del desbalance
```
¿Faltan márgenes de comercio?
¿Impuestos mal asignados?
¿Datos de importación incorrectos?
```

**Paso 2**: Corregir datos específicos
- Agregar filas de márgenes si faltan
- Redistribuir impuestos según estructura tributaria real
- Validar importaciones con datos de aduanas

**Paso 3**: Aplicar RAS solo para discrepancias residuales pequeñas

### 3.3. Validación Post-Balance

Después de balancear, verificar:
```python
# 1. Identidad PIB
va_total = sum(remuneraciones + excedente + impuestos)
gdp_gasto = C + G + I + ΔS + (X - M)
assert abs(va_total - gdp_gasto) < 1.0

# 2. Balance oferta-demanda por producto
for producto in productos:
    oferta = produccion[producto] + importaciones[producto]
    demanda = uso_intermedio[producto] + demanda_final[producto]
    assert abs(oferta - demanda) < 0.1

# 3. Balance por sector
for sector in sectores:
    insumos = compras_intermedias[sector] + va[sector]
    produccion_total = ventas[sector]
    assert abs(insumos - produccion_total) < 0.1
```

---

## 4. Conversión a SAM (Después de Balancear)

### 4.1. Qué Tiene la MIP vs Qué Necesita una SAM

| Elemento | MIP Bolivia | SAM PEP | Acción Necesaria |
|----------|-------------|---------|------------------|
| **Flujos intermedios I×J** | ✅ Presente (70×70) | ✅ Requerido | Mantener |
| **Factores L y K** | ✅ Desagregado | ✅ Requerido | **VENTAJA: Ya está** |
| **Demanda final** | ✅ C, G, I, ΔS, X | ❌ Debe ser institucional | **Transformar** |
| **Importaciones** | ✅ Por sector | ✅ Cuenta ROW | **Transformar** |
| **Instituciones** | ❌ No existe | ✅ hh, gvt, firm, row | **CREAR** |
| **Distribución ingreso** | ❌ No existe | ✅ Factores → Instituciones | **CREAR** |
| **Flujos fiscales** | ⚠️ Agregados | ✅ ti, tm desagregados | **Desagregar** |
| **Ahorro-Inversión** | ❌ No existe | ✅ Cierre macro | **CREAR** |

### 4.2. Estructura de la SAM Objetivo

**Dimensión estimada**: ~90 cuentas (vs 76 en MIP)

**Cuentas nuevas a crear**:
```
FACTORES (2 - YA EXISTEN):
  ✅ L.labor              (de "Remuneraciones")
  ✅ K.capital            (de "Excedente Bruto")

INSTITUCIONES (6 - CREAR):
  ❌ AG.hh                (Hogares - de "Consumo Final Hogares")
  ❌ AG.gvt               (Gobierno - de "Consumo Final Gobierno")
  ❌ AG.firm              (Empresas - nueva)
  ❌ AG.row               (Resto del Mundo - de "Exportaciones" + "Importaciones")
  ❌ AG.ti                (Impuestos indirectos - de "Otros impuestos")
  ❌ AG.tm                (Aranceles - nueva)

OTRAS CUENTAS (2 - CREAR):
  ❌ OTH.inv              (Inversión - de "Form. Bruta Capital" + "Var. Stock")
  ✅ X.* (70 cuentas)     (Exportaciones por producto - crear desde "Exportaciones FOB")
```

### 4.3. Datos Externos Necesarios

A diferencia de MIPs estándar, como **el VA ya está desagregado**, solo necesitamos:

#### 1. Distribución de Ingresos Factoriales (**CRÍTICO**)

**Qué se necesita**: ¿Quién recibe el ingreso de L y K?

```python
factor_to_household_shares = {
    "L": {  # Remuneraciones (19,092.54)
        "hh": 0.95,      # 95% a hogares (18,137.91)
        "gvt": 0.05      # 5% impuestos directos (954.63)
    },
    "K": {  # Excedente Bruto (29,275.35)
        "hh": 0.40,      # 40% a hogares (11,710.14)
        "firm": 0.55,    # 55% empresas (16,101.44)
        "gvt": 0.05      # 5% impuestos capital (1,463.77)
    }
}
```

**Fuentes para Bolivia**:
- **Encuestas de hogares**: ENAHO (Encuesta Nacional de Hogares)
- **Cuentas nacionales**: Instituto Nacional de Estadística (INE)
- **Datos fiscales**: Servicio de Impuestos Nacionales (SIN)

**Default si no disponible**: Usar promedios de América Latina

#### 2. Tasas de Aranceles (para desagregar impuestos)

**Qué se necesita**: Separar "Otros impuestos" en componentes

```python
# La MIP tiene total de impuestos = 246.21
# Necesitamos dividirlo en:

impuestos_produccion = 246.21 * 0.60  # ~148 (impuestos indirectos)
aranceles = 246.21 * 0.40              # ~98 (aranceles sobre importaciones)
```

**Fuentes**:
- Aduana Nacional de Bolivia (ANB)
- Estadísticas de comercio exterior

**Default**: Arancel promedio = 5-10% de importaciones

#### 3. Propensión a Ahorrar (para cierre ahorro-inversión)

**Qué se necesita**: ¿Cuánto ahorra cada institución?

```python
savings_rates = {
    "hh": 0.15,      # Hogares ahorran 15% de ingreso disponible
    "firm": 0.60,    # Empresas retienen 60% de beneficios
    "gvt": 0.05      # Gobierno ahorra 5% (si hay superávit)
}
```

**Fuentes**:
- Banco Central de Bolivia (BCB)
- Cuentas nacionales financieras

**Default**: Calibrar residualmente para que Ahorro = Inversión

### 4.4. Pipeline de Conversión

**Opción A: Usar `run_mip_to_sam()` (Automático)**

```python
from equilibria.sam_tools import run_mip_to_sam

result = run_mip_to_sam(
    "mip_bol_balanced.xlsx",  # MIP ya balanceada

    # VENTAJA: No necesitamos va_factor_shares porque ya está desagregado
    # La función debe detectar que L y K ya existen

    # Solo necesitamos distribución de ingresos
    factor_to_household_shares={
        "L": {"hh": 0.95, "gvt": 0.05},
        "K": {"hh": 0.40, "firm": 0.55, "gvt": 0.05}
    },

    # Y tasas fiscales
    tax_rates={
        "import_tariff": 0.10,  # 10% arancel sobre importaciones
        # production_tax se calcula de los datos existentes
    },

    output_path="sam_bolivia_2020.xlsx",
    report_path="conversion_report.json"
)
```

**Opción B: Transformación Manual (Más Control)**

```python
from equilibria.sam_tools import MIPRawSAM
from equilibria.sam_tools.mip_to_sam_transforms import (
    normalize_mip_accounts,
    create_factor_income_distribution,
    create_household_expenditure,
    create_government_flows,
    create_row_account,
    create_investment_account
)
from equilibria.sam_tools.sam_transforms import (
    create_x_block_on_sam,
    convert_exports_to_x_on_sam
)

# Cargar MIP balanceada
sam = MIPRawSAM.from_mip_excel("mip_bol_balanced.xlsx")

# Paso 1: Normalizar (L y K ya existen, solo renombrar)
normalize_mip_accounts(sam, {})

# Paso 2: SKIP disaggregate_va_to_factors (ya está hecho!)

# Paso 3: Distribuir ingresos factoriales
create_factor_income_distribution(sam, {
    "factor_to_household_shares": {...}
})

# Paso 4: Crear consumo de hogares
create_household_expenditure(sam, {})

# Paso 5: Flujos de gobierno
create_government_flows(sam, {
    "tax_rates": {"import_tariff": 0.10}
})

# Paso 6: Cuenta resto del mundo
create_row_account(sam, {})

# Paso 7: Cierre ahorro-inversión
create_investment_account(sam, {})

# Paso 8: Bloque de exportaciones
create_x_block_on_sam(sam)
convert_exports_to_x_on_sam(sam)

# Exportar
sam.to_excel("sam_bolivia_2020.xlsx")
```

---

## 5. Plan de Acción Recomendado

### Fase 1: Balancear la MIP (1-2 días)

**Objetivo**: Corregir desbalances para tener MIP consistente

1. ✅ Analizar fuentes de desbalance (HECHO - este documento)
2. ⚠️ Investigar datos originales (¿hay errores conocidos?)
3. 🔧 Aplicar RAS para balancear
4. ✅ Validar identidades contables

**Criterio de éxito**:
- Diferencia PIB < 0.1%
- Max desbalance producto < 1.0

### Fase 2: Preparar Datos Externos (2-3 días)

**Objetivo**: Obtener parámetros para conversión a SAM

1. 📊 **Distribución de ingresos**:
   - Buscar ENAHO más reciente
   - Calcular shares L→hh/gvt y K→hh/firm/gvt
   - Default si no disponible: Usar promedios regionales

2. 💰 **Aranceles**:
   - Obtener de ANB (Aduana)
   - Calcular arancel efectivo promedio
   - Default: 8-10% sobre importaciones

3. 💵 **Propensiones a ahorrar**:
   - Datos BCB si disponible
   - Calibrar residualmente: S = I

### Fase 3: Conversión a SAM (1 día)

**Objetivo**: Ejecutar transformación MIP→SAM

1. 🔧 Ejecutar `run_mip_to_sam()` con parámetros
2. ✅ Validar SAM resultante
3. 📄 Documentar supuestos y fuentes

### Fase 4: Validación en Modelo PEP (1-2 días)

**Objetivo**: Verificar que SAM funciona en CGE

1. 🧪 Cargar SAM en PEP
2. ✅ Calibrar modelo
3. 📊 Simular shock de prueba
4. 🔍 Revisar resultados (¿son razonables?)

**Tiempo total estimado**: 5-8 días hábiles

---

## 6. Recomendaciones Específicas para Bolivia

### 6.1. Consideraciones Económicas

**Estructura productiva**:
- Bolivia es economía extractiva (gas, minerales)
- Alta dependencia de importaciones en bienes manufacturados
- Sector informal significativo

**Implicaciones para SAM**:
- ✅ Desagregación L/K apropiada (sector extractivo = alto K)
- ⚠️ Sector informal puede no estar bien capturado en MIP
- ⚠️ Subsidios a combustibles pueden afectar precios relativos

### 6.2. Fuentes de Datos Recomendadas

| Dato Necesario | Fuente Primaria | Fuente Alternativa |
|----------------|-----------------|-------------------|
| Distribución ingreso L | ENAHO (INE) | Cuentas Nacionales |
| Distribución ingreso K | SIN (declaraciones) | Estimación sectorial |
| Aranceles | ANB | Promedio LAC (8%) |
| Propensión ahorrar | BCB | Calibración |
| Márgenes comercio | Encuestas comercio | % estándar (15-20%) |

### 6.3. Validaciones Clave

Post-conversión, verificar:

1. **Shares factoriales razonables**:
   ```
   L/PIB ≈ 35-45% (Bolivia es intensiva en capital)
   K/PIB ≈ 55-65%
   ```

2. **Propensión a consumir**:
   ```
   C/PIB ≈ 65-70%
   G/PIB ≈ 15-20%
   I/PIB ≈ 15-20%
   ```

3. **Apertura económica**:
   ```
   (X+M)/PIB ≈ 60-80% (economía pequeña y abierta)
   ```

---

## 7. Ejemplo de Uso Completo

```python
# ============================================================================
# SCRIPT COMPLETO: MIP BOLIVIA → SAM PEP
# ============================================================================

from equilibria.sam_tools import MIPRawSAM, run_mip_to_sam
from equilibria.sam_tools.balancing import RASBalancer
from equilibria.templates import PEP

# ---------------------------------------------------------------------------
# PASO 1: BALANCEAR MIP
# ---------------------------------------------------------------------------

print("Paso 1: Balanceando MIP...")

# Cargar MIP desbalanceada
mip_raw = MIPRawSAM.from_mip_excel(
    "mip_bol_unbalanced.xlsx",
    sheet_name="mip",
    va_row_label="Remuneraciones"  # Primera fila de VA
)

# Balancear con RAS
mip_balanced = mip_raw.balance_ras(
    ras_type="arithmetic",
    tolerance=1e-6,
    max_iterations=500
)

# Guardar MIP balanceada
mip_balanced.to_excel("mip_bol_balanced.xlsx")

print(f"✓ MIP balanceada. Max diff: {mip_balanced.max_imbalance():.2e}")

# ---------------------------------------------------------------------------
# PASO 2: CONVERTIR A SAM
# ---------------------------------------------------------------------------

print("\nPaso 2: Convirtiendo MIP → SAM...")

# Parámetros basados en datos de Bolivia (ajustar según fuentes reales)
result = run_mip_to_sam(
    "mip_bol_balanced.xlsx",

    # Distribución de ingresos (de ENAHO o estimación)
    factor_to_household_shares={
        "L": {
            "hh": 0.93,   # 93% salarios a hogares
            "gvt": 0.07   # 7% impuestos sobre salarios
        },
        "K": {
            "hh": 0.35,   # 35% capital a hogares (rentas, dividendos)
            "firm": 0.60, # 60% retenido por empresas
            "gvt": 0.05   # 5% impuestos sobre capital
        }
    },

    # Tasas fiscales (de ANB y SIN)
    tax_rates={
        "import_tariff": 0.09,      # Arancel efectivo 9%
        "production_tax": 0.13,     # IVA efectivo 13%
    },

    # Opciones de balanceo
    ras_type="arithmetic",
    ras_max_iter=300,

    # Outputs
    output_path="sam_bolivia_2020.xlsx",
    report_path="sam_conversion_report.json"
)

print(f"✓ SAM creada: {result.output_path}")
print(f"  - Pasos ejecutados: {len(result.steps)}")
print(f"  - Balance final: {result.steps[-1]['balance']['max_row_col_abs_diff']:.2e}")

# ---------------------------------------------------------------------------
# PASO 3: VALIDAR EN MODELO PEP
# ---------------------------------------------------------------------------

print("\nPaso 3: Validando SAM en modelo PEP...")

# Cargar en modelo CGE
model = PEP.from_sam("sam_bolivia_2020.xlsx")

# Calibrar
model.calibrate()
assert model.is_calibrated(), "Modelo no calibró correctamente"
print("✓ Modelo calibrado exitosamente")

# Simular shock de prueba (10% aumento productividad sector gas)
baseline = model.solve()
shocked = model.simulate(
    shocks={"tfp": {"ind-08": 1.10}},  # ind-08 = gas natural
    name="Shock productividad gas"
)

# Resultados
gdp_change = (shocked["GDP"] / baseline["GDP"] - 1) * 100
print(f"✓ Simulación exitosa")
print(f"  - Cambio PIB: {gdp_change:.2f}%")
print(f"  - Cambio ingreso hogares: {(shocked['Y_hh']/baseline['Y_hh']-1)*100:.2f}%")

# ---------------------------------------------------------------------------
# PASO 4: EXPORTAR RESULTADOS
# ---------------------------------------------------------------------------

# Exportar para análisis
result.sam.to_excel("sam_bolivia_final.xlsx")
shocked.to_excel("resultados_simulacion.xlsx")

print("\n" + "="*70)
print("PROCESO COMPLETADO EXITOSAMENTE")
print("="*70)
print(f"Archivos generados:")
print(f"  - mip_bol_balanced.xlsx       (MIP balanceada)")
print(f"  - sam_bolivia_2020.xlsx       (SAM completa)")
print(f"  - sam_conversion_report.json  (Detalles conversión)")
print(f"  - resultados_simulacion.xlsx  (Resultados CGE)")
```

---

## 8. Referencias

### Metodología
- **Lofgren, H., Harris, R. L., & Robinson, S. (2002)**. "A Standard CGE Model in GAMS". IFPRI.
- **Miller, R. E., & Blair, P. D. (2009)**. "Input-Output Analysis: Foundations and Extensions". Cambridge University Press.
- **Stone, R. (1961)**. "Input-Output and National Accounts". OECD.

### Datos Bolivia
- **INE Bolivia**: Instituto Nacional de Estadística - https://www.ine.gob.bo/
- **BCB**: Banco Central de Bolivia - https://www.bcb.gob.bo/
- **SIN**: Servicio de Impuestos Nacionales
- **ANB**: Aduana Nacional de Bolivia

### Software
- Equilibria documentation: `/docs/guides/mip_to_sam_guide_en.md`
- RAS balancing: `/docs/technical/ras_algorithm.md`

---

## Apéndice: Comandos Útiles

### Explorar la MIP
```python
import pandas as pd
df = pd.read_excel("mip_bol_unbalanced.xlsx", sheet_name="mip", header=None)

# Ver estructura
print(df.shape)
print(df.iloc[:10, :10])

# Identificar filas especiales
for i, label in enumerate(df.iloc[:, 0]):
    if pd.notna(label) and any(x in str(label).lower() for x in ['remun', 'excedente', 'impuesto']):
        print(f"Fila {i}: {label}")
```

### Verificar Balance Rápido
```python
from equilibria.sam_tools import MIPRawSAM

mip = MIPRawSAM.from_mip_excel("mip_bol_unbalanced.xlsx", sheet_name="mip")

# Estadísticas de balance
balance_stats = mip.balance_statistics()
print(f"Max imbalance: {balance_stats['max_diff']:.2f}")
print(f"Balanced: {mip.is_balanced(tol=1.0)}")
```

### Exportar para Excel
```python
# Después de balancear
mip_balanced.to_excel(
    "mip_bol_balanced.xlsx",
    include_metadata=True,
    format="detailed"
)
```

---

**Documento generado**: 2026-04-09
**Herramienta**: Equilibria SAM Tools
**Contacto**: equilibria-project/equilibria
