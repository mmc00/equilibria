# MIP Bolivia: Problema con Targets y Solución

## Problema Identificado

Los vectores de targets (objetivos) provistos tienen una inconsistencia fundamental:

```
Row target total:    149,867.17
Column target total: 151,380.65
Diferencia:           1,513.48 (1.0%)
```

### ¿Por qué es un problema?

Una matriz balanceada DEBE satisfacer:
```
Σ(row sums) = Σ(column sums) = Grand Total
```

Si los targets de filas y columnas suman valores diferentes, es matemáticamente imposible lograr ambos simultáneamente.

## Diagnosis de los Targets

### Estructura de Targets de Filas (143 elementos)
- 70 productos (ind-01 a ind-70)
- 70 importaciones (importaciones ind-01 a ind-70)
- 3 componentes de VA (Remuneraciones, Excedente Bruto, Impuestos)

### Estructura de Targets de Columnas (75 elementos)
- 70 sectores (ind-01 a ind-70)
- 5 componentes de demanda final (HH, GOV, INV, Stock, EXP)

### Interpretación Económica

Los targets parecen representar:

**Filas**: Oferta total por producto
- Producción doméstica + Importaciones = Targets de productos
- VA targets = Valor agregado por componente

**Columnas**: Uso total
- Por sector: Insumos intermedios + VA
- Por demanda final: Consumo, inversión, exportaciones

## Soluciones Posibles

### Opción 1: Balancear solo el cuadro de uso intermedio (70×70)

**Ventaja**: Compatible con targets de sectores
**Desventaja**: No balancea demanda final ni importaciones

```python
# Balance solo la matriz Z (intermedia)
Z_balanced = balance_ras(Z, row_targets[:70], col_targets[:70])
```

### Opción 2: Ajustar targets para que sean consistentes

Hay dos sub-opciones:

#### 2a. Usar promedio como grand total
```python
grand_total = (row_targets.sum() + col_targets.sum()) / 2  # 150,623.91
scale_row = grand_total / row_targets.sum()
scale_col = grand_total / col_targets.sum()

row_targets_adjusted = row_targets * scale_row
col_targets_adjusted = col_targets * scale_col
```

#### 2b. Mantener VA fijo y ajustar demanda final
```python
# Opción más realista económicamente
# 1. Fijar VA (más confiable)
# 2. Ajustar demanda final para cerrar la brecha
```

### Opción 3: Balancear con discrepancia estadística explícita

**Método de Cuentas Nacionales estándar**: Crear cuenta "discrepancia estadística"

```
PIB (gasto) = C + G + I + ΔS + (X - M) + DISCREPANCY
```

La diferencia de 1,513 se asigna explícitamente como error estadístico.

## Recomendación para Bolivia

### Paso 1: Verificar origen de targets

**Preguntas críticas**:
1. ¿De dónde vienen estos targets? ¿Son:
   - Calculados desde otra fuente (Cuentas Nacionales oficiales)?
   - Derivados de la propia MIP?
   - Benchmarks externos?

2. ¿Qué representa exactamente cada target?
   - Targets de filas para productos: ¿son oferta total (prod + imp)?
   - Targets de columnas: ¿son demanda total (intermedia + final)?

### Paso 2: Estrategia de balance recomendada

Si no podemos modificar los targets, usar enfoque **híbrido**:

1. **Preservar componentes más confiables**:
   ```python
   # Fijar:
   VA_targets = row_targets[140:143]  # VA es más confiable
   Export_targets = col_targets[74]    # Exportaciones de aduanas
   ```

2. **Balance en dos etapas**:

   **Etapa 1**: Balancear cuadro intermedio (70×70)
   ```python
   # RAS sobre Z con targets de productos y sectores
   Z_balanced = balance_ras(
       Z,
       row_targets=product_supply_targets[:70],
       col_targets=sector_cost_targets[:70] - VA_targets.sum()/70
   )
   ```

   **Etapa 2**: Ajustar demanda final residualmente
   ```python
   # Para cada producto i:
   # F[i] = (prod_target[i] + import_target[i]) - Z_balanced[i,:].sum()
   ```

3. **Validar identidad PIB**:
   ```python
   PIB_VA = VA_targets.sum()
   PIB_gasto = F.sum() + exports - imports
   discrepancy = PIB_VA - PIB_gasto

   if abs(discrepancy) < 0.01 * PIB_VA:  # < 1%
       # Aceptable
   else:
       # Crear cuenta de discrepancia estadística
   ```

## Código de Implementación

```python
import numpy as np
import pandas as pd

def balance_mip_with_inconsistent_targets(
    M: np.ndarray,
    row_targets: np.ndarray,
    col_targets: np.ndarray,
    *,
    n_products: int = 70,
    n_sectors: int = 70,
    va_row_idx: list[int],
    method: str = "hierarchical"
) -> np.ndarray:
    """
    Balance MIP when row and column targets are inconsistent.

    Args:
        M: MIP matrix (143 × 75)
        row_targets: Row targets (143,) - may be inconsistent with col
        col_targets: Column targets (75,) - may be inconsistent with row
        method: 'hierarchical' (recommended) or 'average'

    Returns:
        Balanced MIP matrix
    """

    if method == "hierarchical":
        # Preserve VA (most reliable)
        VA = M[va_row_idx, :n_sectors].copy()

        # Step 1: Balance intermediate flows (70×70)
        Z = M[:n_products, :n_sectors]

        # Adjust targets for intermediate flows
        Z_row_targets = row_targets[:n_products]
        Z_col_targets = col_targets[:n_sectors] - VA.sum(axis=0)

        # Standard RAS on Z
        Z_balanced = ras_balance(Z, Z_row_targets, Z_col_targets)

        # Step 2: Adjust final demand residually
        # Supply = Production + Imports
        # Use = Intermediate + Final
        # => Final = Supply - Intermediate

        for i in range(n_products):
            supply_target = row_targets[i] + row_targets[n_products + i]  # prod + imp
            intermediate_use = Z_balanced[i, :].sum()
            fd_target = supply_target - intermediate_use

            # Distribute across FD categories proportionally
            fd_current = M[i, n_sectors:]
            if fd_current.sum() > 0:
                M[i, n_sectors:] = fd_current * (fd_target / fd_current.sum())

        M[:n_products, :n_sectors] = Z_balanced
        M[va_row_idx, :n_sectors] = VA  # Restore VA

    elif method == "average":
        # Use average of row and column target sums
        grand_total = (row_targets.sum() + col_targets.sum()) / 2

        row_scale = grand_total / row_targets.sum()
        col_scale = grand_total / col_targets.sum()

        row_targets_adj = row_targets * row_scale
        col_targets_adj = col_targets * col_scale

        # Standard GRAS with adjusted targets
        M = gras_balance(M, row_targets_adj, col_targets_adj)

    return M
```

## Siguiente Paso

**Antes de continuar con el balanceo**, necesitamos clarificar:

1. ¿Cuál es la fuente/metodología de cálculo de los targets?
2. ¿Es aceptable la discrepancia de 1,513 (1.0%)?
3. ¿Qué componentes son "fijos" (no se pueden ajustar)?

Una vez clarificado, podemos implementar el método de balance apropiado.

## Referencias

- **SNA 2008 Chapter 26**: Manejo de discrepancias estadísticas
- **IMF GFS Manual 2014**: Reconciliación de flujos y stocks
- **Eurostat Manual on Supply-Use Tables**: Balanceo con targets inconsistentes
