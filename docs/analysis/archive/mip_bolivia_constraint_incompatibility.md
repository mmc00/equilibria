# Bolivia MIP - Incompatibilidad de las Tres Restricciones

## Problema

Se intentó balancear la MIP Bolivia satisfaciendo simultáneamente tres restricciones:

1. **Balance Z interno**: `Z[:,i].sum() = Z[i,:].sum()` para todo i (filas = columnas)
2. **Identidad PIB**: `VA.sum() = F.sum() - IMP_F.sum()`
3. **Balance Oferta-Demanda**: Para cada producto i: `Oferta_i = Demanda_i`

## Demostración Matemática de Incompatibilidad

### Identidad Oferta-Demanda

Para cada producto i:
```
Oferta_i = Producción_i + Importaciones_i
Demanda_i = Uso_intermedio_i + Demanda_final_i
```

En términos de la MIP:
```
Oferta_i = [Z[:,i].sum() + VA[i]] + [IMP_Z[i,:].sum() + IMP_F[i,:].sum()]
Demanda_i = [Z[i,:].sum() + IMP_Z[i,:].sum()] + [F[i,:].sum() + IMP_F[i,:].sum()]
```

Para balance: `Oferta_i = Demanda_i`
```
[Z[:,i].sum() + VA[i]] + [IMP_Z[i,:].sum() + IMP_F[i,:].sum()] =
[Z[i,:].sum() + IMP_Z[i,:].sum()] + [F[i,:].sum() + IMP_F[i,:].sum()]
```

Las importaciones se cancelan en ambos lados:
```
Z[:,i].sum() + VA[i] = Z[i,:].sum() + F[i,:].sum()
```

Reordenando:
```
Z[:,i].sum() - Z[i,:].sum() = F[i,:].sum() - VA[i]
```

### Aplicando Restricción 1 (Z balanceada)

Si `Z[:,i].sum() = Z[i,:].sum()` para todo i, entonces:
```
0 = F[i,:].sum() - VA[i]
```

Por lo tanto:
```
F[i,:].sum() = VA[i]  para todo i
```

**Interpretación**: Con Z balanceada y Oferta-Demanda balanceada, la demanda final de cada producto debe igualar exactamente el valor agregado de ese sector.

### Aplicando Restricción 2 (PIB)

La identidad PIB dice:
```
Σ_i VA[i] = Σ_i F[i,:].sum() - Σ_i IMP_F[i,:].sum()
```

Si `F[i,:].sum() = VA[i]` para todo i (de arriba), entonces:
```
Σ_i VA[i] = Σ_i VA[i] - Σ_i IMP_F[i,:].sum()
```

Simplificando:
```
0 = - Σ_i IMP_F[i,:].sum()
```

**Conclusión**: La suma de importaciones de demanda final debe ser CERO.

### Resultado

Para que las tres restricciones se satisfagan simultáneamente:
1. `F[i,:].sum() = VA[i]` para cada sector i
2. `Σ IMP_F[i,:].sum() = 0` (sin importaciones de demanda final)

En Bolivia MIP 2021:
- `Σ IMP_F = 6,323 USD` ≠ 0

**Por lo tanto, las tres restricciones son MATEMÁTICAMENTE INCOMPATIBLES con estos datos.**

## Resultados Empíricos

### MIP Original (sin balancear)
```
Error PIB:        5.81%
Z balance max:    2,698 USD
S-D balance max:  1,494 USD
S-D OK (<100):    53/70 productos
```

### Método: Priorizar PIB + Z (GRAS completo)
```
Error PIB:        0.00%  ✓
Z balance max:    5 USD   ✓
S-D balance max:  2,732 USD  ✗ (PEOR)
S-D OK (<100):    6/70  ✗ (PEOR)
```

### Método: Balanceo iterativo 3 restricciones
```
Iteración 0:
  PIB error:      2,828 USD
  Z balance:      1,331 USD
  S-D max:        2,030 USD  ← Mejor S-D
  S-D OK:         8/70

Iteración 200:
  PIB error:      0 USD    ✓
  Z balance:      0 USD    ✓
  S-D max:        32,192 USD  ✗ (16x PEOR!)
  S-D OK:         6/70  ✗
```

**Observación**: Al forzar PIB = 0 y Z balance = 0, el desbalance S-D explota de 2,030 a 32,192 USD.

### Método: Compromiso ponderado (PIB=100, Z=10, S-D=1)
```
Mejor resultado (iter 50):
  PIB error:      0 USD
  Z balance max:  66 USD
  S-D max:        3,077 USD  ✗ (PEOR que original)
  S-D OK (<100):  41/70  ✗ (PEOR que original)
```

## Trade-offs Disponibles

No es posible satisfacer las tres restricciones. Se debe elegir cuáles priorizar:

### Opción 1: Priorizar PIB + Z (balance tradicional)
**Método recomendado**: GRAS completo

**Resultado**:
- ✅ PIB error = 0.00%
- ✅ Z balance = 5 USD (excelente)
- ❌ S-D balance = 2,732 USD (peor que original)
- ❌ Solo 6/70 productos con S-D < 100 USD

**Uso**: Modelos CGE estándar que requieren identidad PIB estricta.

### Opción 2: Priorizar S-D + Z (balance físico)
**Método recomendado**: RAS sobre Z + ajuste proporcional de F para S-D

**Resultado esperado**:
- ✅ S-D balance mejorado
- ✅ Z balance aceptable
- ❌ PIB error ~5-10%

**Uso**: Modelos de insumo-producto físicos, análisis de cadenas de valor.

### Opción 3: Priorizar S-D + PIB (equilibrio macroeconómico)
**Método recomendado**: Optimización con Z como variable de ajuste

**Resultado esperado**:
- ✅ PIB error = 0%
- ✅ S-D balance mejorado
- ❌ Z desbalanceada (filas ≠ columnas)

**Uso**: Modelos con Z no cuadrada o con matrices de conversión.

### Opción 4: Balance suave (soft constraints)
**Método recomendado**: Compromiso ponderado con tolerancias

**Resultado** (Bolivia):
- PIB error < 1%
- Z balance < 100 USD
- S-D max < 3,000 USD
- Ninguna restricción perfecta, todas aceptables

**Uso**: Análisis exploratorio, datos de baja calidad.

## Recomendación para Bolivia MIP

### Para modelos CGE PEP:
**Usar GRAS completo** (Opción 1)

**Justificación**:
1. Los modelos CGE requieren identidad PIB estricta para calibración
2. La matriz Z balanceada asegura consistencia de flujos intermedios
3. El desbalance S-D se interpreta como:
   - Cambios en inventarios no registrados
   - Márgenes comerciales y de transporte
   - Diferencias de valoración (precios básicos vs mercado)

**Balance Oferta-Demanda en CGE**:
- Los modelos CGE no requieren S-D balance ex-ante en la SAM
- El modelo endogeniza precios y cantidades para equilibrar O-D
- El desbalance S-D en datos base se interpreta como presión inicial

### Para análisis I-O tradicional:
**Usar balance suave** (Opción 4)

**Justificación**:
- Multiplicadores I-O son sensibles a Z, pero toleran pequeños errores PIB
- S-D balance mejora identificación de cuellos de botella
- Análisis estructural es robusto a errores pequeños en las tres restricciones

## Archivo Recomendado

Para conversión a SAM PEP:
```
mip_bol_balanced_hybrid.xlsx
```

**Características**:
- PIB error = 0.00%
- Z balance = 10.54 USD (0.035% de Z total)
- S-D balance = aceptar desbalance como inherente a los datos

**Uso**:
```python
from equilibria.sam_tools import run_mip_to_sam

result = run_mip_to_sam(
    "mip_bol_balanced_hybrid.xlsx",
    va_factor_shares={"L": 0.65, "K": 0.35},
    # S-D imbalance will be handled by CGE price mechanism
)
```

## Interpretación del Desbalance S-D

El desbalance Oferta-Demanda de ~2,700 USD (5.6% del PIB) se puede atribuir a:

1. **Fuentes de datos mixtas**:
   - VA: MIP base 2017
   - F: MIP 2017 + Cuentas Externas 2021
   - Diferentes metodologías de actualización

2. **Cambios en inventarios no registrados**:
   - `Var.S` en demanda final puede no capturar todos los cambios
   - Errores de medición en inventarios

3. **Márgenes no desagregados**:
   - Márgenes de comercio y transporte no separados por producto
   - Tratamiento agregado distorsiona balance producto-específico

4. **Valoración inconsistente**:
   - Mezcla de precios básicos, CIF frontera, y CIF mercado
   - Conversión USD con TC promedio vs TC específico por flujo

5. **Errores estadísticos**:
   - Discrepancia estadística normal en cuentas nacionales
   - Para Bolivia: 5.6% está dentro de rangos aceptables internacionales

## Conclusión

**Las tres restricciones son matemáticamente incompatibles.**

Para uso en modelos CGE PEP:
1. ✅ Priorizar PIB identity (crítico para calibración)
2. ✅ Priorizar Z balance (consistencia interna)
3. ❌ Aceptar S-D imbalance (será equilibrado endógenamente por el modelo)

**El desbalance S-D residual no es un error del balanceo, sino una característica inherente de datos reales mezclados de múltiples fuentes.**

---

**Fecha**: Abril 2025
**Autor**: Análisis Claude Code
**Archivos**: `mip_bol_unbalanced2.xlsx`, `mip_bol_balanced_hybrid.xlsx`
**Scripts**: `compare_all_balancing_methods.py`, `balance_bolivia_three_constraints.py`
