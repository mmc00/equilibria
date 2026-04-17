# Bolivia MIP - Diagnóstico de Error Estructural

## Resumen Ejecutivo

**Error previo**: Creía que el error PIB era 5.81% calculando `PIB_gasto = F - IMP_F`

**Error real**: El error PIB es **3.11%** con la identidad correcta `PIB_gasto = F - IMP_Z`

**Problema principal**: Desbalance Oferta-Demanda de **-9,149.19 millones USD** (la demanda excede a la oferta)

---

## Identidad PIB Correcta

### ❌ INCORRECTO (lo que usaba antes):
```
PIB (gasto) = F - IMP_F
= 57,763.29 - 6,322.63
= 51,440.67

Error vs VA: 5.81%
```

### ✅ CORRECTO:
```
PIB (gasto) = (F + IMP_F) - (IMP_Z + IMP_F)
            = F - IMP_Z
            = 57,763.29 - 7,635.71
            = 50,127.58

Error vs VA: 3.11%
```

**Explicación**:
- **F** = Demanda Final NACIONAL (C_nal + I_nal + G_nal + X)
- **IMP_F** = Componente importado de DF (C_imp + I_imp + G_imp)
- **IMP_Z** = Importaciones usadas en Consumo Intermedio

En cuentas nacionales:
```
PIB = C + I + G + X - M_total

Donde:
  C + I + G = F + IMP_F (componentes nacionales + importados)
  X = Exportaciones (ya en F)
  M_total = IMP_Z + IMP_F (importaciones intermedias + finales)

Por lo tanto:
  PIB = (F + IMP_F) + X - (IMP_Z + IMP_F)
      = F - IMP_Z
```

---

## Balance Oferta-Demanda

### Ecuación por producto i:

```
OFERTA_i = DEMANDA_i

Donde:
  OFERTA_i = Producción_i + Importaciones_i
           = VBP_i + (IMP_Z_i + IMP_F_i)

  DEMANDA_i = Uso_Intermedio_i + Demanda_Final_i
            = (Z_i + IMP_Z_i) + (F_i + IMP_F_i)
```

### Valores agregados:

| Concepto | Valor (millones USD) |
|----------|---------------------:|
| **OFERTA** ||
| Producción (VBP) | 78,902.28 |
| Importaciones (IMP_Z + IMP_F) | 13,958.34 |
| **Total Oferta** | **92,860.61** |
| | |
| **DEMANDA** | |
| Uso Intermedio (Z + IMP_Z) | 37,923.89 |
| Demanda Final (F + IMP_F) | 64,085.92 |
| **Total Demanda** | **102,009.81** |
| ||
| **BALANCE** | **-9,149.19** ❌ |

**Interpretación**: La demanda excede a la oferta en 9,149 millones USD.

---

## Desbalances por Producto

Productos con mayor desbalance (Oferta - Demanda):

| Producto | Oferta | Demanda | Diferencia |
|----------|-------:|--------:|-----------:|
| 45 | 4,714 | 6,207 | **-1,494** |
| 49 | 3,298 | 4,584 | **-1,286** |
| 1 | 7,784 | 8,710 | **-926** |
| 61 | 4,660 | 5,515 | **-855** |
| 8 | 4,274 | 4,793 | **-519** |

Estadísticas:
- Max diff: 1,494
- Mean |diff|: 134.66
- Productos con deficit (demanda > oferta): La mayoría

---

## ¿De Dónde Viene el Error?

### 1. Error PIB (-1,513.48)

Indica que `F` (demanda final nacional) está **sub-estimada** o `IMP_Z` está **sobre-estimada** relativo al VA.

Posibles causas:
- F proviene de fuentes mixtas (MIP 2017 + Cuentas Externas)
- IMP_Z puede incluir márgenes o ajustes
- VA es el dato más confiable (directo de cuentas nacionales)

### 2. Desbalance Oferta-Demanda (-9,149.19)

Mucho mayor que el error PIB. Indica problemas estructurales:

**Posibilidad A: VBP mal calculado**
- VBP = Z.sum(axis=0) + VA.sum(axis=0) = 78,902.28
- ¿Es correcto asumir MIP cuadrada (producción_i = VBP_sector_i)?

**Posibilidad B: Demanda sobre-estimada**
- F + IMP_F = 64,085.92 parece muy alto
- Puede haber doble conteo entre F e IMP_F

**Posibilidad C: Importaciones mal asignadas**
- IMP_Z + IMP_F = 13,958.34
- ¿Están correctamente clasificadas entre Z y F?

---

## Verificación en Excel

Revisando las hojas del Excel de construcción:

### Hoja "Consistencia":

La hoja muestra que **a precios básicos todo está balanceado** (todos los checks = 0.0).

Esto sugiere que el problema NO está en el Excel sino en cómo se **extrajo** o **interpretó** la MIP.

### Posible error de extracción:

¿La MIP original (`mip_bol_unbalanced.xlsx`) fue extraída correctamente desde el Excel?

Déjame verificar:
- ¿Z incluye SOLO nacional o nacional + importado?
- ¿F incluye SOLO nacional o nacional + importado?
- ¿IMP_Z e IMP_F están en las filas correctas?

---

## Hipótesis Principal

**El problema NO está en las fórmulas del Excel** (que están correctas según verificamos).

**El problema PUEDE estar en**:
1. Cómo se ensambló la MIP desde las hojas del Excel
2. Qué columnas/filas se tomaron para cada bloque
3. Si se están mezclando flujos "totales" vs "nacionales"

**Acción requerida**:
- Verificar EXACTAMENTE qué celdas del Excel se usaron para crear `mip_bol_unbalanced.xlsx`
- Comparar con los valores en las hojas del Excel
- Identificar discrepancias

---

## Comparación: MIP Unbalanced vs Excel

| Bloque | MIP Unbalanced | Excel (reconstrucción) | Match? |
|--------|---------------:|-----------------------:|:------:|
| Z | 30,288.17 | 30,288.17 (CI nal básicos) | ✅ |
| F | 57,763.29 | 57,763.29 (DF nal básicos) | ✅ |
| IMP_Z | 7,635.71 | 7,635.71 (CI imp básicos) | ✅ |
| IMP_F | 6,322.63 | 6,322.63 (DF imp básicos) | ✅ |
| VA | 48,614.10 | 48,614.10 (VA básicos) | ✅ |

**Conclusión**: La MIP fue extraída correctamente del Excel.

**Entonces el problema está en el EXCEL original**, no en la extracción.

---

## Análisis del Excel - Hoja "Consistencia"

La hoja "Consistencia" muestra balance 0.0, pero:

**Pregunta crítica**: ¿Qué versión de precios verifica?

Si verifica solo precios de mercado en Bs, pero nosotros usamos precios básicos en USD, podría haber discrepancias.

**Acción**: Verificar en qué columnas del Excel se hacen las verificaciones de consistencia.

---

## Hipótesis Alternativa: MIP No Cuadrada

¿Y si la MIP NO es cuadrada?

En una MIP **no cuadrada**:
- Productos ≠ Sectores
- Un sector puede producir múltiples productos
- Necesitamos una **matriz MAKE** (Producción por sector×producto)

Si la MIP es no cuadrada:
```
OFERTA_producto_i ≠ VBP_sector_i
```

Necesitaríamos:
```
OFERTA_i = sum_j(MAKE[i,j])  # Producción del producto i por todos los sectores
```

**Verificar**: ¿Existe una matriz MAKE en el Excel?

---

## Conclusión Provisional

**Errores identificados**:
1. ✅ Error PIB real: **3.11%** (corregido de 5.81%)
2. ❌ Desbalance Oferta-Demanda: **-9,149.19 millones USD**

**Próximos pasos**:
1. Verificar si MIP es cuadrada o no cuadrada
2. Buscar matriz MAKE si existe
3. Revisar hoja "Consistencia" para entender qué verifica exactamente
4. Analizar si hay doble conteo entre F e IMP_F

**Status**: ⚠️ **Problema estructural identificado - requiere investigación adicional**
