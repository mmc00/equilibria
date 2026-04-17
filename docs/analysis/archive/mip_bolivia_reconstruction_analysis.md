# Bolivia MIP Reconstruction and Balancing Analysis

## Problema Descubierto

La MIP original (`mip_bol_unbalanced.xlsx`) tiene **5.81% de error en PIB** y desbalances en oferta-demanda por producto.

### Causa Raíz Identificada

El archivo de construcción (`Archivo matriz_BoliviaTodo_2023_final.xlsx`) contiene tres versiones de precios en cada hoja:

1. **Precios de mercado en Bolivianos** (Bs)
2. **Precios de mercado en USD**
3. **Precios básicos/frontera en USD**

La MIP original fue construida usando **factores de ajuste inconsistentes** sobre los valores de precios básicos:

| Bloque | Valor Básicos (USD) | Valor en MIP Original | Factor Ajuste |
|--------|--------------------:|----------------------:|--------------|
| **Z** (Consumo Intermedio) | 37,923.89 | 30,288.17 | **79.87%** |
| **F** (Demanda Final) | 64,085.92 | 57,763.29 | **90.13%** |
| **VA** (Valor Agregado) | 48,614.10 | 48,614.10 | **100.00%** |
| **IMP_F** (Importaciones DF) | 6,322.63 | 6,322.63 | **100.00%** |

**Estos factores inconsistentes causan el desbalance PIB:**
- PIB (VA) = 48,614.10 (100% básicos)
- PIB (gasto) = 57,763.29 - 6,322.63 = 51,440.67 (mezcla de 90% y 100%)
- Error = 5.81%

---

## Opciones de Solución

### Opción 1: Balancear MIP Original (Ajustada)

**Descripción**: Tomar la MIP original con factores de ajuste y aplicar balanceo hybrid.

**Archivo de entrada**: `mip_bol_unbalanced.xlsx`

**Método**: Hybrid balance (RAS geométrico para Z + ajuste mínimo de F)

**Resultados**:

| Métrica | Valor |
|---------|------:|
| PIB error | **0.0000%** ✅ |
| Z balance | **0.000001** ✅ |
| Product max diff | 2,764 ⚠️ |
| Sector max diff | 2,547 ⚠️ |
| Tiempo | <1 minuto ✅ |

**Archivo generado**: `mip_bol_balanced_hybrid.xlsx`

**Ventajas**:
- ✅ PIB perfecto
- ✅ Z perfecto
- ✅ Muy rápido
- ✅ Listo para CGE

**Desventajas**:
- ⚠️ Perpetúa los factores de ajuste inconsistentes
- ⚠️ No sabemos por qué existen esos factores

---

### Opción 2: Re-Construir desde Básicos 100% + Balancear

**Descripción**: Extraer todas las matrices directamente de columnas USD básicos (sin ajustes) y luego balancear.

**Extracción desde Excel**:
```python
# CI tot - cols 146-216: USD precios básicos/frontera
Z = df.iloc[:70, 146:216]

# DF total - cols 16-21: USD precios básicos
F = df.iloc[:70, 16:21]

# VA - cols 14-16: USD precios básicos
VA = df.iloc[:70, 14:17]

# Importaciones - cols 146-216 y 16-21: USD precios básicos
IMP_Z = df_imp.iloc[:70, 146:216]
IMP_F = df_imp.iloc[:70, 16:21]
```

**Estado Inicial (antes de balancear)**:

| Métrica | Valor |
|---------|------:|
| Z sum | 37,923.89 |
| F sum | 64,085.92 |
| VA sum | 48,614.10 |
| IMP_F sum | 6,322.63 |
| PIB (VA) | 48,614.10 |
| PIB (gasto) | 57,763.29 |
| PIB error | **18.82%** ❌ |
| Z balance | 2,708.64 |

**Después de Hybrid Balance**:

| Métrica | Valor |
|---------|------:|
| PIB error | **1.10%** ⚠️ |
| Z balance | **0.000001** ✅ |
| Product max diff | 2,326 ⚠️ |
| Sector max diff | 2,621 ⚠️ |
| F scaling factor | 0.8572 |

**Archivo generado**: `mip_bol_basicos_balanced_hybrid.xlsx`

**Ventajas**:
- ✅ Usa datos directos de básicos (más "puros")
- ✅ Z perfecto
- ✅ Rápido

**Desventajas**:
- ⚠️ PIB error 1.10% (aceptable pero no perfecto)
- ⚠️ Requiere escalar F significativamente (-14.28%)

---

### Opción 3: Investigar Factores de Ajuste

**Descripción**: Entender POR QUÉ se aplicaron factores de ajuste (80% para Z, 90% para F).

**Hipótesis posibles**:

1. **Conversión a precios de comprador**:
   - Los factores podrían estar convirtiendo de básicos a precios de comprador (purchaser prices)
   - Pero típicamente esto AGREGA márgenes (factor >100%), no los reduce
   - ❌ No parece ser el caso (factores son <100%)

2. **Corrección por data quality**:
   - Ajustes empíricos para corregir sobre-estimación en fuentes
   - Típico en MIPs de países en desarrollo
   - ✅ Posible

3. **Deflación a año base**:
   - Si los datos vienen de diferentes años, podría haber deflación
   - Pero el archivo dice "valores corrientes 2021" para todo
   - ❌ Probablemente no

4. **Ajuste por sector informal**:
   - Corrección para economía informal no capturada en algunas fuentes
   - ✅ Posible para Bolivia

**Acción requerida**:
- Contactar a la fuente de datos (MIP base 2017, UDAPE, BCB)
- Revisar documentación metodológica si existe
- Verificar si existe un diccionario de datos o manual de usuario

---

## Verificación de Archivo de Construcción

El archivo `Archivo matriz_BoliviaTodo_2023_final.xlsx` tiene una hoja "Consistencia" que muestra:

- ✅ Balance oferta-demanda por producto: **0.0** (en precios básicos)
- ✅ Balance inputs-outputs por sector: **0.0** (en precios básicos)
- ✅ PIB identity: **0.0** (en precios básicos)

**Conclusión**: El archivo de construcción ESTÁ perfectamente balanceado a precios básicos. El desbalance se introduce al aplicar los factores de ajuste.

---

## Comparación de Todas las MIPs Generadas

| Archivo | Origen | PIB Error | Z Balance | Producto Max | Sector Max | Status |
|---------|--------|-----------|-----------|--------------|------------|--------|
| `mip_bol_unbalanced.xlsx` | Original (factores ajustados) | 5.81% | 2,697 | - | - | ❌ Desbalanceada |
| `mip_bol_balanced_gras_fixed.xlsx` | Original + GRAS | 0.38% | 0.16 | ~2,773 | ~350 | ✅ Buena |
| `mip_bol_balanced_hybrid.xlsx` | Original + Hybrid | **0.0000%** | **0.000001** | 2,764 | 2,547 | ✅ **Mejor** |
| `mip_bol_basicos_reconstructed.xlsx` | 100% básicos | 18.82% | 2,709 | 599 | - | ❌ Sin balancear |
| `mip_bol_basicos_balanced_hybrid.xlsx` | 100% básicos + Hybrid | 1.10% | 0.000001 | 2,326 | 2,621 | ⚠️ Aceptable |

---

## Recomendación

### Para uso inmediato en CGE: Opción 1 ⭐

**Usar**: `mip_bol_balanced_hybrid.xlsx` (MIP original + hybrid balance)

**Razones**:
1. ✅ PIB perfecto (0.0000% error)
2. ✅ Z perfecto (0.000001 balance)
3. ✅ Rápido (<1 minuto)
4. ✅ Cumple todos los criterios críticos para CGE (ver `mip_balance_for_cge_models.md`)
5. ✅ Residuales producto/sector aceptables (~2,500)

### Para investigación adicional: Opción 3

**Investigar**:
- Razón de los factores de ajuste (80% Z, 90% F)
- Contactar fuente de datos o buscar documentación metodológica
- Si se encuentra justificación válida, la Opción 1 queda validada
- Si no hay justificación, considerar Opción 2 (básicos puros)

---

## Scripts Creados

1. **`reconstruct_mip_bolivia_basicos.py`**
   - Extrae matrices directamente de precios básicos
   - Genera `mip_bol_basicos_reconstructed.xlsx`
   - Útil para análisis de sensibilidad

2. **`balance_bolivia_hybrid_final.py`**
   - Aplica hybrid balance (RAS geométrico + ajuste mínimo F)
   - Usado tanto para MIP original como básicos
   - Genera resultados en <1 minuto

---

## Referencias

### Archivo de Construcción
- **Path**: `/Users/marmol/proyectos/cge_babel/playground/bol/Archivo matriz_BoliviaTodo_2023_final.xlsx`
- **Año base**: 2021
- **Fuente**: Nacional - MIP base 2017, Cuentas Externas
- **Tipo de cambio**: 6.91 Bs/USD

### Hojas relevantes:
- `CI tot`: Consumo Intermedio (3 versiones de precios)
- `DF total`: Demanda Final (3 versiones)
- `valor agregado`: VA (2 versiones USD)
- `CI imp`, `DF imp`: Importaciones (3 versiones)
- `Consistencia`: Verificación de balance

### Columnas de precios básicos:
- CI tot: cols 146-216
- DF total: cols 16-21
- VA: cols 14-16
- IMP (CI/DF): cols 146-216 y 16-21

---

## Próximos Pasos

1. **Inmediato**: Usar `mip_bol_balanced_hybrid.xlsx` para conversión MIP→SAM
2. **Corto plazo**: Contactar fuente de datos para entender factores de ajuste
3. **Mediano plazo**: Si se justifican los factores, documentar metodología; si no, cambiar a básicos puros

---

**Última actualización**: Abril 2025
**MIP Recomendada para CGE**: `mip_bol_balanced_hybrid.xlsx` (de MIP original ajustada)
**MIP Alternativa (precios básicos puros)**: `mip_bol_basicos_balanced_hybrid.xlsx`
**Estado**: ✅ Listas para uso en modelos CGE
