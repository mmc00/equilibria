# Bolivia MIP - Análisis Final de Demanda Final

## Hallazgo Clave

✓✓✓ **DF Total coincide perfectamente con CN 2023** (diferencia -0.1%)

## Estructura de Demanda Final en el Excel

El archivo `Archivo matriz_BoliviaTodo_2023_final.xlsx` tiene **tres hojas separadas**:

### 1. DF total (nacional + importado)
Refleja el **total gastado** en cada componente, incluyendo bienes nacionales e importados.

### 2. DF nal (solo nacional)
Refleja solo la **parte nacional** del gasto (excluye componentes importados).

### 3. DF imp (componente importado)
Refleja la **parte importada** del gasto.

**Identidad verificada**: `DF total = DF nal + DF imp` ✓

---

## Comparación DF Total vs CN 2023

| Componente | CN 2023 (Bs) | DF Total (Bs) | Diferencia | % |
|------------|-------------:|--------------:|-----------:|--:|
| C_hh | 249,430 | 250,764 | -1,334 | **-0.5%** ✓✓✓ |
| C_gov | 60,953 | 60,953 | 0 | **0.0%** ✓✓✓ |
| FBKF | 64,045 | 66,427 | -2,382 | **-3.7%** ✓ |
| Var.S | 2,231 | 2,231 | 0 | **0.0%** ✓✓✓ |
| X | 81,024 | 77,749 | 3,275 | **4.0%** ✓ |
| **TOTAL** | **457,683** | **458,123** | **-440** | **-0.1%** ✓✓✓ |

**Conclusión**: DF Total refleja muy bien las Cuentas Nacionales 2023.

---

## Descomposición de Demanda Final (en USD, precios básicos)

### DF Nacional (cols 16-20, hoja 'DF nal')

| Componente | USD básicos |
|------------|------------:|
| C_hh | 32,347 |
| C_gov | 8,531 |
| FBKF | 5,990 |
| Var.S | 397 |
| X | 10,500 |
| **Total DF Nal** | **57,765** |

### DF Importado (cols 16-20, hoja 'DF imp')

| Componente | USD básicos |
|------------|------------:|
| C_hh | 4,471 |
| C_gov | 128 |
| FBKF | 3,309 |
| Var.S | -77 |
| X | 80 |
| **Total DF Imp** | **7,911** |

*Nota: DF Imp negativo en Var.S indica ajuste de inventarios de bienes importados.*

### Importaciones Totales

| Tipo | USD básicos |
|------|------------:|
| IMP intermedias (Z) | 7,636 |
| IMP finales (F) | 6,323 |
| **Total M** | **13,959** |

**Relación**: DF Imp (7,911) es **menor** que IMP finales (6,323) por diferencias de valoración (mercado vs básicos/frontera).

---

## Cálculo PIB (método del gasto)

```
PIB = C_nal + I_nal + G_nal + X - M_total

Donde:
  C_nal = 32,347  (Consumo hogares nacional)
  I_nal = 6,387   (FBKF + Var.S nacional)
  G_nal = 8,531   (Gasto gobierno nacional)
  X     = 10,500  (Exportaciones)
  M     = 13,959  (IMP_Z + IMP_F)

PIB (gasto, básicos) = 32,347 + 6,387 + 8,531 + 10,500 - 13,959
                     = 57,765 - 13,959
                     = 43,806 USD
```

**Alternativamente**:
```
PIB = DF_nal - M_total
    = 57,765 - 13,959
    = 43,806 USD (precios básicos)
```

---

## Comparación VAB vs PIB (gasto)

| Medida | Valor (USD) | Fuente |
|--------|------------:|--------|
| **VAB** (producción) | 48,614 | Suma de VA por sector en MIP |
| **PIB** (gasto) | 43,806 | DF_nal - M_total |
| **Discrepancia** | **4,808** | **9.9% error** |

### ¿Por Qué Hay Discrepancia?

**En una economía balanceada**:
```
PIB (producción) = PIB (gasto)
VA = C + I + G + (X - M)
```

**En la MIP Bolivia**:
```
VA = 48,614 USD
PIB (gasto) = 43,806 USD
Error = 4,808 USD (9.9%)
```

**Causas identificadas** (de análisis previo):

1. **Fuentes de datos mixtas**:
   - VA: MIP base 2017
   - DF: MIP base 2017 + **Cuentas Externas** (fuente adicional)
   - Diferentes metodologías de actualización a 2021

2. **Desbalances en consumo intermedio (Z)**:
   - Balance Z (max |row-col|) = 2,698 USD
   - Contribuye al error PIB

3. **Balance Oferta-Demanda**:
   - Balance O-D = -9,149 USD (demanda > oferta)
   - Error estructural que afecta identidad PIB

---

## Comparación con CN 2023 (PIB)

| Concepto | CN 2023 | MIP Excel | Diferencia |
|----------|--------:|----------:|-----------:|
| **PIB mercado** | 52,340 USD | - | - |
| **VAB básicos** | 48,614 USD | 48,614 USD | **0 USD** ✓ |
| **Impuestos sobre productos** | 3,726 USD | 0 USD | **3,726 USD** |
| **PIB gasto básicos** | ~48,614 USD | 43,806 USD | **4,808 USD** |

**Notas**:
- VAB en MIP **coincide exactamente** con VAB en CN 2023 ✓
- Falta agregar 3,726 USD de impuestos sobre productos para llegar a PIB mercado
- Hay 4,808 USD de discrepancia entre VAB y PIB(gasto) en la MIP

---

## Respuestas a las Dos Preguntas Originales

### 1. ¿DF es total o nacional en la MIP?

**Respuesta**: La MIP usa **DF nacional** (correcto ✓)

**Verificación**:
- Hoja 'DF nal' suma 57,765 USD (básicos)
- MIP `mip_bol_unbalanced.xlsx` F cols 70-74 suma 57,763 USD ✓
- Coincidencia perfecta: la MIP usa solo componente nacional

### 2. ¿Usa VAB en vez de PIB?

**Respuesta**: **SÍ** (correcto ✓)

**Verificación**:
- MIP tiene VA = 48,614 USD
- Esto es **VAB a precios básicos**
- Falta agregar 3,726 USD de impuestos sobre productos para PIB mercado
- PIB (mercado) = VAB + impuestos = 48,614 + 3,726 = 52,340 USD

---

## Impuestos sobre Productos

### Encontrados en Excel

**Hoja 'valor agregado'**:
- Impuestos sobre **producción**: 1,701 Bs = 246 USD
- Estos **YA están incluidos** en el VAB

**NO encontrados**:
- Impuestos sobre **productos**: 25,748 Bs = 3,726 USD
- No están desagregados por sector en el Excel
- Solo existen como agregado en CN 2023

### Componentes (de CN 2023)

```
Impuestos sobre productos netos = 25,748 Bs = 3,726 USD

Incluye:
  - IVA no deducible:     19,487 Bs = 2,820 USD
  - Otros impuestos netos: 6,261 Bs =   906 USD
```

**Para conversión a PIB mercado**:
```
PIB (mercado) = VAB + Impuestos sobre productos
52,340 USD = 48,614 USD + 3,726 USD
```

---

## Estructura Correcta de la MIP

La MIP está construida con:

1. **Flujos intermedios**: Z = CI nacional (básicos)
2. **Demanda final**: F = DF nacional (básicos)
3. **Importaciones**: IMP_Z, IMP_F separados (básicos/frontera)
4. **Valor agregado**: VA = VAB (básicos)

**Precios**:
- Todos los flujos a **precios básicos** (sin impuestos sobre productos)
- Para PIB completo, necesitas agregar impuestos sobre productos

**Separación nacional/importado**:
- ✓ Correcta en todas las hojas (DF total = DF nal + DF imp)
- ✓ Usado correctamente en la MIP (F = DF nal, no DF total)

---

## Discrepancia PIB: 4,808 USD (9.9%)

### Identidad que debería cumplirse:
```
VAB = PIB (gasto) en economía balanceada
48,614 = 43,806 ❌ (diferencia 4,808)
```

### Origen del error:

**Del análisis de balance previo**:
- Balance Oferta-Demanda = -9,149 USD
- Balance Z interno = 2,698 USD
- Error PIB (VA vs gasto) = 4,808 USD

**Fuentes de datos mixtas**:
- VA: MIP base 2017
- DF: MIP base 2017 + Cuentas Externas
- Diferentes actualizaciones a valores corrientes 2021

### Solución aplicada:

**Balanceo Hybrid** (de análisis previo):
- RAS geométrico para Z
- Ajuste mínimo de F para cerrar PIB
- Resultado: PIB error = 0.0000% ✓

**Archivo balanceado**: `mip_bol_balanced_hybrid.xlsx`

---

## Recomendación para MIP→SAM

### 1. Usar MIP balanceada
```python
run_mip_to_sam(
    input_path="mip_bol_balanced_hybrid.xlsx",
    # PIB error = 0%, listo para conversión
)
```

### 2. Agregar impuestos sobre productos (opcional)
```python
run_mip_to_sam(
    input_path="mip_bol_balanced_hybrid.xlsx",
    product_tax_rates={...},  # Si necesitas PIB mercado
    # Agrega 3,726 USD para llegar a PIB = 52,340
)
```

### 3. Decisión de diseño

Hacer `product_tax_rates` **OPCIONAL**:

- **Si None** (default): Trabaja con VAB básicos (48,614 USD)
- **Si se provee**: Agrega impuestos y alcanza PIB mercado (52,340 USD)

**Justificación**:
- No todos los CGE requieren impuestos explícitos
- No todos los países tienen impuestos desagregados
- Permite flexibilidad según datos disponibles

---

## Conclusión Final

### ✅ Datos Correctos

1. **DF Total = DF Nal + DF Imp** ✓ (verificado)
2. **DF Total ≈ CN 2023** ✓ (diferencia -0.1%)
3. **MIP usa DF Nal** ✓ (correcto, excluye importados)
4. **VAB = VAB CN 2023** ✓ (coincidencia exacta)

### ⚠️ Discrepancias

1. **VAB ≠ PIB(gasto)**: 4,808 USD (9.9%)
   - Causa: Fuentes de datos mixtas, desbalances en Z
   - Solución: Balanceo Hybrid aplicado

2. **Faltan impuestos sobre productos**: 3,726 USD
   - No están en Excel desagregados
   - Necesarios solo si modelo usa precios mercado

### 🎯 Para Pipeline MIP→SAM

**Usar**:
- `mip_bol_balanced_hybrid.xlsx` (PIB error = 0%)
- Parámetro `product_tax_rates` opcional
- Si None → VAB básicos (48,614 USD)
- Si se provee → PIB mercado (52,340 USD)

---

**Fecha**: Abril 2025
**Archivos analizados**:
- `Archivo matriz_BoliviaTodo_2023_final.xlsx` (construcción)
- `mip_bol_unbalanced.xlsx` (original)
- `mip_bol_balanced_hybrid.xlsx` (balanceada)

**Estado**: ✅ Análisis completo verificado
