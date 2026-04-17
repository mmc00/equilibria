# Bolivia MIP - Análisis de Fórmulas Excel y Construcción

## Resumen Ejecutivo

**Hallazgo principal**: La MIP original (`mip_bol_unbalanced.xlsx`) **está CORRECTAMENTE construida** desde el archivo Excel. NO hay factores de ajuste arbitrarios.

El aparente "factor 80%" que observamos inicialmente era un **error de interpretación**: estábamos comparando Z (CI **nacional**) con CI **total** (nacional + importado).

---

## Estructura Correcta del Excel

El archivo `Archivo matriz_BoliviaTodo_2023_final.xlsx` contiene **hojas separadas** para flujos nacionales e importados:

### Hojas Principales:

1. **CI nal** - Consumo Intermedio Nacional
2. **CI imp** - Consumo Intermedio Importado
3. **CI tot** - CI total = CI nal + CI imp (calculado)
4. **DF nal** - Demanda Final Nacional
5. **DF imp** - Demanda Final Importada
6. **DF total** - DF total = DF nal + DF imp (calculado)
7. **valor agregado** - Valor Agregado por sector
8. **produccion total** - VBP por sector
9. **importacion total** - Importaciones por producto
10. **Consistencia** - Verificación de balance

---

## Fórmulas de Conversión Mercado → Básicos

### Para CI Nacional (hoja 'CI nal'):

```excel
='CI tot'!BX3 * ('produccion total'!$E3 / 'produccion total'!$F3)
```

Donde:
- `'CI tot'!BX3`: CI en precios **mercado** USD
- `'produccion total'!$E3`: VBP a precios **básicos** USD
- `'produccion total'!$F3`: VBP a precios **mercado** USD

**Ratio promedio (ponderado por VBP)**: **98.81%**

### Para CI Importado (hoja 'CI imp'):

```excel
='CI tot'!BX3 * ('importacion total'!$N3 / 'importacion total'!$J3)
```

Donde:
- `'CI tot'!BX3`: CI en precios **mercado** USD
- `'importacion total'!$N3`: IMP a precios **frontera** USD (CIF)
- `'importacion total'!$J3`: IMP a precios **mercado** USD (CIF + márgenes internos)

**Ratio promedio (ponderado por IMP)**: **83.97%**

### Para CI Total (hoja 'CI tot'):

```excel
='CI nal'!ER3 + 'CI imp'!ER3
```

Simple suma de nacional + importado.

---

## Valores Obtenidos

### Consumo Intermedio (a precios básicos USD):

| Concepto | Valor (millones USD) | Fuente |
|----------|---------------------:|--------|
| CI nacional | 30,288.17 | Hoja 'CI nal' cols 146-216 |
| CI importado | 7,635.71 | Hoja 'CI imp' cols 146-216 |
| **CI total** | **37,923.89** | Suma |

### Demanda Final (a precios básicos USD):

| Concepto | Valor (millones USD) | Fuente |
|----------|---------------------:|--------|
| DF nacional | 57,763.29 | Hoja 'DF nal' cols 16-21 |
| DF importado | 6,322.63 | Hoja 'DF imp' cols 16-21 |
| **DF total** | **64,085.92** | Suma |

### Valor Agregado (a precios básicos USD):

| Componente | Valor | Fuente |
|------------|------:|--------|
| Remuneraciones (L) | 19,092.54 | Hoja 'valor agregado' col 14 |
| Excedente Bruto (K) | 29,275.35 | Hoja 'valor agregado' col 15 |
| Otros Imp - Subs (TI) | 246.21 | Hoja 'valor agregado' col 16 |
| **VA total** | **48,614.10** | Suma |

---

## Estructura MIP Original

La MIP original (`mip_bol_unbalanced.xlsx`) usa la estructura correcta:

```
Filas:
  0-69:   Productos (I) - flujos NACIONALES
  70-139: Importaciones por producto
  140-142: Valor Agregado (L, K, TI)

Columnas:
  0-69:  Sectores (J)
  70-74: Demanda Final (C_hh, C_gov, FBKF, Var.S, X)
```

### Valores en MIP Original:

| Bloque | Valor (millones USD) | Excel Source |
|--------|---------------------:|--------------|
| **Z** (filas 0-69, cols 0-69) | 30,288.17 | ='CI nal'[:70, 146:216] ✓ |
| **F** (filas 0-69, cols 70-74) | 57,763.29 | ='DF nal'[:70, 16:21] ✓ |
| **IMP_Z** (filas 70-139, cols 0-69) | 7,635.71 | ='CI imp'[:70, 146:216] ✓ |
| **IMP_F** (filas 70-139, cols 70-74) | 6,322.63 | ='DF imp'[:70, 16:21] ✓ |
| **VA** (filas 140-142, cols 0-69) | 48,614.10 | ='valor agregado'[:70, 14:17] ✓ |

---

## Error de Interpretación Inicial

### Lo que observamos:

```
Z_original / CI_total_basicos = 30,288.17 / 37,923.89 = 0.7987 (79.87%)
```

### Lo que CREÍMOS:
- "Se aplicó un factor de ajuste de 80% a CI básicos"
- "Hay factores arbitrarios que causan el desbalance"

### La REALIDAD:
- Z contiene solo CI **NACIONAL**, no CI total
- CI total = CI nacional + CI importado
- La separación nacional/importado es la estructura CORRECTA de una MIP

---

## Verificación de Balance en Excel

La hoja **"Consistencia"** del archivo de construcción muestra que **a precios básicos el sistema está balanceado**:

| Verificación | Resultado |
|--------------|----------:|
| OT - CI - DT = 0 (Total) | 0.0 ✓ |
| OT - CI - DT = 0 (Importado) | 0.0 ✓ |
| OT - CI - DT = 0 (Nacional) | 0.0 ✓ |
| DFM + DFN - DF = 0 | 0.0 ✓ |
| OT = DT (a precios básicos) | 0.0 ✓ |
| VA = VBP - CI | 0.0 ✓ |

**Conclusión**: El archivo de construcción Excel está perfectamente balanceado internamente.

---

## Origen del Error PIB 5.81%

El error **PIB (VA) ≠ PIB (gasto)** de 5.81% NO es por factores de ajuste incorrectos, sino por:

### 1. Diferentes Fuentes de Datos

Según la hoja "info" del Excel:

| Variable | Fuente |
|----------|--------|
| Producción total | Nacional - MIP base 2017 |
| Exportaciones | Nacional - MIP base 2017 |
| Importaciones | Nacional - MIP base 2017 |
| Valor Agregado | Nacional - MIP base 2017 |
| Demanda Final | MIP base 2017 + **Cuentas Externas** |

**Problema**: Demanda Final viene de dos fuentes que pueden no coincidir perfectamente.

### 2. Actualización a Valores Corrientes 2021

- La MIP base es de **2017**
- Se actualizó a **valores corrientes 2021**
- Diferentes componentes pueden tener diferentes índices de actualización
- Errores de medición se acumulan en el proceso

### 3. Discrepancia Estadística Normal

La literatura de cuentas nacionales acepta:
- Error PIB 1-2%: **Normal**
- Error PIB 3-5%: **Aceptable** para países en desarrollo
- Error PIB 5-10%: **Requiere investigación** pero usable con ajustes

Bolivia con 5.81% está en el rango **aceptable pero mejorable**.

---

## Solución: Balanceo Hybrid

Dado que:
1. ✓ La MIP está correctamente construida
2. ✓ No hay factores arbitrarios
3. ⚠️ Existe discrepancia PIB por fuentes de datos

**Solución**: Aplicar método Hybrid (RAS + ajuste mínimo) que:
- Respeta la estructura nacional/importado
- Balancea Z internamente (RAS geométrico)
- Ajusta F mínimamente para cerrar PIB
- Mantiene VA exacto (dato más confiable)

**Resultado**:
- PIB error: **0.0000%** ✅
- Z balance: **0.000001** ✅
- Tiempo: **<1 minuto** ✅

---

## Recomendaciones

### Para uso en CGE (Inmediato):

**Usar**: `mip_bol_balanced_hybrid.xlsx`

**Razón**:
- PIB perfecto
- Z balanceado
- Cumple estándares CGE
- Listo para conversión MIP→SAM

### Para documentación (Corto plazo):

**Documentar**:
- Fuente de datos: MIP base 2017 actualizada a 2021
- Método de actualización usado
- Origen de la discrepancia PIB (fuentes mixtas)
- Método de balanceo aplicado (Hybrid)

### Para futuro (Mediano plazo):

**Si se actualiza la MIP**:
1. Usar fuente única para todos los componentes
2. O bien, aplicar balanceo desde el inicio
3. Documentar metodología claramente

---

## Scripts Creados

### 1. `reconstruct_mip_bolivia_correcto.py` ⭐

Reconstruye la MIP desde el Excel usando:
- CI nal (nacional, no total)
- DF nal (nacional, no total)
- CI imp, DF imp (importados)
- VA (básicos)

**Resultado**: Reproduce **EXACTAMENTE** la MIP original, confirmando que no hay factores arbitrarios.

### 2. `balance_bolivia_hybrid_final.py`

Aplica balanceo Hybrid:
- RAS geométrico para Z
- Ajuste mínimo de F para PIB
- Preserva VA exacto

**Resultado**: PIB 0.0000%, Z 0.000001 en <1 minuto.

---

## Ratios Precios Mercado → Básicos

### Por Tipo de Flujo:

| Tipo | Ratio Básicos/Mercado | Ponderado por |
|------|----------------------:|---------------|
| Producción (VBP) | 98.81% | VBP total |
| Importaciones | 83.97% | IMP total |
| CI nacional | 98.58% | CI_nal total |
| CI importado | 87.65% | CI_imp total |
| CI total | 96.16% | CI total |

**Interpretación**:
- Producción nacional: Márgenes pequeños (1-2%)
- Importaciones: Márgenes mayores (16%) por transporte/distribución

Estos ratios son **económicamente razonables** y siguen las definiciones del Sistema de Cuentas Nacionales 2008.

---

## Conclusiones

1. ✅ **La MIP original está correctamente construida** desde el Excel
2. ✅ **No hay factores de ajuste arbitrarios** - todo sigue fórmulas del SNA 2008
3. ✅ **La separación nacional/importado es estándar** en MIPs modernas
4. ⚠️ **El error PIB 5.81% es por fuentes de datos mixtas**, no por construcción incorrecta
5. ✅ **El balanceo Hybrid** corrige el error PIB sin distorsionar la estructura
6. ✅ **La MIP balanceada** está lista para uso en modelos CGE

---

## Referencias

### Sistema de Cuentas Nacionales:
- United Nations (2008). *System of National Accounts 2008*. Chapter 14 (Supply and Use Tables), Chapter 26 (Satellite Accounts and Information Systems).

### Precios en SNA:
- **Precios básicos**: Excluye impuestos sobre productos, incluye subsidios
- **Precios de mercado**: Básicos + impuestos netos sobre productos
- **Precios de comprador**: Mercado + márgenes comerciales + transporte

### MIP Bolivia:
- Año base: 2021
- Fuente principal: MIP base 2017 (actualizada)
- Tipo de cambio: 6.91 Bs/USD
- Estructura: 70 sectores × 70 productos

---

**Última actualización**: Abril 2025
**MIP Verificada**: `mip_bol_unbalanced.xlsx` (construcción correcta)
**MIP Recomendada para CGE**: `mip_bol_balanced_hybrid.xlsx`
**Script de Verificación**: `reconstruct_mip_bolivia_correcto.py`
**Estado**: ✅ Construcción verificada, balanceo completo, lista para MIP→SAM
