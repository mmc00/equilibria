# Bolivia MIP - Análisis de Impuestos y VAB vs PIB

## Pregunta Original

¿La MIP tiene dos problemas?
1. La DF es total (nacional + importado) en vez de solo nacional?
2. Usa VAB en vez de PIB?

¿Son necesarios IVA no deducible y otros impuestos para la MIP?

---

## Respuestas

### 1. DF es total o nacional? ✅ CORRECTO

**Respuesta**: La DF en la MIP (`mip_bol_unbalanced.xlsx`) **ES nacional**, no total.

**Verificación**:
- MIP F sum = 57,763.29 USD
- Excel hoja 'DF nal' cols 16-21 sum = 57,763.29 USD ✅ Match perfecto
- Excel hoja 'DF total' = 64,085.92 USD (incluye importados)
- Excel hoja 'DF imp' = 6,322.63 USD

**Conclusión**: Este aspecto está **correcto**. La MIP usa solo demanda final nacional.

---

### 2. Usa VAB en vez de PIB? ✅ CORRECTO

**Respuesta**: **SÍ**, la MIP usa VAB (Valor Agregado Bruto) a precios básicos, **NO** PIB.

**Cuentas Nacionales oficiales (2023)**:
| Concepto | Bs | USD (TC=6.91) |
|----------|----:|---------------:|
| PIB (a precios de mercado) | 361,671 | 52,340.23 |
| VAB (a precios básicos) | 335,923 | 48,614.04 |
| **Diferencia** | **25,748** | **3,726.19** |

**MIP actual**:
- VA en MIP = 48,614.10 USD
- Esto corresponde al **VAB**, no al PIB
- Falta agregar 3,726 USD de impuestos sobre productos netos

---

## Impuestos Encontrados en el Excel

### A. Impuestos sobre la PRODUCCIÓN (Ya incluidos en VAB)

**Ubicación**: Hoja `valor agregado`, cols 3-4, 16

| Concepto | Bs | USD |
|----------|----:|-----:|
| Impuestos sobre producción e importaciones | 3,912.83 | 566.18 |
| Subsidios | 2,211.54 | 320.05 |
| **Neto (Otros impuestos menos subsidios)** | **1,701.29** | **246.21** |

**Nota**: Estos impuestos **YA están incluidos en el VAB**. Son impuestos sobre el proceso de producción, no sobre los productos finales.

### B. Impuestos sobre PRODUCTOS (Faltan en la MIP)

**Ubicación**: **NO están en el Excel** desagregados por sector.

Solo existen como agregado macro en Cuentas Nacionales:

| Componente | Bs | USD |
|------------|----:|-----:|
| IVA no deducible | 19,487 | 2,820 |
| Otros impuestos netos sobre productos | 6,261 | 906 |
| **Total** | **25,748** | **3,726** |

**Explicación**:
- **Impuestos sobre producción**: Aplicados durante el proceso productivo (ej: impuestos a la nómina, patentes). Se incluyen en VAB.
- **Impuestos sobre productos**: Aplicados sobre bienes/servicios finales (ej: IVA, impuestos específicos). Se suman al VAB para obtener PIB.

---

## ¿Son Necesarios para la MIP?

### Respuesta Corta

**Depende del uso que le vayas a dar:**

1. **Para CGE a precios básicos**: **NO** son necesarios
   - La MIP actual está completa y correcta
   - Muchos modelos CGE trabajan solo a precios básicos

2. **Para CGE a precios de mercado**: **SÍ** son necesarios
   - Necesitas agregar los 3,726 USD de impuestos sobre productos
   - La SAM debe incluir cuenta de impuestos indirectos

3. **Para SAM PEP estándar**: **SÍ**, probablemente
   - Los modelos PEP típicamente requieren impuestos sobre productos explícitos
   - Ver documentación del modelo PEP específico

### Respuesta Larga

#### Precios Básicos vs Precios de Mercado

La MIP actual usa **precios básicos**:

```
Precios básicos:
- Excluyen impuestos sobre productos
- Excluyen márgenes comerciales y de transporte
- Incluyen subsidios sobre productos
- Usado en tablas de oferta-utilización (SUT)

PIB a precios básicos = VAB

Precios de mercado:
- Incluyen impuestos sobre productos
- Incluyen márgenes comerciales
- Usado para PIB final

PIB a precios de mercado = VAB + Impuestos netos sobre productos
```

#### Identidades de Cuentas Nacionales

```
PIB (precios mercado) = VAB + Impuestos sobre productos - Subsidios sobre productos

Para Bolivia 2021:
52,340 USD = 48,614 USD + 3,726 USD
```

#### ¿Qué Hace Cada Tipo de Modelo?

**Modelos CGE a precios básicos** (ej: GTAP, algunos modelos del World Bank):
- Trabajan con VAB directamente
- Impuestos sobre productos se modelan como wedges entre precios básicos y precios al consumidor
- **La MIP actual es suficiente**

**Modelos CGE a precios de mercado** (ej: muchos modelos PEP):
- Requieren PIB completo
- Necesitan cuentas explícitas de:
  - Impuestos indirectos (IVA, impuestos específicos)
  - Aranceles de importación
  - Subsidios
- **Necesitan agregar los 3,726 USD**

---

## ¿Cómo Agregar Impuestos sobre Productos?

### Problema

Los 3,726 USD de impuestos sobre productos **NO están desagregados por sector** en el Excel.

Solo tenemos el agregado macroeconómico de Cuentas Nacionales.

### Opciones de Solución

#### Opción 1: Usar Tasa Efectiva Uniforme (Simple)

```python
# Total impuestos sobre productos = 3,726 USD
# Total demanda final nacional = 57,763 USD
# Tasa efectiva = 3,726 / 57,763 = 6.45%

# Aplicar uniformemente a toda la demanda final
tax_rate = 0.0645
product_taxes_by_sector = F * tax_rate
```

**Ventajas**: Simple, conserva proporciones de demanda
**Desventajas**: No refleja tasas reales diferenciadas por producto

#### Opción 2: Estimar Tasas por Producto (Recomendado)

Usar información adicional:

1. **IVA**: Tasa general 13% en Bolivia, pero:
   - Algunos productos exentos (salud, educación)
   - Tasa efectiva ~40-60% del total recaudado

2. **Impuestos específicos**:
   - Bebidas alcohólicas: ~10-20%
   - Tabaco: ~15-25%
   - Combustibles: variable

3. **Datos de recaudación tributaria**:
   - Servicio de Impuestos Nacionales (SIN)
   - Desagregación de IVA por sector económico

**Implementación**:
```python
# Definir tasas efectivas por producto
effective_rates = {
    "ind-01": 0.03,  # Agricultura (IVA reducido/exento)
    "ind-10": 0.08,  # Alimentos procesados
    "ind-14": 0.15,  # Bebidas (impuestos específicos)
    # ... etc
}

# Aplicar
for i, sector in enumerate(sectors):
    rate = effective_rates.get(sector, 0.0645)  # Default 6.45%
    product_taxes[i] = F[i] * rate
```

#### Opción 3: RAS Balancing con PIB Target (Híbrido)

```python
# Balancear la MIP para que:
# 1. VA_ajustado + Impuestos = PIB = 52,340
# 2. Se conserven proporciones relativas
# 3. Se agregue fila de impuestos sobre productos

# Implementado en balance_bolivia_hybrid_final.py
```

---

## Estructura SAM con Impuestos sobre Productos

Para convertir a SAM PEP completa:

```
Cuentas adicionales requeridas:

1. AG.ti - Impuestos indirectos (IVA, impuestos específicos)
   Ingreso: De commodities I
   Gasto: A gobierno (AG.gvt)

2. AG.tm - Aranceles de importación
   Ingreso: De importaciones
   Gasto: A gobierno (AG.gvt)

3. AG.gvt - Gobierno
   Ingresos:
   - Impuestos indirectos (ti)
   - Aranceles (tm)
   - Impuestos directos (de factores/hogares)
   Gastos:
   - Consumo gobierno
   - Transferencias
   - Subsidios
```

### Ejemplo de Flujos con Impuestos

**Sin impuestos sobre productos** (MIP actual):
```
("I", "agr") → ("AG", "hh") = 1,000  # Hogares compran agricultura
```

**Con impuestos sobre productos** (SAM completa):
```
("AG", "hh") → ("I", "agr") = 1,000  # Demanda de hogares (precios básicos)
("AG", "hh") → ("AG", "ti") = 65     # IVA pagado (6.5%)
Total pagado por hogares = 1,065     # Precio de mercado
```

---

## Recomendación para MIP→SAM Pipeline

Dado que estás implementando `run_mip_to_sam()`:

### Para Bolivia específicamente:

1. **Input mínimo**: Usar MIP actual (VAB = 48,614 USD)
2. **Parámetro adicional** en `run_mip_to_sam()`:
   ```python
   product_tax_rates: dict[str, float] | None = None
   ```
3. **Si `product_tax_rates` es None**: Trabajar solo con VAB (precios básicos)
4. **Si se provee**: Agregar cuentas AG.ti y AG.tm

### Implementación sugerida:

```python
def run_mip_to_sam(
    input_path,
    *,
    va_factor_shares = {"L": 0.65, "K": 0.35},
    product_tax_rates = None,  # Nuevo parámetro
    import_tariff_rate = 0.05,
    **kwargs
):
    # ... conversiones estándar ...

    if product_tax_rates:
        # Agregar impuestos sobre productos
        create_product_tax_account(sam, product_tax_rates)
        create_import_tariff_account(sam, import_tariff_rate)

        # Ahora PIB = VAB + Impuestos netos
        # Validar que PIB_target se alcanza si se provee
```

### Uso para Bolivia:

**Opción A - Sin impuestos (más simple)**:
```python
result = run_mip_to_sam(
    "mip_bol_balanced_hybrid.xlsx"
    # No product_tax_rates → Solo VAB
)
```

**Opción B - Con impuestos estimados**:
```python
# Estimar tasas efectivas por sector
tax_rates = estimate_product_tax_rates(
    total_tax_needed=3726,  # USD
    final_demand=F,
    method="uniform"  # o "by_sector"
)

result = run_mip_to_sam(
    "mip_bol_balanced_hybrid.xlsx",
    product_tax_rates=tax_rates
)

# Validar que PIB resultante ≈ 52,340 USD
```

---

## Conclusión Final

### Dos Problemas Identificados

| Problema | Status | Acción Requerida |
|----------|--------|------------------|
| 1. DF es total en vez de nacional | ❌ **Falso** | Ninguna - está correcto |
| 2. Usa VAB en vez de PIB | ✅ **Verdadero** | Agregar 3,726 USD impuestos |

### Impuestos para la MIP

| Tipo de Impuesto | En Excel? | En MIP? | Necesario? |
|------------------|-----------|---------|------------|
| Impuestos sobre producción | ✅ Sí | ✅ Incluido en VAB | Ya presente |
| Impuestos sobre productos | ❌ No | ❌ No | Depende del modelo |
| IVA no deducible | ❌ No | ❌ No | Si modelo usa precios mercado |
| Aranceles | ⚠️ Implícito | ⚠️ En IMP_F | Separar para SAM |

### Para el Pipeline MIP→SAM

**Decisión de diseño**:

1. **Hacer parámetro `product_tax_rates` OPCIONAL**
2. **Default = None** → Trabajar solo con VAB (precios básicos)
3. **Si se provee** → Agregar cuentas de impuestos y alcanzar PIB

**Justificación**:
- No todos los modelos CGE requieren impuestos explícitos
- No todos los países tienen datos desagregados de impuestos
- Permite uso flexible dependiendo de datos disponibles

---

**Fecha**: Abril 2025
**MIP Analizada**: `mip_bol_unbalanced.xlsx` y `Archivo matriz_BoliviaTodo_2023_final.xlsx`
**Estado**: ✅ Análisis completo de impuestos
**Próximo paso**: Decidir si agregar impuestos sobre productos en implementación MIP→SAM
