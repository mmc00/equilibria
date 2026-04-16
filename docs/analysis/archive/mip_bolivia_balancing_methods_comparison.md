# Comparación Completa de Métodos de Balanceo - Bolivia MIP

## Resumen Ejecutivo

Se probaron 7+ métodos de balanceo sobre `mip_bol_unbalanced2.xlsx` con el objetivo de satisfacer tres restricciones:

1. **Z balance**: Filas = Columnas en consumo intermedio
2. **PIB identity**: VA = F - IMP_F
3. **Supply-Demand balance**: Oferta = Demanda por producto

**Hallazgo clave**: Las tres restricciones son **matemáticamente incompatibles** con estos datos. Se debe elegir qué priorizar.

## Comparación de Resultados

### MIP Original (sin balancear)
| Métrica | Valor |
|---------|------:|
| Error PIB | 5.81% (2,828 USD) |
| Error PIB (absoluto) | 2,828 USD |
| Z balance max | 2,698 USD |
| Z balance mean | 52.41 USD |
| S-D balance max | 1,494 USD |
| S-D balance total | 15,455 USD |
| Productos S-D OK (<1 USD) | 11/70 |
| Productos S-D OK (<100 USD) | 53/70 |

### Métodos Tradicionales (Solo Z + PIB)

#### 1. GRAS con damping
```
Error PIB:      0.93%
Z balance max:  10.54
S-D balance:    -6,775 (PEOR que original)
S-D OK (<100):  6/70 (PEOR)
Tiempo:         0.35s
```

#### 2. Hybrid RAS
```
Error PIB:      1.38%
Z balance max:  10.54
S-D balance:    -6,994 (PEOR)
S-D OK (<100):  6/70
Tiempo:         0.05s
```

#### 3. GRAS original
```
Error PIB:      1.23%
Z balance max:  10.54
S-D balance:    -6,881 (PEOR)
S-D OK (<100):  6/70
Tiempo:         0.34s
```

#### 4. Cross-Entropy (simplificado)
```
Error PIB:      1.23%
Z balance max:  10.54
S-D balance:    -6,881 (PEOR)
S-D OK (<100):  6/70
Tiempo:         0.05s
```

#### 5. RAS jerárquico
```
Error PIB:      1.49%
Z balance max:  9.37
S-D balance:    -7,053 (PEOR)
S-D OK (<100):  6/70
Tiempo:         0.04s
```

#### 6. RAS simple
```
Error PIB:      6.81%
Z balance max:  12.73
S-D balance:    -4,048 (PEOR)
S-D OK (<100):  7/70
Tiempo:         0.05s
```

#### 7. GRAS completo ⭐ MEJOR para CGE
```
Error PIB:      0.00%
Z balance max:  5.27
S-D balance:    -6,323 (PEOR que original)
S-D OK (<100):  6/70
Tiempo:         0.29s
```

**Observación**: Todos los métodos tradicionales EMPEORAN el balance S-D.

### Métodos con Restricción S-D Explícita

#### 8. GRAS damping + S-D enforcement
```
Error PIB:      0.00%
Z balance max:  0.06
S-D balance:    19,146 (10x PEOR!)
S-D OK (<100):  7/70
Tiempo:         0.35s
```

#### 9. Iterativo 3 restricciones
```
Inicio:
  PIB error:    2,828 USD
  Z balance:    1,331 USD
  S-D max:      2,030 USD
  S-D OK:       8/70

Final (iter 200):
  PIB error:    0 USD      ✓
  Z balance:    0 USD      ✓
  S-D max:      32,192 USD  ✗ (16x PEOR!)
  S-D OK:       6/70  ✗

Tiempo:         0.03s
```

**Observación**: Forzar PIB = 0 y Z = 0 hace explotar S-D de 2,030 a 32,192.

#### 10. Compromiso ponderado (w_PIB=100, w_Z=10, w_SD=1)
```
Mejor resultado:
  PIB error:    0.00%
  Z balance:    66 USD
  S-D max:      3,077 USD (2x PEOR)
  S-D OK (<100): 41/70 (PEOR)
Tiempo:         0.11s
```

### Método Especial: Cross-Entropy PURO (solo PIB)

```
Error PIB:      0.00%  ✓
Z balance max:  2,435 USD  ✗ (destruye estructura Z)
S-D balance:    No evaluado
Tiempo:         ~300s (100 iters optimization)
```

**Nota**: Optimiza solo macro constraint, ignora estructura interna.

## Tabla Resumen Comparativa

| Método | PIB Error (%) | Z Balance | S-D Max | S-D OK <100 | Tiempo | Uso Recomendado |
|--------|---------------|-----------|---------|-------------|--------|-----------------|
| **Original** | 5.81 | 2,698 | 1,494 | 53/70 | - | Baseline |
| GRAS completo ⭐ | **0.00** | **5.27** | 6,323 ↓ | 6/70 ↓ | 0.29s | **CGE models** |
| GRAS damping | 0.93 | 10.54 | 6,775 ↓ | 6/70 ↓ | 0.35s | Alternative CGE |
| Hybrid RAS | 1.38 | 10.54 | 6,994 ↓ | 6/70 ↓ | 0.05s | Fast CGE |
| RAS simple | 6.81 | 12.73 | 4,048 ↓ | 7/70 ↓ | 0.05s | Exploratory |
| Compromiso ponderado | **0.00** | 66 | 3,077 ↓ | 41/70 ↓ | 0.11s | Soft constraints |
| Iterativo 3-const | **0.00** | **0.00** | 32,192 ↓↓ | 6/70 ↓ | 0.03s | ⚠️ No usar |
| Cross-Entropy puro | **0.00** | 2,435 ↓ | ? | ? | 300s | ⚠️ No usar |

**Leyenda**:
- ⭐ = Recomendado
- ↓ = Empeora vs original
- ↓↓ = Empeora significativamente
- ⚠️ = No recomendado

## Análisis por Tipo de Uso

### Para Modelos CGE (PEP, GTAP, etc.)

**Método recomendado**: GRAS completo

**Razón**:
1. PIB error = 0.00% (crítico para calibración)
2. Z balance = 5.27 USD (excelente para consistencia interna)
3. S-D imbalance se acepta como:
   - Datos reales tienen errores de medición
   - CGE equilibra O-D endógenamente vía precios
   - Desbalance inicial = presión sobre el modelo

**Archivo**: `mip_bol_balanced_hybrid.xlsx` o `mip_balanced_7_gras_completo.xlsx`

### Para Análisis Insumo-Producto Tradicional

**Método recomendado**: Original sin balancear o RAS simple

**Razón**:
1. I-O multiplicadores son robustos a errores PIB <10%
2. S-D balance original (1,494) es mejor que balanceados
3. Estructura interna preservada
4. Identificación de cuellos de botella más realista

**Archivo**: `mip_bol_unbalanced2.xlsx`

### Para Análisis de Cadenas de Valor

**Opción 1**: Original (mejor S-D)
**Opción 2**: RAS simple (balance moderado)

**No usar**: GRAS completo (S-D muy desbalanceado)

## Trade-off Fundamental

```
Restricciones:
  [1] Z balance
  [2] PIB identity
  [3] S-D balance

Prioridades:
  CGE:     [2] > [1] >> [3]  → GRAS completo
  I-O:     [3] > [1] > [2]   → Original o RAS simple
  Físico:  [3] > [2] > [1]   → Requiere método nuevo*

* No implementado actualmente
```

## Explicación Matemática

Con Z balanceada ([1]) y PIB satisfecho ([2]), el balance S-D ([3]) requiere:

```
F[i,:].sum() = VA[i]  para cada producto i

Y además:
Σ IMP_F = 0  (importaciones finales = 0)
```

En Bolivia:
- `Σ IMP_F = 6,323 USD` ≠ 0

**Por lo tanto, [1] + [2] + [3] es IMPOSIBLE con estos datos.**

## Recomendación Final

### Para pipeline MIP → SAM:

```python
from equilibria.sam_tools import run_mip_to_sam

# Opción A: GRAS completo (recomendado para CGE)
result = run_mip_to_sam(
    "mip_bol_balanced_hybrid.xlsx",  # GRAS completo
    va_factor_shares={"L": 0.65, "K": 0.35},
    # PIB = 0% error, Z balance = 5 USD
    # S-D imbalance aceptado como realidad de datos mixtos
)

# Opción B: Original (para I-O análisis)
result = run_mip_to_sam(
    "mip_bol_unbalanced2.xlsx",  # Sin balancear
    va_factor_shares={"L": 0.65, "K": 0.35},
    # Mejor S-D balance (1,494 vs 6,323)
    # Acepta 5.81% PIB error para I-O multiplicadores
)
```

### Parámetro opcional: `enforce_supply_demand_balance`

**Decisión de diseño**: NO implementar en versión inicial

**Razón**:
1. Matemáticamente incompatible con PIB + Z balance
2. Empeora resultados en todos los tests empíricos
3. No es requerido por modelos CGE PEP estándar
4. Añade complejidad sin beneficio para usuarios típicos

**Alternativa**: Documentar que S-D balance puede estar desbalanceado y por qué esto es aceptable (errores de medición, márgenes, cambios en inventarios).

## Archivos Generados

| Archivo | Método | Uso |
|---------|--------|-----|
| `mip_bol_unbalanced2.xlsx` | Original | Baseline, I-O |
| `mip_bol_balanced_hybrid.xlsx` | Hybrid balancing (previo) | CGE ⭐ |
| `mip_balanced_7_gras_completo.xlsx` | GRAS completo | CGE ⭐ |
| `mip_balanced_weighted_compromise.xlsx` | Compromiso ponderado | Experimental |
| `mip_balanced_iterative_3constraints.xlsx` | Iterativo 3-const | ⚠️ No usar |
| `balance_history_3constraints.xlsx` | Historia iterativo | Análisis |

---

**Fecha**: Abril 2025
**Análisis**: Claude Code
**Scripts**:
- `compare_all_balancing_methods.py` - Métodos tradicionales
- `balance_bolivia_three_constraints.py` - Iterativo
- `balance_bolivia_weighted_compromise.py` - Compromiso ponderado

**Conclusión**: Usar GRAS completo para CGE, aceptar S-D imbalance como característica de datos reales.
