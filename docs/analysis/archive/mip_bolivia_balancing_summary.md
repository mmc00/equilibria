# MIP Bolivia: Resumen del Problema de Balanceo

## Intentos Realizados

He implementado tres enfoques diferentes para balancear la MIP de Bolivia:

1. **GRAS completo** (`balance_bolivia_mip.py`): No convergió, produjo NaNs
2. **GRAS con targets explícitos** (`balance_bolivia_with_targets.py`): Divergió, targets incompatibles
3. **Balanceo jer árquico** (`balance_bolivia_hierarchical.py`): Resultados incoherentes

## Problema Fundamental

Los vectores de targets proporcionados son **matemáticamente incompatibles**:

```
Suma de targets de filas:    149,867.17
Suma de targets de columnas: 151,380.65
Diferencia:                    1,513.48 (1.0%)
```

**Para una matriz balanceada**: `Σ(filas) = Σ(columnas)` SIEMPRE.

## Pregunta Crítica

**¿De dónde vienen estos targets?**

Necesito entender:

1. **Targets de filas** (143 valores):
   - ¿Qué representan exactamente?
   - ¿Oferta total por producto (producción + importaciones)?
   - ¿Son fijos o se pueden ajustar?

2. **Targets de columnas** (75 valores):
   - ¿Qué representan exactamente?
   - ¿Demanda total por sector/categoria?
   - ¿Son fijos o se pueden ajustar?

3. **Metodología de cálculo**:
   - ¿Cómo se calcularon?
   - ¿Hay componentes que NO se deben modificar?

## Tres Caminos Posibles

### Opción 1: Balancear SIN usar los targets

**Enfoque**: Balancear la MIP internamente sin restricciones externas.

**Método**:
1. Preservar VA (más confiable)
2. Aplicar RAS estándar a flujos intermedios (70×70)
3. Ajustar demanda final para cerrar las cuentas

**Ventaja**: Simple, no requiere entender targets
**Desventaja**: Resultado puede diferir de benchmarks oficiales

### Opción 2: Usar targets como guía (no como restricción exacta)

**Enfoque**: Balancear lo más cerca posible de targets, aceptando pequeñas desviaciones.

**Método**:
1. Usar targets como punto de partida
2. Aplicar GRAS permitiendo <1% de desviación
3. Crear "discrepancia estadística" explícita si necesario

**Ventaja**: Usa información de targets
**Desventaja**: Targets deben ser razonables

### Opción 3: Clarificar targets primero, luego balancear

**Enfoque**: Entender qué representan los targets antes de usarlos.

**Preguntas a responder**:
- ¿Estos targets vienen de Cuentas Nacionales oficiales?
- ¿Son proyecciones/ajustes de una MIP antigua?
- ¿Hay jerarquía de confiabilidad? (ej: VA > Exportaciones > Flujos intermedios)

**Ventaja**: Solución correcta
**Desventaja**: Requiere más contexto

## Recomendación

**Necesito que me aclares**:

1. ¿Cuál es la fuente de los archivos `.npy` con targets?
2. ¿Estos targets son "oficiales" (de instituto de estadística) o fueron calculados internamente?
3. ¿Hay alguna documentación sobre cómo se calcularon?
4. ¿Qué nivel de discrepancia es aceptable? (1%, 5%, 10%?)

## Alternativa Pragmática

Si los targets no son críticos, puedo implementar **Opción 1** (balanceo interno) que:

1. ✓ Preserva VA exactamente (48,614.10)
2. ✓ Balancea flujos intermedios con RAS estándar
3. ✓ Ajusta demanda final residualmente
4. ✓ Garantiza identidades contables
5. ✓ PIB discrepancy < 1%

**¿Procedo con esta opción?** O prefieres que espere hasta clarificar el tema de los targets.

## Código ya implementado

He creado:
- `src/equilibria/sam_tools/balancing.py`: Función `balance_complete_mip()` (tiene bugs por targets incompatibles)
- `docs/technical/mip_balancing_methods.md`: Revisión completa de literatura
- `docs/analysis/mip_bolivia_balancing_issue.md`: Análisis del problema de targets

**Siguiente paso**: Decidir estrategia antes de continuar implementación.

