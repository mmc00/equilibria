# Comparación de Métodos RAS y Variantes

## Resumen Ejecutivo

| Método | ¿Qué balancea? | Valores negativos | Aplicación típica |
|--------|----------------|-------------------|-------------------|
| **RAS** | Solo matriz cuadrada (Z) | ❌ No permite | Actualizar flujos intermedios |
| **MRAS** | Matriz cuadrada con incertidumbre | ❌ No permite | Datos con error de medición |
| **GRAS** | Matriz cuadrada con negativos | ✅ Sí permite | Inventarios, exportaciones netas |
| **KRAS** | Sistema completo con conflictos | ✅ Sí permite | SUTs con datos inconsistentes |

## 1. RAS Clásico (Stone, 1961)

### ¿Qué hace?

Balancea UNA matriz cuadrada **Z** (n×n) para que:
```
row_sums = col_sums = targets
```

### Algoritmo

```python
def ras_classic(Z, row_targets, col_targets, max_iter=1000):
    """
    RAS clásico - solo para matriz cuadrada positiva.

    Args:
        Z: Matriz cuadrada (n×n) de flujos intermedios
        row_targets: Vector (n,) de sumas objetivo por fila
        col_targets: Vector (n,) de sumas objetivo por columna

    Returns:
        Z_balanced: Matriz balanceada
    """
    for iter in range(max_iter):
        # Paso 1: Ajustar filas
        r = row_targets / Z.sum(axis=1)
        Z = diag(r) @ Z

        # Paso 2: Ajustar columnas
        s = col_targets / Z.sum(axis=0)
        Z = Z @ diag(s)

        if converged:
            break

    return Z
```

### ¿Qué NO hace?

- ❌ NO balancea demanda final (F)
- ❌ NO balancea valor agregado (VA)
- ❌ NO balancea importaciones (M)
- ❌ NO maneja valores negativos
- ❌ NO garantiza identidad PIB

**Resultado**: Solo la submatriz Z está balanceada, el resto del sistema puede quedar inconsistente.

## 2. MRAS - Modified RAS (Lecomber, 1975)

### ¿Qué hace?

Igual que RAS pero incorpora **incertidumbre** en los datos.

### Diferencia clave

```python
# RAS clásico
Z_new = r * Z * s  # Ajuste directo

# MRAS
Z_new = r * (Z + error_adjustment) * s  # Ajuste con corrección de error
```

### Aplicación

Cuando los datos base tienen errores de medición conocidos.

### Limitaciones

- ❌ Sigue siendo solo para matriz cuadrada
- ❌ No maneja negativos
- ❌ No balancea sistema completo

## 3. GRAS - Generalized RAS (Junius & Oosterhaven, 2003)

### ¿Qué hace?

Balancea UNA matriz **Z** (puede tener negativos) usando transformación exponencial.

### Innovación clave

**Problema con RAS clásico**:
```
Si Z[i,j] = -10 y necesitas multiplicar por 1.5:
Z[i,j] * 1.5 = -15  ← ¡El negativo se hace más negativo!
```

**Solución GRAS**:
```python
# En lugar de multiplicar directamente, usa exponenciales
r_gras = exp(lambda_r)  # Multiplicadores de Lagrange
s_gras = exp(lambda_s)

Z_new[i,j] = sign(Z[i,j]) * |Z[i,j]|^(r_gras[i] * s_gras[j])
```

Esto **preserva el signo** de los valores originales.

### Algoritmo

```python
def gras(Z, row_targets, col_targets):
    """
    GRAS - permite valores negativos en la matriz.

    Características:
    - Sign-preserving (preserva signos)
    - Minimiza cross-entropy
    - Único para cada solución
    """
    # Separar positivos y negativos
    Z_pos = maximum(Z, 0)
    Z_neg = maximum(-Z, 0)

    # Aplicar RAS a cada parte
    Z_pos_balanced = ras(Z_pos, ...)
    Z_neg_balanced = ras(Z_neg, ...)

    # Recombinar preservando signos
    Z_balanced = Z_pos_balanced - Z_neg_balanced

    return Z_balanced
```

### ¿Qué NO hace?

- ❌ Sigue siendo solo para matriz cuadrada Z
- ❌ NO balancea demanda final
- ❌ NO balancea VA
- ❌ NO garantiza identidad PIB global

**Ventaja sobre RAS**: Puede manejar inventarios (ΔStock) y exportaciones netas (X-M) que pueden ser negativos.

## 4. KRAS - Konfliktfreies RAS (Lenzen et al., 2009)

### ¿Qué hace?

Balancea **sistema completo** con datos conflictivos.

### Características

- Combina RAS + MRAS + GRAS
- Maneja conflictos entre fuentes de datos
- Asigna pesos de confiabilidad

```python
def kras(system, weights):
    """
    KRAS - sistema completo con prioridades.

    Args:
        system: {Z, F, VA, M} - todos los componentes
        weights: Confiabilidad de cada dato
            - VA: weight = 1.0 (más confiable - cuentas nacionales)
            - X, M: weight = 0.8 (confiable - aduanas)
            - Z: weight = 0.5 (menos confiable - encuestas)
    """
    # Ajusta cada componente según su peso
    # Datos con mayor peso se modifican menos
```

### Ventaja

✅ Balancea TODO el sistema respetando jerarquía de confiabilidad

## 5. Stone's Method (implícito en SNA)

### ¿Qué hace?

Balancea **Supply-Use Tables completas** incluyendo:
- Matriz de uso (Z)
- Demanda final (F)
- Valor agregado (VA)
- Márgenes comerciales
- Impuestos sobre productos

### Identidades enforced

```
1. Supply = Use (por producto)
   Producción + Importaciones = Uso intermedio + Demanda final

2. Inputs = Output (por sector)
   Insumos intermedios + VA = Producción

3. PIB identity
   Σ(VA) = Σ(Demanda final) - Σ(Importaciones)
```

### Algoritmo (simplificado)

```python
def balance_complete_sut(Z, F, VA, M):
    """
    Balancea sistema completo de oferta-uso.

    Enfoque jerárquico (SNA 2008):
    1. Fijar VA (más confiable)
    2. Fijar totales de X, M (de aduanas)
    3. Ajustar Z con RAS/GRAS
    4. Ajustar F residualmente
    5. Iterar hasta convergencia
    """
    for iteration in range(max_iter):
        # Paso 1: Balance de producción (columnas)
        production = Z.sum(axis=0) + VA.sum(axis=0)

        # Paso 2: Balance de uso (filas)
        use = Z.sum(axis=1) + F.sum(axis=1)

        # Paso 3: Ajustar Z
        Z = gras(Z, use_targets, production_targets)

        # Paso 4: Ajustar F residualmente
        F = adjust_final_demand(...)

        # Paso 5: Verificar identidad PIB
        if pib_converged:
            break

    return Z, F, VA, M
```

## Respuesta a tu Pregunta: ¿Qué estaba aplicando?

En `balance_bolivia_pragmatic.py` apliqué **RAS CLÁSICO** solamente:

```python
Z_balanced = ras_square(Z)  # ← Solo esto
```

**Resultado**:
- ✅ Matriz Z balanceada (row sums = col sums)
- ❌ F, VA, M sin balancear
- ❌ Identidades globales NO satisfechas

## Lo Que Necesitas: Balanceo Completo

Para balancear una MIP completa necesitas un método que ajuste **TODOS** los componentes:

```
┌─────────────────────────────────────────┐
│  MATRIZ MIP COMPLETA (143 × 75)         │
├─────────────────────────────────────────┤
│  Z (70×70)     │  F (70×5)              │ ← Productos
│  Intermedia    │  Demanda Final         │
├────────────────┼────────────────────────┤
│  M_Z (70×70)   │  M_F (70×5)            │ ← Importaciones
│  Imp a sectors │  Imp a FD              │
├────────────────┼────────────────────────┤
│  VA (3×70)     │  (vacío)               │ ← Valor Agregado
│  Rem/Exc/Imp   │                        │
└─────────────────────────────────────────┘
```

**Método recomendado**: Combinación Stone + GRAS

```python
def balance_complete_mip(mip_matrix):
    # 1. Preservar VA (fijo)
    VA_fixed = VA.copy()

    # 2. Calcular targets consistentes
    production_targets = calculate_from_expenditure_side()
    use_targets = calculate_from_production_side()

    # 3. Aplicar GRAS a bloques ajustables
    for iteration in range(max_iter):
        # 3a. Balance Z (intermedia)
        Z = gras(Z, row_targets, col_targets)

        # 3b. Ajustar F (demanda final)
        F = adjust_proportionally(F, targets)

        # 3c. Ajustar M (importaciones)
        M = adjust_proportionally(M, targets)

        # 3d. Restaurar VA
        VA = VA_fixed

        # 3e. Verificar las 3 identidades
        if all_identities_satisfied:
            break

    return balanced_matrix
```

## Fuentes

- [RAS Technique - Open Risk Manual](https://www.openriskmanual.org/wiki/RAS_Technique)
- [IMPLAN - The RAS Method](https://support.implan.com/hc/en-us/articles/11646312866075-The-RAS-Method)
- [GRAS MATLAB Implementation](https://www.mathworks.com/matlabcentral/fileexchange/43231-generalized-ras-matrix-balancing-updating-biproportional-method)
- [IMF - Algorithm to Balance Supply and Use Tables](https://www.imf.org/-/media/files/publications/tnm/2018/tnm-1803.pdf)
- [Junius & Oosterhaven (2003) - Semantic Scholar](https://www.semanticscholar.org/paper/The-Solution-of-Updating-or-Regionalizing-a-Matrix-Junius-Oosterhaven/e2ccf6bbe11abe7a0521c4a680ce53607262c749)
- [Efficient updating of regional supply and use tables](https://journalofeconomicstructures.springeropen.com/articles/10.1186/s40008-022-00274-8)

## Conclusión

**Tu MIP necesita**: Método tipo "Stone + GRAS" que balancea **todo el sistema**, no solo Z.

**Lo que implementé**: Solo RAS clásico en Z.

**Por eso**: Solo 1 de 4 identidades está balanceada.

**Siguiente paso**: Implementar el algoritmo completo que ajuste Z, F, M simultáneamente mientras preserva VA.
