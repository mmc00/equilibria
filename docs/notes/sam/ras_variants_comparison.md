# Comparison of RAS methods and variants

## Executive summary

| Method | What it balances | Negative values | Typical application |
|--------|----------------|-------------------|-------------------|
| **RAS** | Only the square matrix (Z) | ❌ Not allowed | Updating intermediate flows |
| **MRAS** | Square matrix with uncertainty | ❌ Not allowed | Data with measurement error |
| **GRAS** | Square matrix with negatives | ✅ Allowed | Inventories, net exports |
| **KRAS** | Whole system with conflicts | ✅ Allowed | SUTs with inconsistent data |

## 1. Classic RAS (Stone, 1961)

### What it does

Balances ONE square matrix **Z** (n×n) so that:
```
row_sums = col_sums = targets
```

### Algorithm

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

### What it does NOT do

- ❌ Does NOT balance final demand (F)
- ❌ Does NOT balance value added (VA)
- ❌ Does NOT balance imports (M)
- ❌ Does NOT handle negative values
- ❌ Does NOT guarantee the GDP identity

**Result**: only the Z submatrix is balanced; the rest of the system can stay inconsistent.

## 2. MRAS - Modified RAS (Lecomber, 1975)

### What it does

Like RAS, but it brings **uncertainty** in the data into account.

### Key difference

```python
# RAS clásico
Z_new = r * Z * s  # Ajuste directo

# MRAS
Z_new = r * (Z + error_adjustment) * s  # Ajuste con corrección de error
```

### Application

When the base data have known measurement errors.

### Limitations

- ❌ Still only for a square matrix
- ❌ Does not handle negatives
- ❌ Does not balance the whole system

## 3. GRAS - Generalized RAS (Junius & Oosterhaven, 2003)

### What it does

Balances ONE matrix **Z** (it may have negatives) using an exponential transformation.

### Key innovation

**Problem with classic RAS**:
```
If Z[i,j] = -10 and you need to multiply by 1.5:
Z[i,j] * 1.5 = -15  ← The negative gets more negative!
```

**GRAS solution**:
```python
# En lugar de multiplicar directamente, usa exponenciales
r_gras = exp(lambda_r)  # Multiplicadores de Lagrange
s_gras = exp(lambda_s)

Z_new[i,j] = sign(Z[i,j]) * |Z[i,j]|^(r_gras[i] * s_gras[j])
```

This **preserves the sign** of the original values.

### Algorithm

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

### What it does NOT do

- ❌ Still only for the square matrix Z
- ❌ Does NOT balance final demand
- ❌ Does NOT balance value added
- ❌ Does NOT guarantee the overall GDP identity

**Advantage over RAS**: it can handle inventories (ΔStock) and net exports (X-M), which can be negative.

## 4. KRAS - Konfliktfreies RAS (Lenzen et al., 2009)

### What it does

Balances the **whole system** with conflicting data.

### Features

- Combines RAS + MRAS + GRAS
- Handles conflicts between data sources
- Assigns reliability weights

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

### Advantage

✅ Balances the WHOLE system, respecting the reliability ranking

## 5. Stone's method (implicit in the SNA)

### What it does

Balances **complete supply-use tables**, including:
- Use matrix (Z)
- Final demand (F)
- Value added (VA)
- Trade margins
- Taxes on products

### Identities enforced

```
1. Supply = Use (by product)
   Output + Imports = Intermediate use + Final demand

2. Inputs = Output (by sector)
   Intermediate inputs + VA = Output

3. PIB identity
   Σ(VA) = Σ(Final demand) - Σ(Imports)
```

### Algorithm (simplified)

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

## Answer to your question: what was I applying?

In `balance_bolivia_pragmatic.py` I applied **CLASSIC RAS** only:

```python
Z_balanced = ras_square(Z)  # ← Solo esto
```

**Result**:
- ✅ Matrix Z balanced (row sums = col sums)
- ❌ F, VA, M unbalanced
- ❌ Overall identities NOT satisfied

## What you need: full balancing

To balance a whole MIP you need a method that adjusts **ALL** the components:

```
┌─────────────────────────────────────────┐
│  COMPLETE MIP MATRIX (143 × 75)         │
├─────────────────────────────────────────┤
│  Z (70×70)     │  F (70×5)              │ ← Products
│  Intermediate  │  Final demand          │
├────────────────┼────────────────────────┤
│  M_Z (70×70)   │  M_F (70×5)            │ ← Imports
│  Imp to sectors│  Imp to FD             │
├────────────────┼────────────────────────┤
│  VA (3×70)     │  (empty)               │ ← Value added
│  Comp/GOS/Tax  │                        │
└─────────────────────────────────────────┘
```

**Recommended method**: Stone + GRAS combined

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

## Sources

- [RAS Technique - Open Risk Manual](https://www.openriskmanual.org/wiki/RAS_Technique)
- [IMPLAN - The RAS Method](https://support.implan.com/hc/en-us/articles/11646312866075-The-RAS-Method)
- [GRAS MATLAB Implementation](https://www.mathworks.com/matlabcentral/fileexchange/43231-generalized-ras-matrix-balancing-updating-biproportional-method)
- [IMF - Algorithm to Balance Supply and Use Tables](https://www.imf.org/-/media/files/publications/tnm/2018/tnm-1803.pdf)
- [Junius & Oosterhaven (2003) - Semantic Scholar](https://www.semanticscholar.org/paper/The-Solution-of-Updating-or-Regionalizing-a-Matrix-Junius-Oosterhaven/e2ccf6bbe11abe7a0521c4a680ce53607262c749)
- [Efficient updating of regional supply and use tables](https://journalofeconomicstructures.springeropen.com/articles/10.1186/s40008-022-00274-8)

## Conclusion

**Your MIP needs**: a "Stone + GRAS" kind of method that balances **the whole system**, not only Z.

**What I implemented**: only classic RAS on Z.

**That is why**: only 1 of 4 identities is balanced.

**Next step**: implement the complete algorithm that adjusts Z, F, M at once while preserving value added.
