# MIP balancing methods: literature and practice

## 1. Identities a balanced MIP must meet

A balanced input-output matrix (MIP) must satisfy, at the same time:

### 1.1. Balance by product (rows)

For each product `i`:
```
Total supply = Total demand

Domestic_output[i] + Imports[i] =
    Σ(Intermediate_use[i,j]) + C[i] + G[i] + I[i] + ΔS[i] + X[i]
```

### 1.2. Balance by sector (columns)

For each sector `j`:
```
Total inputs = Total output

Σ(Intermediate_inputs[i,j]) + VA[j] + Taxes[j] = Output[j]
```

### 1.3. GDP identity

```
GDP (production) = Σ(VA[j] + Taxes[j])

GDP (expenditure) = C + G + I + ΔS + (X - M)

GDP (production) = GDP (expenditure)
```

## 2. Standard methods in the literature

### 2.1. The RAS method (biproportional scaling)

**Reference**: Stone (1961), Bacharach (1970)

**Limitation**: it only balances a square matrix (intermediate flows), not the whole system.

**Appropriate use**:
- Updating an old MIP with new row/column totals
- Preliminary balancing of intermediate flows

**NOT appropriate for**: fully balancing a system with final demand and value added.

### 2.2. Cross-Entropy Minimization (minimum-information method)

**Reference**: Robinson, Cattaneo & El-Said (2001), Golan, Judge & Miller (1996)

**Idea**: minimize the "information distance" between the unbalanced and the balanced matrix, subject to balance constraints.

**Objective function**:
```
min Σ Σ x[i,j] * log(x[i,j] / x₀[i,j])

subject to:
- Row balance (supply = demand)
- Column balance (inputs = output)
- GDP identity
- x[i,j] ≥ 0
```

**Advantages**:
- ✅ Balances the whole system at once
- ✅ Minimizes changes to the original data
- ✅ Can take further constraints
- ✅ Formal treatment of uncertainty

**Disadvantages**:
- ⚠️ Needs nonlinear optimization (computationally more complex)
- ⚠️ May have several local solutions

### 2.3. Generalized proportional adjustment (GRAS)

**Reference**: Junius & Oosterhaven (2003)

**Idea**: an extension of RAS that adjusts both intermediate flows and final demand/value added.

**Algorithm**:
1. Compute the target totals by row and column
2. Apply RAS to the intermediate-flows submatrix
3. Adjust final demand proportionally
4. Adjust value added proportionally
5. Iterate until convergence

**Advantages**:
- ✅ Simpler than cross-entropy
- ✅ Balances the whole system
- ✅ Converges fast

### 2.4. National accounts reconciliation (SNA)

**Reference**: United Nations SNA 2008, Chapter 26

**Idea**: adjust components according to their statistical reliability.

**Typical reliability ranking**:
1. **Most reliable**: value added (from the national accounts)
2. **Reliable**: exports, imports (from customs)
3. **Less reliable**: intermediate flows (from surveys)
4. **Variable**: final demand components (various sources)

**Procedure**:
1. Fix the most reliable data (VA, X, M)
2. Adjust the less reliable data (Z, C, G, I)
3. Apply RAS/GRAS to the residual

## 3. Recommendation for the Bolivia MIP

### 3.1. Diagnosis

The Bolivia MIP has:
- ✅ Disaggregated value added (reliable - from the national accounts)
- ✅ Imports by sector (reliable - from customs)
- ⚠️ Intermediate flows (less reliable - surveys)
- ⚠️ Final demand (mixed - several sources)

**Imbalance found**:
```
PIB (VA) = 48,614
PIB (gasto) = 50,128
Diferencia = 1,514 (3.1%)
```

### 3.2. Recommended method: GRAS + final demand adjustment

**Step 1: fix the reliable components**
- Value added by sector: FIXED (comes from the national accounts)
- Total imports: FIXED (comes from customs)

**Step 2: compute the statistical discrepancy**
```
Discrepancia = PIB(gasto) - PIB(VA)
             = 50,128 - 48,614
             = 1,514
```

**Step 3: distribute the discrepancy**

Option A: adjust final demand proportionally
```
C_ajustado = C * (PIB_VA / PIB_gasto)
G_ajustado = G * (PIB_VA / PIB_gasto)
...
```

Option B: create a "statistical discrepancy" account
```
PIB = C + G + I + ΔS + (X - M) + DISC
```

**Step 4: apply GRAS for fine balancing**

### 3.3. Practical implementation

```python
# Algoritmo GRAS simplificado

def balance_mip_complete(Z, F, VA, M):
    """
    Balance completo de MIP.

    Args:
        Z: Flujos intermedios (n×n)
        F: Demanda final (n×m) - m componentes
        VA: Valor agregado por sector (n,)
        M: Importaciones por producto (n,)

    Returns:
        Z_bal, F_bal: Matrices balanceadas
    """
    n = Z.shape[0]

    # Paso 1: Calcular producción total target
    # X = uso_intermedio + demanda_final
    X_demanda = Z.sum(axis=0) + F.sum(axis=1)

    # X = insumos_intermedios + VA
    X_oferta = Z.sum(axis=1) + VA

    # Target = promedio (o usar el más confiable)
    X_target = (X_demanda + X_oferta) / 2

    # Paso 2: Calcular oferta total (producción + importaciones)
    Q = X_target + M

    # Paso 3: Iterar GRAS
    for iteration in range(max_iter):
        # 3a. Ajustar Z para que columnas = X_target - VA
        col_target = X_target - VA
        col_sums = Z.sum(axis=0)
        col_factors = col_target / col_sums
        Z = Z * col_factors

        # 3b. Ajustar Z y F para que filas = Q - M
        row_target = Q - M
        row_sums = Z.sum(axis=1) + F.sum(axis=1)
        row_factors = row_target / row_sums

        Z = Z * row_factors[:, None]
        F = F * row_factors[:, None]

        # Verificar convergencia
        diff = abs(X_target - (Z.sum(axis=0) + VA)).max()
        if diff < tolerance:
            break

    return Z, F
```

## 4. Alternatives when the data are very inconsistent

### 4.1. Hybrid method (recommended for Bolivia)

1. **Preserve the reliable components**:
   - Total value added and its sector structure
   - Total imports
   - Total exports

2. **Adjust the intermediate components**:
   - Matrix Z: RAS with constraints
   - Private consumption: proportional adjustment
   - Investment: residual adjustment

3. **Accept a small statistical discrepancy** (< 1%)

### 4.2. Building a synthetic MIP

If the inconsistency is very large (>5%), consider:

1. **Rebuild from value added**:
   - Use value added as the base
   - Apply average technical coefficients (regional/international)
   - Calibrate with known totals

2. **Supply/use tables method (SUTs)**:
   - Build separate use and supply matrices
   - Balance each one independently
   - Derive a symmetric MIP

## 5. Post-balancing validation criteria

A balanced MIP must pass:

### 5.1. Numerical tests

```python
# 1. Balance de filas (oferta = demanda)
for i in range(n):
    oferta = X[i] + M[i]
    demanda = Z[i,:].sum() + F[i,:].sum()
    assert abs(oferta - demanda) < 0.01

# 2. Balance de columnas (insumos = producción)
for j in range(n):
    insumos = Z[:,j].sum() + VA[j]
    produccion = X[j]
    assert abs(insumos - produccion) < 0.01

# 3. Identidad PIB
pib_va = VA.sum()
pib_gasto = F.sum() + (X.sum() - M.sum())
assert abs(pib_va - pib_gasto) < 0.01
```

### 5.2. Economic tests

- ✅ Reasonable technical coefficients (0 ≤ a[i,j] ≤ 1)
- ✅ Sector structure preserved
- ✅ No negative flows introduced
- ✅ Changes < 10% in individual cells (ideally)

## 6. Software and tools

### Existing implementations

- **Python**:
  - `pymrio` - Multi-Regional Input-Output analysis
  - `iotables` - European Input-Output tables

- **R**:
  - `ioanalysis` package
  - `sna` (System of National Accounts)

- **GAMS**:
  - GTAP database balancing procedures
  - IFPRI SAM balancing tools

### For Equilibria

We should implement:
```python
def balance_full_mip(
    mip: MIPRawSAM,
    *,
    method: str = "gras",  # "ras", "gras", "cross-entropy"
    fix_va: bool = True,   # Fijar VA (más confiable)
    fix_imports: bool = True,  # Fijar importaciones
    fix_exports: bool = True,  # Fijar exportaciones
    tolerance: float = 1e-4,
    max_iterations: int = 1000
) -> MIPBalanceResult:
    """Balance completo de MIP siguiendo literatura."""
    ...
```

## References

### Foundational papers

1. **Stone, R. (1961)**. "Input-Output and National Accounts". OECD.
   - Introduces the RAS method

2. **Bacharach, M. (1970)**. "Biproportional Matrices and Input-Output Change". Cambridge.
   - Mathematical theory of RAS

3. **Robinson, S., Cattaneo, A., & El-Said, M. (2001)**. "Updating and Estimating a Social Accounting Matrix Using Cross Entropy Methods". Economic Systems Research.
   - Cross-entropy for SAMs

4. **Junius, T., & Oosterhaven, J. (2003)**. "The Solution of Updating or Regionalizing a Matrix with both Positive and Negative Entries". Economic Systems Research.
   - The GRAS method

5. **Golan, A., Judge, G., & Miller, D. (1996)**. "Maximum Entropy Econometrics". Wiley.
   - Theoretical foundations of cross-entropy

6. **United Nations (2008)**. "System of National Accounts 2008", Chapter 14 & 26.
   - Official standards for balancing MIPs

### Software

- **pymrio**: https://github.com/konstantinstadler/pymrio
- **GTAP**: https://www.gtap.agecon.purdue.edu/
- **IFPRI SAM tools**: https://www.ifpri.org/

## Conclusion

For the Bolivia MIP:

1. Balancing only intermediate flows **is not enough**
2. The whole system **must** be balanced
3. **Recommended method**: GRAS with fixed value added
4. **Target**: max diff < 0.1, GDP discrepancy < 0.1%

The next step is to implement `balance_full_mip()`, which balances the whole system correctly.
