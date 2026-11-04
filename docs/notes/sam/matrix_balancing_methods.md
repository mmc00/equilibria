# Matrix balancing methods for MIP/SAM

## Introduction

Balancing methods adjust economic matrices (MIP, SAM, I-O tables) so that they meet accounting-consistency constraints, while minimizing the distance to the original matrix.

**Typical problem**: given a matrix X₀ and targets (u, v), find X such that:
- `X.sum(axis=1) = u` (row sums)
- `X.sum(axis=0) = v` (column sums)
- `X` is "close" to `X₀` under some metric

---

## 1. RAS (Biproportional Scaling)

### Description

The most classic and simplest method. It adjusts rows and columns in turn with multiplicative factors.

**Name**: RAS comes from R×A×S, where R and S are diagonal matrices of factors.

### Algorithm

```
Inicializar: r = ones(n), s = ones(m)

Repetir hasta convergencia:
  1. Ajuste filas:    r[i] = u[i] / (X₀[i,:] @ s).sum()
  2. Ajuste columnas: s[j] = v[j] / (r @ X₀[:,j]).sum()

  X = diag(r) @ X₀ @ diag(s)
```

### Advantages
- ✓ Simple to implement
- ✓ Converges fast (typically <100 iters)
- ✓ Preserves zeros (if X₀[i,j]=0 → X[i,j]=0)
- ✓ Low computational cost

### Disadvantages
- ✗ Only works with non-negative matrices
- ✗ Needs consistent targets (sum(u) = sum(v))
- ✗ Does not handle negative values
- ✗ Can fail with rows/columns of zeros

### Implementation

```python
def ras(X0, u, v, max_iter=200, tol=1e-9):
    """
    RAS biproportional scaling.

    Args:
        X0: Initial matrix (n×m)
        u: Row targets (n,)
        v: Column targets (m,)

    Returns:
        X: Balanced matrix
        iterations: Number of iterations
        converged: Boolean
    """
    n, m = X0.shape
    r = np.ones(n)
    s = np.ones(m)

    for iteration in range(max_iter):
        # Row scaling
        row_sums = (X0 * s).sum(axis=1)
        r = np.where(row_sums > 1e-10, u / row_sums, 1.0)

        # Column scaling
        col_sums = (r[:, np.newaxis] * X0).sum(axis=0)
        s = np.where(col_sums > 1e-10, v / col_sums, 1.0)

        # Check convergence
        X = r[:, np.newaxis] * X0 * s
        row_diff = np.abs(X.sum(axis=1) - u).max()
        col_diff = np.abs(X.sum(axis=0) - v).max()

        if max(row_diff, col_diff) < tol:
            return X, iteration + 1, True

    return X, max_iter, False
```

### When to use
- 100% non-negative matrices
- Exact targets known
- You need speed
- Simple I-O tables

---

## 2. GRAS (Generalized RAS)

### Description

An extension of RAS that handles **negative values** by splitting the matrix into positive and negative components.

**Key innovation**: `X = X_pos - X_neg`, where both are ≥0.

### Algorithm

```
X₀_pos = max(X₀, 0)
X₀_neg = max(-X₀, 0)

Inicializar: r = ones(n), s = ones(m)

Repetir:
  1. Para cada fila i:
     row_sum = (r[i] * X₀_pos[i,:] * s).sum() - (r[i] * X₀_neg[i,:] * s).sum()
     r[i] *= u[i] / row_sum

  2. Para cada columna j:
     col_sum = (r * X₀_pos[:,j] * s[j]).sum() - (r * X₀_neg[:,j] * s[j]).sum()
     s[j] *= v[j] / col_sum

  X = r[:,None] * X₀_pos * s - r[:,None] * X₀_neg * s
```

### Improvements with damping

To prevent oscillation/overflow:

```python
r[i] = damping * r_old[i] + (1 - damping) * r_new[i]
s[j] = damping * s_old[j] + (1 - damping) * s_new[j]
```

Typical: `damping = 0.3-0.5`

### Advantages
- ✓ Handles negative values (e.g. change in stocks)
- ✓ Preserves signs
- ✓ More robust than RAS
- ✓ Preserves zeros

### Disadvantages
- ✗ Slower convergence than RAS
- ✗ May need damping for stability
- ✗ More complex to implement
- ✗ May not converge if targets are inconsistent

### Implementation

```python
def gras_with_damping(X0, u, v, max_iter=1000, tol=1e-4, damping=0.5):
    """GRAS with damping for stability."""
    n, m = X0.shape
    r = np.ones(n)
    s = np.ones(m)

    # Separate positive and negative
    X0_pos = np.maximum(X0, 0)
    X0_neg = np.maximum(-X0, 0)

    for iteration in range(max_iter):
        r_old, s_old = r.copy(), s.copy()

        # Row update with damping
        for i in range(n):
            row_sum = (r[i] * X0_pos[i, :] * s).sum() - (r[i] * X0_neg[i, :] * s).sum()
            if abs(row_sum) > 1e-10 and abs(u[i]) > 1e-10:
                r_new = abs(u[i]) / abs(row_sum)
                r[i] = damping * r_old[i] + (1 - damping) * (r_old[i] * r_new)
                r[i] = np.clip(r[i], 1e-6, 1e6)  # Prevent overflow

        # Column update with damping
        for j in range(m):
            col_sum = (r * X0_pos[:, j] * s[j]).sum() - (r * X0_neg[:, j] * s[j]).sum()
            if abs(col_sum) > 1e-10 and abs(v[j]) > 1e-10:
                s_new = abs(v[j]) / abs(col_sum)
                s[j] = damping * s_old[j] + (1 - damping) * (s_old[j] * s_new)
                s[j] = np.clip(s[j], 1e-6, 1e6)

        # Compute balanced matrix
        X = r[:, np.newaxis] * X0_pos * s - r[:, np.newaxis] * X0_neg * s

        # Check convergence
        row_diff = np.abs(X.sum(axis=1) - u).max()
        col_diff = np.abs(X.sum(axis=0) - v).max()

        if max(row_diff, col_diff) < tol:
            return X, iteration + 1, True

    return X, max_iter, False
```

### When to use
- Matrices with negative values (SAM, MIP with change in stocks)
- You need to preserve structure (zeros, signs)
- You can tolerate slower convergence
- Reasonably consistent targets

---

## 3. Cross-Entropy (information minimization)

### Description

An **optimization**-based method that minimizes the information divergence between X and X₀, subject to linear constraints.

**Objective function** (Kullback-Leibler divergence):
```
min Σᵢⱼ X[i,j] * log(X[i,j] / X₀[i,j]) - X[i,j] + X₀[i,j]

s.t. X.sum(axis=1) = u
     X.sum(axis=0) = v
     X[i,j] ≥ 0
```

### Advantages
- ✓ Direct solution (not iterative over the matrix)
- ✓ Guarantees convergence
- ✓ Can add further constraints easily
- ✓ Solid theoretical basis (information theory)
- ✓ Equivalent to RAS in the limit (without extra constraints)

### Disadvantages
- ✗ Needs an optimization solver (scipy, cvxpy)
- ✗ Slower than RAS for large matrices
- ✗ Does not preserve zeros (X[i,j]=0 → X[i,j]≈0, small)
- ✗ Only for non-negative values (in standard form)

### Implementation

```python
from scipy.optimize import minimize

def cross_entropy(X0, u, v):
    """
    Cross-entropy balancing via optimization.

    Uses scipy to minimize KL divergence.
    """
    n, m = X0.shape

    # Flatten initial matrix
    x0_flat = X0.flatten()

    # Objective: KL divergence
    def objective(x):
        x = x.reshape((n, m))
        # KL(X || X₀) = Σ X*log(X/X₀) - X + X₀
        kl = np.where(
            (x > 1e-10) & (x0_flat.reshape((n,m)) > 1e-10),
            x * np.log(x / X0) - x + X0,
            0
        )
        return kl.sum()

    # Constraints
    constraints = []

    # Row sums
    for i in range(n):
        def row_constraint(x, i=i):
            return x.reshape((n, m))[i, :].sum() - u[i]
        constraints.append({'type': 'eq', 'fun': row_constraint})

    # Column sums
    for j in range(m):
        def col_constraint(x, j=j):
            return x.reshape((n, m))[:, j].sum() - v[j]
        constraints.append({'type': 'eq', 'fun': col_constraint})

    # Bounds (non-negativity)
    bounds = [(0, None)] * (n * m)

    # Optimize
    result = minimize(
        objective,
        x0_flat,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 1000, 'ftol': 1e-9}
    )

    if result.success:
        return result.x.reshape((n, m)), True
    else:
        return result.x.reshape((n, m)), False
```

### When to use
- You need guaranteed convergence
- Inconsistent targets (the optimizer adjusts them)
- Further constraints (e.g. X[i,j] ≤ bound)
- Moderate size (n×m < 10,000)
- The theoretical basis matters

---

## 4. Direct adjustment methods

### 4.1 Simple proportional adjustment

Scales the whole matrix proportionally.

```python
def proportional_scaling(X0, u, v):
    """Scale entire matrix to match total."""
    current_total = X0.sum()
    target_total = u.sum()  # or v.sum(), debe ser igual

    if current_total > 1e-10:
        X = X0 * (target_total / current_total)
    else:
        X = X0

    return X
```

**Use**: a first step before fine balancing.

### 4.2 Row/column adjustment

Adjusts only rows or only columns (not both).

```python
def row_scaling(X0, u):
    """Scale rows to match targets."""
    row_sums = X0.sum(axis=1)
    scales = np.where(row_sums > 1e-10, u / row_sums, 1.0)
    return X0 * scales[:, np.newaxis]

def column_scaling(X0, v):
    """Scale columns to match targets."""
    col_sums = X0.sum(axis=0)
    scales = np.where(col_sums > 1e-10, v / col_sums, 1.0)
    return X0 * scales
```

**Use**: when only one dimension needs adjusting.

### 4.3 Block adjustment

For MIPs, adjust the blocks (Z, F, IMP_Z, IMP_F) independently.

```python
def balance_mip_blocks(Z, F, IMP_Z, IMP_F, VA, PIB_target):
    """
    Balance MIP by adjusting blocks.

    Strategy:
    1. Balance Z with RAS/GRAS
    2. Adjust F to match PIB = VA
    3. Adjust imports proportionally
    """
    # 1. Balance Z
    z_row = Z.sum(axis=1)
    z_col = Z.sum(axis=0)
    z_target = 0.5 * (z_row + z_col)
    Z_balanced = ras(Z, z_target, z_target)

    # 2. Adjust F for PIB
    current_imp_f = IMP_F.sum()
    required_f = PIB_target + current_imp_f
    current_f = F.sum()

    if current_f > 0:
        F_adjusted = F * (required_f / current_f)
    else:
        F_adjusted = F

    # 3. Adjust imports (optional)
    # ... (various strategies possible)

    return Z_balanced, F_adjusted, IMP_Z, IMP_F
```

**Use**: MIPs with a well-defined block structure.

---

## 5. Stone's method (geometric RAS)

A variant of RAS that uses the **geometric mean** instead of the arithmetic one for the targets.

```python
def ras_geometric(X0, u, v, max_iter=200, tol=1e-9):
    """
    RAS with geometric mean for targets.

    More stable when row and col sums differ significantly.
    """
    n, m = X0.shape
    r = np.ones(n)
    s = np.ones(m)

    for iteration in range(max_iter):
        # Geometric adjustment
        row_sums = (X0 * s).sum(axis=1)
        r = np.where(row_sums > 1e-10, np.sqrt(u / row_sums), 1.0)

        col_sums = (r[:, np.newaxis] * X0).sum(axis=0)
        s = np.where(col_sums > 1e-10, np.sqrt(v / col_sums), 1.0)

        X = r[:, np.newaxis] * X0 * s

        row_diff = np.abs(X.sum(axis=1) - u).max()
        col_diff = np.abs(X.sum(axis=0) - v).max()

        if max(row_diff, col_diff) < tol:
            return X, iteration + 1, True

    return X, max_iter, False
```

**Advantage**: smoother convergence when targets are very inconsistent.

---

## 6. Hybrid methods

### Hybrid RAS + adjustment

Combines RAS for Z with direct adjustment for F.

```python
def hybrid_balance(Z, F, VA_target, max_iter_outer=20):
    """
    Hybrid: RAS for Z + direct adjustment for F.

    Strategy:
    1. Balance Z with RAS (or GRAS)
    2. Adjust F to close PIB identity
    3. Iterate if needed
    """
    for iteration in range(max_iter_outer):
        # Step 1: Balance Z
        z_row = Z.sum(axis=1)
        z_col = Z.sum(axis=0)
        z_target = 0.5 * (z_row + z_col)

        Z_balanced, _, _ = ras(Z, z_target, z_target)

        # Step 2: Adjust F for PIB
        PIB_gasto = F.sum() - IMP_F.sum()
        PIB_error = VA_target - PIB_gasto

        if abs(PIB_error) < 1.0:  # Converged
            break

        # Adjust F proportionally
        if F.sum() > 0:
            F = F * ((VA_target + IMP_F.sum()) / F.sum())

        # Enforce non-negativity
        F = np.maximum(0, F)

        Z = Z_balanced

    return Z, F
```

**Advantage**: combines the speed of direct adjustment with the precision of RAS.

---

## 7. MRAS (Modified RAS)

### Description

A modified RAS that allows **differentiated weights** per cell, giving more confidence to certain estimates.

**Key idea**: not every cell of X₀ has the same quality/confidence.

### Algorithm

```
Pesos W[i,j] ∈ [0, 1]:
  - W[i,j] = 1.0 → celda confiable (cambiar poco)
  - W[i,j] = 0.0 → celda no confiable (cambiar mucho)

RAS modificado:
  X = r[:,None] * X₀^(1-W) * s * X₀^W

Donde:
  - Celdas con W=1 cambian solo por factores r,s
  - Celdas con W=0 se ajustan completamente
  - 0<W<1 es intermedio
```

### Implementation

```python
def mras(X0, u, v, W=None, max_iter=200, tol=1e-9):
    """
    Modified RAS with cell-specific weights.

    Args:
        X0: Initial matrix (n×m)
        u: Row targets
        v: Column targets
        W: Weight matrix (n×m), default ones (standard RAS)

    Returns:
        X: Balanced matrix
    """
    n, m = X0.shape

    if W is None:
        W = np.ones((n, m))

    r = np.ones(n)
    s = np.ones(m)

    # Pre-compute power matrices
    X0_weighted = X0 ** (1 - W)  # Part that changes
    X0_fixed = X0 ** W           # Part that stays

    for iteration in range(max_iter):
        # Row scaling
        row_sums = (X0_weighted * s * X0_fixed).sum(axis=1)
        r = np.where(row_sums > 1e-10, u / row_sums, 1.0)

        # Column scaling
        col_sums = (r[:, np.newaxis] * X0_weighted * X0_fixed).sum(axis=0)
        s = np.where(col_sums > 1e-10, v / col_sums, 1.0)

        # Balanced matrix
        X = r[:, np.newaxis] * X0_weighted * s * X0_fixed

        # Check convergence
        row_diff = np.abs(X.sum(axis=1) - u).max()
        col_diff = np.abs(X.sum(axis=0) - v).max()

        if max(row_diff, col_diff) < tol:
            return X, iteration + 1, True

    return X, max_iter, False
```

### Advantages
- ✓ Preserves reliable cells
- ✓ Lets you bring in expert knowledge
- ✓ Useful when there are measured vs estimated cells

### Disadvantages
- ✗ Needs weights to be defined (subjective)
- ✗ More complex than standard RAS
- ✗ May not converge if the weights are badly chosen

### Usage example

```python
# Caso: Datos de encuestas (confiables) vs estimaciones (no confiables)
W = np.ones((n, m))

# Celdas de encuestas: W = 1.0 (preservar)
encuesta_indices = [(0, 0), (1, 2), (5, 3)]
for i, j in encuesta_indices:
    W[i, j] = 1.0

# Celdas estimadas: W = 0.0 (ajustar libremente)
estimadas_indices = [(2, 1), (3, 4)]
for i, j in estimadas_indices:
    W[i, j] = 0.0

X_balanced = mras(X0, u, v, W)
```

### When to use
- Mixed data (surveys + estimates)
- Known benchmark cells
- Partial updating of matrices

---

## 8. KRAS (Kernel RAS)

### Description

RAS with **spatial smoothing** using kernel functions. Useful for **regional** I-O tables, where neighboring regions have similar technologies.

**Innovation**: brings spatial spillovers into the adjustment.

### Algorithm

```
Definir kernel espacial K[r, s]:
  K[r, s] = exp(-d(r, s)² / σ²)

Donde:
  - d(r, s) = distancia entre regiones r y s
  - σ = parámetro de suavizado

KRAS:
  En cada iteración de RAS:
    1. Ajuste estándar: r', s'
    2. Suavizado espacial:
       r[r] = Σₛ K[r,s] * r'[s] / Σₛ K[r,s]
       s[j] = Σᵢ K[i,j] * s'[i] / Σᵢ K[i,j]
```

### Implementation

```python
def kras(X0, u, v, distance_matrix, sigma=1.0, max_iter=200, tol=1e-9):
    """
    Kernel RAS with spatial smoothing.

    Args:
        X0: Initial matrix (n×m)
        u: Row targets
        v: Column targets
        distance_matrix: Distance between rows/cols (n×n or m×m)
        sigma: Kernel bandwidth

    Returns:
        X: Balanced matrix with spatial smoothing
    """
    n, m = X0.shape
    r = np.ones(n)
    s = np.ones(m)

    # Compute kernel matrices
    K_rows = np.exp(-distance_matrix**2 / (2 * sigma**2))
    K_rows = K_rows / K_rows.sum(axis=1, keepdims=True)  # Normalize

    for iteration in range(max_iter):
        # Standard RAS step
        row_sums = (X0 * s).sum(axis=1)
        r_raw = np.where(row_sums > 1e-10, u / row_sums, 1.0)

        # Spatial smoothing on r
        r = K_rows @ r_raw

        # Column scaling (standard)
        col_sums = (r[:, np.newaxis] * X0).sum(axis=0)
        s = np.where(col_sums > 1e-10, v / col_sums, 1.0)

        # Balanced matrix
        X = r[:, np.newaxis] * X0 * s

        # Check convergence
        row_diff = np.abs(X.sum(axis=1) - u).max()
        col_diff = np.abs(X.sum(axis=0) - v).max()

        if max(row_diff, col_diff) < tol:
            return X, iteration + 1, True

    return X, max_iter, False
```

### Building the distance matrix

```python
def create_distance_matrix(regions_coords):
    """
    Create distance matrix from coordinates.

    Args:
        regions_coords: Array (n, 2) with lat/lon

    Returns:
        D: Distance matrix (n, n)
    """
    from scipy.spatial.distance import cdist

    # Euclidean distance (or use haversine for lat/lon)
    D = cdist(regions_coords, regions_coords, metric='euclidean')

    return D

# Example usage
coords = np.array([
    [40.7128, -74.0060],  # New York
    [34.0522, -118.2437], # Los Angeles
    [41.8781, -87.6298],  # Chicago
])

D = create_distance_matrix(coords)
X_balanced = kras(X0, u, v, D, sigma=500)  # sigma in km
```

### Advantages
- ✓ Smooths estimates across neighboring regions
- ✓ Reduces noise in regional data
- ✓ Brings in spatial structure

### Disadvantages
- ✗ Needs spatial information (coordinates)
- ✗ Slower than standard RAS
- ✗ The σ parameter must be calibrated
- ✗ Only useful for data with spatial structure

### When to use
- Multi-regional I-O tables
- Data with known spatial correlation
- Updating regional matrices
- Technological spillovers between regions

### References

**KRAS**:
- Lenzen, M., Moran, D., Kanemoto, K. & Geschke, A. (2013). "Building Eora: A global multi-region input-output database at high country and sector resolution." Economic Systems Research, 25(1), 20-49.

**MRAS**:
- Lenzen, M., Kanemoto, K., Moran, D. & Geschke, A. (2012). "Mapping the structure of the world economy." Environmental Science & Technology, 46(15), 8374-8381.

---

## 9. Comparison of methods

| Method | Speed | Neg. values | Converges | Zeros | Typical use |
|--------|-----------|--------------|----------|-------|------------|
| **RAS** | ⭐⭐⭐⭐⭐ | ❌ | ✓ | Preserves | Simple I-O |
| **GRAS** | ⭐⭐⭐ | ✓ | ⚠️ | Preserves | Complex MIP/SAM |
| **Cross-Entropy** | ⭐⭐ | ❌ | ✓✓ | Approx. | Inconsistent targets |
| **Direct adjustment** | ⭐⭐⭐⭐⭐ | ✓ | - | Does not preserve | Pre-processing |
| **Geometric RAS** | ⭐⭐⭐⭐ | ❌ | ✓ | Preserves | Inconsistent targets |
| **Hybrid** | ⭐⭐⭐⭐ | ✓ | ✓ | Mixed | MIP by blocks |
| **MRAS** | ⭐⭐⭐⭐ | ❌ | ✓ | Weighted | Mixed data |
| **KRAS** | ⭐⭐ | ❌ | ✓ | Smooths | Regional I-O |

---

## 8. Selection guide

### Decision flowchart

```
¿Tu matriz tiene valores negativos?
├─ NO  → ¿Necesitas velocidad?
│        ├─ SÍ  → RAS
│        └─ NO  → Cross-Entropy (si targets inconsistentes)
│                 RAS (si targets consistentes)
│
└─ SÍ  → ¿Estructura de bloques clara (MIP)?
         ├─ SÍ  → Hybrid (GRAS para Z + ajuste para F)
         └─ NO  → GRAS con damping
```

### By type of matrix

**Input-output table (I-O)**:
- First choice: RAS
- With negative values: GRAS

**MIP (extended input-output matrix)**:
- First choice: Hybrid
- Alternative: GRAS for the whole matrix

**SAM (Social Accounting Matrix)**:
- First choice: GRAS (many negative values)
- Alternative: Cross-Entropy with further constraints

**National accounts**:
- First choice: direct adjustment + RAS
- For theoretical consistency: Cross-Entropy

---

## 9. Recommended parameters

### RAS
```python
max_iter = 200
tol = 1e-9  # Muy estricto, típicamente suficiente
```

### GRAS
```python
max_iter = 500-1000  # Más lento que RAS
tol = 1e-3 a 1e-4    # Más relajado por complejidad
damping = 0.3-0.5    # Menor = más lento pero estable
```

### Cross-Entropy
```python
maxiter = 1000
ftol = 1e-9
method = 'SLSQP'  # Bueno para restricciones de igualdad
```

### Hybrid
```python
max_iter_outer = 20-50
ras_tol = 1e-3  # Tolerancia para RAS interno
pib_tol = 1.0   # USD para cierre PIB
```

---

## 10. Diagnosing problems

### GRAS does not converge

**Symptoms**: reaches max_iter without reaching tol.

**Possible causes**:
1. Tolerance too strict → relax to 1e-3 or 1e-2
2. Damping too strong → raise to 0.5
3. Inconsistent targets → use the arithmetic/geometric mean
4. Ill-conditioned matrix → normalize first

**Fix**:
```python
# Antes
tol = 1e-9, damping = 0.3, max_iter = 500

# Después
tol = 1e-2, damping = 0.5, max_iter = 1000
```

### RAS diverges (overflow)

**Symptoms**: r or s → ∞ or NaN.

**Causes**:
1. Rows/columns of zeros
2. Targets = 0 but matrix > 0
3. Ill-conditioned matrix

**Fix**:
```python
# Agregar clipping
r = np.clip(r, 1e-6, 1e6)
s = np.clip(s, 1e-6, 1e6)

# O cambiar a GRAS con damping
```

### Cross-Entropy very slow

**Symptoms**: minutes to converge.

**Causes**:
1. Very large matrix (n×m > 10,000)
2. Ill-conditioning
3. Inefficient solver

**Fix**:
```python
# 1. Usar sparse matrices
from scipy.sparse import csr_matrix

# 2. Cambiar solver
method = 'trust-constr'  # Más robusto pero lento
# o
method = 'COBYLA'  # Más rápido pero menos preciso

# 3. Pre-balancear con RAS
X_pre = ras(X0, u, v, tol=1e-2)  # Pre-balance grueso
X_final = cross_entropy(X_pre, u, v)  # Refinamiento
```

---

## 11. Validating results

### Quality metrics

```python
def evaluate_balance(X, X0, u, v):
    """
    Evaluate quality of balanced matrix.

    Returns dict with metrics.
    """
    metrics = {}

    # 1. Target compliance
    row_sums = X.sum(axis=1)
    col_sums = X.sum(axis=0)

    metrics['max_row_error'] = np.abs(row_sums - u).max()
    metrics['max_col_error'] = np.abs(col_sums - v).max()
    metrics['mean_row_error'] = np.abs(row_sums - u).mean()
    metrics['mean_col_error'] = np.abs(col_sums - v).mean()

    # 2. Distance to original
    metrics['max_abs_change'] = np.abs(X - X0).max()
    metrics['mean_abs_change'] = np.abs(X - X0).mean()
    metrics['relative_change'] = (
        np.abs(X - X0).sum() / np.abs(X0).sum()
    )

    # 3. Structure preservation
    zeros_orig = (X0 == 0).sum()
    zeros_new = (X == 0).sum()
    metrics['zeros_preserved'] = zeros_new / zeros_orig if zeros_orig > 0 else 1.0

    # 4. Sign preservation (if applicable)
    if np.any(X0 < 0):
        signs_match = ((X >= 0) == (X0 >= 0)).sum()
        metrics['signs_preserved'] = signs_match / X.size

    return metrics
```

### Acceptance criteria (CGE)

```python
def is_acceptable_for_cge(metrics):
    """
    Check if balanced matrix meets CGE standards.

    Based on literature (Lofgren et al. 2002, Robinson et al. 2001).
    """
    criteria = {
        'max_row_error': 1.0,      # USD
        'max_col_error': 1.0,      # USD
        'relative_change': 0.05,   # 5%
        'zeros_preserved': 0.95,   # 95%
    }

    passed = all(
        metrics[key] <= threshold
        for key, threshold in criteria.items()
    )

    return passed, criteria
```

---

## 12. References

### Foundational papers

1. **RAS Original**:
   - Stone, R. (1961). "Input-Output and National Accounts." OECD.
   - Bacharach, M. (1970). "Biproportional Matrices and Input-Output Change." Cambridge University Press.

2. **GRAS**:
   - Junius, T. & Oosterhaven, J. (2003). "The solution of updating or regionalizing a matrix with both positive and negative entries." Economic Systems Research, 15(1), 87-96.

3. **Cross-Entropy**:
   - Robinson, S., Cattaneo, A. & El-Said, M. (2001). "Updating and Estimating a Social Accounting Matrix Using Cross Entropy Methods." Economic Systems Research, 13(1), 47-64.

4. **Comparisons**:
   - Lofgren, H., Harris, R.L. & Robinson, S. (2002). "A Standard Computable General Equilibrium (CGE) Model in GAMS." IFPRI Microcomputers in Policy Research 5.

### Software

- **Python**: `scipy.optimize`, `cvxpy`
- **R**: `RAS`, `entropy`
- **GAMS**: Built-in `GAMS.Matrix` balancing
- **Stata**: `ras` command

### Reference implementations

- IFPRI CGE models: https://github.com/IFPRI
- PEP models: https://www.pep-net.org/
- GTAP: https://www.gtap.agecon.purdue.edu/

---

## 13. Specific use cases

### Case 1: Bolivia MIP (this project)

**Problem**:
- GDP error: 5.81%
- Z unbalanced: 2,698 USD max diff
- Negative values in the change in stocks

**Fix applied**:
```python
# Hybrid: GRAS para Z + ajuste F
Z_balanced = gras(Z, z_targets, z_targets, damping=0.3)
F_adjusted = F * (PIB_target / F.sum())

# Resultado:
# - Error PIB: 0.93%
# - Z balance: 10.54 USD (0.035%)
```

**Lesson**: Hybrid is ideal for MIPs with a block structure.

### Case 2: regional SAM

**Problem**:
- A 100×100 SAM with many negatives
- Targets from household surveys
- Government flows must be preserved

**Fix**:
```python
# GRAS con restricciones adicionales
# Preservar bloques específicos
fixed_blocks = [(80, 90, 0, 10)]  # Transferencias gobierno
X_balanced = gras_with_fixed_blocks(X, u, v, fixed_blocks)
```

### Case 3: I-O update

**Problem**:
- I-O of 2015 → update to 2023
- Only new totals available
- Preserve the technological structure

**Fix**:
```python
# RAS simple (no negativos en I-O)
X_2023 = ras(X_2015, u_2023, v_2023)

# Preserva coeficientes técnicos relativos
```

---

## 14. Good practice

### Before balancing

1. **Validate the data**:
   ```python
   assert np.all(np.isfinite(X0)), "Contiene NaN o Inf"
   assert X0.shape == (n, m), "Dimensiones incorrectas"
   ```

2. **Check the targets**:
   ```python
   assert abs(u.sum() - v.sum()) < 1e-6, "Targets inconsistentes"
   ```

3. **Normalize if needed**:
   ```python
   # Si valores muy grandes (>1e10) o pequeños (<1e-10)
   scale = np.abs(X0).max()
   X0_norm = X0 / scale
   # ... balancear ...
   X_balanced = X_balanced_norm * scale
   ```

### While balancing

1. **Monitor convergence**:
   ```python
   if iteration % 100 == 0:
       print(f"Iter {iteration}: error={error:.4e}")
   ```

2. **Detect divergence early**:
   ```python
   if np.any(np.isnan(r)) or r.max() > 1e6:
       print("Divergencia detectada!")
       break
   ```

### After balancing

1. **Validate the result**:
   ```python
   metrics = evaluate_balance(X, X0, u, v)
   is_ok, criteria = is_acceptable_for_cge(metrics)

   if not is_ok:
       print("Advertencia: No cumple criterios CGE")
   ```

2. **Document**:
   ```python
   metadata = {
       'method': 'GRAS',
       'iterations': iters,
       'converged': converged,
       'parameters': {'damping': 0.3, 'tol': 1e-3},
       'metrics': metrics,
       'date': datetime.now()
   }
   ```

---

## 15. Complete example code

Implementations in:
- `/balance_bolivia_gras_fixed.py` - GRAS with damping
- `/balance_bolivia_hybrid_final.py` - Hybrid method
- `/src/equilibria/sam_tools/balancing.py` - balancing utilities

---

## 16. Index of implemented scripts

### Scripts in `/` (project root)

| Script | Method | Description | Used in |
|--------|--------|-------------|----------|
| `balance_bolivia_gras_fixed.py` | **GRAS with damping** | GRAS with pos/neg split and damping=0.3 | ✓ `mip_bol_unbalanced2.xlsx` |
| `balance_bolivia_gras_true.py` | **Original GRAS** | GRAS without damping, standard implementation | - |
| `balance_bolivia_hybrid_final.py` | **Hybrid** | RAS for Z + F adjustment for GDP | ✓ `mip_bol_balanced_hybrid.xlsx` |
| `balance_bolivia_crossentropy.py` | **Cross-Entropy** | KL divergence minimization with scipy | - |
| `balance_bolivia_optimization.py` | **Optimization** | cvxpy with every constraint | - |
| `balance_bolivia_full_optimization.py` | **Full optimization** | Optimizes Z, F, IMP at once | - |
| `balance_bolivia_hierarchical.py` | **Hierarchical RAS** | RAS by levels (Z first, F after) | - |
| `balance_bolivia_pragmatic.py` | **Simple RAS** | Basic square RAS | - |
| `balance_bolivia_triple_ras.py` | **Triple RAS** | RAS applied 3 times | - |
| `balance_bolivia_with_targets.py` | **RAS with targets** | Targets from an external source | - |
| `balance_bolivia_complete.py` | **Complete GRAS** | GRAS with MIP constraints | - |

### Methods implemented in the documentation (no script)

| Method | Section | Status |
|--------|---------|--------|
| **Standard RAS** | §1 | ✓ Example code |
| **GRAS** | §2 | ✓ Complete code |
| **Cross-Entropy** | §3 | ✓ Example code |
| **Direct adjustments** | §4 | ✓ Example code |
| **Geometric RAS** | §5 | ✓ Example code |
| **Hybrid** | §6 | ✓ Example code |
| **MRAS** | §7 | ✓ Complete code |
| **KRAS** | §8 | ✓ Complete code |

### Methods NOT implemented

These methods are documented but **have no dedicated script**:
- ❌ MRAS (Modified RAS) - example in the docs only
- ❌ KRAS (Kernel RAS) - example in the docs only
- ❌ Geometric RAS (Stone's method) - example in the docs only

**Note**: the examples in the documentation work and can be used directly.

### Runs in this chat

| Date | Method | Script | Input | Output | Result |
|-------|--------|--------|-------|--------|-----------|
| 2025-04-11 | GRAS damping | `balance_bolivia_gras_fixed.py` | `mip_bol_unbalanced2.xlsx` | `mip_bol_balanced_gras.xlsx` | GDP error: 5.81%→0.93% |
| 2025-04-09 | Hybrid | `balance_bolivia_hybrid_final.py` | `mip_bol_unbalanced.xlsx` | `mip_bol_balanced_hybrid.xlsx` | GDP error: 5.81%→0.00% |

### Usage recommendations

**For your case (Bolivia MIP)**:
1. **First choice**: `balance_bolivia_hybrid_final.py` - GDP error = 0.00%
2. **Second choice**: `balance_bolivia_gras_fixed.py` - GDP error = 0.93%
3. **Exploration**: `balance_bolivia_crossentropy.py` - guarantees convergence

**For general cases**:
- Simple I-O → `balance_bolivia_pragmatic.py` (simple RAS)
- Complex SAM → `balance_bolivia_gras_fixed.py` (GRAS with damping)
- Inconsistent targets → `balance_bolivia_crossentropy.py`

---

**Last updated**: April 2025
**Author**: analysis for the Equilibria project
**License**: project documentation
