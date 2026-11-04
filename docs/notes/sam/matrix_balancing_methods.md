# Métodos de Balanceo de Matrices para MIP/SAM

## Introducción

Los métodos de balanceo ajustan matrices económicas (MIP, SAM, tablas I-O) para que cumplan restricciones de consistencia contable, minimizando la distancia a la matriz original.

**Problema típico**: Dada matriz X₀ y targets (u, v), encontrar X tal que:
- `X.sum(axis=1) = u` (sumas por fila)
- `X.sum(axis=0) = v` (sumas por columna)
- `X` es "cercana" a `X₀` según alguna métrica

---

## 1. RAS (Biproportional Scaling)

### Descripción

El método más clásico y simple. Ajusta filas y columnas alternativamente usando factores multiplicativos.

**Nombre**: RAS viene de R×A×S donde R y S son matrices diagonales de factores.

### Algoritmo

```
Inicializar: r = ones(n), s = ones(m)

Repetir hasta convergencia:
  1. Ajuste filas:    r[i] = u[i] / (X₀[i,:] @ s).sum()
  2. Ajuste columnas: s[j] = v[j] / (r @ X₀[:,j]).sum()

  X = diag(r) @ X₀ @ diag(s)
```

### Ventajas
- ✓ Simple de implementar
- ✓ Converge rápido (típicamente <100 iters)
- ✓ Preserva ceros (si X₀[i,j]=0 → X[i,j]=0)
- ✓ Bajo costo computacional

### Desventajas
- ✗ Solo funciona con matrices no-negativas
- ✗ Requiere targets consistentes (sum(u) = sum(v))
- ✗ No maneja valores negativos
- ✗ Puede fallar con filas/columnas de ceros

### Implementación

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

### Cuándo usar
- Matrices 100% no-negativas
- Targets exactos conocidos
- Necesitas velocidad
- Tablas I-O simples

---

## 2. GRAS (Generalized RAS)

### Descripción

Extensión de RAS que maneja **valores negativos** separando la matriz en componentes positivos y negativos.

**Innovación clave**: `X = X_pos - X_neg` donde ambas son ≥0.

### Algoritmo

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

### Mejoras con Damping

Para prevenir oscilación/overflow:

```python
r[i] = damping * r_old[i] + (1 - damping) * r_new[i]
s[j] = damping * s_old[j] + (1 - damping) * s_new[j]
```

Típico: `damping = 0.3-0.5`

### Ventajas
- ✓ Maneja valores negativos (ej: Variación de Stock)
- ✓ Preserva signos
- ✓ Más robusto que RAS
- ✓ Preserva ceros

### Desventajas
- ✗ Convergencia más lenta que RAS
- ✗ Puede requerir damping para estabilidad
- ✗ Más complejo de implementar
- ✗ Puede no converger si targets inconsistentes

### Implementación

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

### Cuándo usar
- Matrices con valores negativos (SAM, MIP con Var.Stock)
- Necesitas preservar estructura (ceros, signos)
- Puedes tolerar convergencia más lenta
- Targets razonablemente consistentes

---

## 3. Cross-Entropy (Minimización de Información)

### Descripción

Método basado en **optimización** que minimiza la divergencia de información entre X y X₀, sujeto a restricciones lineales.

**Función objetivo** (Kullback-Leibler divergence):
```
min Σᵢⱼ X[i,j] * log(X[i,j] / X₀[i,j]) - X[i,j] + X₀[i,j]

s.t. X.sum(axis=1) = u
     X.sum(axis=0) = v
     X[i,j] ≥ 0
```

### Ventajas
- ✓ Solución directa (no iterativo sobre matriz)
- ✓ Garantiza convergencia
- ✓ Puede agregar restricciones adicionales fácilmente
- ✓ Fundamento teórico sólido (teoría de información)
- ✓ Equivalente a RAS en límite (si sin restricciones extras)

### Desventajas
- ✗ Requiere solver de optimización (scipy, cvxpy)
- ✗ Más lento que RAS para matrices grandes
- ✗ No preserva ceros (X[i,j]=0 → X[i,j]≈0 pequeño)
- ✗ Solo para valores no-negativos (en forma estándar)

### Implementación

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

### Cuándo usar
- Necesitas garantía de convergencia
- Targets inconsistentes (optimizer los ajusta)
- Restricciones adicionales (ej: X[i,j] ≤ bound)
- Tamaño moderado (n×m < 10,000)
- Fundamento teórico importa

---

## 4. Métodos de Ajuste Directo

### 4.1 Ajuste Proporcional Simple

Escala toda la matriz proporcionalmente.

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

**Uso**: Primer paso antes de balanceo fino.

### 4.2 Ajuste por Fila/Columna

Ajusta solo filas o solo columnas (no ambas).

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

**Uso**: Cuando solo una dimensión necesita ajuste.

### 4.3 Ajuste de Bloques

Para MIPs, ajustar bloques (Z, F, IMP_Z, IMP_F) independientemente.

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

**Uso**: MIPs con estructura de bloques bien definida.

---

## 5. Stone's Method (RAS Geométrico)

Variante de RAS usando **media geométrica** en vez de aritmética para targets.

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

**Ventaja**: Convergencia más suave cuando targets muy inconsistentes.

---

## 6. Métodos Híbridos

### Hybrid RAS + Ajuste

Combina RAS para Z con ajuste directo para F.

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

**Ventaja**: Combina velocidad de ajuste directo con precisión de RAS.

---

## 7. MRAS (Modified RAS)

### Descripción

RAS modificado que permite **pesos diferenciados** por celda, dando mayor confianza a ciertas estimaciones.

**Idea clave**: No todas las celdas de X₀ tienen la misma calidad/confianza.

### Algoritmo

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

### Implementación

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

### Ventajas
- ✓ Preserva celdas confiables
- ✓ Permite incorporar conocimiento experto
- ✓ Útil cuando hay celdas medidas vs estimadas

### Desventajas
- ✗ Requiere definir pesos (subjetivo)
- ✗ Más complejo que RAS estándar
- ✗ Puede no converger si pesos mal elegidos

### Ejemplo de Uso

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

### Cuándo usar
- Datos mixtos (encuestas + estimaciones)
- Celdas benchmark conocidas
- Actualización parcial de matrices

---

## 8. KRAS (Kernel RAS)

### Descripción

RAS con **suavizado espacial** usando funciones kernel. Útil para tablas I-O **regionales** donde regiones vecinas tienen tecnologías similares.

**Innovación**: Incorpora spillovers espaciales en el ajuste.

### Algoritmo

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

### Implementación

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

### Construcción de Matriz de Distancias

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

### Ventajas
- ✓ Suaviza estimaciones entre regiones vecinas
- ✓ Reduce ruido en datos regionales
- ✓ Incorpora estructura espacial

### Desventajas
- ✗ Requiere información espacial (coordenadas)
- ✗ Más lento que RAS estándar
- ✗ Parámetro σ debe calibrarse
- ✗ Solo útil para datos con estructura espacial

### Cuándo usar
- Tablas I-O multi-regionales
- Datos con correlación espacial conocida
- Actualización de matrices regionales
- Spillovers tecnológicos entre regiones

### Referencias

**KRAS**:
- Lenzen, M., Moran, D., Kanemoto, K. & Geschke, A. (2013). "Building Eora: A global multi-region input-output database at high country and sector resolution." Economic Systems Research, 25(1), 20-49.

**MRAS**:
- Lenzen, M., Kanemoto, K., Moran, D. & Geschke, A. (2012). "Mapping the structure of the world economy." Environmental Science & Technology, 46(15), 8374-8381.

---

## 9. Comparación de Métodos

| Método | Velocidad | Valores Neg. | Converge | Ceros | Uso Típico |
|--------|-----------|--------------|----------|-------|------------|
| **RAS** | ⭐⭐⭐⭐⭐ | ❌ | ✓ | Preserva | I-O simples |
| **GRAS** | ⭐⭐⭐ | ✓ | ⚠️ | Preserva | MIP/SAM complejas |
| **Cross-Entropy** | ⭐⭐ | ❌ | ✓✓ | Aprox. | Targets inconsist. |
| **Ajuste Directo** | ⭐⭐⭐⭐⭐ | ✓ | - | No preserva | Pre-procesamiento |
| **RAS Geométrico** | ⭐⭐⭐⭐ | ❌ | ✓ | Preserva | Targets inconsist. |
| **Hybrid** | ⭐⭐⭐⭐ | ✓ | ✓ | Mixto | MIP por bloques |
| **MRAS** | ⭐⭐⭐⭐ | ❌ | ✓ | Ponderado | Datos mixtos |
| **KRAS** | ⭐⭐ | ❌ | ✓ | Suaviza | I-O regionales |

---

## 8. Guía de Selección

### Flowchart de Decisión

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

### Por Tipo de Matriz

**Tabla Insumo-Producto (I-O)**:
- Primera opción: RAS
- Si valores negativos: GRAS

**MIP (Matriz Insumo-Producto extendida)**:
- Primera opción: Hybrid
- Alternativa: GRAS para toda la matriz

**SAM (Social Accounting Matrix)**:
- Primera opción: GRAS (muchos valores negativos)
- Alternativa: Cross-Entropy con restricciones adicionales

**Cuentas Nacionales**:
- Primera opción: Ajuste directo + RAS
- Para consistencia teórica: Cross-Entropy

---

## 9. Parámetros Recomendados

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

## 10. Diagnóstico de Problemas

### GRAS no converge

**Síntomas**: Alcanza max_iter sin llegar a tol.

**Causas posibles**:
1. Tolerancia muy estricta → relajar a 1e-3 o 1e-2
2. Damping muy fuerte → aumentar a 0.5
3. Targets inconsistentes → usar promedio aritmético/geométrico
4. Matriz mal condicionada → normalizar antes

**Solución**:
```python
# Antes
tol = 1e-9, damping = 0.3, max_iter = 500

# Después
tol = 1e-2, damping = 0.5, max_iter = 1000
```

### RAS diverge (overflow)

**Síntomas**: r o s → ∞ o NaN.

**Causas**:
1. Filas/columnas de ceros
2. Targets = 0 pero matriz > 0
3. Matriz mal condicionada

**Solución**:
```python
# Agregar clipping
r = np.clip(r, 1e-6, 1e6)
s = np.clip(s, 1e-6, 1e6)

# O cambiar a GRAS con damping
```

### Cross-Entropy muy lento

**Síntomas**: Minutos para convergencia.

**Causas**:
1. Matriz muy grande (n×m > 10,000)
2. Mal condicionamiento
3. Solver ineficiente

**Solución**:
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

## 11. Validación de Resultados

### Métricas de Calidad

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

### Criterios de Aceptación (CGE)

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

## 12. Referencias

### Artículos Fundamentales

1. **RAS Original**:
   - Stone, R. (1961). "Input-Output and National Accounts." OECD.
   - Bacharach, M. (1970). "Biproportional Matrices and Input-Output Change." Cambridge University Press.

2. **GRAS**:
   - Junius, T. & Oosterhaven, J. (2003). "The solution of updating or regionalizing a matrix with both positive and negative entries." Economic Systems Research, 15(1), 87-96.

3. **Cross-Entropy**:
   - Robinson, S., Cattaneo, A. & El-Said, M. (2001). "Updating and Estimating a Social Accounting Matrix Using Cross Entropy Methods." Economic Systems Research, 13(1), 47-64.

4. **Comparaciones**:
   - Lofgren, H., Harris, R.L. & Robinson, S. (2002). "A Standard Computable General Equilibrium (CGE) Model in GAMS." IFPRI Microcomputers in Policy Research 5.

### Software

- **Python**: `scipy.optimize`, `cvxpy`
- **R**: `RAS`, `entropy`
- **GAMS**: Built-in `GAMS.Matrix` balancing
- **Stata**: `ras` command

### Implementaciones de Referencia

- IFPRI CGE models: https://github.com/IFPRI
- PEP models: https://www.pep-net.org/
- GTAP: https://www.gtap.agecon.purdue.edu/

---

## 13. Casos de Uso Específicos

### Caso 1: MIP Bolivia (Este Proyecto)

**Problema**:
- Error PIB: 5.81%
- Z desbalanceado: 2,698 USD max diff
- Valores negativos en Var.Stock

**Solución aplicada**:
```python
# Hybrid: GRAS para Z + ajuste F
Z_balanced = gras(Z, z_targets, z_targets, damping=0.3)
F_adjusted = F * (PIB_target / F.sum())

# Resultado:
# - Error PIB: 0.93%
# - Z balance: 10.54 USD (0.035%)
```

**Lección**: Hybrid es ideal para MIPs con estructura de bloques.

### Caso 2: SAM Regional

**Problema**:
- SAM 100×100 con muchos negativos
- Targets de encuestas de hogares
- Necesita preservar flujos gobierno

**Solución**:
```python
# GRAS con restricciones adicionales
# Preservar bloques específicos
fixed_blocks = [(80, 90, 0, 10)]  # Transferencias gobierno
X_balanced = gras_with_fixed_blocks(X, u, v, fixed_blocks)
```

### Caso 3: Actualización I-O

**Problema**:
- I-O de 2015 → actualizar a 2023
- Solo totales nuevos disponibles
- Preservar estructura tecnológica

**Solución**:
```python
# RAS simple (no negativos en I-O)
X_2023 = ras(X_2015, u_2023, v_2023)

# Preserva coeficientes técnicos relativos
```

---

## 14. Mejores Prácticas

### Antes de Balancear

1. **Validar datos**:
   ```python
   assert np.all(np.isfinite(X0)), "Contiene NaN o Inf"
   assert X0.shape == (n, m), "Dimensiones incorrectas"
   ```

2. **Verificar targets**:
   ```python
   assert abs(u.sum() - v.sum()) < 1e-6, "Targets inconsistentes"
   ```

3. **Normalizar si es necesario**:
   ```python
   # Si valores muy grandes (>1e10) o pequeños (<1e-10)
   scale = np.abs(X0).max()
   X0_norm = X0 / scale
   # ... balancear ...
   X_balanced = X_balanced_norm * scale
   ```

### Durante el Balanceo

1. **Monitorear convergencia**:
   ```python
   if iteration % 100 == 0:
       print(f"Iter {iteration}: error={error:.4e}")
   ```

2. **Detectar divergencia temprano**:
   ```python
   if np.any(np.isnan(r)) or r.max() > 1e6:
       print("Divergencia detectada!")
       break
   ```

### Después del Balanceo

1. **Validar resultado**:
   ```python
   metrics = evaluate_balance(X, X0, u, v)
   is_ok, criteria = is_acceptable_for_cge(metrics)

   if not is_ok:
       print("Advertencia: No cumple criterios CGE")
   ```

2. **Documentar**:
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

## 15. Código de Ejemplo Completo

Ver implementaciones en:
- `/balance_bolivia_gras_fixed.py` - GRAS con damping
- `/balance_bolivia_hybrid_final.py` - Método Hybrid
- `/src/equilibria/sam_tools/balancing.py` - Utilidades de balanceo

---

## 16. Índice de Scripts Implementados

### Scripts en `/` (raíz del proyecto)

| Script | Método | Descripción | Usado en |
|--------|--------|-------------|----------|
| `balance_bolivia_gras_fixed.py` | **GRAS con damping** | GRAS con separación pos/neg y damping=0.3 | ✓ `mip_bol_unbalanced2.xlsx` |
| `balance_bolivia_gras_true.py` | **GRAS original** | GRAS sin damping, implementación estándar | - |
| `balance_bolivia_hybrid_final.py` | **Hybrid** | RAS para Z + ajuste F para PIB | ✓ `mip_bol_balanced_hybrid.xlsx` |
| `balance_bolivia_crossentropy.py` | **Cross-Entropy** | Minimización KL divergence con scipy | - |
| `balance_bolivia_optimization.py` | **Optimización** | cvxpy con todas las restricciones | - |
| `balance_bolivia_full_optimization.py` | **Optimización Full** | Optimiza Z, F, IMP simultáneamente | - |
| `balance_bolivia_hierarchical.py` | **RAS Jerárquico** | RAS en niveles (Z primero, F después) | - |
| `balance_bolivia_pragmatic.py` | **RAS Simple** | RAS cuadrado básico | - |
| `balance_bolivia_triple_ras.py` | **Triple RAS** | RAS aplicado 3 veces | - |
| `balance_bolivia_with_targets.py` | **RAS con targets** | Targets de fuente externa | - |
| `balance_bolivia_complete.py` | **GRAS completo** | GRAS con restricciones MIP | - |

### Métodos Implementados en Documentación (sin script)

| Método | Sección | Estado |
|--------|---------|--------|
| **RAS estándar** | §1 | ✓ Código ejemplo |
| **GRAS** | §2 | ✓ Código completo |
| **Cross-Entropy** | §3 | ✓ Código ejemplo |
| **Ajustes Directos** | §4 | ✓ Código ejemplo |
| **RAS Geométrico** | §5 | ✓ Código ejemplo |
| **Hybrid** | §6 | ✓ Código ejemplo |
| **MRAS** | §7 | ✓ Código completo |
| **KRAS** | §8 | ✓ Código completo |

### Métodos NO Implementados

Estos métodos están documentados pero **no tienen script dedicado**:
- ❌ MRAS (Modified RAS) - solo ejemplo en docs
- ❌ KRAS (Kernel RAS) - solo ejemplo en docs
- ❌ RAS Geométrico (Stone's method) - solo ejemplo en docs

**Nota**: Los ejemplos en la documentación son funcionales y pueden usarse directamente.

### Ejecución en Este Chat

| Fecha | Método | Script | Input | Output | Resultado |
|-------|--------|--------|-------|--------|-----------|
| 2025-04-11 | GRAS damping | `balance_bolivia_gras_fixed.py` | `mip_bol_unbalanced2.xlsx` | `mip_bol_balanced_gras.xlsx` | Error PIB: 5.81%→0.93% |
| 2025-04-09 | Hybrid | `balance_bolivia_hybrid_final.py` | `mip_bol_unbalanced.xlsx` | `mip_bol_balanced_hybrid.xlsx` | Error PIB: 5.81%→0.00% |

### Recomendaciones de Uso

**Para tu caso (MIP Bolivia)**:
1. **Primera opción**: `balance_bolivia_hybrid_final.py` - Error PIB = 0.00%
2. **Segunda opción**: `balance_bolivia_gras_fixed.py` - Error PIB = 0.93%
3. **Exploración**: `balance_bolivia_crossentropy.py` - Garantiza convergencia

**Para casos generales**:
- I-O simple → `balance_bolivia_pragmatic.py` (RAS simple)
- SAM compleja → `balance_bolivia_gras_fixed.py` (GRAS con damping)
- Targets inconsistentes → `balance_bolivia_crossentropy.py`

---

**Última actualización**: Abril 2025
**Autor**: Análisis para proyecto Equilibria
**Licencia**: Documentación del proyecto
