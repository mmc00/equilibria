# Métodos de Balanceo de MIP: Literatura y Práctica

## 1. Identidades que Debe Cumplir una MIP Balanceada

Una Matriz Insumo-Producto (MIP) balanceada debe satisfacer simultáneamente:

### 1.1. Balance por Producto (Filas)

Para cada producto `i`:
```
Oferta Total = Demanda Total

Producción_doméstica[i] + Importaciones[i] =
    Σ(Uso_intermedio[i,j]) + C[i] + G[i] + I[i] + ΔS[i] + X[i]
```

### 1.2. Balance por Sector (Columnas)

Para cada sector `j`:
```
Insumos Totales = Producción Total

Σ(Insumos_intermedios[i,j]) + VA[j] + Impuestos[j] = Producción[j]
```

### 1.3. Identidad PIB

```
PIB (producción) = Σ(VA[j] + Impuestos[j])

PIB (gasto) = C + G + I + ΔS + (X - M)

PIB (producción) = PIB (gasto)
```

## 2. Métodos Estándar en la Literatura

### 2.1. Método RAS (Biproportional Scaling)

**Referencia**: Stone (1961), Bacharach (1970)

**Limitación**: Solo balancea una matriz cuadrada (flujos intermedios), no el sistema completo.

**Uso apropiado**:
- Actualizar MIP antigua con nuevos totales de fila/columna
- Balance preliminar de flujos intermedios

**NO apropiado para**: Balance completo de un sistema con demanda final y VA.

### 2.2. Cross-Entropy Minimization (Método de Mínima Información)

**Referencia**: Robinson, Cattaneo & El-Said (2001), Golan, Judge & Miller (1996)

**Idea**: Minimizar la "distancia informacional" entre la matriz desbalanceada y la balanceada, sujeto a restricciones de balance.

**Función objetivo**:
```
min Σ Σ x[i,j] * log(x[i,j] / x₀[i,j])

sujeto a:
- Balance de filas (oferta = demanda)
- Balance de columnas (insumos = producción)
- Identidad PIB
- x[i,j] ≥ 0
```

**Ventajas**:
- ✅ Balancea todo el sistema simultáneamente
- ✅ Minimiza cambios a los datos originales
- ✅ Puede incorporar restricciones adicionales
- ✅ Tratamiento formal de incertidumbre

**Desventajas**:
- ⚠️ Requiere optimización no lineal (más complejo computacionalmente)
- ⚠️ Puede tener múltiples soluciones locales

### 2.3. Método de Ajuste Proporcional Generalizado (GRAS)

**Referencia**: Junius & Oosterhaven (2003)

**Idea**: Extensión de RAS que ajusta tanto flujos intermedios como demanda final/VA.

**Algoritmo**:
1. Calcular totales objetivo por fila y columna
2. Aplicar RAS a submatriz de flujos intermedios
3. Ajustar proporcionalmente demanda final
4. Ajustar proporcionalmente VA
5. Iterar hasta convergencia

**Ventajas**:
- ✅ Más simple que cross-entropy
- ✅ Balancea sistema completo
- ✅ Converge rápidamente

### 2.4. Método de Reconciliación de Cuentas Nacionales (SNA)

**Referencia**: United Nations SNA 2008, Chapter 26

**Idea**: Ajustar componentes según su confiabilidad estadística.

**Jerarquía típica de confiabilidad**:
1. **Más confiable**: Valor Agregado (de cuentas nacionales)
2. **Confiable**: Exportaciones, Importaciones (de aduanas)
3. **Menos confiable**: Flujos intermedios (de encuestas)
4. **Variable**: Demanda final componentes (diferentes fuentes)

**Procedimiento**:
1. Fijar datos más confiables (VA, X, M)
2. Ajustar datos menos confiables (Z, C, G, I)
3. Aplicar RAS/GRAS al residuo

## 3. Recomendación para MIP Bolivia

### 3.1. Diagnóstico

La MIP Bolivia tiene:
- ✅ VA desagregado (confiable - de cuentas nacionales)
- ✅ Importaciones por sector (confiable - de aduanas)
- ⚠️ Flujos intermedios (menos confiable - encuestas)
- ⚠️ Demanda final (mixta - varias fuentes)

**Desbalance identificado**:
```
PIB (VA) = 48,614
PIB (gasto) = 50,128
Diferencia = 1,514 (3.1%)
```

### 3.2. Método Recomendado: GRAS + Ajuste de Demanda Final

**Paso 1: Fijar componentes confiables**
- VA por sector: FIJO (viene de cuentas nacionales)
- Importaciones totales: FIJO (viene de aduanas)

**Paso 2: Calcular discrepancia estadística**
```
Discrepancia = PIB(gasto) - PIB(VA)
             = 50,128 - 48,614
             = 1,514
```

**Paso 3: Distribuir discrepancia**

Opción A: Ajustar demanda final proporcionalmente
```
C_ajustado = C * (PIB_VA / PIB_gasto)
G_ajustado = G * (PIB_VA / PIB_gasto)
...
```

Opción B: Crear cuenta "discrepancia estadística"
```
PIB = C + G + I + ΔS + (X - M) + DISC
```

**Paso 4: Aplicar GRAS para balance fino**

### 3.3. Implementación Práctica

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

## 4. Alternativas Si Datos Son Muy Inconsistentes

### 4.1. Método Híbrido (Recomendado para Bolivia)

1. **Preservar componentes confiables**:
   - VA total y estructura sectorial
   - Total de importaciones
   - Total de exportaciones

2. **Ajustar componentes intermedios**:
   - Matriz Z: RAS con restricciones
   - Consumo privado: ajuste proporcional
   - Inversión: ajuste residual

3. **Aceptar discrepancia estadística** pequeña (< 1%)

### 4.2. Construcción de MIP Sintética

Si la inconsistencia es muy grande (>5%), considerar:

1. **Reconstruir desde VA**:
   - Usar VA como base
   - Aplicar coeficientes técnicos promedio (regionales/internacionales)
   - Calibrar con totales conocidos

2. **Método de Matriz de Uso/Oferta (SUTs)**:
   - Construir matrices de Uso y Oferta separadas
   - Balancear cada una independientemente
   - Derivar MIP simétrica

## 5. Criterios de Validación Post-Balance

Una MIP balanceada debe pasar:

### 5.1. Tests Numéricos

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

### 5.2. Tests Económicos

- ✅ Coeficientes técnicos razonables (0 ≤ a[i,j] ≤ 1)
- ✅ Estructura sectorial preservada
- ✅ No se introdujeron flujos negativos
- ✅ Cambios < 10% en celdas individuales (idealmente)

## 6. Software y Herramientas

### Implementaciones Existentes

- **Python**:
  - `pymrio` - Multi-Regional Input-Output analysis
  - `iotables` - European Input-Output tables

- **R**:
  - `ioanalysis` package
  - `sna` (System of National Accounts)

- **GAMS**:
  - GTAP database balancing procedures
  - IFPRI SAM balancing tools

### Para Equilibria

Deberíamos implementar:
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

## Referencias

### Papers Fundamentales

1. **Stone, R. (1961)**. "Input-Output and National Accounts". OECD.
   - Introduce método RAS

2. **Bacharach, M. (1970)**. "Biproportional Matrices and Input-Output Change". Cambridge.
   - Teoría matemática de RAS

3. **Robinson, S., Cattaneo, A., & El-Said, M. (2001)**. "Updating and Estimating a Social Accounting Matrix Using Cross Entropy Methods". Economic Systems Research.
   - Cross-entropy para SAMs

4. **Junius, T., & Oosterhaven, J. (2003)**. "The Solution of Updating or Regionalizing a Matrix with both Positive and Negative Entries". Economic Systems Research.
   - Método GRAS

5. **Golan, A., Judge, G., & Miller, D. (1996)**. "Maximum Entropy Econometrics". Wiley.
   - Fundamentos teóricos de cross-entropy

6. **United Nations (2008)**. "System of National Accounts 2008", Chapter 14 & 26.
   - Estándares oficiales de balance de MIPs

### Software

- **pymrio**: https://github.com/konstantinstadler/pymrio
- **GTAP**: https://www.gtap.agecon.purdue.edu/
- **IFPRI SAM tools**: https://www.ifpri.org/

## Conclusión

Para la MIP de Bolivia:

1. **No es suficiente** balancear solo flujos intermedios
2. **Se debe** balancear el sistema completo
3. **Método recomendado**: GRAS con VA fijo
4. **Objetivo**: Max diff < 0.1, discrepancia PIB < 0.1%

El siguiente paso es implementar `balance_full_mip()` que balancea todo el sistema correctamente.
