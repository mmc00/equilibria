# Finding: SAM IEEM vs SAM PEP (CRI)

Fecha: 2026-02-19  
Repositorio: `equilibria`

## Resumen ejecutivo

El problema de no convergencia en CRI no viene de "falta de cuentas" en la SAM raw, sino de una conversion estructural IEEM -> PEP donde:

1. Las exportaciones quedan parcialmente mal ruteadas entre cuentas (`I`, `X`, `ROW`).
2. Parte del impuesto a actividades/produccion queda en bucket de impuesto a productos (`TI`) en lugar de gobierno/produccion.

IEEM cierra porque usa otra arquitectura de comercio/impuestos y valida/balancea antes de resolver. PEP exige otra arquitectura contable y, si no se transforma bien, aparecen inconsistencias de absorcion/comercio/fisco.

## Hallazgo 1: bloque de exportaciones mal representado para PEP

### Lo observado en CRI (caso problematico)

En `SAM-CRI-gams.xlsx` (unfixed):

- Existe exportacion en `X.i -> AG.ROW` (EXDO positivo).
- Existe tambien flujo `I.i -> AG.ROW` con el mismo monto.
- No existe contraparte en `J.j -> X.i` (EXO ~ 0).

Ejemplos (monto de exportacion sin contraparte `J->X`):

- `SER`: 3,927,307.20
- `OTHIND`: 3,369,149.84
- `AGR`: 1,825,011.30
- `FOOD`: 700,934.55
- `ADM`: 9,294.31

Ademas, la cuenta `X` queda desbalanceada internamente en esa version:

- `row(X.i) > 0`, pero `col(X.i) = 0`.

### Por que rompe PEP

PEP calibra comercio con estas lecturas:

- `EXDO(i) = SAM('X',i,'AG','ROW')`
- `EXO(j,i) = SAM('J',j,'X',i)`
- `DDO(i) = SUM_j SAM('J',j,'I',i)`

Implementacion: `src/equilibria/templates/pep_calibration_trade.py:401`, `src/equilibria/templates/pep_calibration_trade.py:415`, `src/equilibria/templates/pep_calibration_trade.py:465`.

Si `EXDO` esta en `X->ROW` pero `EXO` no esta en `J->X`, una parte de lo exportado queda contada como oferta domestica (`J->I`). Economicamente: el modelo interpreta que esos bienes van al mercado interno y externo a la vez. Eso tensiona Armington/CET, absorcion y cierres macro.

## Hallazgo 2: impuesto de produccion mal ruteado para PEP

### Lo observado

En la version no fija:

- Flujo relevante en `AG.TI -> J` (844,105.27).
- `AG.GVT -> J` casi nulo para ese componente.

En la version fija:

- `AG.TI -> J` se mueve a `AG.GVT -> J` (mismo total).
- Se ajusta `AG.GVT -> AG.TI` para preservar balance de cuenta.

### Por que importa

Para PEP, el impuesto a produccion se calibra sobre el bloque de gobierno/actividad. Si cae en bucket de impuesto a productos (`TI`) se deforma:

- el wedge de precios por bloque tributario,
- la recaudacion por tipo de impuesto,
- y el cierre ingreso-gasto del gobierno.

## Por que IEEM si cierra con la misma base economica

IEEM usa otra arquitectura de modelado:

1. Comercio exterior por commodity-socio (`SAM(c,r)` y `SAM(r,c)`), no forzado al mismo esquema `X` de PEP.
2. Reconstruccion interna de `QE`, `QM`, `QD`:
   - exportaciones desde `SAM(c,r)` y tasas/margenes asociadas,
   - `QD = QX - QE`.
3. Separacion tributaria mas rica:
   - `taxact`, `taxcom`, `taxvatc`, `taxfact`, `taxfac`, `taximp`, `taxexp`.
4. Validaciones estructurales + rebalanceo:
   - legalidad de flujos y mapeos (`diagnostics-data.inc`),
   - chequeos VAT/social security,
   - `sambal.inc` para balance exacto cuando aplica.

Referencias:

- `cge_babel/ieem/gams/mod.gms:226`
- `cge_babel/ieem/gams/mod.gms:243`
- `cge_babel/ieem/gams/diagnostics-data.inc:114`
- `cge_babel/ieem/gams/diagnostics-data.inc:371`
- `cge_babel/ieem/gams/sambal.inc:191`

## Respuesta a la pregunta clave ("si la SAM balancea, por que falla el modelo")

Porque hay dos niveles distintos de consistencia:

1. **Consistencia contable (fila = columna)**: puede cumplirse.
2. **Consistencia estructural de ecuaciones del modelo**: puede fallar si un flujo esta en la cuenta "equivocada" para esa arquitectura (PEP vs IEEM).

En CRI, el problema es del nivel 2 durante la transformacion a formato PEP.

## Evidencia de QA en este repo

- `output/qa_cri_unfixed.json`:
  - falla `EXP001` (export value balance),
  - falla `MAC002` (GDP proxy closure).
- `output/qa_cri_fixed_after_fix.json`:
  - todos los checks pasan.

## Implicacion operativa

La conversion IEEM -> PEP debe imponer como regla estructural:

1. Coherencia `X`:
   - todo `X.i -> ROW` debe tener contraparte en `J.j -> X.i` (o `I.ij -> X.i` si aplica margen), sin duplicarlo en `I.i -> ROW`.
2. Coherencia tributaria:
   - impuesto de produccion en bloque gobierno/actividad (`AG.GVT -> J`),
   - `TI` reservado para impuesto a productos (`AG.TI -> I`), salvo reglas explicitas y compensadas.

Si no se aplica esto, puede haber SAM "balanceada" y aun asi no converger en PEP.
