# RESULTADO — validación GEMPACK del GTAP 20x41

**Corrido el 2026-08-18 en Windows, RunGTAP 3.75 / GEMPACK, `gtapv7.exe -cmf tm10.cmf`.**

## Respuesta corta

**GEMPACK SÍ resuelve el 20x41.** El fallo `loge`-de-negativo NO es irreducible: lo causa el
*closure* del `tm10.cmf` de este paquete, no el dataset.

## Los dos desenlaces

| corrida | closure | resultado | max residual ratio |
|---|---|---|---|
| `tm10.cmf` (el del paquete) | swap `dpsave(r)=del_tbalry(r)` × 40 regiones | ❌ aborta a los 16 s | — |
| `noswap.cmf` (control) | idéntico pero SIN los swaps | ✅ converge | `1.0538946E-07` |

El control es el mismo archivo con las 40 líneas `swap dpsave(...)` borradas — mismos `sets.har`,
`basedata.har`, `default.prm`, mismo `Shock tm = uniform 10`, mismos `Steps = 8 16 32`, misma
carpeta. **La única diferencia son los swaps.**

## El error, textual (`tm10.log:2725`)

```
     Arithmetic error report
     -----------------------

  %% Error doing submatrix for variable "dpsave",
      in equation "E_u".

     Attempt to take LOGE of zero or of a negative number.
     This occurred first when:

     value of LOGE argument is -0.2668599
     (There is 1 active index)
     index "r" from set "REG", value "Caribbean" (element 29)

     Expression being evaluated is:

       (ALL,r,REG) -DPARSAVE(r) * LOGE{UTILSAVE(r)} * dpsave(r)
```

## Por qué el closure es la causa

El término que revienta es `-DPARSAVE(r) * LOGE{UTILSAVE(r)} * dpsave(r)`, y GEMPACK lo evalúa
**armando la submatriz de la variable `dpsave`**. Esa submatriz solo se forma si `dpsave` es
ENDÓGENA — que es precisamente lo que hacen los `swap dpsave(r)=del_tbalry(r)`. Bajo el closure
estándar `dpsave` es exógena y sin shock, la columna nunca se arma, y el `LOGE` negativo nunca
se evalúa.

El dato subyacente sí es degenerado, y queda registrado en `tm10-assert-arith-fail.har` (volcado
por el propio GEMPACK al abortar):

- `UTILSAVE` es negativa en **1 de 41** regiones: `Caribbean` (índice 28, 0-based) `= -0.26685989`
- `DPARSAVE` es negativa en **2 de 41**: índices 28 y 32

O sea: la degeneración existe en el benchmark, pero es *inerte* mientras `dpsave` no se libere.
El closure del paquete la activa.

## Corroboración cruzada

La solución del control **coincide exactamente** con la fixture GEMPACK que ya estaba commiteada
en `f3.5-base-calib`:

```
updated_noswap.har  vs  tests/fixtures/gtap7_gempack/updated_gtap7_20x41_tm10_s8-16-32.har
  32 headers comparados, max abs diff = 0
  SAVE(Caribbean) = 412.1044006347656 en ambos
  residual idéntico: 1.0538946E-07
```

Y no es la única: el repo ya tiene **11** soluciones GEMPACK del 20x41 commiteadas, en ambos
closures estándar y en todo el barrido del estudio de linearización —
`updated_gtap7_20x41_tm{10,3,1,0p3,0p1}_s8-16-32.har`, el barrido Gragg `s{4,8,16,32,64}`, y
`updated_gtap7_20x41_tm10_s8-16-32_capfix.har` (RORDELTA=0, residual `1.0868212E-07`).

## Consecuencia para la pregunta original

El README planteaba: *"si falla igual → confirma que el 20x41 es irresoluble en motores estándar y
el solver libre resolvió algo único."* **Esa rama no aplica.** GEMPACK resuelve el 20x41 bajo el
closure estándar, así que la referencia externa existe y la comparación cell-by-cell contra la
solución del solver libre SÍ se puede hacer.

Una advertencia para esa comparación: hay que comparar contra la fixture del closure que
corresponda. `capFix` significa dos cosas distintas en este repo —
en `run_gempack_matrix.py` es RORDELTA=0 (retornos que difieren), mientras que en el `tm10.cmf`
de este paquete es el swap `dpsave`, que es el closure no-estándar retirado en `191a7cf` porque
redistribuye el ahorro. No son el mismo experimento.

## Archivos

Commiteados aquí: `solve_console.txt` (el fallo), `noswap.cmf` (el control), `noswap_console.txt` (la convergencia)
y `tm10-assert-arith-fail.har` (el volcado con `UTILSAVE`/`DPARSAVE`). Los `tm10.log` / `noswap.log`
que produce GEMPACK están en `.gitignore`; los `*_console.txt` llevan el mismo contenido.

`updated_noswap.har` NO se commitea: es byte-por-byte la misma solución que
`tests/fixtures/gtap7_gempack/updated_gtap7_20x41_tm10_s8-16-32.har`, que ya está en el repo.
