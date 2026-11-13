# Validación GEMPACK del GTAP 20x41 — paquete Windows

**Objetivo:** correr el GTAP 20x41 en GEMPACK (RunGTAP) para obtener una solución de referencia
que valide la solución que el solver LIBRE de equilibria produjo (`code=1`, resid `2e-9`, shock
tarifa +10%). Hoy el 20x41 está marcado "blocked" en GEMPACK (aborta con `loge`-of-negative en la
utilidad de Caribbean) — este paquete lo re-intenta para responder UNA pregunta:

> **¿GEMPACK resuelve el 20x41, o el fallo `loge`-negativo es irreducible del dataset?**

- Si **converge** → tenemos la referencia; se puede comparar cell-by-cell (%-change de cantidades,
  1pp de tolerancia) contra la solución libre. Validación externa REAL.
- Si **falla igual** (`loge` negativo en Caribbean) → confirma que el 20x41 es irresoluble en GEMPACK
  por una degeneración del dataset/modelo, y que el solver libre resolvió algo que GEMPACK no puede
  (pendiente entonces auditar económicamente la solución libre, no compararla).

---

## Contenido del paquete

| Archivo | Qué es |
|---|---|
| `sets.har` | conjuntos (20 sectores × 41 regiones) |
| `basedata.har` | datos del benchmark (SAM) |
| `default.prm` | parámetros/elasticidades |
| `tm10.cmf` | el experimento: shock uniforme +10% en `tm` (poder de tarifa de importación), closure capFix, residual = ROW |

**Nota importante:** estos `.har` son los datos del benchmark, sin tocar. El bug de datos que
equilibria arregló (PR #46) era del LECTOR de Python — NO afectaba estos archivos ni a GEMPACK (que
usa su lector Fortran nativo). Así que GEMPACK ve exactamente lo que siempre vio. El `loge` negativo,
si reaparece, NO es por datos corruptos — es una propiedad del dataset+modelo GTAPv7.

---

## Requisitos (máquina Windows)

- **RunGTAP** instalado con GEMPACK (el `.cmf` apunta a `C:\runGTAP375\gtapv7`).
- Si tu instalación está en otra ruta, edita la primera línea del `.cmf`:
  `Auxiliary files = C:\tu\ruta\gtapv7 ;`

---

## Pasos

1. **Copiar el paquete** a una carpeta de trabajo en Windows, ej. `C:\val_20x41\`.

2. **Ejecutar GEMPACK:**
   ```
   cd C:\val_20x41
   gtapv7 -cmf tm10.cmf
   ```
   (o desde RunGTAP: File → Run cmf → seleccionar `tm10.cmf`)

3. **Leer el resultado en `tm10.log`.** Dos desenlaces:

   **✅ CONVERGIÓ** — el log dice *"completed without error"* y reporta un `max residual ratio`
   ~1e-7 (Gragg 8/16/32). Se generó `updated_tm10_s8-16-32.har`. → **Tenemos referencia.**

   **❌ FALLÓ** — el log aborta con algo como
   `arithmetic error: LOG of non-positive number in equation E_u, region Caribbean`.
   → **El fallo es irreducible; el 20x41 no resuelve en GEMPACK con estos datos.**

---

## Qué devolver

**Si CONVERGIÓ:** mándame de vuelta:
- `updated_tm10_s8-16-32.har` (la solución en niveles post-shock)
- `tm10.log` (para confirmar el residual)
- Si tienes `sltoht.exe`: exporta también el SL4 de cantidades (comparación más limpia):
  ```
  gtapv7 -cmf tm10.cmf
  sltoht tm10.sl4 sl4dump_tm10.har -SIC
  ```
  y manda `sl4dump_tm10.har` también.

**Si FALLÓ:** mándame las ~30 líneas del `tm10.log` alrededor del error (`LOG`/`E_u`/`Caribbean`)
— para confirmar la ecuación y variable exactas que revientan, y ver si es la misma degeneración que
nuestro solver libre "toleró" o que nuestro modelo maneja con un clamp.

---

## Contexto (por si lo necesita quien lo corre)

- El shock es +10% uniforme en el poder de tarifa de importación (`tm`), global, sobre todas las
  rutas bilaterales — el mismo que resolvió el solver libre de equilibria (λ=1.0).
- Closure: capFix — `swap dpsave(r) = del_tbalry(r)` para toda región NO-residual; ROW absorbe la
  identidad de cuenta de capital.
- El solver libre (Newton + MUMPS + GMIN) obtuvo `code=1`, resid `2.04e-09`, `pfact[USA]=1.0`,
  3.43M vars. Falta saber si es EL equilibrio (este test lo decide) o solo UN punto factible.
