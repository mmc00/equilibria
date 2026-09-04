# Backend POI — resultado de la Fase 0

**Fecha:** 2026-09-04 · Rama `perf/poi-fase0`
**Spec:** `2026-09-03-poi-backend-design.md` · **Plan:** `2026-09-03-poi-backend-fase0.md`

## Veredicto

**Las tres preguntas de la Fase 0 se responden a favor.** Los bloques corren sobre
PyOptInterface sin reescribirse, el Jacobiano es idéntico al de Pyomo, y el 20×41 se
construye en **21.8s con POI contra 24.80s de Pyomo**.

El muro que bloqueaba el proyecto no era de la herramienta: eran **dos ecuaciones**
anchas y no lineales. Partirlas con variables auxiliares lo elimina.

Lo que esto **no** demuestra: que el wall de 43.6 min del 20×41 baje. Eso requiere el
arnés completo (9 fases + solve) y no está medido. Ver "Lo que falta".

---

## Las tres preguntas

| pregunta | respuesta | evidencia |
|---|---|---|
| ¿los bloques corren sin reescribirse? | **Sí** | 5 líneas tocadas de ~5,900 |
| ¿el Jacobiano sale 44% más denso? | **No** | nnz **idéntico** en 5 datasets |
| ¿POI construye más rápido? | **Sí** | 20×41: 21.8s vs 24.80s |

### Paridad estructural — CERRADA

| dataset | filas | nnz Pyomo | nnz POI |
|---|---|---|---|
| 3×3 | 1,110 | 4,568 | **4,568** |
| 10×7 | 12,505 | 60,393 | **60,393** |
| 15×10 | 34,043 | 183,067 | **183,067** |
| 20×41 | 395,929 | 2,766,728 | **2,766,728** |

Cinco coincidencias exactas por caminos independientes. El riesgo que gobernaba el
spec (POI midió antes 2.58M vs 1.79M, 44% más denso) **no se materializa**.

La paridad de nombres está verificada con prueba de mutación: quitar una sola celda
dispara el assert.

---

## La causa real del muro: dos ecuaciones

`eq_pwfact` (índice de Fisher mundial) y `eq_pfact` (su gemela regional) suman sobre
todo `(r,f,a)` **dentro de una raíz cuadrada sobre un cociente**: 701 variables en una
fila en el 10×7, 1,501 en el 15×10.

El coste de la derivación simbólica es **superlineal en el ancho de UNA fila**:

| variables en la fila | compilar |
|---|---|
| 21 | 0.08s |
| 81 | 2.21s |
| 161 | **66.35s** |
| 701 | no termina |

Las otras 12,502 filas del modelo se declaran en 0.15s **combinadas**. El problema
nunca fue el número de ecuaciones.

### Ancho NO implica coste

De las 5 familias anchas detectadas, solo 2 costaban:

| ecuación | vars (15×10) | tipo | ¿deriva simbólicamente? |
|---|---|---|---|
| `eq_xtmg` | 1,501 | lineal | **No** |
| `eq_gdpmp` | 706 | cuadrática | **No** |
| `eq_ytax` | 481 | cuadrática | **No** |
| `eq_pwfact` | 1,501 | **no lineal** | **Sí** → partida |
| `eq_pfact` | 151 | **no lineal** | **Sí** → partida |

Solo duele si es ancha **y** no lineal. `eq_xtmg` tiene 1,501 variables y es gratis
porque es `constante × variable`.

### El arreglo

Nombrar cada suma con una variable auxiliar, sembrada al valor de benchmark. Las sumas
pasan a ser filas lineales o cuadráticas (que no se derivan simbólicamente) y el índice
queda sobre cuatro variables. **El álgebra es idéntica.**

| dataset | compilar antes | compilar después |
|---|---|---|
| 10×7 | 235s | **4.31s** |
| 15×10 | **>3,881s (TIMEOUT)** | **1.16s** |
| 20×41 | — | **2.65s** |

---

## Fidelidad

Solve completo base→check→shock en el 10×7, `code=1` en las tres fases, con y sin los
splits:

| | |
|---|---|
| variables comparadas | **60,435** |
| diferencia relativa **máxima** | **2.2e-10** |
| celdas > 1e-3 (tolerancia de paridad) | **0** |
| celdas > 1e-9 | **0** |
| mediana | 1.1e-13 |

El peor caso está en el décimo dígito de `xma[USA,Rice,Textiles,shock]`. Residuales en
el punto base tras los splits: **1e-13**.

### El gate que falla

`test_settle_only_seed_identical_to_full` compara un **hash byte-exacto** de la semilla
y falla. No detecta un error de modelo: detecta que Newton recorre otro camino al haber
filas adicionales, lo que mueve el último dígito. El conteo de semilla no cambia
(20,138) y las auxiliares no entran en ella.

Ese gate se creó para atrapar un **atajo** en `calibrate_base` que alterara la semilla.
Una reformulación algebraicamente equivalente no es ese riesgo, pero un hash exacto no
puede distinguir los dos casos. **Decisión de diseño pendiente del usuario** — no se
tocó.

---

## Lo que falta

**Lo medido es construir el modelo una vez.** El wall real del 20×41 son 43.6 min
(17.7 build + 25.9 solve), y esos 17.7 incluyen calibración, escalado y trabajo
repartido a lo largo de las 9 fases:

| componente | ~min |
|---|---|
| `walk_expression` (13.8M llamadas) | ~8.8 |
| `add_component` | ~7.3 |
| `nl_writer` **×9** | ~3.9 |
| `PyomoNLP.__init__` **×9** | ~3.8 |

Los `×9` son la oportunidad: Pyomo reserializa las 395,310 ecuaciones una vez por
fase. POI compila una sola vez. **Si eso se traduce en ahorro real es una hipótesis
sin medir** — requiere correr POI en el arnés multi-período completo.

El solve (25.9 min, 59%) POI no lo toca.

---

## Notas de método

Cinco diagnósticos se reportaron mal antes de dar con la causa. Todos por **restar
totales en vez de instrumentar la pieza**:

| dicho | real |
|---|---|
| "el muro es LLVM `opt=3`" | LLVM es 27% |
| "74% es generar el IR" | es 3% |
| "es la Hessiana" | 3% del IR |
| "el agrupamiento `ny=1`" | irrelevante |
| "compilar no escala, es un compromiso real" | eran 2 ecuaciones |

**La lección: instrumentar cada función del camino y exigir que "sin explicar" sea 0.**
Fue lo que encontró la causa en un solo intento tras cuatro fallidos.

Otras trampas medidas:

1. **El build de POI debe incluir compilar.** Declarar solo registra en el grafo.
   Medir solo el declarar favorecía a POI ~3000×.
2. **`m_jacobian_nnz` es del GRUPO, no por fila.** Multiplicar por `ny` infló 1,978 en
   809,002 y dio un **177× falso** de densidad.
3. **`signal.alarm` no interrumpe código nativo de LLVM** — un corte de 20 min saltó a
   los 3,881s. Hace falta un proceso aparte.
4. **`pip install -q` devuelve rc=0 sin instalar.** Verificar por `import`. POI importa
   `tccbox` incondicionalmente aunque no se use TCC.
5. **El editable install** apunta al checkout principal: un worktree no prueba su
   código sin `PYTHONPATH=<worktree>/src`.
