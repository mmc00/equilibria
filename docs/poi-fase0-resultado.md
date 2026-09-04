# Backend POI — resultado de la Fase 0

**Fecha:** 2026-09-04 · Rama `perf/poi-fase0` (7 commits, 35 tests verdes)
**Spec:** `2026-09-03-poi-backend-design.md` · **Plan:** `2026-09-03-poi-backend-fase0.md`

## Veredicto

**El supuesto central del spec se sostiene y su riesgo principal no existe. El
proyecto no avanza por una razón distinta: ninguna forma de agrupar las funciones
compiladas escala al 20×41.**

## Las tres preguntas de la Fase 0

| pregunta | respuesta | evidencia |
|---|---|---|
| ¿los bloques corren sobre POI sin reescribirse? | **Sí** | 1,106 de 1,108 celdas; 5 líneas tocadas de ~5,900 |
| ¿el Jacobiano sale 44% más denso? | **No** | nnz **idéntico** en 3 datasets |
| ¿POI construye más rápido? | **Declarar sí, compilar no** | 0.43s vs 0.74s declarando; compilar no escala |

### Paridad estructural (Gate 1) — CERRADA

Ambos backends producen el **mismo conjunto de ecuaciones**, celda por celda:

| dataset | filas | nnz Pyomo | nnz POI |
|---|---|---|---|
| 3×3 | 1,110 | 4,568 | **4,568** |
| 10×7 | 12,502 | 60,393 | **60,393** |
| 15×10 | 34,010 | 180,441 | **180,441** |

Tres coincidencias exactas, contadas por caminos independientes. El riesgo que
gobernaba el spec (POI midió antes 2.58M vs 1.79M, 44% más denso) **no se
materializa**: los dos backends construyen la misma matriz.

La paridad de nombres está verificada con prueba de mutación — quitar una sola
celda dispara el assert, así que el test puede fallar.

## Por qué no avanza

POI compila los evaluadores de autodiff generando texto LLVM IR. Ese paso no escala:

| dataset | filas | declarar | compilar |
|---|---|---|---|
| 3×3 | 1,110 | 0.01s | 0.43s |
| 10×7 | 12,502 | 0.15s | 235s |
| 15×10 | 34,010 | 0.43s | **>3,881s (TIMEOUT)** |
| 20×41 | 395,310 | — | 12× más grande que el 15×10 |

**Referencia: Pyomo construye el 20×41 ENTERO en 17.7 min.** POI no compila el
15×10 —doce veces más chico— en 65.

### Dónde está el coste (medido, 10×7)

| | s | % |
|---|---|---|
| `_compile_evaluators` TOTAL | 208.0 | 100% |
| LLVM `compile_module` | 53.9 | 26% |
| **POI generando el IR** | **154.1** | **74%** |

El cuello no es LLVM optimizando: es POI **escribiendo** el IR en Python. Por eso
bajar el nivel de optimización ayuda poco cuando el modelo crece — solo toca ese 26%.

### Las dos formas de agrupar, ambas medidas

POI compila una función por *grupo* de grafos similares. Hay dos extremos:

| modo | compilar 3×3 | `max ny` | IR 3×3 |
|---|---|---|---|
| un grafo por **fila** | 4.29s | 1 | 2.10 MB |
| un grafo por **ecuación** | 8.47s | 57 | 5.85 MB |

- **Por fila**: cada celda compila su propia función. El 10×7 genera ~4,288
  funciones y **80.7 MB** de IR.
- **Por ecuación**: las filas comparten función (`ny=57` lo confirma), pero cada
  función debe manejar 57 casos y sale más grande. **El IR crece 2.8× y compilar
  tarda el doble.**

Es un compromiso real, no un parámetro mal puesto: pocas funciones grandes o muchas
chicas, y ninguna de las dos escala. La hipótesis de que compartir función reduciría
el IR fue **refutada por la medición**.

## Lo que queda utilizable

- `PoiBackend` funcional con paridad estructural probada y 35 tests
- `PoiModelAdapter`: los bloques corren sin reescribirse
- `_backend_math.py`: `exp`/`log`/`sqrt` neutrales al backend, y `build_value()`
  para los guards que leen el valor de una variable en tiempo de construcción
- `bench_poi_build.py`: mide build y nnz de ambos backends

El camino Pyomo (el oráculo de paridad) quedó **intacto**: 66 tests verdes.

## Recomendación

**Cerrar la Fase 0 y no planificar las Fases 1-3 por ahora.**

La palanca que POI atacaba —`nl_writer` + `PyomoNLP` ×9, ~7.7 min de los 17.7 del
build— sigue disponible **dentro de Pyomo**, sin cambiar de framework: la interfaz
NLP se reconstruye 9 veces, una por fase, serializando las 395,310 ecuaciones cada
vez. Es la misma clase de redundancia que el andamiaje ya corregido (~219s), y no
depende de que POI escale.

Si POI se retoma, el punto de partida es el diagnóstico de arriba: el problema es la
**generación del IR** (74% del coste), no LLVM ni la densidad del Jacobiano.

## Notas de método

Cuatro mediciones estuvieron a punto de reportarse mal. Quedan anotadas porque
volverían a ocurrir:

1. **El build de POI debe incluir compilar.** Declarar solo registra en el grafo; la
   compilación ocurre dentro de `optimize()`. Medir solo el declarar favorecía a POI
   ~3000×.
2. **`m_jacobian_nnz` es del GRUPO, no por fila.** Multiplicar por `ny` infló 1,978
   en 809,002 y dio un **177× falso** de densidad.
3. **Con grafos por fila, los contadores de POI subestiman**: deduplica 409 filas en
   32 representantes. El nnz se cuenta fila por fila mientras se construye, igual que
   `identify_variables` en Pyomo.
4. **`signal.alarm` no interrumpe código nativo de LLVM** — un corte de 20 min saltó
   a los 3,881s. Para acotar por tiempo hace falta un proceso aparte.

Y una trampa de entorno: **`pip install -q` devuelve rc=0 sin instalar**. Verificar
siempre por `import`. POI importa `tccbox` incondicionalmente aunque no se use TCC.
