# Benchmark GTAP 20x41 — misma maquina

- Maquina: 13th Gen Intel(R) Core(TM) i7-1365U · 10 hilos logicos · 31.4 GB RAM · Windows-11-10.0.26200-SP0
- Experimento: gtap7_20x41, tm +10% uniforme (imptx_new = (1+imptx)*1.10 - 1), capFix (RORDELTA=0), capital sluggish, residual = ROW, ifSUB=1

Minutos, mediana de las corridas que convergieron.

| herramienta | convergio | carga+build | solve | wall total | paralelismo |
|---|---|---|---|---|---|
| equilibria (warm) | 1/1 | 29.54 | 30.61 | 61.19 | 1 hilo (Python) |
| GEMPACK (default_threads) | 3/3 | — | — | 1.52 | OpenMP [10] |
| GEMPACK (one_thread) | 3/3 | — | — | 1.45 | OpenMP [1] |
| GAMS + PATH | 0/1 | — | — | — | |

## Lo que NO es igual entre las tres (leer antes de citar)

- **Metodo:** equilibria y GAMS resuelven el sistema no lineal en niveles
  (PATH). GEMPACK resuelve el shock linealizado por Gragg 8/16/32 con
  extrapolacion: es otra aproximacion, con su propio error.
- **Periodos:** equilibria y GAMS resuelven check y shock; GEMPACK solo el
  shock (parte del benchmark).
- **Arranque:** `equilibria (warm)` siembra con una solucion previa del
  20x41 (en el 20x41 era la medicion de ~7 min en la Mac);  `cold`, GEMPACK y GAMS
  arrancan del benchmark. La comparacion justa es `cold`.
- **Hilos:** GEMPACK usa OpenMP; equilibria, un hilo. Por eso GEMPACK se
  mide tambien con `OMP_NUM_THREADS=1`.
- **GAMS:** `iterlim` subido de 1000 a 1.000.000 (con 1000 PATH corta y
  GAMS lo reporta como infactible).
