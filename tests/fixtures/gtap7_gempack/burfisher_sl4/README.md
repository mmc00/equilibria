# Soluciones GEMPACK de Burfisher (nus333) para leer niveles directo del .sl4 (#148)

| Archivo | Qué es |
|---|---|
| `TBL45A.sl4`, `ME9C.sl4` | soluciones GEMPACK (gtapuv7.for de 09-SEP-2020, GEMPACK Release 12.1.003, Gragg), corridas el 2026-09-24 con `tools/gempack-oracle` sobre el dataset nus333 de Burfisher 3e |
| `GTAPV7.sl4` | la misma corrida con `gtapv7.for` (no gtapuv7), que **omite 11 variables**: prueba que los argumentos se leen de la lista de la solución (`VCNA`/`VCAR`), no de la del modelo (`VCSP`/`VCSN`) |
| `sl4dump_TBL45A.har`, `sl4dump_ME9C.har`, `sl4dump_GTAPV7.har` | el volcado de cambios % de esos `.sl4` hecho por `sltoht` (SLTOHT 5.53, GEMPACK 11.3.004), el mismo día; cabeceras XX* + una por variable |

Se eligieron dos ejercicios distintos:
- TBL45A tiene un shock simple;
- ME9C tiene shocks de cambio climático en varias variables.

Los `.sl4` son la entrada de la lectura nativa de niveles. Los volcados son el oráculo del cambio (independiente: los escribió `sltoht`, no equilibria).
