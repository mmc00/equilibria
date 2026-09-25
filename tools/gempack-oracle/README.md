# Oráculo GEMPACK para NUS333 (Burfisher 3e)

Rama de uso único: correr los 45 experimentos de NUS333 en una máquina Windows
con GEMPACK + RunGTAP, y traer **los niveles absolutos** para comparar celda a
celda contra equilibria.

**Por qué existe.** Las tablas del libro están impresas en cambios porcentuales
con dos decimales. Eso sirve como referencia pero no como oráculo: sobre bases
chicas un porcentaje exagera (en la tabla 4.5, agricultura es el 0,69% de la
canasta), y el redondeo impreso no permite distinguir un error del modelo de un
error de imprenta. Los `.UPD` de GEMPACK son los datos actualizados **en
niveles**, que es la comparación válida.

## Qué hace falta en la máquina Windows

- GEMPACK con `gemsim`, `tablo` y `sltoht` en el `PATH`
- RunGTAP instalado, por el `GTAPV7.TAB` — **el modelo no está en esta rama**.
  El paquete NUS333 trae datos y experimentos; el modelo es "Standard GTAPv7
  (uncondensed)" y viene con RunGTAP.
- ~2 GB libres

## Pasos

```bat
REM 1. decir donde esta el modelo
set GTAP_MODEL_DIR=C:\RunGTAP\GTAPV7
REM    si no sabes donde:   dir /s /b C:\GTAPV7.TAB

REM 2. verificar (no corre nada, solo reporta)
00-check-env.bat

REM 3. correr los 45
01-run-all.bat

REM 4. empaquetar
02-pack.bat
```

Después, en la Mac:

```bash
python tools/gempack-oracle/read_oracle.py <carpeta-descomprimida>
```

## Qué sale

Por experimento, en `out\<NOMBRE>\`:

| archivo | qué es |
|---|---|
| `.upd` | **datos actualizados = NIVELES post-shock** — lo que importa |
| `.sl4` | solución en cambios % — lo que imprime el libro |
| `.upd.txt` / `.sl4.txt` | los dos anteriores en texto, vía `sltoht` |
| `.log` | log completo del solver |
| `.cmf` | el command file usado, para auditar |
| `SUMMARY.har`, `DECOMP.har`, `GTAPVol.har` | si el modelo los emite |

Y una vez, en `out\_base\`: los **niveles del benchmark sin shock**. Sin eso no
hay contra qué comparar.

`out\_report.txt` lista qué experimento salió bien y cuál falló.

## Decisiones que conviene conocer

**El `.CMF` se arma anexando el `.EXP` verbatim.** Cada `.EXP` ya trae su
`GTAPPARM`, su `Method`, sus `Steps` y su cierre completo — reinterpretarlos
sería introducir un error mío. El script sólo agrega las rutas de archivos y las
líneas `updated file` / `solution file`. Las líneas `!@` son comentarios de
RunGTAP y GEMPACK las ignora.

**No se detiene ante un fallo.** Un `.EXP` que no converge se anota en
`_report.txt` y sigue. Varios experimentos usan `.prm` alternativos y closures no
estándar; si alguno falla, quiero los otros 44.

**El base corre con `default.prm` y método Johansen 1 paso.** Sin shock, el
`.UPD` son los niveles del benchmark. Ojo: los experimentos que usan otro `.prm`
(CES, Cobb-Douglas, elasticidades alteradas) tienen **otra calibración**, así que
comparar sus niveles contra este base único no es válido — para esos hace falta
un base por `.prm`. Está pendiente.

## MEDIDO: 97 celdas en 15 tablas, 0 difieren

```bash
python tools/gempack-oracle/cmp_libro.py          # todas
python tools/gempack-oracle/cmp_libro.py 6.4      # una
```

| tabla | celdas | ok | aprox | difieren | suma \|dif\| | peor \|dif\| |
|---|---|---|---|---|---|---|
| 4.5 | 18 | 18 | 0 | 0 | 0,0372 | 0,0042 |
| 4.6 | 9 | 9 | 0 | 0 | 0,2623 | 0,0481 |
| 5.4 | 4 | 2 | **2** | 0 | 0,0209 | 0,0069 |
| 5.5 | 6 | 6 | 0 | 0 | 0,0809 | 0,0291 |
| 6.2 | 9 | 9 | 0 | 0 | 0,0969 | 0,0474 |
| 6.3 | 4 | 4 | 0 | 0 | 0,1314 | 0,0436 |
| 6.4 | 9 | 9 | 0 | 0 | 0,2836 | 0,0482 |
| 6.5 | 4 | 4 | 0 | 0 | 0,1119 | 0,0489 |
| 6.6 | 3 | 3 | 0 | 0 | 0,0642 | 0,0319 |
| 6.7 | 3 | 3 | 0 | 0 | 0,1102 | 0,0485 |
| 7.5 | 4 | 4 | 0 | 0 | 0,0819 | 0,0465 |
| 7.7 | 6 | 6 | 0 | 0 | 0,0172 | 0,0050 |
| 7.8 | 9 | 9 | 0 | 0 | 0,2795 | 0,0497 |
| 7.9 | 2 | 2 | 0 | 0 | 0,0063 | 0,0035 |
| 9.4 | 7 | 7 | 0 | 0 | 0,1493 | 0,0498 |
| **TOTAL** | **97** | **95** | **2** | **0** | **1,7337** | **0,0498** |

`|dif|` media **0,0179**. La suma de 1,7337 **no es error del modelo**: el libro
publica 1 o 2 decimales, así que hasta media unidad del último decimal es
redondeo de la fuente. La **peor celda es 0,0498** — nunca llega a 0,05, el
umbral de las tablas de 1 decimal.

Contra el umbral estricto de cada tabla, sólo **2 celdas lo exceden**: 0,0058 y
0,0069 contra 0,005, y son el ratio salario/renta de la 5.4, la única fila cuya
fórmula el propio libro declara *"approximately"*. No se aflojó la tolerancia:
se marcan `aprox` y se cuentan aparte.

## Cobertura: las 28 tablas de resultados del libro

El libro tiene 73 tablas numeradas, pero la mayoría no son simulaciones (datos
del SAM, parámetros, ejercicios de práctica, esquemas). De las **28 que
reportan un experimento**:

**15 medidas — todas coinciden.** 4.5, 4.6, 5.4, 5.5, 6.2, 6.3, 6.4, 6.5, 6.6,
6.7, 7.5, 7.7, 7.8, 7.9, 9.4.

**3 que este dataset no permite medir** — y esto es la conclusión, no un
pendiente:

| tabla | por qué |
|---|---|
| **9.3** | Sus tres columnas son **tres modelos distintos**: el libro dice *"base version and two versions with updated tax rates"*. El dataset trae un solo `.EXP` (`TBL93`, shock `ams=2`) y su `qxs[MFG,ROW,USA]` da **2,0031** contra 1,67 / 3,18 / 3,98 de las tres columnas — no coincide con ninguna. Haría falta los `.prm`/`.har` con las tasas rebalanceadas. |
| **5.3** | El `.EXP` **no corre**: usa `= target% 5 from file tfd.shk`, sintaxis que resuelve la GUI de RunGTAP, no el ejecutable. GEMPACK corta con *"Shock value input 'target%' not as expected"*. |
| **8.13** | Igual, con `= rate% 1 from file tpdall.shk`: *"Semantic error in Command file statement 19"*. |

**10 sin `.EXP` en el dataset** — 5.6, 8.2, 8.3, 8.5, 8.7, 8.9, 8.11, 8.12,
8.14, 8.15. El capítulo 8 casi entero (impuestos y bienestar). No hay nada que
correr: el experimento no viene con los datos.

### Las 2 que no corren son ARREGLABLES

Los `.shk` que necesitan **sí están** en `nus333/`, con los valores ya
calculados. Por ejemplo `tpdall.shk`:

```
 3 2  real row_order;
 -4.2836466     -0.98050290    ! %1=AGR
 -9.1956644      -14.864902    ! %1=MFG
-0.64833277      -5.0825901    ! %1=SER
```

Así que el arreglo es reemplazar `= rate% 1 from file X.shk` por el valor
directo del `.shk`. No falta el dato — falta traducir la sintaxis de la GUI.
Con `ME8` son 3 experimentos.

### ⚠️ La letra del `.EXP` NO sigue el orden de filas del libro

En la 5.4, `TBL54A` usa `esubvamfg1.2.prm` y `TBL54B` usa `esubvamfg.8.prm`,
pero el libro lista σ=0,8 primero. Lo mismo en 6.2 (`TBL62B` es capital
específico, `TBL62C` sluggish, y el libro los lista al revés), 6.3 y 6.5.
**Emparejar por el `.prm` o el cierre que declara el `.EXP`, nunca por la
letra.** Emparejado por letra, la 5.4 parece cruzada.

## La tabla 4.5 en detalle

Las **18 celdas** de la tabla 4.5 (pág. 127) —los tres bloques, CDE, CES y
Cobb-Douglas— coinciden **exactamente** con GEMPACK/Johansen: diferencia
**0,00 en las 18** al redondear a los 2 decimales que imprime el libro. La
tercera columna del libro (`qpa+ppa`) también sale de sumar las dos primeras.

| | `ppa` AGR / MFG / SER | `qpa` AGR / MFG / SER |
|---|---|---|
| TBL45A (CDE) | 0,59 / −0,20 / −5,07 | 1,96 / 5,46 / 9,66 |
| TBL45B (CES) | 0,91 / −0,24 / −5,19 | 6,22 / 6,80 / 9,27 |
| TBL45C (C-D) | 0,64 / −0,18 / −5,05 | 4,10 / 4,92 / 9,78 |

Verificado con `cmp_gragg.py` y contra el PDF del libro, no contra notas.

### El método NO es la explicación de nada — medido por contraste

El libro dice al pie de la tabla: *"We use the Johansen solution method"*.
Correr Gragg 2-4-6 lo confirma: **aleja las 6 celdas** de TBL45A.

| celda | Johansen | Gragg | libro | \|J−lib\| | \|G−lib\| |
|---|---|---|---|---|---|
| `ppa[AGR]` | 0,5884 | 0,6152 | 0,59 | 0,0016 | 0,0252 |
| `ppa[MFG]` | −0,1972 | −0,2539 | −0,20 | 0,0028 | 0,0539 |
| `ppa[SER]` | −5,0720 | −4,9004 | −5,07 | 0,0020 | 0,1696 |
| `qpa[AGR]` | 1,9595 | 1,8076 | 1,96 | 0,0005 | 0,1524 |
| `qpa[MFG]` | 5,4572 | 5,1434 | 5,46 | 0,0028 | 0,3166 |
| `qpa[SER]` | 9,6636 | 9,5506 | 9,66 | 0,0036 | 0,1094 |

Johansen **6/6**, Gragg **0/6**. Los ratios Gragg/Johansen van de 0,75 a 1,57
—dispersos, no un factor común— así que Gragg sí aplica corrección no lineal
genuina; simplemente el libro no la usó.

```bat
03-gragg.bat TBL45A
```
```bash
python tools/gempack-oracle/cmp_gragg.py TBL45A
```

`03-gragg.bat` sale a `out-gragg\` para no pisar `out\`, copia el `.EXP`
verbatim **menos** sus líneas `Method`/`Steps`, y verifica que el `.cmf` quede
con **una sola** línea `Method` — con dos, GEMPACK toma la última (Johansen) y
la corrida sería un duplicado silencioso de la base, que se vería como "Gragg
no cambia nada". El guard se ejercitó en la corrida real: quedó una sola.

### ⚠️ Trampa de la fuente: −0,79 / 4,44 NO son de la tabla 4.5

Ese par circuló en notas previas como si fuera el valor del libro para MFG, y
motivó toda la persecución de Gragg. **No está en la tabla 4.5.** El `4,44` es
`qc("MFG","USA")` de la **Tabla ME 3.1** (pág. 318, clave de respuestas pág.
411): otro experimento —un **subsidio del 10% a la producción de manufacturas
de USA**, no el shock de TFP en servicios. Y `−0,79` no aparece en el libro.

La tabla 4.5 da para MFG/CDE **−0,20 y 5,46**, y GEMPACK da −0,1972 y 5,4572.
Nunca hubo brecha. Verificar contra el PDF antes de tomar un valor por "del
libro".

## Lo que esta rama NO resuelve

- No trae el modelo. Si no hay RunGTAP, no hay corrida.
- No compara automáticamente contra equilibria. `read_oracle.py` lee e inventaría;
  el comparador se escribe cuando estén los datos y se sepa qué headers trae el
  `.UPD`.
- El base por `.prm` alternativo, arriba.
- **El escritor HAR no sabe escribir los `.sl4`.** Medido acá: un round-trip
  `read_har` → `write_har` sobre `TBL45A.sl4` colapsa **3 de los 70 headers** a un
  solo valor — y son justo los tres que llevan los resultados:

  | header | original | round-trip |
  |---|---|---|
  | `CUMS` (cambios %) | 1212 | 1 |
  | `LEVB` (niveles pre) | 253 | 1 |
  | `LEVA` (niveles post) | 253 | 1 |

  Son headers sparse (`RESPSE`), un tercer formato además de los 2RFULL/2IFULL que
  arregla el PR #88. Leerlos funciona (es lo que hace `read_sl4.py`); escribirlos
  no, y **falla en silencio** — no tira excepción, devuelve un archivo válido con
  los datos perdidos. Para el oráculo no estorba, porque los `.sl4` sólo se leen,
  pero no uses `write_har` sobre uno. Sin issue abierto todavía.

## No mergear

Esto es andamio de medición, no código de producto. Vive en su rama hasta que los
datos estén traídos; después el comparador puede valer un PR y esto no.
