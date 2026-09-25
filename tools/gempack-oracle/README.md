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

## Gragg 2-4-6: medir el error de linealización

Con `Method = Johansen; Steps = 1` (lo que traen los `.EXP`), TBL45A reproduce
**4 de las 6 celdas** de la tabla 4.5 dentro del redondeo del libro, que trae dos
decimales:

| celda | GEMPACK/Johansen | libro | \|dif\| |
|---|---|---|---|
| `ppa[AGR,USA]` | 0,5884 | 0,59 | 0,0016 |
| `ppa[SER,USA]` | −5,0720 | −5,07 | 0,0020 |
| `qpa[AGR,USA]` | 1,9595 | 1,96 | 0,0005 |
| `qpa[SER,USA]` | 9,6636 | 9,66 | 0,0036 |
| **`ppa[MFG,USA]`** | **−0,1972** | **−0,79** | **0,5928** |
| **`qpa[MFG,USA]`** | **5,4572** | **4,44** | **1,0172** |

O sea: el libro SÍ es este GEMPACK con estos datos — cuatro celdas coinciden al
cuarto decimal. Lo que no cierra es MFG, y sólo MFG.

La **hipótesis** —no el hallazgo— es que el libro corrió con un método multi-paso
y la brecha de MFG es error de linealización de Johansen. Para medirla:

```bat
REM  en Windows, tras 00-check-env.bat
03-gragg.bat TBL45A
```

y después, donde estén las dos corridas:

```bash
python tools/gempack-oracle/cmp_gragg.py TBL45A
```

`03-gragg.bat` escribe a `out-gragg\` para no pisar la corrida Johansen de `out\`,
que es la línea base. Copia el `.EXP` verbatim **menos** sus líneas `Method`/`Steps`
(que traen Johansen 1), y verifica que el `.cmf` final tenga exactamente una línea
`Method` — si quedaran dos, GEMPACK tomaría la última y la corrida sería un
duplicado silencioso de la base.

`cmp_gragg.py` imprime Johansen y Gragg lado a lado contra el libro y dice si MFG
se acerca, se aleja o no se mueve. **Si no se mueve, la hipótesis queda refutada**
y la causa está en otro lado (datos, parámetros o cierre), no en el método.

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
