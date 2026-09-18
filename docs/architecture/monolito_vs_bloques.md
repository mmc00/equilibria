# GTAP: monolito y bloques

Quien llega nuevo al repo suele suponer que `templates/gtap/gtap_model_equations.py` (el
«monolito») y `blocks/gtap/` (los «bloques») son dos implementaciones alternativas del mismo
modelo, y que la primera es legado a la espera de ser borrada.

**No lo son.** Este documento existe para evitar ese error, que ya costó tiempo una vez.

## Quién depende de quién

Los módulos de `blocks/gtap/*.py` sólo importan `gtap_parameters`. Las menciones a
`gtap_model_equations` en sus docstrings son comentarios de procedencia
(`VERBATIM from ... líneas X-Y`): dicen de dónde se copió cada ecuación, no crean una dependencia.

Eso llevaba a concluir —incorrectamente— que el monolito ya no se usa. Desde 2026-09-16 el
composer **ya no lo importa**: el escalado de benchmark es un módulo aparte y la reflexión
multiperiodo pide su modelo SP por método. Lo que queda es de otra naturaleza:

| Sitio | Qué hace |
|---|---|
| `templates/gtap/__init__.py:58` | Cualquier `import equilibria.templates.gtap` carga el monolito |
| 1 de los 5 gates | El monolito es el oráculo contra GAMS (sólo `nl`) |

Borrar `gtap_model_equations.py` provoca *collection error* en ~34 archivos de test — **incluidos
los 9 que ejercitan sólo bloques**, porque el import falla antes de llegar a ellos.

## Cobertura de ecuaciones

101 familias `Constraint(...)` en el monolito frente a 100 `eq_*` en bloques.

De las 7 que sólo existían en el monolito, 6 se declaraban y se desactivaban en la línea siguiente
—nunca entraban al `.nl`— y ya fueron eliminadas. Queda `eq_pmuv` como único gap real, y sólo
muerde cuando `closure.rmuv` e `imuv` son ambos no vacíos, lo que no ocurre en ningún dataset del
gate. Está documentado en `blocks/gtap/__init__.py:47-53`.

Las 6 exclusivas de bloques (`eq_mfr_{bs,sb,ss}`, `eq_mfw_{bs,sb,ss}` en `closure.py:218-341`) no
son funcionalidad nueva: son la descomposición en constraints de lo que el monolito inlinea como
Expressions dentro de `eq_pfact`/`eq_pwfact`.

## Los bloques son un superconjunto funcional

La asimetría va en el sentido contrario al que sugiere la intuición:

| Modo | Monolito | Bloques |
|---|---|---|
| `capFix`, `capSFix` | sí | sí |
| `capFlex` | **no** (cae a un `else` genérico en `:7276`) | sí (`demand_utility.py:584`) |
| `capFixDp` | **no** | sí (`demand_utility.py:116,166`) |
| `base_calibrated=True` | **no existe** | sí (`gtap_block_model.py:372,440-446`) |
| basis `gempack` | parcial | sí (`gtap_contract.py:309-321`) |

`gtap_contract.py:357` declara 5 `savf_flag`; el monolito implementa 2.

## Por qué se conserva el monolito

Es el oráculo de fidelidad contra **GAMS** en 1 de los 5 gates de
`scripts/gtap/run_parity_gates.py` — sólo `nl`. (Eran 3: `nlp` y `mcp` pasaron a bloques el
2026-09-17.)

| Gate | Camino | Referencia |
|---|---|---|
| `test_gtap7_mcp_parity.py` | **bloques** (desde 2026-09-17) | GAMS/PATH |
| `test_gtap7_nlp_parity.py` | **bloques** (desde 2026-09-17) | GAMS/IPOPT |
| `test_gtap7_nl_parity.py` | monolito (vía `nl_compare`) | `.nl` de GAMS |
| `test_gtap7_gempack_parity.py` | **bloques** | GEMPACK SL4 |
| `tests/templates/gtap_logvalue/` | bloques | Julia/port |

El camino de bloques es el oráculo contra **GEMPACK**, y sólo él puede serlo: ese gate necesita
`capFlex` + `base_calibrated=True`, que el monolito no implementa.

## Deuda: cómo se retiraría

**Primer corte — hecho (2026-09-16).** El escalado de benchmark vive ahora en
`templates/gtap/gtap_benchmark_scaling.py`: cuatro funciones libres que reciben un
`ScalingContext` en vez de `self`. El monolito las conserva como delegadores de una línea, así
que su comportamiento no cambia y sigue siendo el oráculo. El composer ya no construye un
`GTAPModelEquations` para tomarle prestados métodos privados.

Verificación de ese corte: los niveles post-escalado de ambos modelos en `gtap7_3x3` son
idénticos bit a bit —3142 celdas, 0 diferencias— contra el commit anterior.

**Segundo corte — hecho (2026-09-16).** La reflexión multiperiodo pide su modelo de período
simple a `GTAPMultiPeriodModel._build_sp()`, que `GTAPBlockMultiPeriodModel` sobrescribe. Antes
le hacía monkey-patch a `GTAPModelEquations.build_model` dentro de un `try/finally`. Con eso
`gtap_block_model.py` **dejó de importar el monolito**.

Verificación: el modelo multiperiodo de bloques —7392 entradas entre niveles de Var y cuerpos de
Constraint— es idéntico contra el commit anterior.

(`_block_sp()` se conserva como el método que sobrescriben las subclases: lo hace
`GTAPLogLevelsMultiPeriodModel`. Renombrarlo habría hecho que el modelo log-levels cayera
calladamente al de bloques normal.)

**Tercer corte — hecho (2026-09-17).** Los gates `nlp` (14/14) y `mcp` (18/18) construyen
bloques. El monolito queda marcado como **referencia manual** en su docstring; el único gate
que lo mide es `nl`.

El gate `nl` sigue leyéndolo, y es deliberado: compara el `.nl` **emitido** contra el de GAMS
—estructura, no solve— y no tiene equivalente en bloques.

### Por qué se pudo migrar: el gap era del denominador

El obstáculo registrado aquí era que bloques puntuaba por debajo del monolito en `altertax`
(−0.17pp en 3x4, −0.35pp en 10x7). **Ese gap no medía calidad de modelo.**

Medido comparando celdas concretas en vez de porcentajes: en `gtap7_10x7` fallan **las mismas 110
celdas en ambos modelos, 0 exclusivas de cualquiera**. El conteo de fallos coincide en los cinco
casos con gap (110, 107, 6, 5, 1). Lo único que difiere es el denominador:

```
110/15314 = 0.72% → 99.28%   (monolito)
110/10349 = 1.06% → 98.94%   (bloques)
```

Las 4965 celdas de diferencia son 28 variables de choque exógeno (`a*` técnicos, `lambda*` de
eficiencia, `*txshft`/`etax`/`mtax`/`kappaf`). No se resuelven: valen lo que el cierre les fija, y
coinciden con GAMS en las 4610 medibles **sin una sola excepción**. Son aciertos automáticos que
inflaban el denominador del monolito.

Bloques pasa los floors actuales sin recalibrarlos **en las 14 filas del gate NLP**: margen mínimo
+0.497pp (3x4 altertax ifsub0) frente a +0.669pp del monolito. Ese margen lo dio el arreglo de
`rsav` (`7eca586`); antes esa fila estaba en 98.24, bajo su floor.

**Esa medición NO cubría el gate MCP**, que incluye `gtap7_15x10` — un dataset ausente de la matriz
NLP. Ahí bloques falla, por la razón de la sección siguiente.

## El MCP con bloques: RESUELTO (2026-09-17)

Durante un tiempo `gtap7_15x10-pure-ifsub1` daba **87.00%** con bloques contra un floor de
99.0, con `pft[USA,Land]` clavado en su floor `1e-3`, `pf[USA,Land,*]` en 0.00104 y
`xf[USA,Land,*]` entre 4x y 12x sobre GAMS. Convergía (`code=1`): aterrizaba en OTRO punto.

**La causa eran 33 filas y 33 columnas MUERTAS.**

`build_equations_fisher` (en la clase base, compartida por ambos modelos) reemplaza
`eq_pfact`/`eq_pwfact` por versiones **cross-período** sobre `mq_factr_*`/`mq_factw_*`, y
borra las intra-período. Pero bloques declara además sus propios agregados auxiliares
`eq_mfr_*`/`eq_mfw_*` (`closure.py:216-341`), con los que parte la suma ancha de **su**
`eq_pfact` intra-período. Al borrarse ese `eq_pfact`, esas filas se quedan sin ningún
consumidor — y nadie las borraba.

Medido: **0 consumidores** de `mfr_*`/`mfw_*`, frente a `eq_pfact` (60) + `eq_pwfact` (6)
para `mq_fact*`.

No rompían la cuadratura —cada fila se apareaba con su propia Var— y por eso **todo** salía
idéntico en las comparaciones habituales: ecuaciones, seed (`pft=0.6463`, el valor de GAMS),
cotas, y el mapa de emparejamiento completo (24818 filas comunes, 0 apareadas distinto). El
único rastro estaba en el Jacobiano:

```
monolito  148,696 nnz      bloques  152,849 nnz      delta +4,153
```

y el desglose por familia los ubicaba enteros en las 33 filas Fisher. **El delta es 4153, no
33**: cada fila no aporta *una* incógnita, aporta ~750 acoplamientos a `pf`/`xf`.

### Por qué rompe en MCP y no en NLP

Medido, no deducido:

La fila `mfr_ss[r] == Σ pf·xf` es **satisfacible gratis** moviendo sólo `mfr_ss`, que es libre
y no la consume nadie. En la solución de GAMS su residual es **0.409**, y mover `mfr_ss` de
`15.605` a `16.014` lo anula **exacto**, sin tocar un solo `pf`/`xf`.

- **IPOPT** resuelve un problema con restricciones: ve la fila violada, ajusta la variable
  huérfana, sigue. El punto de GAMS le queda intacto → **inerte**.
- **PATH** no resuelve restricciones sino **complementariedad emparejada**. `mfr_ss` no es un
  grado de libertad aislado que absorba el desbalance: es una columna más del Jacobiano,
  acoplada a ~750 celdas. El ajuste se reparte por esas derivadas y arrastra `pf[USA,Land]`
  hasta su floor `1e-3`.

Hay además una razón trivial que conviene no olvidar: **el gate NLP nunca midió
`gtap7_15x10`** — su matriz es 3x3/3x4/5x5/10x7. El residual de ~0.43 en esas filas está en
*todos* los datasets, así que el NLP venía pasando 14/14 con ellas dentro.

**El fix** (`gtap_model_multiperiod.build_equations_fisher`): borrar también las seis
familias auxiliares **y sus Vars**, ahí donde el código ya borraba
`eq_pabs`/`eq_pfact`/`eq_pwfact`. Las Vars se **borran**, no se fijan: son sumas definidas, y
congelarlas en un valor rancio estanca la shock en `code=0` (es la razón de ser de
`refresh_fisher_aggregates`).

Resultado: estructura byte-idéntica al monolito —74754 filas, 85857 columnas, `solo-A=0` y
`solo-B=0` en filas y columnas— y **gate MCP 18/18 con bloques**, NLP 14/14.

### Por qué costó tanto encontrarlo

Cinco comparaciones dieron "idéntico" y ninguna era el sistema real: el `.nl` ROW+COL diff
compara **nombres** (y las 99 filas/columnas extra parecían balancearse), el emparejamiento
se autoapareaba, y el seed y las cotas coincidían. La única señal fue contar **nonzeros del
Jacobiano**, no filas ni columnas. Descartes previos, todos medidos: forzar el emparejamiento
de las filas Fisher (87.00% bit-idéntico) y hacer inmutables los `forced_pairs`
(87.00% → 53.76%, activamente dañino — que Hopcroft-Karp pueda reemparejar es *load-bearing*).

**Trampa de probe.** Un probe single-period da «33/33 filas Fisher mal emparejadas, todas
robando `xf[*,Land,*]`», que encaja perfecto con el síntoma (`pf[USA,Land]`) — y es
**artefacto**: truncar `fv[:n]` para cuadrar corta justo las columnas Fisher del final, y el
gate no usa single-period. Enumerar el MP crudo tampoco sirve: 74853 filas × 85956 vars son
los tres períodos sin cruzar (74853 = 3×24951), no el sistema que factoriza el solver
(24851). La señal de que estás mirando el sistema equivocado es que el número de filas sea
múltiplo exacto del número de períodos.

### Diez hipótesis descartadas (para no repetirlas)

Seeds/floors distintos, filas faltantes, `mqfactr_bb` distinto, `pfa` mal fijado, expresiones de
`eq_xweq` distintas, `sqrt` anidado, cuenca/arranque (sembrar bloques en la solución del monolito:
se va igual), cotas de los agregados (`Reals`→`NonNegativeReals`: bit-idéntico), fijar los agregados
(bit-idéntico), y las 33 filas como causa directa (desactivarlas: 3738 igual).

**Trampa de método que costó la sesión:** medir contando `component_data_objects(Constraint,
active=True)` da los TRES períodos sin cuadrar, no el sistema que el solver factoriza. La señal que
lo delata está en `project_btf_probe_y_condensacion_cuadratura_2026_09_02` (dev-tools): *"contar
filas y comprobar si es múltiplo del número de períodos"*. Las mías eran 74754 y 74853 — múltiplos
exactos de 3. El sistema cuadrado real se lee del log de `structural_matching`: 24818 vs 24851.

**Trampa de método, por segunda vez en este repo:** un gap entre dos porcentajes puede ser del
numerador o del denominador, y el porcentaje solo no lo distingue. Ya había pasado con `qga`
(ver memoria `project_gtap_qga_floor_levels_vs_linearization`). Comparar las celdas que fallan
debe ser el primer paso, no el cuarto.

### Lo que se pierde, y por qué se aceptó

Con los gates en bloques, el monolito deja de medirse contra nada. Esa redundancia —dos
implementaciones contra dos oráculos— fue la que hizo visible el bug del snapshot Fisher. La
alternativa evaluada era correr ambos modelos en los gates, al doble de tiempo. Se eligió bloques
solo: el monolito queda accesible como referencia, pero no debe generar ruido ni usarse como si
fuera código vivo.

Ver también: `ROADMAP.md` (registro de deuda técnica) y
`docs/findings/repo_cleanup_spec_2026-09-16.md` (la medición que originó este documento).
