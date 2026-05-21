# GTAP v6.2 vs v7.0 — Diff workspace

TAB files copiados de la instalación local de RunGTAP 3.75 (`C:\runGTAP375\`).
Comparación textual de las dos versiones del modelo GTAP en GEMPACK, paso 1
del plan para construir un reference v6.2 puro en GAMS.

## Archivos

| Archivo | Origen | Versión | Tamaño |
|---|---|---|---|
| `gtap_v62.tab` | `gtap.tab` | **Standard GTAP v6.2** (Hertel/Itakura/McDougall, Sept 2003) | 146 KB |
| `gtap_v70.tab` | `gtapv7.tab` | **Standard GTAP v7.0** (Corong/Hertel/McDougall/Tsigas/vdM, Jun 2017) | 197 KB |
| `decomp_v62.tab` | `decomp.tab` | Welfare decomposition module v6.2 (externo) | 42 KB |
| `altpar_v62.tab` | `altpar.tab` | AlterTax v6.2 | 7 KB |
| `altpar_v70.tab` | `altparv7.tab` | AlterTax v7.0 | 8 KB |
| `equations_v62.txt` | derivado | Nombres únicos de equations en v6.2 (213) | — |
| `equations_v70.txt` | derivado | Nombres únicos de equations en v7.0 (213) | — |
| `equations_v62_map.csv` | derivado | Equation → variable definida (v6.2) | — |
| `equations_v70_map.csv` | derivado | Equation → variable definida (v7.0) | — |
| `crosswalk_v62_to_v70.csv` | derivado | Match v6.2 ↔ v7.0 por variable dependiente | — |

## Conteo estructural

| Item | v6.2 | v7.0 | Δ |
|---|---:|---:|---:|
| Líneas totales | 4,027 | 5,336 | +32% |
| **Equations** | **213** | **213** | **±0** |
| Variables | 239 | 256 | +17 |
| Coefficients | 160 | 274 | +114 |
| Sets declarados | 12 (incl Subset) | 13+ | similar |
| Formulas | 142 | 269 | +127 |

**Conclusión clave:** el **core de ecuaciones es idéntico en cuenta** (213 = 213).
El crecimiento de v7 es casi todo en:

1. **Welfare decomposition absorbido inline** (en v6.2 vive en `decomp.tab` externo).
   Esto explica las equations `E_CNT*` "nuevas" en v7 que en realidad son las
   `CONT_EV_*` movidas desde `decomp.tab`.
2. **Postsim coefficients / formulas** para auxiliares de reporte (no son
   ecuaciones del modelo).
3. **Variables auxiliares** introducidas por el split ACT/COMM (ver abajo).

## Cambio estructural #1 — split ACT vs COMM

El cambio más profundo de v7 vs v6.2 es la **separación de actividades (`ACTS`,
índice `a`) y commodities (`COMM`, índice `c`)**. En v6.2 hay un solo índice
`i` sobre `TRAD_COMM` que cumple ambos roles.

**v6.2 sets (índice único `i`):**

```tablo
Set TRAD_COMM # traded commodities #;     -- usado como input y como output
Set MARG_COMM # margin commodities #;
Set CGDS_COMM # capital goods commodities #;
Set ENDW_COMM # endowment commodities #;
Set PROD_COMM = TRAD_COMM union CGDS_COMM;
Set DEMD_COMM = ENDW_COMM union TRAD_COMM;
Set ENDWS_COMM, ENDWM_COMM, ENDWC_COMM;  -- vía parámetro SLUG binario
```

**v7.0 sets (split a/c):**

```tablo
Set COMM, ACTS;                           -- DOS dimensiones distintas
Set MARG, NMRG, CGDS;
Set ENDW, ENDWC, ENDWT (mobile|sluggish|fixed);
Set ENDWM, ENDWS, ENDWMS, ENDWF;          -- vía matriz ENDOWFLAG(e,t)
Set DEMD = ENDW + COMM;
```

**Consecuencia sobre variables nuevas en v7 (índice `(c,a,r)` o `(a,r)` en vez
de `(i,r)`):**

| v7 variable | índices | qué representa | análogo v6.2 |
|---|---|---|---|
| `qint(a,r)` | (acts, reg) | demand for aggregate intermediate input | implícito en `qf(i,j,r)` agregado |
| `qva(a,r)` | (acts, reg) | demand for value-added composite | `qva(j,r)` |
| `qfa(c,a,r)` | (comm, acts, reg) | armington composite use by activity | `qf(i,j,r)` |
| `qfd(c,a,r)` | (comm, acts, reg) | domestic input use by activity | `qfd(i,j,r)` |
| `qfm(c,a,r)` | (comm, acts, reg) | imported input use by activity | `qfm(i,j,r)` |
| `qpa/qga/qia` | (c,r) | armington composite for priv/gov/inv | implícito en `qp/qg/qcgds` con split |
| `pca` / `pds` / `pms` | (c,r) | activity vs commodity price split | `ps(i,r)` |
| `pe1, pe2` | (e,r) | factor price at distintos niveles | `pm(i,r)` para endowments |
| `qes1, qes2, qes3` | (e,r) | factor supply variants | `qoes(i,r)` |

**Para un rollback a v6.2 en GAMS:** colapsar `a = c` y eliminar el split implica
mapear `qfa→qf`, `qfd→qfd`, `qfm→qfm`, etc., con `a` recorriendo el mismo set
que `c`. Las ecuaciones agregadoras `E_qint`, `E_qva`, `E_qfa` se reducen.

## Cambio estructural #2 — endowment mobility

| v6.2 | v7.0 |
|---|---|
| `SLUG(i)` binario (0=mobile, 1=sluggish) | `ENDOWFLAG(e,t)` matriz 3×N con (mobile, sluggish, fixed) |
| Solo 2 categorías de movilidad | 3 categorías; soporte futuro para fixed factors |

En el v6.2 GAMS proyectado, el SLUG binario es suficiente.

## Cambio estructural #3 — welfare decomposition

| v6.2 | v7.0 |
|---|---|
| `decomp.tab` externo (post-solve, separado) | absorbido inline en `gtap_v70.tab` |
| ~50 ecuaciones `CONT_EV_*` (CNT contributions) | ~50 ecuaciones `E_CNT*` equivalentes |
| Versiones desagregadas: `CONT_EV_qfdir` (i,j,r) y `CONT_EV_qfdijr` | versión única `E_CNTqfd` que el postsim re-mapea |

Para v6.2 GAMS, el welfare decomp puede mantenerse como módulo separado
(`decomp.gms`) o inline. Ambas opciones son válidas; la equivalencia con
`decomp_v62.tab` da una referencia de validación independiente.

## Cambio estructural #4 — adiciones reales en v7 (Nov 2017+)

Documentado en el header del propio TAB v7:

- **Nov 2017:** `pefactreal(e,r)` (tax-exclusive real factor price) y
  `pebfactreal(e,r)` (tax-inclusive). 2 ecuaciones nuevas — `E_pefactreal`,
  `E_pebfactreal`. Útiles para reporting; no son core para equilibrio.
- **Abr 2018:** Refinamiento set definitions de Allocative Efficiency.
- **Jul 2018:** Set `ENDT` (endowment types) con flag binario `ENDOWFLAG`.
- **Dic 2019:** Section C-A25 (factor income tax effect en welfare decomp).

Ninguno es un cambio de cierre o de equilibrio.

## Cambio estructural #5 — pricing taxonomy

v7 introduce un sistema de precios **más estratificado** con 5 niveles:

```
S  = supplier
B  = basic
P  = producer
M  = market
U  = user/agent
```

Esto desdobla `pms(c,r,s)` (basic) vs `pmds(c,r,s)` (market) y `pca(c,a,r)`
(commodity-at-activity) vs `pcif(c,r,s)` (CIF border). v6.2 colapsa varios
de estos en una sola variable de precio.

## Equations renombradas (no estructural)

v7 usa convención sistemática `E_<dep_var>`. Ejemplos:

| v6.2 (descriptivo) | v7.0 (E_<var>) | dep var |
|---|---|---|
| `GPRICEINDEX` | `E_pgov` | `pgov` |
| `PRIVATEU` | `E_up` | `up` |
| `KAPRENTAL` | `E_rental` | `rental` |
| `WRLDPRICE` | `E_psw` o `E_pxw` | precios mundiales |
| `WALRAS_D` / `WALRAS_S` | `E_walras_dem` / `E_walras_sup` | Walras |
| `EV_DECOMPOSITION` | `E_EV` | `EV` |
| `ZEROPROFITS` | `E_po` | `po` |
| `GLOBALINV` | `E_globalcgds` | inversión global |
| `RORCURRENT` / `ROREXPECTED` / `RORGLOBAL` | `E_rorc` / `E_rore` / `E_rorg` | rates of return |
| `OUTPUTPRICES` | `E_pds` o `E_pms` | output prices (split en v7) |
| `INTDEMAND` | `E_qfa` / `E_qfd` / `E_qfm` | intermediate demand (split en v7) |

El crosswalk completo en CSV: `crosswalk_v62_to_v70.csv` (213 filas).

## Resumen ejecutivo para el rollback

Para construir un **GTAP 6.2 GAMS levels reference** en equilibria, el camino
es partir del v7 GAMS (van der Mensbrugghe, JGEA 2018) y aplicar estos
deltas en orden:

1. **Colapsar split ACT/COMM:** unificar `ACTS = COMM`, mapear `qfa→qf`, `pfa→pf`,
   `qint→sum(i,qf)`, `qva→qva`, `qpa→qp`, `qga→qg`, `qia→qcgds`. Eliminar
   `pca/pds/pms` split (volver a una sola `ps(i,r)`).
2. **Endowment flag → SLUG binario:** reemplazar `ENDOWFLAG(e,t)` por `SLUG(i)`
   binario, eliminar el set `ENDWT`, eliminar el caso `fixed`.
3. **Welfare decomp en módulo separado:** mover las equations `E_CNT*` a un
   archivo aparte `gtap_v62_decomp.py` (equivalente a `decomp.tab`).
4. **Eliminar adiciones post-v7.0 base:** quitar `pefactreal`, `pebfactreal`,
   sección C-A25 (factor income tax effect).
5. **Pricing taxonomy plana:** colapsar S/B/P/M/U a la nomenclatura v6.2
   más simple (`ps`, `pm`, `pf`, etc).

**Costo estimado:** dado que el core de equations es 213=213 y la mayoría son
renames, el trabajo de modelo puro es bajo. El esfuerzo grueso está en:

- Validar la equivalencia variable por variable (script crosswalk).
- Reescribir los reads de GTAPSETS (mapeo `COMM↔TRAD_COMM` etc).
- Adaptar el código de equilibria que asume nomenclatura v7.

## Paso 2 — GEMPACK v7 vs GAMS v7 (van der Mensbrugghe)

GAMS v7 vive en [src/equilibria/templates/reference/gtap/scripts/](src/equilibria/templates/reference/gtap/scripts/).
Archivo principal del modelo: [model.gms](src/equilibria/templates/reference/gtap/scripts/model.gms) (1,457 líneas).

| Item | GEMPACK v7.0 | GAMS v7 | Δ |
|---|---:|---:|---:|
| Equations | 213 | **102** | −111 |
| Líneas de modelo | 5,336 (todo) | 1,457 (model.gms) | −73% |

**¿Por qué GAMS v7 tiene la mitad de ecuaciones?** No es por simplificación del
modelo de equilibrio, sino por separación de responsabilidades:

1. **Welfare decomposition no está en `model.gms`.** Las ~50 ecuaciones `E_CNT*`
   inline en GEMPACK v7 NO están como constraints en GAMS v7 — son cálculos
   post-solve (parámetros, no variables). Esto explica ~50 del gap.
2. **Aggregate indices (Appendix B) son postsim.** Las ~30 ecuaciones del
   apéndice B de GEMPACK v7 (factor price indices, ToT decomposition, GDP indices,
   aggregate trade indices, trade balance indices) están en
   [postsim.gms](src/equilibria/templates/reference/gtap/scripts/postsim.gms)
   como asignaciones de parámetros, no como ecuaciones del modelo.
3. **Marg/NMRG variants colapsadas vía `$(condition)`.** GEMPACK tiene
   `EXPPRICE_MARG` y `EXPPRICE_NMRG` (dos ecuaciones). GAMS las une en una sola
   `pmcifeq(r,i,rp,t)` con guard `$tmgFlag(r,i,rp)`. Esto explica ~10 del gap.
4. **Ecuaciones derivadas de pricing absorbidas en otras.** Algunas
   relaciones que GEMPACK pone como ecuaciones independientes, GAMS las
   inline-substituye dentro de la ecuación dependiente.

**Conclusión paso 2:** GAMS v7 = subconjunto de equilibrio puro de GEMPACK v7
(102 ecuaciones core). Las otras 111 son cálculos post-solve. **GAMS v7
preserva el split ACT/COMM** introducido en v7 (índice `a` para actividades,
`i` para commodities en variables como `xap(r,i,a,t)`, `xf(r,fp,a,t)`,
`nd(r,a,t)`, `va(r,a,t)`).

## Paso 3 — GAMS v7 vs equilibria (Python)

equilibria implementa el modelo en [gtap_model_equations.py](src/equilibria/templates/gtap/gtap_model_equations.py)
mediante funciones `def eq_<name>_rule(...)` para Pyomo.

| Item | GAMS v7 | equilibria | Δ |
|---|---:|---:|---:|
| Equations (core) | 102 | **94** | −8 |
| Common (matched) | — | **74** | — |
| Solo en GAMS v7 | — | 28 | — |
| Solo en equilibria | — | 20 | — |

Crosswalk completo: `crosswalk_gamsv7_to_equilibria.csv`.

Las diferencias **no son omisiones**, sino **refactors locales**:

1. **Split por agent type en equilibria.** GAMS tiene equations únicas
   `xageq/xaieq/xaceq` indexadas por agente; equilibria las desdobla en
   `eq_xaa_activity / eq_xaa_hhd / eq_xaa_gov / eq_xaa_inv / eq_xaa_tmg`
   (5 ecuaciones). Misma semántica, distinta organización en Pyomo.
2. **Aggregator equations separadas en equilibria.** GAMS hace agregación
   inline (`xdseq`, `xmteq`); equilibria separa `eq_xd_agg`, `eq_xmt_agg`,
   `eq_xet_agg`, `eq_xiagg` para legibilidad.
3. **Calibration helpers solo en GAMS v7 (no necesarios en equilibria).**
   `afealleq`, `glcaleq`, `incelaseq`, `cedeq`, `apeeq`, `uedeq` son
   calibración de CDE y closure dinámica — equilibria los tiene como Param
   pre-computados (ver CLAUDE.md: "CDE/chiInv elasticities frozen como Param").
4. **lambdaf/lambdaio/lambdand/lambdava** (tech change shifters) — fijados
   como Param en equilibria.
5. **Renames cosméticos.** `paeq` ↔ `eq_paa`, `xdeq` ↔ `eq_xda`,
   `xmeq` ↔ `eq_xma`, `ppeq` ↔ `eq_pp_rai`, `xpeq` ↔ `eq_po` (zero profit).

**Conclusión paso 3:** equilibria es una traducción fiel de GAMS v7 con
refactors organizacionales de Pyomo. El **modelo de equilibrio coincide**.

## Síntesis del rollback v6.2 — plan ejecutable

**⚠️ Importante:** la Table 1 oficial del paper v7 (Corong et al 2017,
transcrita en [notation_crosswalk.md](notation_crosswalk.md) y en
[docs/gtap_7_docs.en.md](../../docs/gtap_7_docs.en.md) líneas 2575–2977)
muestra que el rollback v7→v6.2 **no es solo renames** — hay 5 cambios
estructurales que eliminan ecuaciones y modifican comportamiento.

### Opción A — Rollback en equilibria (no en GAMS)

Crear `templates/gtap_v62/` paralelo a `templates/gtap/`, copiando la
estructura de equations y aplicando los cambios en orden:

#### 🔴 Cambios estructurales (eliminan/reescriben ecuaciones)

| # | Cambio (Tabla 1) | Acción en equilibria |
|---|---|---|
| 3 | **Eliminar intermediate bundle** (`qint, aint, pint, ESUBC`) | Borrar `eq_nd`, `eq_pnd`. Intermedios entran directo a producción (Leontief implícito en v6.2) |
| 5 | **Eliminar MAKE transformation** (`qca, pca, qc, ETRAQ, ESUBQ`) | Borrar ecuaciones de make. Make diagonal: `qo(i,r) = qc(c,r)` con `i=c` |
| 7 | **Government: CES → Cobb-Douglas** | Reescribir `eq_xag`, `eq_ug`, `eq_pg` sin `ESUBG` (default ya es 1.0=CD pero v6.2 lo tiene hard-coded) |
| 10 | **Margins: CES → Cobb-Douglas** | Reescribir `eq_xmgm`, `eq_xtmg`, `eq_ptmg` sin `ESUBS` |
| 12 | **Factor markets a nivel commodity, no activity** | Cambiar `eq_xfeq(r,fp,a)`, `eq_pfeq(r,fp,a)` → indexados por (r,i) con i ∈ ENDW_COMM. `tinc(e,a,r)` desaparece — usar `toi(i,r)` |

#### 🟡 Renames + colapso de índices

| # | Cambio | Acción |
|---|---|---|
| 1 | Sets: COMM→TRAD_COMM, etc. | Aliases. Eliminar sets ACTS, ENDWF, ENDWMS |
| 2 | `qo(a,r) → qo(i,r)`, `po(a,r) → ps(i,r)`, `pb(a,r) → pm(i,r)` | Colapsar dim a en todas las variables |
| 4 | `qfa(c,a,r) → qf(i,j,r)` | i=c, j=a. `ESUBD` pierde dim r |
| 6 | `qpa(c,r) → qp(i,r)`, `ppa→pp` | rename + i=c |
| 8 | Investment como commodity, no agent: `qia/qid/qim → qf/qfd/qfm(i,cgds,r)` | Reformulación de la sub-estructura completa de inversión |
| 9 | `qms/pms(c,r) → qim/pim(i,r)`, `qxs(c,s,d)/pmds → qxs(i,s,r)/pms` | Rename + cambio de índices |
| 11 | Market clearing: `qc(c,r) → qo(i,r)` over TRAD_COMM | Reescribir condición de balance |

#### 🟢 Eliminaciones puras

- Parámetro `ENDOWFLAG(e,t)` → `SLUG(i)` binario
- Variables `pefactreal`, `pebfactreal` (Nov 2017)
- Elasticidades v7: `ESUBC, ESUBG, ESUBS, ETRAQ, ESUBQ`
- Welfare decomp inline → módulo separado `welfare_decomp_v62.py` (equivalente a `decomp.tab` externo en GEMPACK v6.2)

**Costo estimado revisado:** ~3-4 semanas (no 2). El trabajo grueso está en
items 3, 5, 8 y 12 que reescriben sub-estructuras, no solo renombran.

**Ventaja:** se aprovecha toda la infra de validación, parámetros, solver
wrapper de equilibria. La validación tiene referencia clara: el TAB
`gtap_v62.tab` corrido en GEMPACK (RunGTAP local).

### Opción B — GAMS puro nuevo basado en `gtap_v62.tab`

Implementar desde cero un `templates/reference/gtap_v62_gams/` partiendo del
TAB v6.2 GEMPACK. Cada ecuación lineal GEMPACK
`p_x = a*p_y + b*p_z` se levanta a su nivel `X = X0 * (Y/Y0)^a * (Z/Z0)^b`.

**Ventaja:** referencia GAMS independiente para co-validación. **Desventaja:**
mucho más trabajo (~6-8 semanas) y sin valor agregado para equilibria si la
opción A funciona.

### Recomendación

**Empezar por Opción A** y, si se necesita una referencia GAMS independiente
para reportar paridad cruzada (como ya existe `NEOS` y `GAMS local` para v7),
agregar Opción B después.

## Documentos relacionados

- [notation_crosswalk.md](notation_crosswalk.md) — Table 1 oficial del paper v7
  (12 secciones con mapping v6.2 ↔ v7 detallado)
- [docs/gtap_7_docs.en.md](../../docs/gtap_7_docs.en.md) líneas 2575–2977 —
  source original (Corong et al 2017)
- [equations_v62.txt, equations_v70.txt, equations_gams_v7.txt, equations_equilibria.txt](.) —
  listas de ecuaciones por implementación
- [crosswalk_gamsv7_to_equilibria.csv](crosswalk_gamsv7_to_equilibria.csv) —
  match GAMS v7 ↔ equilibria por nombre normalizado

## Próximos pasos

1. ✅ Diff GEMPACK v6.2 vs GEMPACK v7.0
2. ✅ Diff GEMPACK v7.0 vs GAMS v7 (van der Mensbrugghe 2018)
3. ✅ Diff GAMS v7 vs equilibria (Python implementation)
4. ✅ Refinamiento con Table 1 oficial (Corong et al 2017)
5. ⏳ Diseñar `templates/gtap_v62/` con los 5 cambios estructurales aplicados
6. ⏳ (Opcional) Co-validación contra `gtap_v62.tab` corrido en GEMPACK
