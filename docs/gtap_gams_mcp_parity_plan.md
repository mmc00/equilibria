# GTAP GAMS Parity Plan

## Objetivo

Hacer que el modelo GTAP en Pyomo dentro de `equilibria` represente el mismo MCP que la implementación de referencia en GAMS, eliminando simplificaciones y fijaciones artificiales para que ambos resuelvan el mismo sistema.

El objetivo no es solo "mejorar convergencia", sino asegurar equivalencia estructural:

- mismas ecuaciones activas
- mismas variables endógenas
- mismas condiciones de activación tipo `$`
- misma calibración benchmark
- misma lógica de cierre
- mismo mapeo MCP ecuación-variable

## Estado Actual

Hoy el modelo Pyomo todavía difiere de GAMS en puntos estructurales:

- el sistema se fuerza a cuadrado mediante `apply_aggressive_fixing_for_mcp()`
- el bloque CET/exportador sigue con simplificaciones (`ps = pd`, `pet = ps`, `pe = pet`)
- el bloque macro `gdpmp/rgdpmp` no es una réplica literal de GAMS
- el bloque bilateral/importación aún muestra desanclajes en trayectoria PATH
- varias decisiones numéricas actuales son defensivas para que el solve no explote, pero alteran el problema económico

## Principio de Trabajo

No seguir agregando parches de convergencia sobre un MCP distinto al de GAMS.

Orden correcto:

1. igualar estructura
2. igualar benchmark
3. igualar mapeo MCP
4. recién después volver a juzgar convergencia PATH

## Fase 0. Línea Base Reproducible

- [ ] Guardar una línea base reproducible del estado actual
- [ ] Mantener una corrida corta instrumentada con `PATH_CAPI_PROGRESS_HISTORY_FILE`
- [ ] Mantener una corrida larga de referencia
- [ ] Registrar en artifacts:
  - `termination_code`
  - residual final o último residual observado
  - top residuos
  - `progress.jsonl`
  - opciones PATH usadas

Criterio de cierre:

- existe una baseline documentada y repetible para comparar cada cambio

## Fase 1. Inventario Exacto GAMS vs Pyomo

- [ ] Enumerar las ecuaciones que entran al MCP final en GAMS
- [ ] Enumerar las variables complementadas/endógenas en GAMS
- [ ] Enumerar todas las `Constraint` activas en Pyomo
- [ ] Enumerar todas las `Var` libres en Pyomo
- [ ] Construir una tabla diff con:
  - ecuaciones presentes en GAMS y ausentes en Pyomo
  - ecuaciones presentes en Pyomo pero no en GAMS
  - ecuaciones simplificadas respecto a GAMS
  - variables libres en Pyomo que en GAMS están fijas
  - variables fijadas en Pyomo que en GAMS son endógenas

Criterio de cierre:

- existe una tabla MCP parity clara y accionable

### Resultado Inicial de Inventario

Referencia GAMS:

- el modelo `gtap` en [model.gms](/Users/marmol/proyectos2/equilibria/src/equilibria/templates/reference/gtap/scripts/model.gms:1395) declara explícitamente el mapeo ecuación-variable del MCP
- el bloque central relevante incluye, entre otros:
  - `pdpeq.pdp`, `pmpeq.pmp`, `paeq.pa`, `xdeq.xd`, `xmeq.xm`
  - `xmteq.xmt`, `xweq.xw`, `pmteq.pmt`, `pmeq.pm`
  - `xdseq.xds`, `xeteq.xet`, `xseq`, `peeq.pe`, `peteq.pet`, `pefobeq.pefob`
  - `xwmgeq.xwmg`, `xmgmeq.xmgm`, `pwmgeq.pwmg`, `xtmgeq.xtmg`, `ptmgeq.ptmg`, `pmcifeq.pmcif`
  - `pdeq.pd`
  - `xfteq.xft`, `pfeq.pf`, `pfteq`, `pfaeq.pfa`, `pfyeq.pfy`, `kstockeq.kstock`
  - `arenteq.arent`, `kapEndeq.kapEnd`, `rorceq.rorc`, `roreeq.rore`, `xieq.xi`
  - `savfeq`, `rorgeq.rorg`, `chifeq.chif`, `capAccteq`
  - `chiSaveeq.chiSave`, `psaveeq.psave`, `xigbleq.xigbl`, `pigbleq.pigbl`
  - `gdpmpeq.gdpmp`, `rgdpmpeq.rgdpmp`, `pgdpmpeq.pgdpmp`
  - `pabseq.pabs`, `pmuveq.pmuv`, `pfacteq.pfact`, `pwfacteq.pwfact`, `pnumeq`
  - `eveq`, `cveq`, `walraseq`

Estado actual Pyomo:

- después de `apply_closure()` el modelo queda cuadrado con:
  - `17109` constraints activas
  - `17109` variables libres
- componentes activas relevantes:
  - bloque Armington/importación:
    - `eq_paa 1300`
    - `eq_xda 1300`
    - `eq_xma 1300`
    - `eq_xmt_agg 100`
    - `eq_xweq 1000`
    - `eq_pmteq 100`
    - `eq_pmeq 1000`
    - `eq_pmcifeq 1000`
    - `eq_pefobeq 1000`
  - bloque CET/exportador:
    - `eq_xds 100`
    - `eq_xet 100`
    - `eq_xseq 100`
    - `eq_peeq 900`
    - `eq_peteq 100`
    - `eq_pe 100`
    - `eq_pe_route 1000`
    - `eq_ps 100`
  - bloque macro:
    - `eq_gdpmp 10`
    - `eq_rgdpmp 10`
    - `eq_pgdpmp 10`
    - `eq_pabs 10`
    - `eq_pfact 10`
    - `eq_pwfact 1`
    - `eq_pnum 1`
  - bloque capital/ahorro:
    - `eq_arent 10`
    - `eq_kapEnd 10`
    - `eq_rorc 10`
    - `eq_rore 10`
    - `eq_rorg 1`
    - `eq_capAcct 1`
    - `eq_chisave 1`
    - `eq_psave 10`
    - `eq_xigbl 1`
    - `eq_pigbl 1`
  - bloque factores:
    - `eq_xft 30`
    - `eq_xfteq 30`
    - `eq_xfeq 290`
    - `eq_pfeq 30`
    - `eq_kstock 10`

Variables libres Pyomo relevantes:

- libres:
  - `pa 1300`
  - `xda 1001`
  - `xma 1300`
  - `xmt 100`
  - `xw 900`
  - `pmt 100`
  - `pm 900`
  - `pmcif 900`
  - `pefob 900`
  - `xds 100`
  - `xet 100`
  - `xs 100`
  - `pe 900`
  - `pet 100`
  - `ps 100`
  - `pd 100`
  - `gdpmp 10`
  - `rgdpmp 10`
  - `pgdpmp 10`
  - `pabs 10`
- fijadas por cierre actual:
  - `lambdam 1000`
  - `etax 100`
  - `mtax 100`
  - `x 800` de `900`
  - `pwmg 370` de `1000`
  - `xwmg 370` de `1000`
  - `xmgm 370` de `1000`
  - `xw 100` de `1000`
  - `xe 100` de `1000`
  - `pe 100` de `1000`
  - `pm 100` de `1000`
  - `pmcif 100` de `1000`
  - `pefob 100` de `1000`
  - `pf 160`
  - `xf 160`
  - `xft 50`
  - `xda 299` de `1300`

### Diferencias Ya Confirmadas

- Pyomo sigue dependiendo de fijaciones estructurales para cuadrar el MCP; GAMS no usa este `aggressive fixing`
- en Pyomo, `lambdam`, `mtax` y `etax` quedan completamente fijadas, mientras en GAMS aparecen dentro del modelo y su estatus debe verificarse desde el cierre original
- el bloque CET/exportador en Pyomo incluye ecuaciones simplificadas (`eq_ps`, `eq_pe`, `eq_pe_route`) que no son una réplica literal del bloque GAMS
- Pyomo no tiene todavía un espejo uno-a-uno documentado de:
  - `pfaeq/pfyeq`
  - `eveq/cveq`
  - `pmuv`
- Pyomo usa `eq_xd_agg` y `eq_xmt_agg` como reconciliación explícita con `xda/xma`, mientras GAMS usa `pdeq` y `xmteq` dentro de un mapeo distinto

### Tareas Derivadas de Fase 1

- [ ] construir una tabla formal `GAMS equation -> Pyomo equation -> status`
- [ ] construir una tabla formal `GAMS variable -> Pyomo variable -> status`
- [ ] clasificar cada ítem como:
  - `exact_match`
  - `simplified`
  - `missing`
  - `extra_in_pyomo`
  - `fixed_in_pyomo_only`
- [ ] documentar específicamente el cierre de:
  - `lambdam`
  - `mtax`
  - `etax`
  - `xda`
  - `x/pe/xw/xe/pwmg/xwmg/xmgm`

### Tabla Inicial `GAMS equation -> Pyomo equation -> status`

Leyenda:

- `exact_match`: misma intención estructural y sin simplificación obvia
- `simplified`: existe contraparte, pero Pyomo impone una versión simplificada
- `needs_mapping`: existe bloque relacionado, pero el mapeo no es todavía uno-a-uno
- `missing`: no hay espejo claro todavía en Pyomo

#### Producción y Demanda Intermedia

| GAMS | Pyomo | Estado | Nota |
|---|---|---|---|
| `ndeq.nd` | `eq_nd` | `exact_match` | Top nest de producción presente |
| `vaeq.va` | `eq_va` | `exact_match` | Presente |
| `xapeq.xa` | `eq_xaa_activity`, `eq_xaa_hhd`, `eq_xaa_gov`, `eq_xaa_inv`, `eq_xaa_tmg`, `eq_xa` | `needs_mapping` | En Pyomo se descompone en varias ecuaciones |
| `xeq.x` | `eq_x` | `exact_match` | Presente |
| `xpeq.xp` | `prf_y` / `eq_po` / variables `xp` | `needs_mapping` | Pyomo reparte el bloque entre profit y mapping |
| `ppeq.pp` | `eq_pp_rai` / `pp` | `needs_mapping` | No documentado aún uno-a-uno |
| `peq.p` | `p_rai` / `eq_pxeq` ausente | `needs_mapping` | Falta cerrar equivalencia exacta |
| `pseq.ps` | `eq_ps` | `simplified` | Pyomo usa identidad `ps = pd`, no el bloque completo GAMS |
| `lambdaioeq.lambdaio` | sin espejo explícito libre | `missing` | No aparece como variable libre equivalente |
| `lambdandeq.lambdand` | sin espejo explícito libre | `missing` | No aparece como variable libre equivalente |
| `lambdavaeq.lambdava` | sin espejo explícito libre | `missing` | No aparece como variable libre equivalente |
| `lambdafeq.lambdaf` | sin espejo explícito libre | `missing` | No aparece como variable libre equivalente |
| `pxeq.px` | `px`, `eq_pxeq` con cardinalidad 0 | `missing` | Variable existe, ecuación activa equivalente no |
| `pndeq.pnd` | `pnd`, `eq_pndeq` con cardinalidad 0 | `missing` | Variable existe, ecuación activa equivalente no |
| `pvaeq.pva` | `pva`, `eq_pvaeq` con cardinalidad 0 | `missing` | Variable existe, ecuación activa equivalente no |

#### Armington Top Nest

| GAMS | Pyomo | Estado | Nota |
|---|---|---|---|
| `pdpeq.pdp` | implícito en `pdp` + bloque Armington | `needs_mapping` | Falta documentar ecuación exacta vs GAMS |
| `pmpeq.pmp` | implícito en `pmp` + bloque Armington | `needs_mapping` | Falta documentar ecuación exacta vs GAMS |
| `paeq.pa` | `eq_paa` | `exact_match` | Misma intención CES top nest |
| `xdeq.xd` | `eq_xda` | `exact_match` | Misma intención top nest doméstico |
| `xmeq.xm` | `eq_xma` | `exact_match` | Misma intención top nest importado |
| `pdeq.pd` | `eq_pdeq` | `exact_match` | Presente |
| `xmteq.xmt` | `eq_xmt_agg` | `exact_match` | Agregado importado presente |

#### Bilateral Importación

| GAMS | Pyomo | Estado | Nota |
|---|---|---|---|
| `xweq.xw` | `eq_xweq` | `needs_mapping` | Fórmula cercana, pero calibración/uso de `amw` no está cerrada |
| `pmteq.pmt` | `eq_pmteq` | `needs_mapping` | Fórmula cercana, pero acoplada al mismo problema de `amw` |
| `pmeq.pm` | `eq_pmeq` | `needs_mapping` | Existe, pero trayectoria muestra desajuste |
| `pmcifeq.pmcif` | `eq_pmcifeq` | `exact_match` | Intención alineada |
| `pefobeq.pefob` | `eq_pefobeq` | `exact_match` | Intención alineada |
| `lambdam` dentro de `xweq/pmteq` | `lambdam` | `fixed_in_pyomo_only` | En Pyomo queda totalmente fijada por cierre actual |
| `mtax` dentro de `pmeq` | `mtax` | `fixed_in_pyomo_only` | En Pyomo queda totalmente fijada |
| `etax` dentro de `pefobeq` | `etax` | `fixed_in_pyomo_only` | En Pyomo queda totalmente fijada |

#### CET y Exportación

| GAMS | Pyomo | Estado | Nota |
|---|---|---|---|
| `xdseq.xds` | `eq_xds` | `exact_match` | Misma forma CET doméstica |
| `xeteq.xet` | `eq_xet` | `exact_match` | Misma forma CET export |
| `xseq` | `eq_xseq` | `exact_match` | Misma forma CET agregadora |
| `peeq.pe` | `eq_peeq` | `exact_match` | Presente |
| `peteq.pet` | `eq_peteq` | `exact_match` | Presente |
| `peq.p` / `pseq.ps` acoplados al bloque exportador | `eq_pe`, `eq_pe_route`, `eq_ps` | `simplified` | Estas identidades adicionales no reflejan literalmente GAMS y alteran la trayectoria |
| `xe` / `xet` agregación | `eq_xe_xw`, `eq_xet_agg` | `needs_mapping` | Pyomo agrega restricciones auxiliares no documentadas uno-a-uno en GAMS |

#### Márgenes de Comercio

| GAMS | Pyomo | Estado | Nota |
|---|---|---|---|
| `xwmgeq.xwmg` | `eq_xwmg` | `exact_match` | Presente |
| `xmgmeq.xmgm` | `eq_xmgm` | `exact_match` | Presente |
| `pwmgeq.pwmg` | `eq_pwmg` | `exact_match` | Presente |
| `xtmgeq.xtmg` | `eq_xtmg` | `exact_match` | Presente |
| `ptmgeq.ptmg` | `eq_ptmg` | `simplified` | Pyomo fija precio de transporte al numeraire |
| `xatmgeq.xa` | `eq_xaa_tmg` / `eq_xa` | `needs_mapping` | Bloque repartido en Pyomo |

#### Factores

| GAMS | Pyomo | Estado | Nota |
|---|---|---|---|
| `xfteq.xft` | `eq_xft`, `eq_xfteq` | `needs_mapping` | Dos ecuaciones relacionadas en Pyomo |
| `pfeq.pf` | `eq_pfeq` | `exact_match` | Presente |
| `pfteq` | `pft` sin espejo inequívoco | `needs_mapping` | Falta cerrar espejo exacto |
| `pfaeq.pfa` | `pfa` no aparece como bloque explícito equivalente | `missing` | Requiere revisión |
| `pfyeq.pfy` | `pfy` no aparece como bloque explícito equivalente | `missing` | Requiere revisión |
| `kstockeq.kstock` | `eq_kstock` | `exact_match` | Presente |

#### Capital, Ahorro e Inversión

| GAMS | Pyomo | Estado | Nota |
|---|---|---|---|
| `arenteq.arent` | `eq_arent` | `exact_match` | Presente |
| `kapEndeq.kapEnd` | `eq_kapEnd` | `exact_match` | Presente en [gtap_model_equations.py](/Users/marmol/proyectos2/equilibria/src/equilibria/templates/gtap/gtap_model_equations.py:2654) |
| `rorceq.rorc` | `eq_rorc` | `exact_match` | Presente |
| `roreeq.rore` | `eq_rore` | `exact_match` | Presente |
| `xieq.xi` | `eq_xi` | `exact_match` | Presente |
| `savfeq` | `eq_savf` | `exact_match` | Presente |
| `rorgeq.rorg` | `eq_rorg` | `exact_match` | Presente |
| `chifeq.chif` | sin espejo claro | `missing` | Requiere revisión |
| `capAccteq` | `eq_capAcct` | `exact_match` | Presente |
| `chiSaveeq.chiSave` | `eq_chisave` | `exact_match` | Presente |
| `psaveeq.psave` | `eq_psave` | `exact_match` | Presente |
| `xigbleq.xigbl` | `eq_xigbl` | `exact_match` | Presente |
| `pigbleq.pigbl` | `eq_pigbl` | `exact_match` | Presente |

#### Ingreso, Utilidad e Impuestos

| GAMS | Pyomo | Estado | Nota |
|---|---|---|---|
| `yTaxeq.ytax` | `eq_ytax` | `exact_match` | Presente |
| `yTaxToteq.ytaxtot` | `eq_ytax_tot` | `exact_match` | Presente |
| `yTaxIndeq.ytaxind` | `eq_ytax_ind` | `exact_match` | Presente |
| `factYeq.factY` | `eq_facty` | `exact_match` | Presente |
| `regYeq.regY` | `eq_regy` | `exact_match` | Presente |
| `phiPeq.phiP` | sin espejo claro | `missing` | Requiere revisión |
| `phieq.phi` | sin espejo claro | `missing` | Requiere revisión |
| `yceq.yc` | `eq_yc` | `exact_match` | Presente |
| `ygeq.yg` | `eq_yg` | `exact_match` | Presente |
| `rsaveq.rsav` | `eq_rsav` | `exact_match` | Presente |
| `uheq.uh` | `eq_uh` | `exact_match` | Presente |
| `ugeq.ug` | `eq_ug` | `exact_match` | Presente |
| `useq.us` | `eq_us` | `exact_match` | Presente |
| `ueq.u` | `eq_u` | `exact_match` | Presente |
| `zconseq.zcons` | `zcons` no aparece como ecuación explícita | `missing` | Requiere revisión |
| `xcshreq.xcshr` | `eq_xcshr` | `exact_match` | Presente |
| `xaceq.xa` | `eq_xc` / `eq_xaa_hhd` / `eq_xa` | `needs_mapping` | Bloque repartido |
| `pconseq.pcons` | `eq_pcons` | `exact_match` | Presente |
| `xageq.xa` | `eq_xg` / `eq_xaa_gov` / `eq_xa` | `needs_mapping` | Bloque repartido |
| `pgeq.pg` | `pg` sin espejo inequívoco documentado | `needs_mapping` | Requiere revisión |
| `xaieq.xa` | `eq_xi` / `eq_xaa_inv` / `eq_xa` | `needs_mapping` | Bloque repartido |
| `pieq.pi` | `eq_pi` | `exact_match` | Presente |
| `yieq.yi` | `eq_yi` | `exact_match` | Presente |
| `dintxeq.dintx` | sin espejo explícito | `missing` | Requiere revisión |
| `mintxeq.mintx` | sin espejo explícito | `missing` | Requiere revisión |
| `ytaxshreq.ytaxshr` | sin espejo explícito | `missing` | Requiere revisión |

#### Macro y Bienestar

| GAMS | Pyomo | Estado | Nota |
|---|---|---|---|
| `gdpmpeq.gdpmp` | `eq_gdpmp` | `needs_mapping` | Existe, pero formulación aún no es espejo exacto |
| `rgdpmpeq.rgdpmp` | `eq_rgdpmp` | `simplified` | Pyomo usa una forma simplificada/cuadrática |
| `pgdpmpeq.pgdpmp` | `eq_pgdpmp` | `exact_match` | Identidad presente |
| `pabseq.pabs` | `eq_pabs` | `needs_mapping` | Intención Fisher presente, pero no aún espejo literal |
| `pmuveq.pmuv` | sin espejo claro | `missing` | Requiere revisión |
| `pfacteq.pfact` | `eq_pfact` | `exact_match` | Presente |
| `pwfacteq.pwfact` | `eq_pwfact` | `exact_match` | Presente |
| `pnumeq` | `eq_pnum` | `exact_match` | Presente |
| `eveq` | sin espejo claro | `missing` | Requiere revisión |
| `cveq` | sin espejo claro | `missing` | Requiere revisión |
| `walraseq` | `eq_walras` | `exact_match` | Presente |

### Tabla Inicial `GAMS variable -> Pyomo variable -> status`

Leyenda:

- `exact_match`: misma variable económica con rol equivalente
- `simplified`: existe en Pyomo, pero queda determinada por una simplificación adicional
- `fixed_in_pyomo_only`: Pyomo la fija hoy para cerrar o estabilizar el MCP
- `needs_mapping`: existe pero la correspondencia exacta aún no está cerrada
- `missing`: no hay espejo claro hoy

#### Producción, Precios de Producción y Factores

| GAMS | Pyomo | Estado | Nota |
|---|---|---|---|
| `axp` | sin espejo libre claro | `missing` | No aparece como variable MCP en Pyomo actual |
| `lambdand` | sin espejo libre claro | `missing` | No aparece como variable libre equivalente |
| `lambdava` | sin espejo libre claro | `missing` | No aparece como variable libre equivalente |
| `nd` | `nd` | `exact_match` | Libre |
| `va` | `va` | `exact_match` | Libre |
| `px` | `px` | `missing` | Variable existe, pero sin ecuación activa equivalente |
| `lambdaio` | sin espejo libre claro | `missing` | No aparece como variable libre equivalente |
| `pnd` | `pnd` | `missing` | Variable existe, pero sin ecuación activa equivalente |
| `lambdaf` | sin espejo libre claro | `missing` | No aparece como variable libre equivalente |
| `xf` | `xf` | `exact_match` | Parcialmente fijo por cierre actual |
| `pva` | `pva` | `missing` | Variable existe, pero sin ecuación activa equivalente |
| `x` | `x` | `needs_mapping` | La variable coincide con GAMS; su fijación debe evaluarse vía `makb/xFlag` antes de tratarla como diferencia estructural |
| `xp` | `xp` | `exact_match` | Libre |
| `p` | `p_rai` / `p` implícito | `needs_mapping` | Falta cerrar la correspondencia exacta |
| `pp` | `pp_rai` / `pp` | `needs_mapping` | Falta cerrar la correspondencia exacta |
| `xft` | `xft` | `exact_match` | Totalmente fijada hoy en Pyomo |
| `pf` | `pf` | `exact_match` | Parcialmente fijada hoy |
| `pft` | `pft` | `needs_mapping` | Variable existe, pero espejo ecuacional no cerrado |
| `pfa` | sin espejo claro | `missing` | Requiere revisión |
| `pfy` | sin espejo claro | `missing` | Requiere revisión |
| `kstock` | `kstock` | `exact_match` | Libre |

#### Ingreso, Utilidad e Impuestos

| GAMS | Pyomo | Estado | Nota |
|---|---|---|---|
| `ytax` | `ytax` | `exact_match` | Libre |
| `ytaxTot` | `ytaxTot` | `exact_match` | Libre |
| `ytaxInd` | `ytax_ind` | `exact_match` | Libre |
| `factY` | `facty` | `exact_match` | Libre |
| `regY` | `regy` | `exact_match` | Libre |
| `phiP` | sin espejo claro | `missing` | Variable GAMS no aparece como variable Pyomo equivalente |
| `phi` | sin espejo claro | `missing` | Variable GAMS no aparece como variable Pyomo equivalente |
| `yc` | `yc` | `exact_match` | Libre |
| `yg` | `yg` | `exact_match` | Libre |
| `rsav` | `rsav` | `exact_match` | Libre |
| `uh` | `uh` | `exact_match` | Libre |
| `ug` | `ug` | `exact_match` | Libre |
| `us` | `us` | `exact_match` | Libre |
| `u` | `u` | `exact_match` | Libre |
| `zcons` | sin espejo variable claro | `missing` | Hoy no existe variable MCP equivalente |
| `xcshr` | `xcshr` | `exact_match` | Libre |
| `pcons` | `pcons` | `exact_match` | Libre |
| `pg` | `pg` implícita o ausente | `missing` | No hay espejo claro |
| `xg` | `xg` | `exact_match` | Libre |
| `pi` | `pi` | `exact_match` | Libre |
| `yi` | `yi` | `exact_match` | Libre |
| `xi` | `xi` | `exact_match` | Libre |
| `dintx` | sin espejo claro | `missing` | No hay variable MCP equivalente clara |
| `mintx` | sin espejo claro | `missing` | No hay variable MCP equivalente clara |
| `ytaxshr` | sin espejo claro | `missing` | No hay variable MCP equivalente clara |

#### Armington y Demanda Final

| GAMS | Pyomo | Estado | Nota |
|---|---|---|---|
| `xa(r,i,aa)` | `xaa` | `needs_mapping` | Misma idea económica, nombre distinto |
| `xd(r,i,aa)` | `xda` | `exact_match` | Parcialmente fijada hoy (`299`) |
| `xm(r,i,aa)` | `xma` | `exact_match` | Libre |
| `pdp(r,i,aa)` | `pdp` | `exact_match` | Libre |
| `pmp(r,i,aa)` | `pmp` | `exact_match` | Libre |
| `pa(r,i,aa)` | `pa` | `exact_match` | Libre |
| `xmt` | `xmt` | `exact_match` | Libre |
| `xw` | `xw` | `exact_match` | Parcialmente fijada (`100`) |
| `pmt` | `pmt` | `exact_match` | Libre |

#### CET, Oferta y Exportación

| GAMS | Pyomo | Estado | Nota |
|---|---|---|---|
| `xds` | `xds` | `exact_match` | Libre |
| `xet` | `xet` | `exact_match` | Libre |
| `xs` | `xs` | `exact_match` | Libre |
| `ps` | `ps` | `simplified` | Libre, pero sujeto a simplificación adicional `eq_ps` |
| `pe(r,i,rp)` | `pe` | `simplified` | Libre en rutas activas, pero atado por simplificación `eq_pe/eq_pe_route` |
| `pet` | `pet` | `simplified` | Libre, pero atado por simplificación adicional |
| `pd` | `pd` | `exact_match` | Libre |
| `pefob` | `pefob` | `exact_match` | Parcialmente fijada (`100`) |
| `pmcif` | `pmcif` | `exact_match` | Parcialmente fijada (`100`) |
| `pm` | `pm` | `exact_match` | Parcialmente fijada (`100`) |

#### Márgenes

| GAMS | Pyomo | Estado | Nota |
|---|---|---|---|
| `xwmg` | `xwmg` | `exact_match` | Parcialmente fijada (`370`) |
| `xmgm` | `xmgm` | `exact_match` | Parcialmente fijada (`370`) |
| `pwmg` | `pwmg` | `exact_match` | Parcialmente fijada (`370`) |
| `xtmg` | `xtmg` | `exact_match` | Libre |
| `ptmg` | `ptmg` | `simplified` | Libre, pero fijado al numeraire por ecuación simplificada |

#### Capital, Ahorro y Cierre Externo

| GAMS | Pyomo | Estado | Nota |
|---|---|---|---|
| `arent` | `arent` | `exact_match` | Libre |
| `kapEnd` | `kapEnd` | `exact_match` | Libre |
| `rorc` | `rorc` | `exact_match` | Libre |
| `rore` | `rore` | `exact_match` | Libre |
| `rorg` | `rorg` | `exact_match` | Libre |
| `chiSave` | `chiSave` | `exact_match` | Libre |
| `psave` | `psave` | `exact_match` | Libre |
| `xigbl` | `xigbl` | `exact_match` | Libre |
| `pigbl` | `pigbl` | `exact_match` | Libre |
| `chiInv` | sin espejo claro | `missing` | No hay variable MCP equivalente clara |
| `chif` | sin espejo claro | `missing` | No hay variable MCP equivalente clara |
| `savf` | `savf` | `exact_match` | Libre |

#### Macro, Numeraire y Bienestar

| GAMS | Pyomo | Estado | Nota |
|---|---|---|---|
| `pabs` | `pabs` | `needs_mapping` | Variable equivalente existe, pero formulación no es aún espejo literal |
| `pmuv` | sin espejo claro | `missing` | No existe variable equivalente clara |
| `pfact` | `pfact` | `exact_match` | Libre |
| `pwfact` | `pwfact` | `exact_match` | Libre |
| `pnum` | `pnum` | `fixed_in_pyomo_only` | En Pyomo queda fijada |
| `walras` | `walras` | `exact_match` | Libre |
| `gdpmp` | `gdpmp` | `exact_match` | Libre |
| `rgdpmp` | `rgdpmp` | `simplified` | Variable equivalente, ecuación aún simplificada |
| `pgdpmp` | `pgdpmp` | `exact_match` | Libre |
| `ev` | sin espejo claro | `missing` | No existe variable equivalente clara |
| `cv` | sin espejo claro | `missing` | No existe variable equivalente clara |

#### Política y Variables Técnicas Relevantes para el MCP actual

| GAMS | Pyomo | Estado | Nota |
|---|---|---|---|
| `etax` | `etax` | `fixed_in_pyomo_only` | Totalmente fijada hoy |
| `mtax` | `mtax` | `fixed_in_pyomo_only` | Totalmente fijada hoy |
| `imptx` | `imptx` | `exact_match` | Presente |
| `exptx` | `exptx` | `exact_match` | Presente |
| `lambdai` | sin espejo claro | `missing` | Requiere revisión |
| `lambdam` | `lambdam` | `fixed_in_pyomo_only` | Totalmente fijada hoy |
| `lambdamg` | `lambdamg` | `exact_match` | Presente |
| `tmarg` | `tmarg` | `exact_match` | Presente como parámetro |
| `amw` | `import_source_share` derivado desde `p_amw` | `needs_mapping` | Punto crítico: equivalencia exacta aún no cerrada |
| `gw` | `export_destination_share` derivado | `needs_mapping` | Debe confirmarse uso exacto |

### Clasificación de Fijaciones Actuales vs Cierre GAMS

Evidencia base usada para esta clasificación:

- la closure declarativa Pyomo fija por diseño `etax`, `mtax`, `lambdam`, `lambdamg`, `xft` y `tmarg` en [gtap_contract.py](/Users/marmol/proyectos2/equilibria/src/equilibria/templates/gtap/gtap_contract.py:117)
- GAMS fija en calibración:
  - `etax.fx(r,i,t) = 0` en [cal.gms](/Users/marmol/proyectos2/equilibria/src/equilibria/templates/reference/gtap/scripts/cal.gms:315)
  - `mtax.fx(rp,i,t) = 0` en [cal.gms](/Users/marmol/proyectos2/equilibria/src/equilibria/templates/reference/gtap/scripts/cal.gms:346)
  - `lambdam.fx(rp,i,r,t) = 1` en [cal.gms](/Users/marmol/proyectos2/equilibria/src/equilibria/templates/reference/gtap/scripts/cal.gms:347)
  - `lambdamg.fx(m,r,i,rp,t) = 1` en [cal.gms](/Users/marmol/proyectos2/equilibria/src/equilibria/templates/reference/gtap/scripts/cal.gms:382)
  - `tmarg.fx(r,i,rp,t)$xwFlag(r,i,rp)` en [cal.gms](/Users/marmol/proyectos2/equilibria/src/equilibria/templates/reference/gtap/scripts/cal.gms:322)
- GAMS fija por cierre/iteración:
  - `pnum.fx(t) = pnum.l(t)` en [iterloop.gms](/Users/marmol/proyectos2/equilibria/src/equilibria/templates/reference/gtap/scripts/iterloop.gms:41)
  - `xft.fx(r,fm,tsim)$(not xftFlag(r,fm)) = 0` en [iterloop.gms](/Users/marmol/proyectos2/equilibria/src/equilibria/templates/reference/gtap/scripts/iterloop.gms:142)
  - `chif.fx(r,t)$(not rres(r)) = chif.l(r,t)` en [iterloop.gms](/Users/marmol/proyectos2/equilibria/src/equilibria/templates/reference/gtap/scripts/iterloop.gms:46)

#### Tabla `Pyomo fixed now -> parity assessment`

Leyenda adicional:

- `closure_consistent`: fijación compatible con cierre/calibración estándar de GAMS
- `flag_consistent`: fijación compatible con lógica `$` de flags/rutas inactivas en GAMS
- `pyomo_structural_fix`: fijación usada hoy para cuadrar o estabilizar el MCP, pero no justificada como cierre estándar GAMS
- `needs_closure_decision`: depende de la closure final elegida y debe verificarse antes de eliminarla o mantenerla

| Pyomo fijo actual | Evidencia GAMS | Evaluación | Nota de paridad |
|---|---|---|---|
| `etax` | `etax.fx = 0` en calibración | `closure_consistent` | No es una diferencia estructural en cierre estándar |
| `mtax` | `mtax.fx = 0` en calibración | `closure_consistent` | No es una diferencia estructural en cierre estándar |
| `lambdam` | `lambdam.fx = 1` en calibración | `closure_consistent` | Mantener fija es compatible con benchmark estándar |
| `lambdamg` | `lambdamg.fx = 1` en calibración | `closure_consistent` | Compatible con benchmark estándar |
| `tmarg` | `tmarg.fx(...)` en calibración | `closure_consistent` | Tratarlo como exógeno es consistente con GAMS estándar |
| `pnum` | `pnum.fx(t) = pnum.l(t)` | `closure_consistent` | Numeraire fijo sí corresponde a GAMS |
| `xft` | `xft.fx(...)$(not xftFlag)` | `flag_consistent` | Fijar solo entradas inactivas sí es consistente; fijación total debe revisarse por closure |
| `xf` | `xf.fx(...)$(not xfFlag)` | `flag_consistent` | Solo consistente en celdas inactivas; no justifica fijación extra fuera de flags |
| `pf` | `pf.fx(...)$(not xfFlag)` | `flag_consistent` | Igual que `xf`: solo por flags inactivos |
| `xw` | `xw.fx(...)$(not xwFlag)` | `flag_consistent` | Fijación de rutas inactivas sí coincide con GAMS |
| `xe` | depende de flags/rutas no activas | `flag_consistent` | Consistente si se limita a celdas inactivas |
| `pe` | `pe.fx(...)$(not xwFlag)` | `flag_consistent` | Fijación de rutas inactivas sí coincide con GAMS |
| `pm` | `pm.fx(...)$(not xwFlag)` | `flag_consistent` | Fijación de rutas inactivas sí coincide con GAMS |
| `pmcif` | `pmcif.fx(...)$(not xwFlag)` | `flag_consistent` | Fijación de rutas inactivas sí coincide con GAMS |
| `pefob` | `pefob.fx(...)$(not xwFlag)` | `flag_consistent` | Fijación de rutas inactivas sí coincide con GAMS |
| `pwmg` | `pwmg.fx(...)$(not tmgFlag)` | `flag_consistent` | Fijación de rutas de margen inactivas sí coincide con GAMS |
| `xwmg` | `xwmg.fx(...)$(not tmgFlag)` | `flag_consistent` | Fijación de rutas de margen inactivas sí coincide con GAMS |
| `xmgm` | `xmgm.fx(...)$(not amgm)` | `flag_consistent` | Fijación de celdas inactivas sí coincide con GAMS |
| `x` | `x.fx(...)$(not xFlag)` y estructura make `makb` | `flag_consistent` | La fijación observada proviene de `conditional_fixing` sobre celdas make inactivas |
| `xda` | `xd.fx(...)$(not alphad)` en GAMS, pero Pyomo fija `299/1300` | `pyomo_structural_fix` | La fijación adicional de celdas activas es hoy una diferencia estructural clave |
| `pabs` | en GAMS puede fijarse por salida nula o por dinámica previa, no como parche MCP | `needs_closure_decision` | Verificar si alguna fijación actual excede el cierre estándar |
| `chif` | `chif.fx(r,t)$(not rres(r))` | `needs_closure_decision` | Depende del cierre de cuenta de capital elegido; no tratar como bug automático |

#### Conclusión Operativa de Fase 1

Las fijaciones actuales se separan en tres grupos:

- no son bug de paridad y pueden mantenerse mientras reproduzcan la closure estándar:
  - `etax`
  - `mtax`
  - `lambdam`
  - `lambdamg`
  - `tmarg`
  - `pnum`
- son consistentes con la lógica GAMS solo para entradas/rutas inactivas por flags:
  - `xft`
  - `xf`
  - `pf`
  - `xw`
  - `xe`
  - `pe`
  - `pm`
  - `pmcif`
  - `pefob`
  - `pwmg`
  - `xwmg`
  - `xmgm`
- siguen siendo diferencias estructurales Pyomo-only y son el objetivo principal de Fase 2:
  - `xda` fijada masivamente para cuadrar el MCP

Implicación práctica:

- Fase 2 no debe empezar “liberando todo”
- debe empezar eliminando solo las fijaciones `pyomo_structural_fix`
- y preservando las fijaciones justificadas por closure estándar o por flags inactivos

#### Medición Exacta del Gap MCP en la Ruta Nonlinear Actual

Medición reproducida con la misma secuencia que usa la CLI nonlinear:

- construir `GTAPModelEquations`
- construir `build_gtap_contract({"closure": "mcp"})`
- crear `GTAPSolver(..., solver_name="path-capi", params=params)`
- ejecutar `apply_closure()`
- ejecutar luego `apply_aggressive_fixing_for_mcp()` otra vez como sanity check

Resultado observado:

- antes de aplicar closure:
  - constraints activas: `17109`
  - variables libres: `21469`
  - gap MCP: `4360`
- después de `apply_closure()` en la ruta actual:
  - constraints activas: `17109`
  - variables libres: `17109`
  - gap MCP: `0`
- una llamada extra a `apply_aggressive_fixing_for_mcp()` fija `0` variables adicionales

Descomposición relevante del estado final:

| Variable | Total | Fijadas | Libres | Lectura |
|---|---:|---:|---:|---|
| `x` | `900` | `800` | `100` | Fijación estructural masiva |
| `xda` | `1300` | `299` | `1001` | Fijación estructural masiva |
| `xma` | `1300` | `0` | `1300` | Sin fijación extra |
| `xa` | `100` | `0` | `100` | Sin fijación extra |
| `pabs` | `10` | `0` | `10` | Sin fijación extra |
| `pe` | `1000` | `100` | `900` | Fijación por rutas inactivas |
| `xw` | `1000` | `100` | `900` | Fijación por rutas inactivas |
| `xe` | `1000` | `100` | `900` | Fijación por rutas inactivas |

Conclusión precisa:

- el gap estructural completo de `4360` no lo cierra una sola variable
- tras separar `conditional_fixing` de `aggressive fixing`, el bloque se interpreta así:
  - `x`: `800` fijadas por `conditional_fixing` vía `makb/xFlag`; esto es consistente con la lógica GAMS de celdas make inactivas
  - `xda`: `299` fijadas después para cerrar el MCP; esta sí sigue siendo la diferencia estructural relevante
- todas las demás fijaciones grandes visibles en el solve nonlinear actual están explicadas por:
  - closure estándar (`etax`, `mtax`, `lambdam`, `lambdamg`, `tmarg`, `pnum`)
  - flags/rutas inactivas (`xw`, `xe`, `pe`, `pm`, `pmcif`, `pefob`, `pwmg`, `xwmg`, `xmgm`)

Implicación para Fase 2:

- el primer experimento de eliminación estructural debe enfocarse en `xda`
- no en liberar `pe/xw/xe` ni las variables de impuestos/tecnología ya justificadas por GAMS
- la pregunta correcta ya no es “qué tanto fija Pyomo en general”, sino:
  - por qué necesita dejar solo `1001/1300` de `xda` libres

#### Hallazgo Específico del Bloque `x` vs `xda`

Medición separando `conditional_fixing` de `aggressive fixing`:

- antes de `conditional_fixing`:
  - gap: `3129`
  - `x`: `900` libres
  - `xda`: `1300` libres
- después de `conditional_fixing`:
  - gap: `299`
  - `x`: `100` libres, `800` fijadas
  - `xda`: `1300` libres, `0` fijadas

Además, con closure aplicada:

- `eq_x`: `100`
- `x` libres: `100`
- `eq_xda`: `1300`
- `xda` libres antes del cierre extra: `1300`

Interpretación:

- el bloque `x` está razonablemente alineado con GAMS:
  - solo quedan libres las celdas make activas
  - `eq_x` tiene exactamente la misma cardinalidad que `x` libre
- el gap remanente de `299` aparece después de esa limpieza y no proviene de `x`
- por tanto, el principal sospechoso estructural en esta subfase es `xda`

Hipótesis de trabajo para Fase 2:

- el desbalance ya no está en `xeq.x`
- está en cómo Pyomo representa el bloque Armington por agente:
  - `xda`
  - `xma`
  - `xaa`
  - `eq_xd_agg` / `eq_xmt_agg`
  - y el hecho de que `pdp` / `pmp` hoy sean `Expression` y no variables MCP explícitas como en GAMS

#### Hallazgo Nuevo: `xda` parece ser variable de sacrificio, no origen del gap

La comparación por bloque después de `conditional_fixing` muestra:

- `eq_x`: `100` vs `x` libre: `100`
- `eq_xda`: `1300` vs `xda` libre: `1300`
- `eq_xma`: `1300` vs `xma` libre: `1300`
- `eq_paa`: `1300` vs `pa` libre: `1300`
- `eq_xd_agg`: `100` vs `xd` libre: `100`
- `eq_xmt_agg`: `100` vs `xmt` libre: `100`

Eso indica que el subbloque Armington `xda/xma/pa/xd/xmt` está balanceado por cardinalidad.

En cambio, el bloque de precios/ingresos por actividad-commodity muestra:

- `p_rai`: `900` libres
- `pp_rai`: `900` libres
- `px`: `90` libres
- ecuaciones activas relacionadas:
  - `eq_x`: `100`
  - `eq_po`: `90`
  - `eq_pp_rai`: `100`

Lectura:

- Pyomo mantiene `p_rai` y `pp_rai` como variables MCP plenas
- pero no tiene un espejo completo uno-a-uno de `peq.p` y `ppeq.pp` de GAMS
- el resultado es un gran exceso potencial de grados de libertad en el bloque output-price mapping
- luego `apply_aggressive_fixing_for_mcp()` tapa el gap restante fijando `xda` solo porque aparece primero en la lista, no porque `xda` sea necesariamente el origen estructural

Nueva prioridad dentro de Fase 2:

1. revisar `p_rai/pp_rai/px` contra `peq/ppeq/pxeq` de GAMS
2. documentar si `p_rai` y `pp_rai` deben:
   - quedar como variables MCP explícitas con más ecuaciones, o
   - colapsarse/condicionarse como ocurre hoy con otras piezas derivadas
3. recién después volver sobre `xda` como posible consecuencia y no como causa primaria

#### Refinamiento: el problema en `p_rai/pp_rai` está fuera de soporte activo

Medición adicional sobre el soporte activo de `xFlag/makb`:

- soporte activo de `p_rai/pp_rai`: `100`
- `eq_x`: `100`
- `eq_pp_rai`: `100`
- `p_rai` libres sobre soporte activo: `100`
- `pp_rai` libres sobre soporte activo: `100`

Pero globalmente:

- `p_rai`: `900` libres
- `pp_rai`: `900` libres

Interpretación:

- sobre soporte activo, el bloque `x / p_rai / pp_rai` sí está balanceado
- el exceso real es que Pyomo deja libres `800` celdas de `p_rai` y `pp_rai` fuera del soporte make activo
- en GAMS, esas celdas están gobernadas por `$xFlag(r,a,i)` en:
  - `xeq`
  - `ppeq`
  - `peq`
- por tanto, el siguiente ajuste estructural no es “agregar más ecuaciones” para `p_rai/pp_rai`
- es replicar la lógica GAMS de flags también en el dominio/fijación de esas celdas inactivas

Nueva hipótesis principal:

- el gap MCP remanente de esta subfase proviene sobre todo de variables libres fuera de soporte en:
  - `p_rai`
  - `pp_rai`
- `xda` sigue siendo una variable de sacrificio del cierre extra, no la causa primaria del desbalance

Siguiente tarea concreta:

- extender `conditional_fixing()` para fijar `p_rai` y `pp_rai` a benchmark/numeraire cuando `xFlag(r,a,i) = 0` o `makb(r,a,i) = 0`
- recontar el gap post-conditional antes de tocar cualquier variable Armington

#### Resultado del Experimento `fix p_rai/pp_rai fuera de soporte`

Se probó fijar `p_rai` y `pp_rai` fuera de `x_rai_flag` dentro de `conditional_fixing()`.

Resultado:

- antes del cambio:
  - `PRE_CONDITIONAL`: gap `3129`
  - `POST_CONDITIONAL`: gap `299`
- con el cambio experimental:
  - `POST_CONDITIONAL`: gap `-1301`
  - `p_rai`: `800` fijadas, `100` libres
  - `pp_rai`: `800` fijadas, `100` libres

Conclusión:

- fijar simplemente `p_rai/pp_rai` fuera de soporte sobrecorrige el sistema
- eso indica que esas variables libres extra no pueden eliminarse de forma aislada sin reequilibrar el resto del mapeo MCP
- el problema ya no se interpreta como “faltan solo fijaciones por flags”
- se interpreta como una discrepancia estructural entre:
  - cómo GAMS representa `p(r,a,i)` y `pp(r,a,i)` dentro de `peq/ppeq/xeq/xpeq`
  - cómo Pyomo reparte ese bloque entre `p_rai`, `pp_rai`, `eq_x`, `eq_po`, `eq_pp_rai` y `eq_xs`

Decisión:

- el parche experimental fue revertido
- el baseline correcto sigue siendo:
  - `PRE_CONDITIONAL`: gap `3129`
  - `POST_CONDITIONAL`: gap `299`

Siguiente paso real:

- ya no intentar fijar `p_rai/pp_rai` aisladamente
- revisar el espejo estructural del bloque GAMS:
  - `xeq`
  - `xpeq`
  - `ppeq`
  - `peq`
- y decidir si Pyomo necesita:
  - más ecuaciones activas en ese bloque, o
  - menos variables libres en ese bloque, pero con una reformulación coherente y no por fijación local ad hoc

#### Revisión Directa de GAMS: `xeq/xpeq/ppeq/peq`

Comparación estructural:

- GAMS [model.gms](/Users/marmol/proyectos2/equilibria/src/equilibria/templates/reference/gtap/scripts/model.gms:587) define:
  - `xeq(r,a,i)` sobre `xFlag(r,a,i)`
  - `xpeq(r,a)` sobre `xpFlag(r,a)`
  - `ppeq(r,a,i)` sobre `xFlag(r,a,i)`
  - `peq(r,a,i)` sobre `xFlag(r,a,i)`
  - `pseq(r,i)` sobre `xsFlag(r,i)`
- Pyomo [gtap_model_equations.py](/Users/marmol/proyectos2/equilibria/src/equilibria/templates/gtap/gtap_model_equations.py:1887) tiene:
  - `eq_x`
  - `eq_po`
  - `eq_pp_rai`
  - `eq_xs`
  - pero no tiene un espejo explícito de `peq(r,a,i)` que complemente `p(r,a,i)`/`pp(r,a,i)` por ruta make activa

Resultado de soporte:

- soporte activo tipo `xFlag/makb`: `100`
- soporte tipo `peq` en Pyomo: `100`
- `eq_x = 100`
- `eq_pp_rai = 100`
- `eq_xs = 100`
- pero `p_rai` y `pp_rai` permanecen `900` libres cada una

Conclusión estructural:

- el faltante principal no es de soporte activo; el soporte activo coincide
- el faltante es que Pyomo colapsó `peq(r,a,i)` dentro de `eq_xs(r,i)` usando `pp_rai` agregados por commodity
- en GAMS, `peq(r,a,i)` es una ecuación por ruta make activa, no una agregación por commodity
- esa compresión elimina un bloque de ecuaciones MCP y deja libres `p_rai/pp_rai` fuera de la estructura correcta

Conclusión práctica para Fase 2:

- el siguiente cambio de fondo no debe ser sobre `xda`
- debe ser reintroducir un espejo explícito de `peq(r,a,i)` en Pyomo
- solo después tiene sentido reevaluar si siguen sobrando variables y si `xda` continúa siendo la variable sacrificada por el cierre extra

#### Avance Ejecutado: `peq(r,a,i)` reintroducida en Pyomo

Se añadió `eq_peq(r,a,i)` en [gtap_model_equations.py](/Users/marmol/proyectos2/equilibria/src/equilibria/templates/gtap/gtap_model_equations.py) como espejo de `peq(r,a,i)` de GAMS:

- soporte activo observado: `100`
- `eq_peq = 100`

Impacto medido en el conteo MCP:

- antes de `eq_peq`:
  - `PRE_CONDITIONAL`: `17109` constraints, `20238` libres, gap `3129`
  - `POST_CONDITIONAL`: `17109` constraints, `17408` libres, gap `299`
- después de `eq_peq`:
  - `PRE_CONDITIONAL`: `17209` constraints, `20238` libres, gap `3029`
  - `POST_CONDITIONAL`: `17209` constraints, `17408` libres, gap `199`

Conclusión:

- la ausencia de `peq(r,a,i)` sí era una parte real del gap estructural
- reintroducir ese espejo cerró `100` grados de libertad, exactamente como sugería el análisis de soporte
- el remanente bajó de `299` a `199`

Implicación:

- el diagnóstico estructural va en la dirección correcta
- aún faltan bloques GAMS equivalentes para cerrar completamente la brecha
- el siguiente candidato natural sigue siendo revisar qué otra parte de `p/pp/px/xs` quedó comprimida o mal asignada en Pyomo

#### Avance Ejecutado: `$` bilaterales alineados con GAMS

Se reemplazaron igualdades de rutas inactivas por `Constraint.Skip` en:

- `eq_pe_route`
- `eq_xe_xw`
- `eq_xweq`
- `eq_pmeq`
- `eq_pmcifeq`
- `eq_pefobeq`

Esto alinea mejor el comportamiento con GAMS, donde estos bloques están activados por flags tipo `$xwFlag`.

Impacto medido:

- antes de este cambio:
  - `PRE_CONDITIONAL`: gap `2759`
  - `POST_CONDITIONAL`: gap `-71`
  - `ALL_FIXED_CONSTRAINTS`: `630`
- después de este cambio:
  - `PRE_CONDITIONAL`: gap `3359`
  - `POST_CONDITIONAL`: gap `529`
  - `ALL_FIXED_CONSTRAINTS`: `30`

Cardinalidades observadas después del cambio:

- `eq_pe_route = 900`
- `eq_xe_xw = 900`
- `eq_xweq = 900`
- `eq_pmeq = 900`
- `eq_pmcifeq = 900`
- `eq_pefobeq = 900`

Conclusión:

- el sobrecierre anterior venía en gran parte de constraints activas sobre variables ya fijadas en rutas inactivas
- al corregir eso, quedó expuesta una brecha estructural más honesta: faltan aproximadamente `529` grados de libertad por cerrar con ecuaciones MCP equivalentes a GAMS
- esto confirma que la estrategia correcta es seguir reemplazando simplificaciones por espejos GAMS, aunque temporalmente el sistema deje de quedar cuadrado

Siguiente frente prioritario:

- bloques aún marcados como `missing` o `simplified` con peso estructural:
  - `pfaeq`
  - `pfyeq`
  - `dintxeq`
  - `mintxeq`
  - `ytaxshreq`
  - `pmuveq`
  - `eveq`
  - `cveq`

#### Avance Ejecutado: `pfaeq/pfyeq`

Se portó el bloque de precios de factores más cerca de GAMS:

- nuevas variables:
  - `pfa(r,f,a)`
  - `pfy(r,f,a)`
- nuevas ecuaciones:
  - `eq_pfaeq`
  - `eq_pfyeq`
- `eq_xfeq` y `eq_pvaeq` ahora usan `pfa` como término de precio de factor, en línea con `M_PFA` de GAMS
- `conditional_fixing()` ahora también fija `pfa/pfy` fuera de `xfFlag`

Medición:

- con `pfa/pfy` pero sin fixing por `xfFlag`, el gap empeoraba:
  - `POST_CONDITIONAL`: `849`
- tras alinear `pfa/pfy` con `xfFlag`, el gap volvió a:
  - `POST_CONDITIONAL`: `529`

Lectura:

- `pfaeq/pfyeq` sí eran faltantes reales de paridad estructural
- pero su port no reduce por sí mismo el gap neto del MCP, porque agrega variables y ecuaciones casi en la misma magnitud
- aun así, deja el bloque de factores mucho más fiel a GAMS y evita seguir mezclando `pf` con `M_PFA`

Estado neto después de este bloque:

- el gap estructural honesto sigue en `529`
- pero el modelo ya replica mejor dos frentes importantes:
  - make/supply con `peq`
  - factor prices con `pfaeq/pfyeq`

## Fase 2. Eliminar el `aggressive fixing` como mecanismo estructural

- [ ] Identificar por qué Pyomo queda con gap MCP antes del cierre artificial
- [ ] Reemplazar `apply_aggressive_fixing_for_mcp()` por un mapeo MCP explícito
- [ ] Asegurar que el sistema quede cuadrado sin fijaciones artificiales extra
- [ ] Verificar que la cuenta final sea:
  - ecuaciones activas MCP = variables libres MCP

Criterio de cierre:

- el modelo queda cuadrado sin `aggressive fixing`

### Estado Actual de Fase 2

Subbloques ya portados o alineados parcialmente:

- `peq`
- casos Cobb-Douglas de `pxeq/pndeq/pvaeq`
- `$` bilaterales en rutas inactivas
- `pfaeq/pfyeq`
- `dintxeq/mintxeq/ytaxshreq`
- fijación GAMS-consistente de `p_rai` fuera de `xFlag`
- desactivación de restricciones Pyomo-only:
  - `prf_y`
  - `eq_ps`
  - `eq_pe`

Medición vigente más honesta del sistema:

- antes de este ajuste:
  - `POST_CONDITIONAL` gap MCP: `529`
- después de fijar `p_rai` fuera de `xFlag` y sacar del MCP activo `prf_y`, `eq_ps`, `eq_pe`:
  - `POST_CONDITIONAL` gap MCP: `19`
  - variables libres: `19888`
  - constraints activas: `19869`

Estado del cierre artificial restante:

- con `close_mcp_gap=True`, `apply_aggressive_fixing_for_mcp()` ya no fija `299` celdas de `xda`
- ahora el cierre residual se absorbe primero en `pp`
- verificación actual:
  - `pp` ya queda con `19` fijaciones residuales
  - `xda` fijadas adicionalmente: `0`
  - `p_rai` fijadas por soporte `xFlag`: `800`

Lectura:

- Fase 2 ya no está dominada por un hueco estructural masivo
- el remanente de `19` indica que el MCP está muy cerca de quedar cuadrado sin `aggressive fixing`
- eso habilita arrancar Fase 3 sin seguir arrastrando las simplificaciones más obvias del bloque CET/exportador

Bloques GAMS aún ausentes como MCP explícito en Pyomo:

- `pmuveq`
- `eveq`
- `cveq`

Observación:

- `pmuv`, `ev` y `cv` no existen todavía como variables en el template Pyomo actual
- por tanto siguen siendo faltantes reales de paridad de Fase 2
- pero no explican por sí solos el gap `529` hasta que se porten; hoy el siguiente frente con más probabilidad de mover el balance del MCP sigue siendo revisar bloques ya presentes pero aún simplificados o comprimidos
- con el estado nuevo, la principal deuda estructural de Fase 2 ya no es masiva; queda un remanente pequeño (`19`) y el frente siguiente correcto es Fase 3

## Fase 3. Portar el bloque CET/exportador literalmente

Portar sin simplificaciones:

- [ ] `xdseq`
- [ ] `xeteq`
- [ ] `xseq`
- [ ] `peeq`
- [ ] `peteq`

Eliminar simplificaciones actuales:

- [x] quitar `ps = pd` como identidad fija
- [x] quitar `pet = ps` como identidad fija
- [ ] quitar `pe(route) = pet` como identidad fija general

Verificaciones:

- [ ] benchmark residual por ecuación ~ 0
- [ ] los casos hoy conflictivos dejan de estar dominados por `eq_xseq`, `eq_ps`, `eq_pe`, `eq_peteq`

Criterio de cierre:

- bloque CET/exportador alineado algebraicamente con `model.gms`

### Arranque de Fase 3

Acciones ya hechas al abrir esta fase:

- `eq_ps` salió del MCP activo
- `eq_pe` salió del MCP activo
- el soporte de precios make-route (`p_rai`) ya quedó alineado con `xFlag`, como en `iterloop.gms`
- `xe` salió del MCP activo:
  - `xe` quedó totalmente fijada como auxiliar Pyomo-only
  - `eq_xe_xw` quedó desactivada
  - `eq_xet_agg` ya agrega `xw` directamente, no `xe`
- `eq_pe_route` salió del MCP activo
- `pp` quedó fijada como auxiliar Pyomo-only y `pp_rai` quedó fijada fuera de `xFlag`
- `chif` dejó de estar colapsada como parámetro y ya existe como variable explícita con `eq_chif`, más cerca de `chifeq.chif` de GAMS

Próximo frente inmediato dentro de Fase 3:

- revisar el bloque de ecuaciones GAMS que entran "sueltas" al MCP (`savfeq`, `capAccteq`, `pnumeq`, `eveq`, `cveq`) frente a la representación Pyomo actual
- identificar si el gap honesto residual `29` viene de esa diferencia de mapeo, no de otro `Skip` faltante

Medición inicial de ese frente:

- baseline actual sin `close_mcp_gap`: gap `29`
- con el estado actual:
  - `free = 18108`
  - `cons = 18079`
  - `gap = 29`
- corrida corta instrumentada posterior al port explícito de `chif`:
  - `elapsed_sec ≈ 106.9`
  - `function_calls = 51`
  - `jacobian_calls = 8`
  - `latest_function_inf_norm ≈ 9.999`
  - `latest_function_l2_norm ≈ 16.77`
  - `latest_x_inf_norm ≈ 19.845`
  - no apareció un nuevo error de dominio en esa ventana

Lectura:

- ya salieron del MCP activo las simplificaciones exportadoras más obvias (`eq_ps`, `eq_pe`, `eq_pe_route`, `eq_xe_xw`)
- el remanente ya no parece venir de una sola identidad Pyomo-only
- la siguiente hipótesis fuerte es el mapeo GAMS de ecuaciones no emparejadas, en particular:
  - `savfeq`
  - `capAccteq`
  - `pnumeq`
  - `eveq`
  - `cveq`
- antes de tocar más bloques Armington/trade, conviene cerrar esa diferencia de representación del MCP

## Fase 4. Portar el bloque bilateral/importación exactamente como GAMS

Portar/revisar:

- [ ] `xmteq`
- [ ] `xweq`
- [ ] `pmteq`
- [ ] `pmeq`
- [ ] `pmcifeq`
- [ ] `pefobeq`

Revisar consistencia de parámetros:

- [ ] `amw`
- [ ] `lambdam`
- [ ] `chipm`
- [ ] `mtax`
- [ ] `etax`

Punto crítico:

- [ ] confirmar el significado económico exacto de `p_amw` en Pyomo frente a `amw` en GAMS
- [ ] evitar usar como “share normalizado” un parámetro que en GAMS es un parámetro CES calibrado

Verificaciones:

- [ ] benchmark parity de `xmt`, `xw`, `pm`, `pmcif`, `pefob`
- [ ] desaparición del desajuste `xmt` vs `sum(xma/xscale)` en casos benchmark

Criterio de cierre:

- el nest importador y bilateral replica la referencia GAMS

## Fase 5. Portar bloque macro exactamente como compStat GAMS

Portar/revisar:

- [ ] `gdpmpeq`
- [ ] `rgdpmpeq`
- [ ] `pgdpmpeq`
- [ ] `pabs`

Requisitos:

- [ ] usar base-year / compStat tal como en GAMS
- [ ] eliminar proxies simplificados no equivalentes
- [ ] verificar consistencia entre benchmark y solve

Verificaciones:

- [ ] benchmark residual bajo en `gdpmp`, `rgdpmp`, `pgdpmp`
- [ ] `eq_rgdpmp` deja de aparecer como residuo dominante

Criterio de cierre:

- bloque macro alineado con la formulación GAMS compStat

## Fase 6. Replicar lógica de activación tipo `$`

- [ ] revisar ecuación por ecuación dónde GAMS usa `$`
- [ ] mapear esos casos a `Constraint.Skip` o activación condicional exacta
- [ ] cubrir especialmente:
  - `xwFlag`
  - `xmtFlag`
  - `xetFlag`
  - `xdFlag`
  - `tmgFlag`
  - `amgm`

Verificaciones:

- [ ] no quedan ecuaciones activas en Pyomo que GAMS habría desactivado
- [ ] no faltan ecuaciones que GAMS sí activaría

Criterio de cierre:

- activación de ecuaciones equivalente a GAMS

## Fase 7. Alinear inicialización y bounds con GAMS

- [ ] revisar `.l` benchmark en GAMS
- [ ] revisar `.lo` benchmark-relativos en GAMS
- [ ] eliminar floors artificiales no justificados
- [ ] mantener solo bounds positivos que sí correspondan a la política de GAMS

Verificaciones:

- [ ] el benchmark inicial queda consistente sin parches extra
- [ ] no reaparecen errores de dominio en potencias negativas

Criterio de cierre:

- la trayectoria inicial PATH en Pyomo parte del mismo vecindario económico que GAMS

## Fase 8. Suite de pruebas de paridad por bloques

- [ ] prueba del bloque CET/exportador
- [ ] prueba del bloque Armington/importación
- [ ] prueba del bloque de márgenes
- [ ] prueba del bloque macro
- [ ] prueba del bloque ingreso/ahorro

Cada prueba debe verificar:

- [ ] residual benchmark bajo
- [ ] misma identidad algebraica que GAMS
- [ ] mismas ecuaciones activas

Criterio de cierre:

- existe una suite de paridad que detecta regresiones estructurales

## Fase 9. Validación final PATH

Solo después de cerrar las fases anteriores:

- [ ] correr solve corto instrumentado
- [ ] correr solve largo
- [ ] comparar contra GAMS:
  - convergencia
  - residuos
  - bloques dominantes
  - trayectoria PATH

Si todavía hay diferencias:

- [ ] ajustar opciones PATH
- [ ] revisar escalamiento numérico
- [ ] revisar licencia/runtime solo después de descartar diferencias estructurales

Criterio de cierre:

- Pyomo y GAMS están resolviendo el mismo MCP y la comparación de convergencia ya es válida

## Prioridad Recomendada

1. Inventario MCP exacto GAMS vs Pyomo
2. Eliminar `aggressive fixing` como mecanismo estructural
3. Portar bloque CET/exportador
4. Portar bloque bilateral/importación
5. Portar bloque macro
6. Replicar flags `$`
7. Alinear bounds e inicialización
8. Suite de paridad
9. Validación PATH final

## Regla de Aceptación

No considerar “resuelto” ningún bloque solo porque PATH deja de explotar.

Un bloque se considera cerrado solo si:

- la formulación es equivalente a GAMS
- el benchmark pasa
- la activación de ecuaciones coincide
- el bloque no depende de fijaciones artificiales para existir

## Estado Actual Fase 3

- `default-9x10.gdx` ya se carga automáticamente como fuente de elasticidades/CDE cuando el benchmark principal no las trae.
- `incpar` y `subpar` ya están disponibles en `GTAPElasticities`; el port de `eveq/cveq` queda pospuesto hasta definir una formulación numéricamente segura para PATH.
- `eq_ev` y `eq_cv` están construidas pero fuera del MCP activo; `ev/cv` quedan fijadas como auxiliares Pyomo-only por ahora.
- se añadió diagnóstico en `path-capi-python` para reportar la ecuación exacta que falla en callbacks `F/J`.

### Hallazgo nuevo clave

- el error `complex` no venía de `eveq/cveq`; venía de `eq_paa`.
- la causa era estructural: `dintx` y `mintx` seguían libres dentro del MCP aunque la closure estándar los trata como exógenos.
- PATH los estaba empujando a valores imposibles, por ejemplo:
  - `dintx[EU_28,c_Crops,a_HeavyMnfc] = -52.63`
  - `dintx[EU_28,c_Crops,a_OthService] = -119.07`
- eso hacía negativo el término `(1 + dintx) * pd` dentro de `eq_paa`, y por tanto aparecían potencias fraccionales complejas.

### Corrección aplicada

- `dintx` y `mintx` ahora se fijan bajo `closure.fix_taxes`.
- `eq_dintxeq` y `eq_mintxeq` quedan desactivadas en el MCP activo cuando esas cuñas están fijadas.
- con eso el sistema volvió a quedar cuadrado sin ese bloque:
  - `free = 15478`
  - `cons = 15478`
  - `gap = 0`

### Señal de solve corta tras la corrección

- la corrida corta nonlinear ya no cae por dominio en `eq_paa`.
- a ~`70s`:
  - `function_calls = 32`
  - `jacobian_calls = 6`
  - `||F||∞ ≈ 10.84`
  - `||F||2 ≈ 17.89`
  - `||x||∞ ≈ 19.845`

### Próximo frente dentro de Fase 3

- revisar cuál es ahora el siguiente bloque dominante en la trayectoria PATH con el MCP ya libre de `dintx/mintx`.
- prioridad inmediata:
  1. leer el snapshot/top residuos del estado corto actual
  2. identificar la siguiente ecuación que domina `||F||∞`
  3. seguir eliminando diferencias de closure/mapeo antes de tocar tuning numérico de PATH

## Estado Actual Fase 3

### Avances nuevos

- se corrigió la inicialización de `xs` para que siga el benchmark de make de GAMS:
  - `xs` ahora se ancla primero a `sum_a makb(r,a,i)` y no al agregado simplificado de `get_trade_totals()`.
- con eso `eq_peq` dejó de dominar el benchmark:
  - casos que antes fallaban como `EU_28/a_HeavyMnfc/c_HeavyMnfc` ahora quedan en `0.0`.
- luego se corrigió la convención de escalado del bloque make:
  - `xp`, `va` y `nd` ahora arrancan reescaladas por `xscale(r,a)`, alineado con `cal.gms`.
  - `xf` hereda esa convención vía `va`.
- también se alineó la calibración de `gd/ge` al mismo `xs` benchmark que usa ahora el modelo.

### Señal nueva del benchmark

- después de esos cambios, salieron del top:
  - `eq_x`
  - `eq_peq`
  - `eq_xds`
  - `eq_xet`
- el top residual actual en benchmark quedó concentrado en:
  1. `eq_chif[RestofWorld] = 10.4781`
  2. `eq_rgdpmp[EastAsia] = 2.7795`
  3. `eq_xseq[EU_28,c_Extraction] = -1.3242`
  4. `eq_xseq[SouthAsia,c_Extraction] = -1.2827`
  5. varios `eq_pfaeq[...]` sub-1 y `eq_xi[...]` sub-0.4

### Señal nueva de solve

- una corrida nonlinear que estaba activa tras el ajuste del lado productivo terminó con:
  - `Residual = 1.23e+01`
  - `Iterations = 8 / 4636`
  - `Callbacks = F=47 (3.60s) | J=15 (8.58s)`
  - `Post-checks = True`
- esto es una mejora útil frente al desanclaje intermedio que había aparecido tras corregir solo `xs`.

### Hallazgo abierto

- el siguiente diff estructural más claro ahora está en el bloque CET superior:
  - `eq_xseq`
- el patrón sugiere que todavía no estamos replicando exactamente la calibración GAMS de `xds/xet/gd/ge` para algunos commodities, especialmente `c_Extraction`.
- referencia GAMS relevante:
  - `xds.l = sum(aa, xd.l(...))`
  - `xet.l = (ps*xs - pd*xds)/pet`
  - `gd/ge` se calibran sobre esos niveles ya consistentes

### Próximo paso

- portar esa calibración de `xet`/`gd`/`ge` más cerca de GAMS para cerrar `eq_xseq`, empezando por el bloque de `c_Extraction`.

## Estado Actual Fase 3 - Update Export/Import

### Corrección aplicada

- se movió `_refresh_macro_initial_state()` para que corra después de resincronizar
  `xaa[hhd/gov/inv]` con `xc/xg/xi`.
- con eso salieron del top los residuos grandes de:
  - `eq_gdpmp`
  - `eq_rgdpmp`
- también se probó el bloque exportador final:
  - recalibrar `gw_share` y refrescar `pet/xet`
  - luego se evaluó si conviene recalcular `xw` al final del refresh

### Hallazgo importante

- recalcular `xw` al final del refresh exportador deja limpio `peeq`, pero desancla el
  lado importador y hace que el principal residual pase a `eq_xweq`.
- mantener `xw` anclada y refrescar macro después es más fiel a `cal.gms`:
  en GAMS `xw.l` se fija desde `VXSB/pe` y luego se calibran `gw` y `xet`, pero no
  se vuelve a reconstruir `xw` al final.
- por eso se eliminó el último refresh bilateral de `xw` del flujo
  `apply_production_scaling()`.

### Benchmark actual

- top residuales actuales:
  1. `eq_peeq[EU_28,c_HeavyMnfc,NAmerica] ≈ -0.462`
  2. `eq_peeq[EU_28,c_HeavyMnfc,RestofWorld] ≈ -0.442`
  3. `eq_xweq[RestofWorld,c_Extraction,MENA] ≈ -0.393`
  4. `eq_peeq[EU_28,c_HeavyMnfc,EastAsia] ≈ -0.358`
  5. `eq_xweq[MENA,c_Extraction,EastAsia] ≈ 0.337`
  6. `eq_ytax[EastAsia,ic] ≈ -0.287`
  7. `eq_pvaeq[EU_28,a_agricultur] ≈ -0.273`

### Diagnóstico actual

- el siguiente frente real ya no es macro.
- el desajuste dominante está en la reconciliación entre:
  - `peeq` del lado exportador
  - `xweq` del lado importador
- el caso líder muestra que `xw` sigue siendo la variable puente problemática:
  para `('EU_28','c_HeavyMnfc','NAmerica')`, después del refresh:
  - `xw ≈ 0.737`
  - `rhs_xweq ≈ 0.380`
  - residual `≈ 0.357`

### Próximo paso

- atacar la reconciliación estructural de `xw` entre los nests exportador e importador
  sin volver a introducir un refresh final de `xw` que no existe en GAMS.

### Update reciente

- probé una variante más agresiva de anclaje de precios de ruta:
  - `pet = ps`
  - `pe = pet`
- esa variante rompió más de lo que arregló:
  - `eq_peteq` subió a magnitudes de `3.9`
  - `eq_xet` y `eq_xseq` también empeoraron fuerte
- el mejor estado conocido sigue siendo el actual:
  - `xw` anclada al snapshot de referencia en la inicialización
  - sin refresh final de `xw`
  - sin recálculo final de `xet`
  - `peeq`, `xweq` y `xseq` quedan en el rango `0.2-0.5` en lugar de explotar

## Estado Actual Fase 3 - Residual Analysis 2026-04-09

### Resultado global

- la producción quedó alineada en el benchmark:
  - `eq_x = 1.776357e-15`
  - `eq_peq = 6.661338e-16`
  - `eq_pp_rai = 0.0`
- el análisis completo de residuales volvió a mover el cuello dominante al bloque exportador:
  - `eq_peteq` max residual `4.068048e+01`
  - `eq_xfeq` max residual `8.363035e-01`
  - `eq_xaa_hhd` max residual `3.271469e-01`
  - `eq_ytax` max residual `2.800218e-01`
  - `eq_pvaeq` max residual `2.726921e-01`
  - `eq_xaa_activity` max residual `2.660203e-01`
  - `eq_paa` max residual `2.318354e-01`

### Lectura operativa

- el bloque de producción ya no es el problema principal
- el bloque macro tampoco es el cuello dominante
- el frente que sigue empujando el residual es `eq_peteq`
- el siguiente grupo de interés es `eq_xfeq` y luego las demandas Armington por agente (`eq_xaa_hhd`, `eq_xaa_activity`, `eq_xaa_inv`)

### Próximo paso

- atacar `eq_peteq` contra la calibración GAMS del bloque CET exportador
- luego revisar si `eq_xfeq` y las demandas por agente requieren un ajuste de benchmark o de mapeo MCP
