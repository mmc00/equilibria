# GTAP v6.2 ↔ v7.0 — Crosswalk oficial (Table 1 de Corong et al 2017)

Transcripción limpia de [docs/gtap_7_docs.en.md](../../docs/gtap_7_docs.en.md)
líneas 2575–2977 (Table 1, Old vs new notation). Esta tabla es la spec
canónica para construir un reference v6.2 a partir de v7.

## (1) Sets e índices

| v6.2 | v7.0 | Notas |
|---|---|---|
| REG(r) | REG(r) | igual |
| TRAD_COMM(i) | COMM(c) | rename + cambio de índice i→c |
| MARG_COMM(m) | MARG(m) | rename |
| NMRG_COMM(n) | NMRG(n) | rename |
| ENDW_COMM(e) | ENDW(e) | rename |
| DEMD_COMM(i) | DEMD(d) | rename + i→d |
| ENDWC_COMM(i) | ENDWC(e) | rename + i→e |
| ENDWM_COMM(i) | ENDWM(e) | rename + i→e |
| ENDWS_COMM(i) | ENDWS(e) | rename + i→e |
| PROD_COMM(j) | (eliminado) | ya no se usa en v7 |
| CGDS_COMM | (eliminado) | inversión = agente explícito en v7 |
| NSAV_COMM | (eliminado) | ídem |
| — | **ACTS(a)** | NUEVO en v7: actividades, excluye inversión |
| — | **ENDWF(e)** | NUEVO: factor sector-específico (recurso natural) |
| — | **ENDWMS(e)** | NUEVO: mobile + sluggish unificados |

## (2) Top-level production nest

| v6.2 | v7.0 | Cambio |
|---|---|---|
| `qo(i,r)` sobre NSAV_COMM | `qo(a,r)` | output ahora por activity |
| `ps(i,r)` | `po(a,r)` | unit cost of activity a |
| `pm(i,r)` | `pb(a,r)` | activity tax-inclusive price (= pds para make diagonal) |
| (implícito) | **`pds(c,r)`** | NUEVO: basic price of domestic commodity (reemplaza pm) |
| (implícito) | **`ps(c,a,r)`** | NUEVO: supply price commodity-por-activity (production-tax-exclusive) |

## (3) Intermediate demand bundle — **NUEVO en v7, no existe en v6.2**

| v6.2 | v7.0 | Notas |
|---|---|---|
| (no existe — Leontief implícito) | `qint(a,r)`, `aint(a,r)`, `pint(a,r)`, `ESUBC(a,r)` | CES bundle de intermedios por actividad. ESUBC default = 0.0 (Leontief) |

## (4) Commodity sourcing by firms (Armington doméstico vs importado)

| v6.2 | v7.0 |
|---|---|
| `ESUBD(c)` | `ESUBD(c,r)` — ahora region-specific |
| `qf(i,j,r)` | `qfa(c,a,r)` — 'a' = agent |
| `pf(i,j,r)` | `pfa(c,a,r)` |

## (5) MAKE transformation — **NUEVO en v7, no existe en v6.2**

| v6.2 | v7.0 |
|---|---|
| (make matrix diagonal implícita: 1 activity = 1 commodity) | `ETRAQ(a,r)`, `ESUBQ(c,r)`, `ps(c,a,r)`, `qca(c,a,r)`, `pca(c,a,r)`, `qc(c,r)` |

Detalles v7:
- CET para suministro de commodities por activities (`ETRAQ(a,r)`)
- CES para sourcing de commodities por activities (`ESUBQ(c,r)`, default 0 = perfect subs)
- `qc(c,r)` = total commodity supply (reemplaza `qo(i,r)` de v6.2)

## (6) Private demand

| v6.2 | v7.0 |
|---|---|
| `qp(i,r)`, `pp(i,r)` | `qpa(c,r)`, `ppa(c,r)` — sufijo 'a' = agent |

## (7) Government demand — **función cambia: CD → CES**

| v6.2 | v7.0 |
|---|---|
| Cobb-Douglas: `qg(i,r)`, `pg(i,r)` | CES: `qga(c,r)`, `pga(c,r)`, `ESUBG(r)` (default 1.0 = CD) |

## (8) Investment demand — **estructura cambia**

| v6.2 | v7.0 |
|---|---|
| Investment como `PROD_COMM`: `qo(cgds,r)`, `qf(i,cgds,r)`, `qfd(i,cgds,r)`, `qfm(i,cgds,r)` | Agente explícito: `qinv(r)`, `qia(c,r)`, `qid(c,r)`, `qim(c,r)` |
| `pm(cgds,r)`, `pf(i,cgds,r)`, `pfd(i,cgds,r)`, `pfm(i,cgds,r)` | `pinv(r)`, `pia(c,r)`, `pid(c,r)`, `pim(c,r)` |
| `tfd(i,cgds,r)`, `tfm(i,cgds,r)` | `tid(c,r)`, `tim(c,r)` |
| Demanda implícita via cgds como commodity | Función Leontief-CES anidada explícita |

## (9) Sourcing of imports (Armington internacional)

| v6.2 | v7.0 |
|---|---|
| `ESUBM(i)` | `ESUBM(c,r)` — ahora commodity y region-specific |
| `qim(i,r)`, `pim(i,r)` | `qms(c,r)`, `pms(c,r)` |
| `qxs(i,s,r)`, `pms(i,s,r)` | `qxs(c,s,d)`, `pmds(c,s,d)` |
| índices: s=source, r=importer | índices: s=source, d=destination |

## (10) International trade & transport margins — **CD → CES**

| v6.2 | v7.0 |
|---|---|
| Cobb-Douglas | CES con `ESUBS(m)` (default 1.0 = CD) |

## (11) Market clearing for commodities

| v6.2 | v7.0 |
|---|---|
| `qo(i,r)` sobre TRAD_COMM | `qc(c,r)` = total supply para doméstico + exports |

## (12) Income distribution & factor market — **mercados a nivel activity**

| v6.2 | v7.0 |
|---|---|
| `qo(i,r)`, `ps(i,r)`, `toi(i,r)`, `pm(i,r)` sobre ENDW_COMM | `qe(e,r)`, `pe(e,r)` sobre ENDW (agregado) |
| `pmes(i,r)`, `qoes(i,r)`, `ETRAE(i)` sobre ENDWS_COMM | `qes(e,a,r)`, `pes(e,a,r)`, `tinc(e,a,r)`, `peb(e,a,r)` sobre ENDW (a nivel activity) |
| | `ETRAE(e,r)` sobre ENDWS — endowment y region-specific |

Insight clave: la remuneración a factores es ahora **a nivel de actividad** en
v7. Refleja que recursos naturales son sector-específicos. `tinc(e,a,r)`
reemplaza `toi(i,r)` para endowments.

---

## Implicaciones para el rollback v7 → v6.2 (revisado)

Esto **no son solo renames**. Hay 5 cambios estructurales reales:

### 🔴 Cambios estructurales (eliminan ecuaciones)

| # | Cambio | Acción en equilibria |
|---|---|---|
| 3 | **Eliminar intermediate bundle** | Borrar `eq_nd`, `eq_pnd`. Intermedios entran directo a producción (Leontief implícito) |
| 5 | **Eliminar MAKE transformation** | Borrar ecuaciones de `qca, pca, qc`. Make diagonal: `qo(i,r) = qc(c,r)` con i=c |
| 8 | **Sustituir CES de gobierno por CD** | Reemplazar `eq_xag` (con ESUBG) por la forma CD original (sin elasticidad) |
| 10 | **Margins CD en vez de CES** | Reemplazar `eq_xmgm, eq_xtmg, eq_ptmg` por forma CD |
| 12 | **Factor markets a nivel commodity** | Cambiar `eq_xfeq(r,fp,a)`, `eq_pfeq(r,fp,a)` a versiones (r,i) con i ∈ ENDW_COMM. `tinc` desaparece — usar `toi` |

### 🟡 Cambios de indexación (mismo modelo, distintos índices)

| # | Cambio | Acción |
|---|---|---|
| 1 | Renames de sets | Alias `COMM=TRAD_COMM`, `MARG=MARG_COMM`, etc. |
| 2 | `qo(a,r) → qo(i,r)`, `ps→ps`, `pm→pm` | Colapsar ACTS dimension: `a = i` |
| 4 | `qfa(c,a,r) → qf(i,j,r)` | Mapear `(c,a) → (i,j)`. ESUBD pierde dimensión `r` |
| 6 | `qpa(c,r) → qp(i,r)` | Rename + i en lugar de c |
| 7 | `qga(c,r) → qg(i,r)` | Rename + función CD (ver fila 7 arriba) |
| 8 | `qia/qid/qim(c,r) → qf/qfd/qfm(i,cgds,r)` | Inversión vuelve a ser commodity, no agent |
| 9 | `qms/pms(c,r)`, `qxs(c,s,d)/pmds` → `qim/pim(i,r)`, `qxs(i,s,r)/pms` | Rename + s/d → s/r |
| 11 | `qc(c,r) → qo(i,r)` over TRAD_COMM | Mercado total = total output del bien |

### 🟢 Eliminaciones puras

| Eliminar de v6.2 | Por qué |
|---|---|
| Set `ACTS` | Colapsado a TRAD_COMM |
| Set `ENDWF` | v6.2 no tiene factor sector-específico |
| Set `ENDWMS` | v6.2 no unifica mobile + sluggish |
| Parámetro `ENDOWFLAG(e,t)` | v6.2 usa `SLUG(i)` binario |
| `pefactreal`, `pebfactreal` | Adiciones Nov 2017 |
| `ESUBC(a,r)`, `ESUBG(r)`, `ESUBS(m)`, `ETRAQ(a,r)`, `ESUBQ(c,r)` | Elasticidades de v7 |
| Variables `qint, aint, pint, qca, pca, qc, pds, pb, ps(c,a,r), tinc, peb, qes(e,a,r), pes(e,a,r)` | Variables nuevas de v7 |
| Variables `qpa, ppa, qga, pga, qia, pia, qid, pid, qim, pim, tid, tim, qms, pms(c,r), pmds, qxs(c,s,d)` | Renombradas a versiones v6.2 |

## Mapping rápido para implementación

| equilibria v7 (Pyomo) | Forma v6.2 equivalente |
|---|---|
| `eq_nd` (aggregate intermediate demand) | **eliminar** — input directo a `eq_x` Leontief |
| `eq_va` (aggregate value-added) | mantener (existía en v6.2) |
| `eq_pxeq` (unit cost activity) | reescribir: `po(a,r)` → `ps(i,r)` con i=a |
| `eq_xapeq, eq_pndeq` (composite intermediate) | **eliminar** — bundle no existe |
| `eq_xfeq(r,fp,a)`, `eq_pfeq(r,fp,a)` | indexar por `(r,i)` con i ∈ ENDW_COMM, eliminar dim `a` |
| `eq_xc, eq_xcshr, eq_zcons` (CDE/CD private demand) | mantener (mecánica CDE/CD igual; renombrar qpa→qp, ppa→pp) |
| `eq_xg, eq_pg, eq_ug` (govt CES) | reescribir como CD — eliminar elasticidad |
| `eq_xi, eq_xiagg, eq_pi` (inv agent) | reformular: investment como commodity cgds, no agent |
| `eq_xmgm, eq_xtmg, eq_ptmg` (margins CES) | reescribir como CD |
| Cualquier `eq_*` con `(c,a,r)` | colapsar a `(i,r)` con i=c=a |

## Referencia para Francois extensions

El paper v7 menciona explícitamente que Francois (1998, "GTAP-IRTS") fue construido
sobre v6.x con la idea de *"simply adding a few equations to the standard model,
providing a few additional parameters, and altering the model closure"* (Sec 4.1
del paper, línea 3007 del doc). El roadmap personal del proyecto
([versión anterior → Francois]) **requiere v6.2 como base** porque las
extensiones de Francois están escritas para esa estructura (no para v7 con
ACT/COMM split).
