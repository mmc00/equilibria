# GTAP v6.2 — Implementation Plan

**Fecha:** 2026-05-20
**Status:** Draft — pendiente green-light para implementar
**Referencia de diff:** [runs/gtap_v62_vs_v7/](../../runs/gtap_v62_vs_v7/)
(README.md + notation_crosswalk.md + 4 CSVs de equation crosswalk)
**Oracle de validación:** `gtap.exe` v6.2 en `C:\runGTAP375\` (corriendo el TAB
`gtap_v62.tab` con `RunGTAP` o `gemsim`/`runtab` directos)

## Por qué v6.2 (no v7)

equilibria implementa hoy GTAP Standard 7 (Corong et al 2017), siguiendo la
traducción GAMS de van der Mensbrugghe (JGEA 2018). El roadmap personal del
proyecto (memoria `project_equilibria_scope.md`) indica:

> welfare RunGtap → Altertax → versión anterior → extensiones Francois

La "versión anterior" es **GTAP 6.2** (Hertel/Itakura/McDougall 2003), porque:

1. **Francois (1998, "GTAP-IRTS")** fue construido sobre v6.x. El paper v7
   (Sec. 4.1) lo describe como "simply adding a few equations to the standard
   model, providing a few additional parameters, and altering the model
   closure". Las ecuaciones de Francois (monopolistic competition, Cournot,
   scale economies) están escritas para la estructura v6.x — no para v7 con
   split ACT/COMM.
2. **AlterTax v6.2** ya existe en RunGTAP local (`altpar.tab`). Es una
   herramienta complementaria que opera sobre datasets v6.2.
3. Tener v6.2 + v7 lado-a-lado da una **referencia de regresión** para validar
   que los cambios estructurales de v7 (intermediate bundle CES, MAKE, ESUBG,
   factor markets a nivel activity) se implementaron correctamente.

## Estrategia de alto nivel

**No GAMS nuevo.** Construir un nuevo template puro en Python/Pyomo bajo
`src/equilibria/templates/gtap_v62/` que sea **paralelo** a
`templates/gtap/`. Reutilizar babel HAR/GDX, contracts, solver wrappers,
shocks API — todo lo agnóstico de versión.

**Oracle:** correr el TAB v6.2 con el `gtap.exe` que ya está instalado en
`C:\runGTAP375\` sobre el dataset BOOK3X3 (que es el dataset canónico que
acompaña al TAB 6.2). Dump de `.sl4` solution a HAR via `sltohar.exe` (ya
incluido en RunGTAP). Comparar Python vs GEMPACK cell-by-cell, igual que
hoy se hace contra NEOS v7.

**Phasing:**
- **Fase 1 — Scaffold** (1 semana): estructura de archivos, sets, parameters,
  oracle de GEMPACK funcionando.
- **Fase 2 — Modelo core** (1.5 semanas): equations adaptadas con los 5
  cambios estructurales aplicados.
- **Fase 3 — Validación** (1 semana): parity tests vs GEMPACK, dataset
  BOOK3X3 + NUS333.
- **Fase 4 — Welfare decomp v6.2** (0.5 semana): equivalente a `decomp.tab`
  externo.

**Effort total:** ~4 semanas (con buffer realista 5 semanas).

## Estructura de archivos

Crear `src/equilibria/templates/gtap_v62/`:

```
src/equilibria/templates/gtap_v62/
├── __init__.py                  # exports paralelos a gtap/__init__.py
├── README.md                    # diferencias clave vs gtap/ + ejemplo quickstart
├── gtap_v62_sets.py             # GTAPv62Sets — sin ACTS, con SLUG binario
├── gtap_v62_parameters.py       # GTAPv62Parameters — read HAR v6.2 + GDX
├── gtap_v62_contract.py         # GTAPv62Contract — closures v6.2
├── gtap_v62_model_equations.py  # GTAPv62ModelEquations — equations adaptadas
├── gtap_v62_solver.py           # thin wrapper reusing gtap.gtap_solver
├── welfare_decomp_v62.py        # decomp.tab equivalente (post-solve)
└── shocks_v62.py                # shocks API (reusa core de gtap/shocks.py)
```

Y `tests/templates/gtap_v62/`:

```
tests/templates/gtap_v62/
├── test_sets.py                 # carga BOOK3X3 sets correctamente
├── test_parameters.py           # carga benchmark BOOK3X3
├── test_model_build.py          # modelo construye sin errores
├── test_baseline_parity.py      # baseline vs gtap.exe BOOK3X3
├── test_shock_parity.py         # 10% tariff shock vs gtap.exe
└── test_v62_vs_v7_diff.py       # sanity check: cells que deben diferir
```

Y `scripts/gtap_v62/`:

```
scripts/gtap_v62/
├── run_gtap_v62.py              # CLI: info, solve, shock
├── validate_v62_parity.py       # corre Python + GEMPACK + compara
└── run_gempack_oracle.py        # invoca gtap.exe v6.2 y dumpea sol
```

## File-by-file: qué copiar, qué cambiar

### 1. `gtap_v62_sets.py` (~400 LOC)

**Base:** copiar `gtap/gtap_sets.py` (561 LOC) y simplificar.

**Cambios clave (Tabla 1, item 1):**

| Field/método de v7 | Cambio en v6.2 |
|---|---|
| `a: List[str]` (activities) | **Eliminar** — colapsar `a = i` siempre. Set diagonal forzado. |
| `i_to_a, a_to_i, output_pairs, activity_commodities` | **Eliminar** — make matrix siempre diagonal en v6.2 |
| `mf, sf` derivados de `ENDOWFLAG(e,t)` | **Reescribir** — derivar de `SLUG(i)` binario (header "SLUG" en `default.prm`/GTAPPARM) |
| `_determine_factor_mobility` con ENDOWFLAG | Cambiar a leer `SLUG` y separar `if SLUG[f] == 0: mf.append; else: sf.append` |
| `load_from_har(sets_path, default_path)` | Mantener pero **sin header ACTS**. Sólo: REG, COMM (=TRAD_COMM), MARG, ENDW_COMM, CGDS_COMM |
| Properties `is_diagonal, is_bijective_output_structure` | **Eliminar** (siempre True) |

**Sets v6.2 a declarar:**

```python
@dataclass
class GTAPv62Sets:
    r: List[str]              # REG — regions
    i: List[str]              # TRAD_COMM — traded commodities
    marg: List[str]           # MARG_COMM — margin commodities (subset of i)
    cgds: List[str]           # CGDS_COMM — capital goods (single element)
    f: List[str]              # ENDW_COMM — primary factors
    mf: List[str]             # ENDWM_COMM (SLUG=0)
    sf: List[str]             # ENDWS_COMM (SLUG=1)
    # Derivados
    prod_comm: List[str]      # TRAD_COMM ∪ CGDS_COMM
    demd_comm: List[str]      # ENDW_COMM ∪ TRAD_COMM
    nsav_comm: List[str]      # DEMD_COMM ∪ CGDS_COMM
    m: List[str]              # alias de i para margins
    s: List[str]              # alias de r para bilaterals
```

**Nota:** `ACTS` no se almacena. Si código consumidor pregunta por `a`,
devolver `i` (property `a → return self.i`).

### 2. `gtap_v62_parameters.py` (~1,200 LOC, vs 2,312 en v7)

**Base:** copiar `gtap/gtap_parameters.py` y eliminar todo lo activity-level.

**Cambios clave:**

| Container/parámetro de v7 | Cambio en v6.2 |
|---|---|
| `esubc, etraq, esubq` (intermediate bundle, MAKE) | **Eliminar** — no existen en v6.2 |
| `esubg` (Cobb-Douglas implícito → CES en v7) | **Eliminar** — gobierno es CD en v6.2, sin elasticidad |
| `sigmam` (CES margins) | **Eliminar** — margins son CD en v6.2 |
| `esubd: Dict[Tuple[str, str], float]` (region-specific) | Cambiar a `Dict[str, float]` — solo por commodity en v6.2 |
| `esubm: Dict[Tuple[str, str], float]` | Cambiar a `Dict[str, float]` — solo por commodity |
| `tinc(e,a,r)` | **Eliminar** — usar `toi(i,r)` para income tax on endowments |
| `vfa, vfm, vdfa, vdfm` (con dimensión a) | Reformular: leer `VFA(i,j,r)`, `VDFA(i,j,r)` con i,j sobre TRAD_COMM. `j` reemplaza `a` |
| `peb, pes, qes` (activity-level factor) | **Eliminar** — usar `pmes(i,r)`, `qoes(i,r)` sobre ENDW_COMM |
| Reader benchmarks: `VFM, VDFM, VIFM, VPA, VGA, VIA, VXMD, VIMS` | Mantener nombres mayúscula (header en HAR son los mismos en v6.2) pero asegurar índices `(i,j,r)` no `(c,a,r)` |
| `vom(a,r)` (output by activity) | Reformular a `VOM(i,r)` sobre TRAD_COMM ∪ CGDS_COMM |

**Calibrated shares a recalibrar** (los nombres se mantienen pero la
indexación cambia):
- `nd_share, va_share` (production nest) — igual, pero indexadas (r,i) con i=a
- `alphad, alpham` (Armington) — indexadas (r,i) — igual pero sin agent splitting
- `gx` (output shares) — ya no hay make matrix, son trivial 1.0 si i ∈ activity output

**Dataset compat:** los GDX/HAR de v6.2 usan headers como `VFM`, `VFA`,
`VDFA`, `VIFA`, `VOM`, `VPA`, `VGA`, `VXMD`, `VIMS`, `MAKB`. Estos
**mismos headers existen** en datasets v7 (basedata-9x10.gdx). La
diferencia es que en v6.2 `j` ∈ TRAD_COMM (no ACTS), y `make` es diagonal.

**Estrategia recomendada:** reutilizar `babel.har.read_har` y
`babel.gdx.read_gdx`. La capa de mapeo va en `gtap_v62_parameters.py`.

### 3. `gtap_v62_contract.py` (~200 LOC, vs 466 en v7)

**Base:** copiar `gtap/gtap_contract.py`.

**Cambios:**

- Lista de equation IDs reducida: eliminar `e_lambdaio`, `e_xftReg`, `e_pftFnm`, `e_xftFnm`, `e_capAcct` que dependen de la estructura activity-level extendida que no existe en v6.2.
- Closure `gtap_standard_v62` con numeraire `pgdpwld` (v6.2 default) en lugar de `pnum`.
- Eliminar referencias a `ifSUB` macro — v6.2 no tiene la dualidad
  pdp/pmp formulation (todas las relaciones de precio son inline).

### 4. `gtap_v62_model_equations.py` (~3,500 LOC, vs 5,706 en v7)

**Base:** copiar `gtap/gtap_model_equations.py` como punto de partida.

#### Cambios estructurales por sección de Tabla 1:

**§3 — Eliminar intermediate bundle (qint, aint, pint, ESUBC)**

| equilibria v7 | Cambio |
|---|---|
| `eq_nd_rule` (línea 3921) | **Eliminar** — no hay agregación de intermedios |
| `eq_pndeq_rule` (línea 3990) | **Eliminar** |
| `eq_xapeq_rule` (línea ~ línea ?) | Renombrar a `eq_xf_rule`. Función directa: `qf(i,j,r) = qo(j,r) * io(i,j,r)` (Leontief, no CES) |
| Variables `nd, pnd, ava_param, and_param` | **Eliminar** |

En su lugar, **producción v6.2** es:
```
qf(i,j,r) = afall(i,j,r) * qo(j,r)  ! Leontief intermediates
qfe(f,j,r) = aff(f,j,r) * qo(j,r) ^ esubt(j,r) ...  ! CES VA
```
con `esubt` (entre VA y composite intermediate) reemplazando `sigmap`.

**§5 — Eliminar MAKE transformation (qca, pca, qc, ETRAQ, ESUBQ)**

| equilibria v7 | Cambio |
|---|---|
| `eq_xseq_rule` (sourcing commodities por activities) | **Eliminar** — make diagonal forzada |
| `eq_pseq_rule` (commodity price por activity) | **Eliminar** — colapsar `ps(c,a,r) → ps(i,r)` |
| `eq_pxeq_rule` (unit cost por activity) | Renombrar `po(a,r) → ps(i,r)` con i=a |
| Variables `qca, pca, qc, pds, pb, ps(c,a,r)` | **Eliminar** todas |

Production en v6.2 es de un solo commodity por sector: `qo(i,r) = output del sector i` directo.

**§7 — Gobierno: CES → Cobb-Douglas**

| equilibria v7 | Cambio |
|---|---|
| `eq_xg_rule` (CES con esubg) | Reescribir como Cobb-Douglas: `pg(i,r) * qg(i,r) = sg(i,r) * yg(r)` donde `sg(i,r)` es share calibrado |
| `eq_pg_rule` (CES price index) | Reescribir como `pg(r) = prod(i, pg(i,r)**sg(i,r))` (CD geometric mean) |
| `eq_ug_rule` | Mantener — utility de gobierno es lineal en gasto |
| Parámetro `esubg` | **Eliminar** |

**§10 — Margins: CES → Cobb-Douglas**

| equilibria v7 | Cambio |
|---|---|
| `eq_xtmg_rule, eq_ptmg_rule, eq_xatmg_rule` | Reescribir como CD: `xtmg(m) = chi(m) * Y / ptmg(m)` y `ptmg = prod(i, pa(i,r)**share(m,i,r))` |
| Parámetro `sigmam` (ESUBS) | **Eliminar** |

**§12 — Factor markets a nivel commodity (no activity)**

Este es el cambio más grande estructuralmente.

| equilibria v7 | Cambio |
|---|---|
| `eq_xfeq_rule(r, fp, a)` | Renombrar a `eq_xfeq_rule(r, fp, i)` — i ∈ TRAD_COMM |
| `eq_pfeq_rule(r, fp, a)` | Reformular: `pf(r, fp, i)` para mobile factor en sector i. Para sluggish, `pmes(i,r)` único. |
| `eq_xft_rule, eq_xfteq_rule` (factor aggregation) | Mantener pero indexados sobre i, no a |
| `eq_pfact_rule` | Mantener — agregado regional |
| `tinc(e,a,r)` en cualquier ecuación | Reemplazar por `toi(i,r)` (income tax sobre endowment use por sector) |
| `peb(e,a,r)` (tax-inclusive activity-level factor price) | **Eliminar** — usar `pm(i,r)` para ENDW_COMM |

**§8 — Investment como commodity, no agente**

| equilibria v7 | Cambio |
|---|---|
| `eq_xc, eq_xg, eq_xi` (3 agent demand equations) | Unificar `eq_xi` con la estructura v6.2: `qi(r)` es un sector productivo `cgds`, no un agente |
| `eq_xiagg_rule` (composite investment demand) | Reformular: `qcgds(r) = qo(cgds, r)` — output del sector cgds |
| `eq_paa_rule` para agente `inv` | Eliminar — sustituido por demanda interpretada en el sector cgds |
| Variables `qia, qid, qim, pia, pid, pim, tid, tim` | **Eliminar** — reemplazar con `qf(i,cgds,r), qfd(i,cgds,r), qfm(i,cgds,r), pf(i,cgds,r), tfd(i,cgds,r), tfm(i,cgds,r)` |

**§4,§6,§9 — Renames (mecánica idéntica)**

| equilibria v7 | v6.2 |
|---|---|
| `qfa, pfa, qfd, qfm, pfd, pfm` (c,a,r) | `qf, pf, qfd, qfm, pfd, pfm` (i,j,r) |
| `qpa, ppa, qpd, qpm, ppd, ppm` (c,r) | `qp, pp, qpd, qpm, ppd, ppm` (i,r) |
| `qms, pms, pmds, qxs(c,s,d)` | `qim, pim, pms, qxs(i,s,r)` |
| `pmds(c,s,d)` | `pms(i,s,r)` |
| `ESUBM(c,r)` | `ESUBM(i)` (sin dim region) |
| `ESUBD(c,r)` | `ESUBD(i)` (sin dim region) |

**§11 — Market clearing colapsa**

`qc(c,r) → qo(i,r)` sobre TRAD_COMM. Una sola ecuación de balance:
`qo(i,r) = qds(i,r) + sum(rp, qxs(i,r,rp))`.

**Welfare**: mover `eq_ev`, `eq_cv` y todas las `eq_cnt*` a `welfare_decomp_v62.py`
(equivalente al `decomp.tab` externo en GEMPACK v6.2).

### 5. `gtap_v62_solver.py` (~80 LOC)

Wrapper delgado que extiende `gtap.GTAPSolver` overriding solo:
- Las variables de Walras check (mismo nombre `walras`)
- Las variables fijas en numeraire (`pgdpwld` en lugar de `pnum`)

El motor de PATH/IPOPT/aggressive-fixing es reutilizado sin cambios.

### 6. `welfare_decomp_v62.py` (~250 LOC)

Migra la lógica de `decomp.tab` (42 KB de GEMPACK) al estilo postsim.

**Ecuaciones a portar de `decomp_v62.tab`:**
- `CONT_EV_*` (50 contribuciones de welfare por categoría)
- `EVREG, EVWLD` (regional y mundial)
- `EV_DECOMPOSITION` (descomposición Huff/McDougall)

**Función pública:**
```python
def compute_welfare_decomp_v62(
    base_params: GTAPv62Parameters,
    base_model: ConcreteModel,
    shock_params: GTAPv62Parameters,
    shock_model: ConcreteModel,
) -> Dict[str, V62WelfareDecomp]:
```

Devuelve la descomposición clásica de v6.2 (sin las refinaciones de v7 como
factor income tax effect — esa solo existe desde Dec 2019).

### 7. `shocks_v62.py` (~100 LOC)

Reutiliza `apply_shock` de `gtap.shocks` con un registry específico v6.2
(sin `taxes.rtfd, rtfi, rtpd, rtpi, rtgd, rtgi` que son v7-specific).

```python
from equilibria.templates.gtap.shocks import apply_shock as _apply_shock
from equilibria.templates.gtap.shocks import _ContainerSpec

_REGISTRY_V62 = {
    "taxes.tm": _ContainerSpec(  # import tariff
        path="taxes.tm",
        dim_names=("commodities", "destinations"),
    ),
    "taxes.tx": _ContainerSpec(  # export tax
        path="taxes.tx",
        dim_names=("sources", "commodities", "destinations"),
    ),
    "taxes.to": _ContainerSpec(  # output tax
        path="taxes.to",
        dim_names=("regions", "sectors"),
    ),
    # etc — solo los taxes que v6.2 tiene
}
```

## Oracle de validación — GEMPACK local

equilibria ya tiene infra para correr GAMS local. Hay que agregar la
equivalente para GEMPACK v6.2 desde `C:\runGTAP375\`.

### `scripts/gtap_v62/run_gempack_oracle.py`

```python
def run_gtap_v62_gempack(
    cmf_path: Path,        # e.g. C:\runGTAP375\BOOK3X3\Exp1a.exp
    dataset_dir: Path,     # e.g. C:\runGTAP375\BOOK3X3\
    output_har: Path,      # donde dumpear la solución
) -> Dict[str, float]:
    """Invoca gtap.exe v6.2 sobre el CMF dado, dumpea .sl4 a HAR."""
    gtap_exe = Path("C:/runGTAP375/gtap.exe")
    # 1. Run: gtap.exe -cmf <cmf>
    subprocess.run([str(gtap_exe), "-cmf", str(cmf_path)], cwd=dataset_dir, check=True)
    # 2. Convert .sl4 → .har via sltohar.exe
    sl4 = cmf_path.with_suffix(".sl4")
    subprocess.run(["C:/runGTAP375/sltohta.exe", str(sl4), str(output_har)], check=True)
    # 3. Read .har with babel.har.read_har
    return read_har(output_har)
```

### `scripts/gtap_v62/validate_v62_parity.py`

Espejo de `scripts/gtap/validate_gams_parity.py` pero comparando contra
GEMPACK en lugar de GAMS:

1. Cargar dataset BOOK3X3.har.
2. Correr Python: solve baseline → solve shock 10% tariff.
3. Correr GEMPACK: `gtap.exe Exp1a.cmf` (baseline) y `Exp1b.cmf` (shock).
4. Diff cell-by-cell de ~30 variables clave.
5. Output: tabla de paridad como `runs/gtap_v62_compare/`.

### Datasets de validación

| Dataset | Origen | Tamaño | Uso |
|---|---|---|---|
| **BOOK3X3** | `C:\runGTAP375\BOOK3X3\basedata.har` | 3×3 (3 regions, 3 commodities) | Test minimal, ya viene con experimentos (Exp1a–4b) |
| **NUS333** | `C:\runGTAP375\NUS333\GTAPSAM.har` | 3×3 (US-centric) | Equilibria ya valida NUS333 contra v7 — co-validación |
| **3×2** | mismo NUS333 después de agregar a 2 sectors | 3×2 | El dataset que ya usa equilibria — directa comparación |

**Recomendación de orden:**
1. Arrancar con BOOK3X3 — viene con experimentos GEMPACK pre-configurados.
2. Luego NUS333 — para co-validar con equilibria v7.
3. 9x10 NO se valida contra GEMPACK v6.2 porque el dataset fue construido
   para v7 (ACTS dimension, ENDWFLAG). Esa es la única limitación.

## Equation-level mapping (resumen)

Total de equations en `templates/gtap` (v7): **94 funciones `eq_*_rule`**.
Estimado para v6.2: **~75 funciones** (después de eliminar ~25 v7-only y
agregar ~6 v6.2-specific).

| v7 (equilibria) | v6.2 acción | Notas |
|---|---|---|
| `eq_nd, eq_pndeq` | ❌ eliminar | §3 intermediate bundle |
| `eq_xseq, eq_pseq` | ❌ eliminar | §5 MAKE |
| `eq_xpeq → eq_po` | ✏️ renombrar | unit cost de activity → ps(i,r) |
| `eq_paa, eq_pp_rai, eq_pmpeq, eq_pdpeq` | ✏️ reescribir | §4 sin agent split |
| `eq_xaa_activity, eq_xaa_hhd, eq_xaa_gov, eq_xaa_inv, eq_xaa_tmg` | ✏️ fusionar 5→2 | v6.2 tiene xf/qp/qg/qcgds — un agent menos (no inv) |
| `eq_xfeq, eq_pfeq` (activity-level) | ✏️ reescribir (i,r) | §12 commodity-level |
| `eq_xc, eq_xcshr, eq_zcons, eq_uh, eq_u, eq_us, eq_pcons` | 🟢 mantener | CDE/CD privado igual |
| `eq_xg, eq_pg, eq_ug` | ✏️ reescribir CD | §7 |
| `eq_xtmg, eq_ptmg, eq_xatmg, eq_pwmg, eq_xmgm, eq_xwmg` | ✏️ reescribir CD | §10 |
| `eq_xeq, eq_xweq, eq_peeq, eq_peteq, eq_pefobeq, eq_pmcifeq, eq_pmeq, eq_pdeq, eq_pmteq` | ✏️ renombrar índices | §9 (s,d) → (s,r) |
| `eq_xi, eq_xiagg, eq_pi` | ✏️ reformular | §8 investment como sector |
| `eq_facty, eq_regy, eq_yc, eq_yg, eq_yi, eq_ytax, eq_ytax_tot, eq_ytax_ind` | 🟢 mantener | identidades de ingreso |
| `eq_savf, eq_rorc, eq_rore, eq_rorg, eq_chif, eq_capacct, eq_chisave, eq_psave, eq_xigbl, eq_pigbl` | 🟢 mantener | mecanismo de inversión global igual |
| `eq_kstock, eq_kapend, eq_arent` | 🟢 mantener | capital igual |
| `eq_pabs, eq_gdpmp, eq_rgdpmp, eq_pgdpmp, eq_pmuv, eq_pwfact, eq_pnum → eq_pgdpwld` | ✏️ numeraire change | pnum → pgdpwld |
| `eq_walras` | 🟢 mantener | Walras check |
| `eq_ev, eq_cv` | 📦 mover a welfare_decomp_v62.py | postsim |
| `eq_dintxeq, eq_mintxeq` | 🟢 mantener | tax revenues |

**Conteo final:** ~75 ecuaciones core en `gtap_v62_model_equations.py`, en
línea con el conteo GEMPACK v6.2 si excluimos welfare decomp (213 − 50 ≈ 163,
pero ~75 Pyomo por el mismo collapsing que ya hace v7).

## Plan por fases

### Fase 1 — Scaffold (semana 1)

**Goal:** Estructura de archivos creada, sets/parameters cargan BOOK3X3,
oracle GEMPACK corre.

1. Crear directorio `templates/gtap_v62/` con todos los archivos vacíos.
2. Copiar `__init__.py`, `gtap_sets.py`, `gtap_parameters.py`,
   `gtap_contract.py` desde `templates/gtap/` y renombrar clases.
3. Eliminar campos activity-level de Sets/Parameters (ver §1, §2).
4. Implementar `_determine_factor_mobility` con SLUG binario.
5. Implementar `gtap_v62_parameters.load_from_har(BOOK3X3/basedata.har)`.
6. Implementar `scripts/gtap_v62/run_gempack_oracle.py` y verificar que
   corre `BOOK3X3/Exp1a.exp` y dumpea HAR válido.
7. Tests unitarios: `test_sets.py`, `test_parameters.py`.

**Gate:** equilibria puede leer BOOK3X3 y `gtap.exe` corre desde Python.

### Fase 2 — Modelo core (semanas 2-2.5)

**Goal:** `GTAPv62ModelEquations.build_model()` produce un modelo Pyomo
que carga y al menos arranca el solver.

1. Copiar `gtap_model_equations.py` → `gtap_v62_model_equations.py`.
2. Aplicar cambios §3, §5 (eliminar intermediate bundle y MAKE) — los más
   destructivos primero.
3. Aplicar §12 (factor markets commodity-level) — el segundo más profundo.
4. Aplicar §7, §10 (gobierno y margins CD).
5. Aplicar §8 (investment como sector).
6. Aplicar §1, §2, §4, §6, §9, §11 (renames + index changes).
7. Solver wrapper en `gtap_v62_solver.py`.

**Gate:** baseline BOOK3X3 corre y converge (walras < 1e-6).

### Fase 3 — Validación (semana 3.5)

**Goal:** parity contra GEMPACK BOOK3X3 (baseline + shock 10% tariff)
y NUS333.

1. `validate_v62_parity.py` — script de comparación cell-by-cell.
2. Correr baseline BOOK3X3 Python vs `gtap.exe BOOK3X3/Exp1a.cmf` → 0
   cells diverge.
3. Correr shock 10% tariff Python vs `gtap.exe BOOK3X3/Exp1b.cmf` (Exp1b
   ya incluye un shock — verificar cuál es exactamente; si no es 10%
   tariff, crear un CMF custom).
4. Repetir contra NUS333.
5. Documentar paridad en `docs/site/benchmarks_v62.md`.

**Gate:** paridad 100% en BOOK3X3 y NUS333 (baseline + shock vs GEMPACK).

### Fase 4 — Welfare decomp v6.2 (semana 4)

1. Migrar `decomp_v62.tab` a `welfare_decomp_v62.py`.
2. Validar contra `gtap.exe + decomp.exe` BOOK3X3 (RunGTAP usa decomp.tab
   post-solve).
3. Test: `test_welfare_decomp_parity.py`.

**Gate:** EV decomposition Python vs RunGTAP coincide < 0.01%.

## Riesgos y mitigaciones

| Riesgo | Probabilidad | Mitigación |
|---|---|---|
| Calibration shares de v6.2 no son derivables del benchmark v7 dump | Media | Phase 1 carga benchmark v6.2 directo desde HAR, no de GDX v7. Esto fuerza re-calibration desde cero |
| Variable `tinc(e,a,r)` está implícita en el SAM en muchos lugares | Alta | Tratar como tax revenue stream agregado: `tincr = sum(i, toi(i,r) * pm(i,r) * qoes(i,r))` |
| Convergencia con PATH cambia (degrees of freedom diferentes) | Media | Reutilizar `apply_aggressive_fixing_for_mcp` de `gtap_solver.py` adaptado al closure v6.2 |
| Dataset BOOK3X3 demasiado pequeño para detectar errores | Baja | Validar también NUS333 antes de declarar fase completa |
| AlterTax v6.2 (`altpar.tab`) tiene su propia complejidad | Baja | Plan separado — no incluido en este scope. Ver `gtap_altertax_implementation_plan_2026-05-13.md` para la versión v7 |

## Diferencias con el plan AlterTax y plan welfare-decomp existentes

Estos planes son **complementarios**:

- `gtap_altertax_implementation_plan_2026-05-13.md` — implementa AlterTax
  sobre v7 actual. Si después se quiere AlterTax v6.2, se reutiliza el
  patrón pero apuntando a `templates/gtap_v62/`.
- `gtap_welfare_decomp_cgebox_implementation_plan_2026-05-13.md` —
  implementa welfare decomp v7 estilo cgebox. La Fase 4 de este plan
  hace lo mismo pero apuntando al `decomp.tab` v6.2 (más simple).

## Tests y CI

Convención existente en `tests/templates/gtap/` se replica en
`tests/templates/gtap_v62/`. Pytest discovery automático.

**Marcadores propuestos:**
- `@pytest.mark.gempack` — tests que requieren `gtap.exe` instalado
  (skip en CI Linux/macOS, solo se ejecutan en máquinas Windows con RunGTAP).
- `@pytest.mark.slow` — parity tests con shock (ya existe el marker).

## Referencias

- [runs/gtap_v62_vs_v7/README.md](../../runs/gtap_v62_vs_v7/README.md) — análisis comparativo completo
- [runs/gtap_v62_vs_v7/notation_crosswalk.md](../../runs/gtap_v62_vs_v7/notation_crosswalk.md) — Table 1 oficial transcrita
- [docs/gtap_7_docs.en.md](../../docs/gtap_7_docs.en.md) líneas 2575–2977 — fuente original
- `C:\runGTAP375\gtap.tab` — TAB v6.2 oficial (Hertel et al, 2003)
- `C:\runGTAP375\decomp.tab` — welfare decomposition module v6.2
- `C:\runGTAP375\BOOK3X3\` — dataset 3×3 con experimentos GEMPACK
- `C:\runGTAP375\NUS333\` — dataset NUS333 con SAM v6.2
- `src/equilibria/templates/gtap/` — implementación v7 actual (referencia para refactor)

## Próximos pasos

1. Revisión y green-light del plan (este documento)
2. Crear branch `gtap/v62-rollback`
3. Ejecutar Fase 1
