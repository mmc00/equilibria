# Plan para cerrar el gap del shock gtap7_3x3 (78.69% → 100%)

**Estado:** baseline honesto 78.69% (code=1), `--holdfix-pva` 97.68% (validación, hardcoded). HEAD `51b390f`.

---

## Lo que se VERIFICÓ (no asumir — esto reemplaza diagnósticos viejos)

1. **El modelo Python ES multi-período secuencial** (`diff_altertax` [1/3] base → [2/3] check → [3/3] shock, encadenado vía `t0_snapshot`). NO es single-period; NO resuelve de un golpe. Lo único single es la representación interna de cada slice (`pva[r,a]`, sin índice `t`). Ver `feedback_gtap_IS_multiperiod`.

2. **Los índices Fisher inter-temporales YA están bien.** `eq_pfact/eq_pwfact/eq_pabs/eq_rgdpmp` usan `pf0/xf0/pa0` (params del `t0_snapshot` = base), que es exactamente lo que GAMS compStat hace (`sum(t0, ...)`). Verificado: residuales 1e-12 en el punto GAMS. **NO tocar.**

3. **El "holdfix de 25 variables" de GAMS NO es replicable ni necesario tal cual.** GAMS resuelve los 3 períodos en UN modelo MCP; `var.fx(tsim-1)` saca las columnas del período previo para que PATH no las mueva. **Python ya resuelve cada período en un modelo separado** → esas columnas no existen en el modelo del período actual. Confirmado: ninguna ecuación de demanda/oferta del período t lee `pf(t-1)/xf(t-1)/pa(t-1)`; solo los Fisher leen el período previo (vía pf0/xf0, ya implementado).

4. **El "fix" de congelar pf-base en el check (pva=0.854, 97%) era un ANCLA ESPURIA.** GAMS tiene `pf[check]=1.368 ≠ pf[base]=1.328`. Congelar la única `pf` de Python al valor base fuerza `pf[check]=pf[base]`, que es INCORRECTO — daba pva correcto por coincidencia estructural, no por fidelidad. **NO es el camino.** (code=None confirma que rompe el cuadrado.)

5. **El sistema MCP de Python está exactamente cuadrado** (962 eq = 962 var, DOF=0). No falta ni sobra ninguna ecuación.

6. **El punto de GAMS NO satisface las ecuaciones de Python al resolver:** sembrando la respuesta exacta de GAMS y resolviendo, Python se MUEVE (75.7%, 527 vars driftan). Pero las ecuaciones que viola son agregados Python-only (xc/xd_agg/xaa — no existen en GAMS) que cierran perfecto cuando se derivan de sus componentes (eq_xc verificada: diff=0 con valores GAMS). Las de comportamiento (precios/demanda/oferta) NO las viola.

---

## El diagnóstico real (consolidado de toda la investigación)

El gap es **selección de raíz en un sistema con un trade-off de nivel pwfact↔pva**, no un bug de ecuación ni DOF faltante:

- `eq_pwfact` está en forma **cuadrática** (raíz espuria) en vez de la **sqrt** de GAMS. Forma cuadrática → 78.69% (pva correcto por compensación, pwfact mal). Forma sqrt → 60.54% (pwfact correcto, pva mal). Es un trade-off de UN nivel que se reparte pwfact↔pva.
- Bajo CD (sigmav=1), `eq_pvaeq` es tautología → pva sin fuerza restauradora local → PATH lo coloca según el basin del warm-start.
- El nivel patina entre regiones (USA/EU vs ROW residual), ligado al balance externo (chif) y al mercado de factores (el clearing `eq_xft` está desactivado por una suposición falsa — verificado: el mercado NO cierra por región).

---

## Plan de cambio (en orden, cada paso verificable)

### Paso 1 — Arreglar `eq_pwfact` a la forma sqrt de GAMS (fidelidad real)
- **Qué:** reemplazar `pwfact²·mqfactw_bb·m_bs == m_sb·m_ss` por `pwfact == sqrt[(m_sb/mqfactw_bb)·(m_ss/m_bs)]` (gtap_model_equations.py ~5781).
- **Por qué:** es la forma compStat de GAMS (linear-sqrt, raíz única), misma clase que los fixes ya hechos en eq_pfact/eq_pabs/eq_rgdpmp. El residual cuadrático actual es ~97 en el punto GAMS (mal condicionado).
- **Riesgo:** baja el match seeded a 60% AISLADAMENTE (porque destapa el problema de pva). NO commitear solo; es prerequisito de los pasos 2-3.
- **Verificar:** `.nl` gate verde; pwfact=1.0 exacto en el punto GAMS.

### Paso 2 — Arreglar el clearing del mercado de factores (`eq_xft`)
- **Qué:** en `run_gtap.py` ~2012, en vez de desactivar `eq_xft` (el clearing `xft=Σxf/xscale`), mantenerlo activo y fijar `xft=aft` + desactivar la trivial `eq_xfteq` (etaf=0). Mismo conteo DOF (15 eq off + 15 var fix), pero ancla pft por región.
- **Por qué:** verificado que sin el clearing, el mercado de factores NO cierra (USA/EU ociosos, ROW de más) → pft regional patina. El comentario actual ("clearing implícito") es FALSO.
- **Verificar:** `xft == Σxf/xscale` por región tras el solve (cleared=True).

### Paso 3 — Anclar pva per-actividad (el DOF libre bajo CD)
- **Qué:** el nudo. Con pwfact (paso 1) y mercado de factores (paso 2) anclados, re-evaluar si pva todavía patina. Si sí, dar a pva una fuerza restauradora que NO sobre-determine. Opciones a probar EN ORDEN:
  - (a) σ=1−ε localizado en eq_pvaeq SOLO para actividades que ligan (ya probado: empuja mal, pero re-evaluar CON pasos 1-2).
  - (b) chif.fx(not rres) (verificado: ayuda el check, rompe el seeded shock — re-evaluar con pwfact-sqrt).
  - (c) Si nada local funciona, el ancla de pva es genuinamente la estructura de selección de raíz → requiere homotopía (paso 4).

### Paso 4 (si 1-3 no cierran) — Homotopía de la rampa con re-seed
- **Qué:** rampa imptx en N pasos + re-seed `pva=(px/pnd^and)^(1/ava)` entre solves (GAMS cleanup loop). Infra parcial en `homotopy_shock.py`.
- **Riesgo:** ya probado frágil; requiere los pasos 1-2 primero para que el check converja a la rama correcta.

---

## Cómo verificar el progreso (sin auto-engaño)

- **`.nl` gate** (`pytest tests/templates/gtap/test_gtap7_nl_parity.py`) tras CADA cambio de ecuación.
- **Match del shock** via `diff_altertax --dataset gtap7_3x3 --gdx <ifsub0 ref>`.
- **drift_test.py** (tool #7): tras cada paso, ver qué variable patina ahora (el DOF se mueve; rastrearlo).
- **diff_holdfixed.py** (tool #8): confirma el cierre de secuencia.
- Regla: un cambio se queda SOLO si sube el match del shock O es prerequisito verificable de uno que sí. NUNCA excluir celdas. GAMS es la fuente de verdad.

## Lo que NO hacer (errores ya cometidos esta sesión)
- NO decir "single-period" / "resuelve de un golpe" (es multi-período secuencial).
- NO congelar `pf` al valor base (ancla espuria, pisa pf[check]≠pf[base]).
- NO concluir "multiplicidad, nada que hacer" sin correr el drift test.
- NO presentar `--holdfix-pva` (97.68%) como solve fiel — es validación hardcoded.
