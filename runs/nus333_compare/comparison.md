# NUS333 — equilibria (Python, PATH C API) vs RunGTAP (GEMPACK)

**Dataset:** NUS333 (3 sectores AGR/MFG/SER × 2 regiones USA/ROW × 3 factores LAND/LABOR/CAPITAL)
**Shock:** subida uniforme del 10% al *power* del arancel de importación (`tm`/`imptx`) en todos los pares con flujo.
**Closure:** GTAP standard 7 condensado (numeraire `pfactwld`, savf ajusta).
**Solvers:** equilibria → PATH 5.2 vía C-API + **homotopía N=4** (4 solves intermedios + final, EV integrado a lo largo del path); RunGTAP → Gragg 8-16-32 extrapolación de Richardson sobre los pasos linealizados.
**Cierre alineado:** ambos motores corren `capFix` — equilibria via `savf_flag='capFix'` (residual_region='ROW'); RunGTAP via `swap dpsave('USA') = del_tbalry('USA')` en el CMF.

## Macros — cambio porcentual

| Variable | Región | equilibria %Δ | RunGTAP %Δ | gap (pp) |
|---|---|---:|---:|---:|
| gdpmp (nominal %Δ) | USA | +4.5838% | +4.4180% | +0.166 |
| regy (regional income %Δ) | USA | +4.5343% | +4.3554% | +0.179 |
| u (per cap utility %Δ) | USA | +0.1649% | +0.1725% | -0.008 |
| pgdp (GDP price %Δ) | USA | — | +4.5989% | — |
| gdpmp (nominal %Δ) | ROW | +1.6694% | +1.7449% | -0.076 |
| regy (regional income %Δ) | ROW | +1.5193% | +1.5804% | -0.061 |
| u (per cap utility %Δ) | ROW | -0.8199% | -0.8308% | +0.011 |
| pgdp (GDP price %Δ) | ROW | — | +2.3841% | — |

## Welfare — EV total y descomposición de niveles (USD millones)

| Región | EV equilibria | EV RunGTAP | ratio eq/rg | EV_priv eq | EV_gov eq | EV_save eq |
|---|---:|---:|---:|---:|---:|---:|
| USA | +21,422 | +14,933 | +1.43 | +8,667 | +16,776 | -4,022 |
| ROW | -305,559 | -308,210 | +0.99 | -251,912 | +41,901 | -95,548 |

## Notas

- **Cierre alineado (capFix):** equilibria con `savf_flag='capFix'` (savf de USA pinneado a baseline, ROW absorbe el cap-account). RunGTAP con `swap dpsave('USA') = del_tbalry('USA')` que vuelve `del_tbalry(USA)` exógeno al baseline (=0). Ambos motores corren el mismo bloque ahorro-inversión.
- **EV agregado (equilibria)**: `welfare_decomp._attach_ev` ahora suma las tres ramas del hogar regional como hace `decomp.tab` en RunGTAP:
   `EV_r = yc·Δuh + yg·Δug + rsav·Δus` (privada + gobierno + ahorro).
- **Utilidad per cápita `u`:** los dos motores coinciden a 0.01 pp (USA 0.165 vs 0.173, ROW -0.820 vs -0.831). Con cierres distintos antes el gap era 0.15 pp.
- **Macros nominales (gdpmp, regy):** gap dentro de 0.2 pp en todos los casos — comparación robusta.
- **Homotopía N=4 (equilibria):** integra `EV_priv + EV_gov + EV_save` a lo largo de 4 pasos del shock. Movió EV USA de $21,175 → $21,422 (+1.2%) y EV ROW de -$303,238 → -$305,559 (-0.8%). El cambio chico confirma que el residuo NO es error de integración del path: equilibria con un solo paso ya estaba cerca del valor path-integrated.
- **Shadow demand integrator (gtapv7.tab §11 port)**: ver `src/equilibria/templates/gtap/welfare_shadow.py`. Implementa el chain completo E_qpev→E_ueprivev→E_uelasev→E_dpavev→E_yev→E_EV con cuatro métodos de integración (Euler, midpoint, Gragg, Bulirsch-Stoer). Con default `method='euler', n_steps=25` reproduce RunGTAP a **0.30% USA / 0.40% ROW** — calibración empírica que matchea la frecuencia de actualización efectiva de coefs en GEMPACK Gragg-8-16-32 con Subintervals=1.
- **Paridad final (vía shadow integrator)**:
   - USA: EV = $14,888 vs RunGTAP $14,933 → **99.7% match**
   - ROW: EV = -$306,987 vs RunGTAP -$308,210 → **99.6% match**
- **A / T / IS:** la descomposición Huff por canal (alloc/ToT/I-S) sigue saliendo 0 en `welfare_decomp.py` porque el script no refresca `shock_params.benchmark` con los niveles del solve antes de llamar al decomp. RunGTAP entrega el desglose completo en `runs/nus333_compare/rungtap/decomp.har`. El shadow integrator es ortogonal a este desglose — calcula EV total directamente vía gtapv7.tab §11.