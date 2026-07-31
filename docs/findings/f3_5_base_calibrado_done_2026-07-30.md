# F3.5 — Base-calibrado sin check: adopt the settled point as base (2026-07-30)

**Status:** Done. A selectable `base_calibrated` mode on the GTAP block model pre-settles
the specific-factor prices (Land/NatRes) at calibration time, so the shock runs
`base→shock` (no `check`) and the land-price response matches GEMPACK (−3.0% vs −2.68%)
instead of the check-contaminated GAMS path (−18%). The faithful-to-GAMS default
(`base→check→shock`) is untouched (parity gates green, 0-diff vs GAMS).

Branch `f3.5-base-calib` (from `main` `e681d26`, after F3 merged).

## The problem

GTAP normalizes base prices to `1.0`, but for the **specific factors** with a sloped
supply nest (Land/NatRes, CET `eq_pfteq`, ω=`etrae`) that `1.0` is a normalization
artifact — not the model-consistent price. GAMS (and equilibria, faithful to it) settles
it in a `check` period: a no-shock solve that lets the land price fall to its equilibrium.
On `gtap7_3x3`, `pft[EU_28,Land]` slides `1.0 → 0.845` (−15.5%) in the check, then the
shock moves it a further −3%. GEMPACK/Julia/CGEBox instead pre-calibrate the base
consistent, so their land price only responds to the shock (GEMPACK: −2.68%).

Against GEMPACK, the −15.5% check re-settlement reads as a ~15pp discrepancy that is
really a *methodology* difference between engines, not a bug (confirmed in F3 by reading
5 implementations — see `memory: project_gempack_shock_sweep_check_gap`).

## The spike that redefined the mechanism

The design's first mechanism (re-derive `gf_share` to freeze the settled price) was
**refuted by a spike before implementing it**:

- In gtap-mode `etaf=0` for all factors incl. Land (`_aft_etaf`: "GAMS uses etaf=0 for ALL
  fm incl sluggish Land") → the specific-factor supply curve is **vertical**.
- Land in EU_28 feeds a **single** sector (Food) → `gf_share≈1`; re-deriving it is a no-op.
- The settled `pft` (0.845) is a **whole-system result**, not a re-derivable parameter.

Verified numerically on the reference GDX:

| `pft[EU_28,Land]` response | value |
|---|---|
| shock vs RAW base (1.0) — the check-contaminated GAMS path | **−18.09%** |
| shock vs SETTLED base (=check 0.845) | **−3.03%** |
| GEMPACK `pfe[Land,Food,EU_28]` (sl4dump) | **−2.681%** |

−3.03% ≈ −2.68%. **Correct mechanism: adopt the settled (check) point as the base.**

## Design

`FactorBlock.calibrate_base()` runs the settle solve ONCE (the `base→check→shock` stack
reused as a calibration tool) and returns the settled check-period point. The composer's
`build_block_model(base_calibrated=True)` runs it and stamps `m._settled_seed` +
`m._base_calibrated`. The driver, when `_base_calibrated`, seeds the base period from the
settled point, **skips the check phase**, and seeds the shock from base. The check
*calculation* is used once at calibration time, outside the simulation — it is not
re-introduced into the sim path.

The settled point is a given base (already an equilibrium from the settle solve), so the
calibrated mode uses `skip_base_solve=True` — like the faithful-to-GAMS base.

## Results (gtap7_3x3)

- **Non-regression (gate #1):** `run_parity_gates.py` GREEN — the default
  `base→check→shock` is 0-diff vs GAMS, untouched. Form gate 14/14.
- **Against GEMPACK (gate #2):** base-calibrated `pft[EU_28,Land]` base=0.845 →
  shock=0.819 = **−3.03%**, all periods `code=1`, no check phase. GEMPACK: −2.68%.
  `measure_gempack_blocks.py` reports `gap_vs_gempack_pp = −0.352`.
- **Against GEMPACK — the WHOLE quantity set (not just the land price):** across all
  15 mapped `Q_TO_VAR` variables (190 cells), the overall match within 1pp rises
  **75.8% (default) → 96.3% (base-calibrated)**, median error 0.38pp → 0.17pp — the
  same figure F3 measured "excluding check-movers". The Food/Land-chain quantities
  (`xp`/`xda`/`xaa`/`xma`/`xd`) go to 100%.

### Don't re-base the derived-demand vars

An early version seeded ALL 83 settled var families into the base; that regressed
`xg`/`xc` (government/household demand) by ~1.3pp on the EU_28 cells (78% match).
Those are DERIVED-demand vars (computed from prices+incomes), so seeding them at the
check value is a small inconsistency. Skipping them from the base seed
(`_F35_DERIVED_DEMAND` = xg/xc/xg_agg/xi/xiagg/yg/yc/zcons/u*/ev/cv) lets them
re-derive consistently → `xg`/`xc` 78%→100%, overall 94%→96.3%. The equilibrium core
(prices, factor/production/trade quantities) IS seeded. This is exactly the "seed
only the equilibrium vars" narrowing flagged as a risk in the plan.

### Cross-check against Julia (the same-methodology engine)

Julia (mivanic/GlobalTradeAnalysisProjectModelV7.jl) uses the SAME base-calibrated
methodology as F3.5 (an explicit `calibrate()` → `calibrated_data` → single shock
solve, no check period) — so it is the natural oracle for the mechanism (it uses the
same exact power-CET as GAMS/equilibria, and calibrates the base like F3.5). Running Julia's own sample
data end-to-end (`calibrate → +10% tariff → run_model!`, `scratchpad/
julia_mechanism_probe.jl`, Julia instantiated + Ipopt solve `Optimal Solution
Found`): the **largest specific-factor price move calibrated-base→shock is 5.36%**
(single-digit) — Julia's calibrated base does NOT re-settle the specific factor the
−15% way. This confirms F3.5's mechanism with Julia's engine executing (not just
reading its code): calibrate the base → the specific-factor price responds only to
the shock. F3.5's −3.03% on gtap7_3x3 is the same behavior.

**The decisive test — Julia's calibrated base does NOT re-settle:** run Julia
`calibrate → rebuild → solve with NO shock`. Result: land price move
calibrated-base → no-shock solve = **0.0% (max and median)**. Julia's calibrated
base has zero re-settlement, exactly like F3.5's settled base:

| Engine | land price, solve with NO shock |
|---|---|
| GAMS / equilibria default (`base→check→shock`) | **−15.5%** (the check re-settlement) |
| Julia (calibrated base) | **0.0%** (no re-settlement) |
| equilibria F3.5 (base-calibrated) | **0.0%** (settled base — no re-settlement) |

The shock-period land move (Julia gives 7–17% for a large global +10%-all-routes
tariff; F3.5 gives −3% for the single tm10 fixture — different shocks, different
magnitudes) is genuine SHOCK RESPONSE, not re-settlement. F3.5 replicates Julia's
methodology exactly: base-calibration removes the −15% re-settlement, leaving only
the shock response.

**Cell-by-cell port status:** porting our gtap7_3x3 HAR into Julia reached ~85%
(all 46 headers load; `generate_initial_model` + `generate_calibration_inputs`
converge; sets/dims/factor-mobility/factor-names aligned; etrae 0→−1e-5 fix).
Blocked by a deterministic overflow (dual infeas 1.9e25) inside Julia's calibration
`run_model!` (an internal CES-share phase, needing Julia-side instrumentation to
bisect). The no-shock test above supersedes it: it confirms the mechanism more
cleanly than a cross-dataset cell compare would. Harness (export + loader +
diagnostics) is in scratchpad for a future exact-number pass.

### The residual against GEMPACK — what it is NOT (three hypotheses tested + refuted)

The land price is ours −3.033% vs GEMPACK −2.681% (0.35pp). Three natural explanations
were each tested against the real sources and **refuted**:

**(1) NOT linearization / sub-steps.** GEMPACK's land price is flat at −2.681% across
`sl4dump_gtap7_3x3_tm10_s{4,8,16,32,64}` (all mobile factor prices flat too, e.g.
UnSkLab −3.422% s4→s64). GEMPACK is fully converged; more steps do not move it toward
ours. Structural, not numerical.

**(2) NOT the specific-factor CET aggregation.** Read from both sources: GEMPACK
`E_pe2` is `pe=Σ_a REVSHR·pes` (income-share-weighted, `REVSHR=EVOS/VES`); ours
`eq_pfteq` is `pft=[Σ_a gf·pfy^(1+ω)]^(1/(1+ω))`. Linearizing our power-CET gives
weight `w_a = θ_a·p_a^κ / Σ θ_b·p_b^κ`, which **is** the CET income share — i.e. GEMPACK's
`E_pe2` IS the first-order linearization of our `eq_pfteq`, same `ETRAE`, same shares.
BUT: Land in EU_28 is used in a **single** sector (Food), so `REVSHR=[1,0,0]` and our
`w=[1,0,0]` — both collapse to `pft = pes_Food` **identically**, exponent κ irrelevant.
So the aggregation formula CANNOT be the source for single-sector Land.

**(3) NOT the numeraire.** GEMPACK's numeraire is `pfactwld` (world primary-factor
price index, held at 0%); ours is `pnum`. But our own world factor price index moves
only −0.0004% under the shock (≈GEMPACK's 0%), so re-normalizing our prices to GEMPACK's
numeraire shifts them by <0.001pp — the gaps are unchanged.

### What the residual actually is — the full gap map (traced)

Measured the median |gap| vs GEMPACK for every mapped quantity variable (190 cells,
`Q_TO_VAR`). The gap is **NOT uniform** — it has a clear structure:

| Group | Vars (median gap) |
|---|---|
| ~zero | `qe` factor supply 0.000, `qgdp` real GDP 0.004, `qtm` margins 0.077 |
| low (~0.1pp) | `qfa` Armington 0.11, `qva` VA 0.11, `qfd` 0.09, `qc` supply 0.11 |
| medium (~0.2pp) | `qds` dom sales 0.25, `qxs` exports 0.25, `qms` imports 0.06 |
| **high (~0.45pp)** | **`qpa` private demand 0.43, `qga` gov demand 0.47, `qxw` export-agg 0.43** |
| **outlier** | **`qinv` investment 1.24** |

Two decisive facts: (1) **real GDP matches to 0.004pp and factor quantity to 0.000pp**
— the model's anchor quantities agree, which REFUTES a uniform "systemic method
residual" (that would move GDP too). (2) The gap **concentrates on FINAL DEMAND** —
private (`qpa`), government (`qga`), and especially investment/savings (`qinv` 1.24pp,
`qsave` swings USA +2.7%, EU_28 −10%, ROW −1.2%) — while production, factors, trade,
and GDP agree well (~0–0.1pp).

**The locus:** GEMPACK's private demand is **CDE** (Constant Difference of Elasticities,
a non-homothetic minimum-expenditure function with `INCPAR`/`SUBPAR`, `gtapv7.tab`
910–1621), and the savings/investment allocation. The CDE is strongly non-linear, so
levels-exact (ours/GAMS) vs linearized-multistep (GEMPACK) differs most there — exactly
where the gap lives. Production/factor/trade nests (Leontief, CES, Armington) agree to
~0.1pp; the specific-factor equation is NOT the source.

**Corrections of earlier claims (both wrong, found by pursuing the user's skepticism):**
- NOT "exact power-CET vs GEMPACK linearized for the specific factor": single-sector Land
  makes both aggregations identical (`pft=pes_Food`), and the gap is on ALL factors.
- NOT a uniform systemic method residual: real GDP and factor quantities match to
  <0.01pp. The gap is **concentrated in final demand (CDE + savings/investment)**, a
  genuinely non-linear nest where the two engines' solution methods diverge — consistent
  with F5's "structural GAMS↔GEMPACK difference", now pinned to the demand side.

Ruled out with evidence along the way: linearization/steps (GEMPACK flat s4→s64),
CET aggregation, numeraire (world factor index moves −0.0004%), technical shifters
(afe/ava=0), output tax (to=0), top-nest elasticity (ESUBT=sigmap=0, Leontief).

### The exact CDE difference (root cause, verified against both sources)

Read both demand equations from source. GEMPACK `E_qpa` (`gtapv7.tab`): `qpa(c)-pop =
Σ_k EP(c,k)·ppa(k) + EY(c)·[yp-pop]`, where the elasticities `EP`/`EY` are `Formula`
coefficients computed **once** from `ALPHA=1-SUBPAR`, `INCPAR`, and the **base**
consumption shares `CONSHR` — and held fixed. Ours `eq_zcons` is the CDE in **levels**:
`zcons(i) = α·bh·pa^bh·uh^(eh·bh)·(yc/pop)^(−bh)`, with the consumption shares `xcshr`
an **endogenous variable** re-solved under the shock.

Decisive test: our `xcshr[EU_28,·]` MOVES under the +10% shock — Food +0.99%, Mnfcs
+1.76%, Svces −0.61%. So our CDE re-evaluates the effective demand elasticities at the
post-shock shares (level-exact, non-homothetic), while GEMPACK evaluates its `EP`/`EY`
at the base shares and freezes them. This is a genuine structural difference (not just
levels-vs-linearized), and it explains all three observations: (1) the gap concentrates
on final demand (`qpa`/`qga`/`qinv`) because the CDE is the strongly non-homothetic
nest where shares move most; (2) it does NOT shrink with GEMPACK sub-steps because
`EP`/`EY` are computed once, not re-evaluated per step; (3) production/factor/trade
nests agree (~0.1pp) because they are homothetic or move little. **Tested "do it like GEMPACK = freeze the shares":** froze our `xcshr` at its base value
for the shock (deactivated `eq_xcshr[shock]`, pinned via a patched
`freeze_inactive_periods`) — shares verified flat (Δ=0.0000%). Result: qpa gap vs GEMPACK
went **0.429pp → 0.652pp — WORSE, not better.** So freezing the shares does NOT reproduce
GEMPACK. Why: GEMPACK doesn't just "use base shares" — it uses a *consistent* first-order
linearization where the elasticities `EP`/`EY` are derived analytically from the CDE at
the base. Freezing our shares while keeping `eq_zcons` in exact levels makes an
inconsistent hybrid (a third point, worse than either). Reproducing GEMPACK would require
replacing our levels `eq_zcons` with GEMPACK's linearized `E_qpa` outright — i.e. changing
the solution method to GEMPACK's, which breaks `0-diff vs GAMS`. The against-GEMPACK
residual is a genuine levels-exact (GAMS/us/Julia) vs linearized-CDE (GEMPACK) difference
concentrated in final demand — irreducible without abandoning fidelity to GAMS.

**Tested "use base-fixed elasticities like GEMPACK" (the linearized demand equation):**
computed GEMPACK's own `EP`/`EY` from `ALPHA`/`INCPAR`/base `CONSHR`, and applied its
linearized `qpa=Σ EP·ppa+EY·ŷ`. Two facts: (a) fed GEMPACK's OWN prices, the formula
reproduces GEMPACK's qpa (Food −1.19% vs −1.14% actual) — so the elasticities are right;
(b) fed OUR prices, it gives ~our qpa — median gap vs GEMPACK 0.429pp → 0.423pp,
**essentially unchanged.** So swapping the demand equation does NOT close the gap either.

**Final conclusion (by total elimination):** the gap is NOT the demand equation — it is
inherited from the PRICES, which already differ. The consumption-price gap (ppa/pa) is
0.049pp median across all cells; the demand equation merely inherits it. Every
single-equation hypothesis was tested and refuted (specific-factor CET, numeraire,
shifters, output tax, top-nest elasticity, demand shares, linearized demand).

**Precise formulation — it is a documented GAMS↔GEMPACK IMPLEMENTATION difference, not
multi-equilibrium and not our bug.** The GTAP system is uniquely solvable (PATH=IPOPT to
10 digits). Key facts, checked: (a) our GAMS reference itself gives `pft[EU_28,Land]`
shock/check = **−3.03%** — i.e. we reproduce GAMS exactly; (b) GEMPACK's extrapolated
(Richardson) solution is −2.68% and equal to its raw steps s8/s16/s32/s64 (verified), so
it is GEMPACK's accurate answer, not an unconverged step. So the −3.03% vs −2.68% is a
**GAMS-vs-GEMPACK difference**, not equilibria-vs-GEMPACK.

The literature resolves it. GEMPACK-for-GAMS-users material states GAMS and GEMPACK "give
the same numerical solution", but the authoritative GTAP-in-GAMS papers (van der
Mensbrugghe, *The Standard GTAP Model in GAMS*, JGEA; GTAP TP/19/01) state there are
**"substantive differences between the GEMPACK and GAMS implementations"** and that an
*exact* replication of the standard GTAP model in GAMS was an ongoing effort (van der
Mensbrugghe 2016). The specific-factor / sluggish-endowment pricing is one such
implementation difference. So: the two are the same model *conceptually* but the GAMS
and GEMPACK *implementations* differ substantively in places — and the specific-factor
price is one. We faithfully port the GAMS implementation (−3.03%); GAMS differs from
GEMPACK there by 0.35pp; that is a known engine-implementation difference, not our defect
and not multi-equilibrium. (Earlier drafts of this section were wrong twice — "exact CET
vs linearized" and "two different models / systemic" — corrected by the user's insistence
and by checking the literature + our own GAMS reference directly.)

Sources: van der Mensbrugghe, *The Standard GTAP Model in GAMS, Version 7*
(jgea.org/ojs/index.php/jgea/article/download/62/61/394); GTAP7Gams TP/19/01
(mygeohub.org/groups/gtap/.../GTAP7Gams.pdf); Kohlhaas & Pearson, *Introduction to
GEMPACK for GAMS Users* (copsmodels.com/ftp/gamsgp.pdf).

### Exactitude — verified against van der Mensbrugghe's actual GAMS source

The goal is exactness. Read the specific-factor equation directly from van der
Mensbrugghe's GAMS (`gams_code_20121206/gtap_model.gms`, `ENDW_PRICE` HT50):

    pm(es,r) = ( Σ_j REVSHR(es,j,r)·pmes(es,j,r)^(1-ETRAE) )^(1/(1-ETRAE))

This is **exactly our `eq_pfteq`** (`pft = (Σ gf_share·pfy^(1+ω))^(1/(1+ω))`, ω=etrae;
our `gf_share` = his `REVSHR`; `1+ω` = `1-ETRAE`). We are byte-faithful to the official
GAMS specific-factor equation. And the official GAMS supplementary files
(`gft.gms`, `GFTLnd.gms`, …) all use `t / base, check, shock /` with `loop(tsim)` — the
check period and its −15% land re-settlement ARE the standard official behavior.

So there are **two distinct exactness targets, and they cannot both hold**, because
GAMS and GEMPACK are each exact to a DIFFERENT specific-factor equation:
- **Exact to GAMS** (van der Mensbrugghe — the model we port): **achieved, 100%.** Our
  equation is his equation; our check re-settlement is his; we reproduce −3.03%.
- **Exact to GEMPACK** (−2.68%): requires (a) calibrating the base with no check —
  **done, that is F3.5** — PLUS (b) replacing the exact power-CET with GEMPACK's
  *linearized* `E_pe2`, which is a different equation. van der Mensbrugghe's own "exact
  replication of the standard GTAP model in GAMS" was still *underway* (2016) precisely
  because of such differences; the supplementary v7 files we port still use the power-CET.

Conclusion for exactness: we are 100% exact to the GAMS reference (the authoritative
levels model). The 4% against GEMPACK is the residual GAMS↔GEMPACK equation difference
on the specific factor — closing it means adopting GEMPACK's linearized equation, which
would make us no longer exact to GAMS. F3.5 closes the *calibration-methodology* half of
the against-GEMPACK gap (−18%→−3%); the remaining ~0.35pp is the equation-form half, an
open item in van der Mensbrugghe's own GAMS↔GEMPACK reconciliation, not a defect here.

### Is the −3.03% a bug in our port, or does GAMS itself give it? — GAMS-native, proven

The decisive test for "can it be closed": is −3.03% our Python reproduction, or what GAMS
itself gives? Our reference `out_gtap_shock_ifsub1.gdx` is a **GAMS-native local solve**
(`build_gtap7_pure_local_bundle.py`, run with GAMS locally — not our Python). `gdxdump` on
it directly:

    pft[EU_28,Land]: base=1  check=0.844755974281121  shock=0.819131926318381

→ check→shock = **−3.03%**, straight from GAMS. We reproduce it to 6 digits. So −3.03% is
**what van der Mensbrugghe's GAMS gives**, and −2.68% is what GEMPACK gives — two reference
engines, two numbers, for the same dataset + shock, **verified directly (not inferred)**.
This settles "bug vs engine-difference": it is NOT a bug in our port (GAMS-native = ours),
it is a real GAMS≠GEMPACK difference. And it settles "can it be closed maintaining
fidelity": the correct number *per GAMS* IS −3.03%; there is no third "more correct" value
to converge to — closing to −2.68% means no longer reproducing GAMS, i.e. abandoning the
reference we port. Every fidelity-preserving route was tried and none gives −2.68% (raw
base −18%, check base −3.03%, frozen shares worse, linearized demand unchanged, linearized
factor eq identical). "Irreducible without adopting GEMPACK's equations" is therefore a
proven fact, not an excuse.

### Second-opinion (fable) + reading van der Mensbrugghe v7.1 directly — the real answer

A fresh-model review (fable) rightly attacked the "irreducible" framing and produced two
corrections that hold up:

1. **Converged GEMPACK is NOT linearized.** Richardson-extrapolated multi-step GEMPACK
   (Gragg, coefficients updated between steps) converges to the EXACT levels solution of
   the TABLO model — CONSHR/EP/EY/REVSHR are re-evaluated each step. So the earlier
   "levels-exact vs linearized" framing is wrong for a *converged* gap; my mimicry
   experiments failed because they tested the solution METHOD, not the model.
2. **Small-shock scaling test is decisive.** Ran both engines' land response per unit of
   shock at 0.1% / 1% / 10%:

   | shock | ours per-unit | GEMPACK per-unit |
   |---|---|---|
   | 0.1% | −0.3288 | −0.3075 |
   | 1% | −0.3261 | −0.3034 |
   | 10% | −0.3033 | −0.2681 |

   The ~0.02 per-unit gap **persists at shock→0** (and is the same from raw vs check base).
   So it is NOT large-shock curvature and NOT the base point — it is a difference in the
   **local derivatives (Jacobian) at the base**, upstream of the (passthrough) factor
   aggregation. That is a real, located difference, not a mystery.

**Read v7.1 (van der Mensbrugghe 2019, TP #92, GTAP7Gams.pdf) directly.** It names the
exact difference — Equation **(F-1)** for aggregate factor supply `XFT = Aft·(PFT/PABS)^ηft`:
> "There is **no equivalent in the GEMPACK code, where the aggregate supply of all factors
> is exogenous**. Setting the supply elasticity (ηft) to zero would have the same impact as
> exogenizing total supply."

So the GAMS↔GEMPACK factor-supply difference is documented by the model's own author. Two
further v7.1 facts settle the "is it fixed in a newer version" question:
- v7.1 **still has** `/ base, check, shock /` with the check period (paper line 503/518) —
  normalization did NOT remove the specific-factor re-settlement.
- v7.1's normalization "**brings the GAMS model somewhat closer to the GEMPACK version**"
  and improves numerical convergence — explicitly "somewhat closer", NOT identical.

**Net:** it is a documented GAMS↔GEMPACK implementation difference in specific-factor
supply (F-1/F-2), persisting through v7.1, with the local-derivative signature confirmed
by the small-shock test. Not our bug (GAMS-native = ours), not fixed by v7.1. Our `ηft`
(etaf) is already 0 (aggregate supply exogenous, like GEMPACK), so the residual is the F-2
CET-vs-linearized sluggish distribution — which for single-sector Land is passthrough, so
the ~0.02 derivative gap propagates from the upstream VA/Armington nests' local slopes.
**Closing it to GEMPACK's number requires GEMPACK's equation forms — still an open item in
vdM's own GAMS↔GEMPACK reconciliation.** This is now a *characterized* difference (F-1/F-2,
derivative-level), not "irreducible/mysterious".

### Would a "GEMPACK version" of the specific-factor equation close it? No — traced.

Considered adding a selectable `specific_factor_form = 'vdm' | 'gempack'` block flag.
Measured first (the right call). Two facts kill it: (1) for single-sector Land the CET
aggregation is a **passthrough** — `pft`(agg)=`pfy`(sectoral)=−3.03% for us, and
`pe`=`pes`=−2.68% for GEMPACK; the specific-factor equation changes nothing, so a
linearized version of it would not move Land. (2) Tracing the chain that sets the factor
price, the gap is **distributed upstream**, not in the factor nest:

| chain link (EU_28/Food) | ours | GEMPACK | gap |
|---|---|---|---|
| output price (px/po) | +0.468 | +0.893 | −0.425pp |
| intermediates (pnd/pint) | +2.724 | +2.957 | −0.234pp |
| VA (pva) | −3.392 | −3.324 | −0.068pp |
| factor Land (pf/pfe) | −3.033 | −2.681 | −0.352pp |

The gap is already large at the OUTPUT price and the INTERMEDIATES (Armington), which
feed the factor via zero-profit. No single upstream nest dominates — it is spread across
output/intermediate/factor. So a "GEMPACK version" of one equation closes nothing; a
full GEMPACK version = linearizing the entire system = rebuilding GEMPACK in Python
(which already exists). We keep the one faithful-to-GAMS specific-factor equation (van
der Mensbrugghe's exact `ENDW_PRICE`), plus the F3.5 base-calibrated calibration mode —
that is the correct, bounded deliverable; a second full linearized model is out of scope.

## What shipped

- `src/equilibria/blocks/gtap/factor.py` — `FactorBlock.calibrate_base()`
- `src/equilibria/templates/gtap/gtap_block_model.py` — `build_block_model(base_calibrated=)`
- `src/equilibria/templates/gtap/gtap_multiperiod_driver.py` — seed base from settled + skip check
- `scripts/gtap/measure_gempack_blocks.py` — against-GEMPACK land-response measurement
- `tests/templates/gtap/test_f3_5_base_calib.py` — 5 tests (spike oracle + calibrate + composer + end-to-end + against-GEMPACK)

## Reproduction

```bash
# the against-GEMPACK measurement (needs PATH + the committed sl4dump)
uv run python scripts/gtap/measure_gempack_blocks.py --dataset gtap7_3x3 \
    --base-calibrated --gempack-har tests/fixtures/gtap7_gempack/sl4dump_gtap7_3x3_tm10.har
# → land_resp_pct -3.03, gempack_land_pct -2.681, gap_vs_gempack_pp -0.352

# the F3.5 tests
uv run python -m pytest tests/templates/gtap/test_f3_5_base_calib.py -v

# non-regression (default 0-diff vs GAMS)
uv run python scripts/gtap/run_parity_gates.py
```

## Scope / YAGNI

- Only the specific factors are calibrated (the 97 base→check movers all cascade from one
  root: the land price). No Julia-style 6-phase calibration of Armington/production/demand.
- `gtap7_3x3` only for now; extend to other datasets in follow-up.
- The default mode is never changed — F3.5 is a selectable variant.
