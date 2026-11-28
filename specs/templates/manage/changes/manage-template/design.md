# MANAGE-WB as a blocks template (Ghana, one year) design

## Summary for the maintainer

**What.** Port the World Bank's MANAGE-WB (GAMS, `data/manage/gams/`) for
Ghana to an equilibria blocks template on Pyomo: base 2014 calibrated, 2015
solved, dynamics, vintages, forward-looking, endogenous technology, climate
damages and links off. Every cell of 2014 and 2015 must match the recorded GAMS
run (`data/manage/gha_gtap9/gams_mcp_baseline_y2015_*`).

**How big.** The 2015 solve is **9,452 rows in 167 equation blocks over 136
variable blocks** (CONVERT dictionary, `gha2015_dict.txt`). 235 equations are
defined in `model.inc`; 68 of them generate no row in this run. By role:
43 behavioural blocks (4,164 rows), 75 identities (4,329 rows), 46
reporting-only blocks (652 rows) that feed nothing else, and 3 blocks that exist
only because of a switch (307 rows). Nine blocks hold 78 % of the rows (`e_xfac`
1,995, `e_paf` 1,199, `e_xap` and `e_pap` 1,134 each, ...).
equilibria covers almost none of it: no shared block has MANAGE's equation forms.
The plan adds **5 shared blocks** (a CES tree read from a nest table, used twice;
dual-form Armington; dual-form CET; agent prices with ad valorem taxes; CDE
demand) and **13 model-specific blocks** (production links, make matrix, border
prices, final demand, factor supply, income, government, funds and debt,
capital, macro, demography, emissions and land, reporting). No group of
equations is covered exactly by an existing block. It is **11 PRs**, each
sized S, M or L in the plan (two L: the production tree and the full solve).

**Main risks.**

1. equilibria's GDX reader cannot read the reference. Measured on this branch:
   `read_variable_values` returns 0 of 102 records for `xp` in `GHA_BaU.gdx`, and
   only 54 of the 465 non-empty parameters and variables decode completely.
   `gdxdump` reads them all. PR 1 fixes this first.
2. Units. GAMS solves with GDP = 1e5 (`scaleSAM`) and writes the GDX with only
   the `.l` of a fixed list of symbols multiplied back by `reScale` = 3.954e-4
   (`inc/cal/rescale.inc:21-137`); bounds and some entries stay in solve units.
   Some constants in the equations (`1e-8` in `e_FundG`, `1e-12` in `e_FundF`,
   the `1e-3` scale floor, the smooth-max and sigmoid kinks) are not scale-free,
   so **Python solves in GAMS's solve units**, and every comparison converts.
3. The 7-level production tree and the 9-bundle CDE with its CES sub-nests come
   from data. A wrong child-parent row fails no solve; it moves the numbers.
4. The calibration contains three internal NLP solves (one infeasible, one a
   least-squares CDE fit that misses its targets, objective 0.096). Re-solving
   them in Python would not reproduce GAMS to 1e-9. So this change reads GAMS's
   calibrated parameters from `GHA_BaU.gdx`; porting `inical.inc` is a
   separate, later change (decision 6).
5. The size of the solve: 9,452 rows on Pyomo and IPOPT, with no row scaling, at
   a new point (a-ele output x10.5 in 2015). GAMS needed its row scaling and
   PATH on NEOS for the cross-check.

**First PR.** `fix(data): the GDX reader reads MANAGE's GDX` (Task 1 of the
plan): the reader reads every parameter and variable of `GHA_BaU.gdx` and
`BaseBridge_GHA_GTAP_9.gdx` exactly as `gdxdump` does. It changes no model and
unblocks everything else.

**Decisions taken** with the maintainer are listed at the end.

## Context

### The reference

- The model that runs is `data/manage/gams/` (`source.yaml` lists every change
  made so it runs on macOS). `run_gha_one_year.sh` runs it in under a minute.
- The settings are `mac/gui_baseline_one_year.gms`: task Baseline (line 6),
  `scaleModel on` and `scaleSAM on` (110-111), and dynamics off (43-57).
- The dataset is `data/manage/gha_gtap9/`. Its two result folders are
  `gams_mcp_baseline_y2015_nlpec-ipopt` (local NLPEC/IPOPT; it also keeps the
  CONVERT scalar model at the start and at the solution) and
  `gams_mcp_baseline_y2015_path` (PATH on NEOS from the same start point). The two
  agree to 1.1e-6 absolute and 7.9e-9 relative over the 9,452 variables
  (`gams_mcp_baseline_y2015_path/source.yaml`, notes).
- `GHA_BaU.gdx` holds every symbol after `rescale.inc`: 1,166 symbols (270
  sets, 46 aliases, 264 parameters, 331 variables, 252 equations). Its year
  labels are `2014` (the calibrated base, never solved) and `2015`.

### The MCP is a square system

- All 9,452 rows are `=E=`, and all columns are continuous, free and unbounded
  (`gha2015_scalar.gms:1-11`: E = 9,452, FX 0).
- The model statement is unpaired: `paired` is off (`model.inc:3612`). The
  embedded Python strips the `.var` suffixes into `model_def_copy.inc`
  (`model.inc:3622-3635`), and the model is `cge_ / model_def_copy.inc,
  e_rrat.rrat /` (`model.inc:3638-3641`).
- The pairings in `model_def.inc` document intent; they do not bind. Applied
  anyway, they reproduce every variable block's count.
- So Python can solve the same system as a square NLP with a zero objective,
  as `simple_open` does (`templates/simple_open/template.py:1-7`), with IPOPT.
  PATH is not needed for the Python side.

### Closure of the 2015 solve

- The numeraire is the exchange rate, `er.fx(ts) = er.l(ts)` = 1, with
  `pgdpmp` free (`inc/solve/iterloop.inc:385-387`).
- Real GDP is fixed to the target, `rgdpmp.fx(ts) = p_GDP(ts)` (+2.17821 %,
  `iterloop.inc:470`). The uniform labor-productivity shifter `chixFacL`,
  freed in `inc/cal/inical.inc:2084-2085`, is paired with `e_rgdpmp`
  (`model_def.inc:264-270`); GAMS finds -2.86 %.
- Foreign saving is a fixed share of nominal GDP: `savfGDP.fx`
  (`iterloop.inc:651-652`) and `e_savf` (`model.inc:3148`).
- Government real consumption is the base share of real GDP, `e_rshrGDPGov`
  (`model.inc:3328-3330`).
- Tax rates are fixed wherever all their shifters are fixed
  (`iterloop.inc:20-129`, mode `fixTaxesLambda`). With `holdfixed = 1`
  (`inc/solve/prepSolve.inc:117`), fixed variables become constants and do not
  appear among the 9,452 columns.

## Equation inventory (one-year Ghana run)

How it was built: the block list and row counts come from the CONVERT
dictionary of the 2015 solve
(`data/manage/gha_gtap9/gams_mcp_baseline_y2015_nlpec-ipopt/gha2015_dict.txt`,
`Equations 1 to 9452`). "model.inc" is the line of the `e_x(...) ..`
definition in `data/manage/gams/inc/model.inc`, and "paired with" is the
`e_x.var` entry in `inc/model_def.inc`, with its `$$if` branches resolved for
these settings.

Roles:

- **C**: behaviour, market clearing or closure.
- **I**: an identity, a definition that could be substituted out.
- **pin**: the 2015 row holds only its own variable, so it fixes that variable
  to a number.
- **R**: reporting only. The defined variable feeds no other active block,
  directly or through other R blocks. This was found by a graph pass over the
  scalar model (`gha2015_scalar.gms`), not by reading the code.
- **S**: present, or shaped, only because of a switch or a data condition.

Indices in the GHA instance: 61 activities `a` (51 active), 60 commodities `i`
(51 produced, 50 exported, 57 imported, 58 Armington), 41 factors `fp` (11 in
the SAM), 1 household `h-hhold`, 3 other final demanders `f` (gov, inv, ginv),
19 production parent nests `fpPar`, 17 demand nests `kNest` (10 top), one
vintage `v = ""` (`gha2015_dict.txt`; `GHA_BaU.gdx`).

#### A. Production: nested CES technology (xfac/pxfac nests, lambdaXfac)

| equation | model.inc | paired with (model_def.inc) | rows | role |
|---|---:|---|---:|---|
| `e_xfac` | 1253 | `xFac` (:26) | 1995 | C |
| `e_pxfac` | 1307 | `pxFac` (:27) | 570 | C |
| `e_pxfacInd` | 1361 | `pxFacInd` (:28) | 65 | C |
| `e_pap` | 1385 | `pxFac` (:25) | 1134 | I |
| `e_xap` | 1379 | `xa` (:24) | 1134 | I |
| `e_lambdaxfac` | 1428 | `lambdaXfac` (:29) | 255 | S (labor only, because chixFacL is endogenous in Baseline; others fixed by iterloop.inc:88-115) |
| `e_px` | 1455 | `px` (:22) | 51 | C |
| `e_pp` | 1466 | `pp` (:23) | 51 | I |
| `e_xpNew` | 2997 | `xfac` (:187) | 51 | I (S: vintages off branch) |
| `e_yieldLnd` | 1480 | `yieldLnd` (:30) | 8 | R |

#### B. Make matrix: activity output to commodities

| equation | model.inc | paired with (model_def.inc) | rows | role |
|---|---:|---|---:|---|
| `e_x` | 1528 | `x` (:37) | 51 | C |
| `e_xp` | 1534 | `xp` (:38) | 51 | C |
| `e_p` | 1544 | `p` (:39) | 51 | C |
| `e_xs` | 1557 | `xs` (:40) | 51 | C |
| `e_ElecMix` | 1581 | `elecmix` (:262) | 1 | R |

#### C. Trade: Armington, CET, border prices

| equation | model.inc | paired with (model_def.inc) | rows | role |
|---|---:|---|---:|---|
| `e_pma` | 2499 | `pma` (:151) | 57 | I |
| `e_pda` | 2505 | `pda` (:152) | 51 | I |
| `e_xat` | 2512 | `xat` (:153) | 58 | I |
| `e_xdd` | 2519 | `xd` (:154) | 51 | C |
| `e_xm` | 2527 | `xm` (:155) | 57 | C |
| `e_pa` | 2535 | `pa` (:156) | 58 | C |
| `e_paf` | 2644 | `paf` (:157) | 1199 | I |
| `e_xds` | 2660 | `pd` (:161) | 51 | C |
| `e_eds` | 2672 | `xe` (:162) | 50 | C |
| `e_ps` | 2682 | `ps` (:160) | 51 | C |
| `e_pe` | 2703 | `pe` (:163) | 50 | I |
| `e_pwe` | 2710 | `pwe` (:164) | 50 | C (49 rows pin a constant) |
| `e_pwm` | 2717 | `pwm` (:165) | 57 | I (pin) |

#### D. Final demand: households (CDE), government and investment (CES nests)

| equation | model.inc | paired with (model_def.inc) | rows | role |
|---|---:|---|---:|---|
| `e_cdeInt` | 2216 | `cdeInt` (:129) | 9 | C |
| `e_consScale` | 2228 | `consScale` (:128) | 1 | I |
| `e_xakh` | 2238 | `xak` (:130) | 9 | C |
| `e_uh` | 2244 | `uh` (:131) | 1 | C |
| `e_mmuh` | 2249 | `mmuh` (:132) | 1 | R |
| `e_xakShr` | 2256 | `xakShr` (:133) | 14 | R |
| `e_pfh` | 2261 | `pf` (:134) | 1 | C (feeds e_pindh, e_yf) |
| `e_xakf` | 2284 | `xak` (:137) | 5 | C |
| `e_pf` | 2294 | `pf` (:140) | 2 | C |
| `e_xak` | 2303 | `xak` (:136) | 7 | C |
| `e_pxak` | 2315 | `pxak` (:138) | 21 | C |
| `e_xaoa` | 2335 | `xa` (:135) | 65 | C |
| `e_yf` | 2435 | `yf` (:145) | 3 | I |

#### E. Institutions: income, saving, transfers (households, enterprises)

| equation | model.inc | paired with (model_def.inc) | rows | role |
|---|---:|---|---:|---|
| `e_yh` | 1603 | `yh` (:44) | 1 | I |
| `e_yhDsp` | 1623 | `yhDsp` (:46) | 1 | I |
| `e_savh` | 1629 | `savh` (:52) | 1 | C |
| `e_pindh` | 1638 | `pindh` (:53) | 1 | C |
| `e_aps` | 1651 | `aps` (:49) | 1 | R |
| `e_hhtr` | 1656 | `tottr` (:50) | 1 | I (constant 0) |
| `e_yfh` | 1662 | `xfd` (:51) | 1 | I |
| `e_entrY` | 1890 | `entrY` (:106) | 1 | I |
| `e_savEntr` | 1909 | `savEntr` (:107) | 1 | C |
| `e_entrTr` | 1916 | `totTr` (:108) | 1 | I |
| `e_dtransfers` | 1931 | `transfers` (:110) | 2 | I |
| `e_rowtransfers` | 1938 | `transfers` (:111) | 2 | I (pin) |

#### F. Government and taxes

| equation | model.inc | paired with (model_def.inc) | rows | role |
|---|---:|---|---:|---|
| `e_tPit` | 2618 | `tPit` (:72) | 1 | I (pin) |
| `e_tCIT` | 2624 | `tCit` (:73) | 1 | I (pin) |
| `e_tVat` | 2630 | `tVat` (:74) | 21 | S (pin) |
| `e_tPaf` | 2636 | `tPaf` (:75) | 31 | S (pin) |
| `e_ygPit` | 1792 | `ygPit` (:78) | 1 | I |
| `e_ygCit` | 1802 | `ygCit` (:79) | 1 | I |
| `e_ygPaf` | 1751 | `ygPaf` (:80) | 58 | I; 1/58 rows have a single variable |
| `e_ygTotFacLand` | 1708 | `yg` (:81) | 1 | I |
| `e_ygTotFacNatres` | 1690 | `yg` (:82) | 1 | I |
| `e_ygTotFacLab` | 1696 | `yg` (:83) | 1 | I |
| `e_ygTotFacCap` | 1702 | `yg` (:84) | 1 | I |
| `e_ygTotpsb` | 1714 | `yg` (:85) | 1 | I; 1/1 rows have a single variable |
| `e_ygTotPrd` | 1719 | `yg` (:86) | 1 | I |
| `e_ygTotPhn` | 1724 | `yg` (:87) | 1 | I; 1/1 rows have a single variable |
| `e_ygTotExp` | 1741 | `yg` (:88) | 1 | I |
| `e_ygTotImp` | 1746 | `yg` (:89) | 1 | I |
| `e_ygTotEmi` | 1772 | `yg` (:90) | 1 | I; 1/1 rows have a single variable |
| `e_ygTotEmiXP` | 1787 | `yg` (:91) | 1 | I; 1/1 rows have a single variable |
| `e_ygTotPit` | 1797 | `yg` (:92) | 1 | I |
| `e_ygTotCit` | 1807 | `yg` (:93) | 1 | I |
| `e_ygTotVatDom` | 1729 | `yg` (:94) | 1 | I; 1/1 rows have a single variable |
| `e_ygTotVatImp` | 1735 | `yg` (:95) | 1 | I |
| `e_ygTotPaf` | 1756 | `yg` (:96) | 1 | I |
| `e_ygTot` | 1827 | `ygTot` (:97) | 1 | R |
| `e_ygov` | 1812 | `ygov` (:61) | 1 | I |
| `e_gtax` | 1832 | `gtax` (:99) | 1 | I; 1/1 rows have a single variable |
| `e_govTr` | 1838 | `totTr` (:100) | 1 | C |
| `e_primaryBalance` | 1844 | `primaryBalance` (:101) | 1 | I |
| `e_fiscalBalance` | 1866 | `fiscalBalance` (:102) | 1 | I |
| `e_govBalance` | 1873 | `govBalance` (:103) | 1 | I |
| `e_rgovBalance` | 1879 | `rgovBalance` (:104) | 1 | R |

#### G. Loanable funds, public debt, interest rate

| equation | model.inc | paired with (model_def.inc) | rows | role |
|---|---:|---|---:|---|
| `e_FundT` | 1950 | `fundt` (:54) | 1 | I |
| `e_FundP` | 1956 | `FundP` (:56) | 1 | I |
| `e_FundPriceD` | 1961 | `FundPriceD` (:59) | 1 | C |
| `e_FundG` | 1970 | `FundG` (:55) | 1 | C |
| `e_FundPrice` | 1979 | `FundPrice` (:58) | 1 | C |
| `e_FundF` | 2066 | `FundF` (:57) | 1 | I (pin) |
| `e_interest` | 2076 | `interest` (:60) | 1 | C |
| `e_debtNew` | 2085 | `debtNew` (:248) | 2 | I |
| `e_DebtStk` | 2099 | `DebtStk` (:244) | 2 | I |
| `e_DebtStkT` | 2106 | `DebtStkT` (:250) | 1 | R |
| `e_debtGDP` | 2111 | `debtGDP` (:236) | 1 | R |
| `e_debtPay` | 2116 | `debtPay` (:245) | 2 | I |
| `e_debtInterest` | 2122 | `debtInterest` (:246) | 2 | I |
| `e_debtPrincipal` | 2129 | `debtPrincipal` (:247) | 2 | I (pin) |
| `e_avgDebtInt` | 2135 | `avgDebtInt` (:243) | 2 | I (pin) |
| `e_debtNewT` | 2146 | `debtNewT` (:249) | 1 | I |

#### H. Investment, capital stock, rates of return

| equation | model.inc | paired with (model_def.inc) | rows | role |
|---|---:|---|---:|---|
| `e_inv` | 2157 | `xfd` (:112) | 1 | C |
| `e_prodInv` | 2008 | `prodinv` (:222) | 1 | I |
| `e_kstocke` | 2002 | `kstocke` (:117) | 1 | I |
| `e_rorc` | 2026 | `rorc` (:119) | 1 | C |
| `e_rore` | 1985 | `rore` (:120) | 1 | C |
| `e_ror` | 2046 | `ror` (:118) | 1 | I (pin) |
| `e_kaps` | 2052 | `kaps` (:220) | 1 | I |
| `e_depr` | 2031 | `depr` (:221) | 1 | I |
| `e_kstock` | 3558 | `kstock` (:223) | 1 | I (pin; S: forward-looking off branch) |
| `e_grK` | 3186 | `grK` (:122) | 1 | R |
| `e_trent` | 2918 | `trent` (:169) | 1 | I (S: vintages off) |
| `e_trentAvg` | 3017 | `trentAvg` (:184) | 1 | R |
| `e_atCap` | 2903 | `atFmSup` (:167) | 1 | C (S: vintages off) |

#### I. Factor markets and factor income

| equation | model.inc | paired with (model_def.inc) | rows | role |
|---|---:|---|---:|---|
| `e_pxfac_preTax` | 3116 | `pxFac` (:166) | 342 | I |
| `e_xfacFp` | 2736 | `pxfac_pretax` (:172) | 342 | C |
| `e_xfmSup` | 2751 | `xfmsup` (:174) | 380 | C |
| `e_pfmSup` | 2833 | `pfmsup` (:176) | 38 | C |
| `e_pfmSupD` | 2877 | `pfmsupD` (:177) | 16 | C |
| `e_facty` | 1592 | `facty` (:42) | 11 | I |
| `e_kapy` | 1598 | `kapy` (:43) | 1 | R |
| `e_unemp` | 3578 | `unemp` (:178) | 1 | R |

#### J. Macro aggregates, closure, numeraire-related prices

| equation | model.inc | paired with (model_def.inc) | rows | role |
|---|---:|---|---:|---|
| `e_totexp` | 3233 | `totexp` (:194) | 1 | I |
| `e_rtotexp` | 3238 | `rtotexp` (:195) | 1 | I |
| `e_totimp` | 3248 | `totimp` (:197) | 1 | I |
| `e_rtotimp` | 3253 | `rtotimp` (:198) | 1 | I |
| `e_gdpmp` | 3263 | `gdpmp` (:200) | 1 | I |
| `e_rgdpmp` | 3268 | `ChixFacL` (:269) | 1 | C (S: Baseline GDP-targeting closure) |
| `e_pgdpmp` | 3273 | `pgdpmp` (:204) | 1 | I |
| `e_savf` | 3148 | `savf` (:205) | 1 | C (S: capAccClosure) |
| `e_rshrGDPGov` | 3328 | `xfd` (:214) | 1 | C (S: GDPCompGovCons; pin) |
| `e_walras` | 3192 | `walras` (:255) | 1 | R |
| `e_BOT` | 3339 | `bot` (:256) | 1 | R |
| `e_BOTGdp` | 3344 | `botGDP` (:260) | 1 | R |
| `e_pexp` | 3243 | `pexp` (:196) | 1 | R |
| `e_pimp` | 3258 | `pimp` (:199) | 1 | R |
| `e_cpi` | 3364 | `cpi` (:219) | 1 | R |

#### K. Demography

| equation | model.inc | paired with (model_def.inc) | rows | role |
|---|---:|---|---:|---|
| `e_popAge` | 2190 | `popAge` (:124) | 2 | I (pin) |
| `e_Birth` | 2184 | `popAge` (:125) | 1 | I (pin) |
| `e_pop` | 2199 | `pop` (:126) | 4 | I |
| `e_hpop` | 2205 | `hpop` (:127) | 1 | I |
| `e_deprate` | 1673 | `deprate` (:48) | 4 | R |

#### L. Emissions, energy and land cover (all reporting in this run)

| equation | model.inc | paired with (model_def.inc) | rows | role |
|---|---:|---|---:|---|
| `e_emiComm` | 3375 | `emiComm` (:226) | 185 | R |
| `e_emiFact` | 3381 | `emiFact` (:227) | 54 | R |
| `e_emiProd` | 3387 | `emiXP` (:228) | 7 | R |
| `e_emiSecA` | 3398 | `emiSecA` (:235) | 151 | R |
| `e_emiTot` | 3430 | `emiTot` (:229) | 13 | R; 9/13 rows have a single variable |
| `e_emiGHG` | 3494 | `emighg` (:230) | 6 | R; 4/6 rows have a single variable |
| `e_emiInv` | 3409 | `emiInv` (:231) | 117 | R; 98/117 rows have a single variable |
| `e_emiInt` | 3499 | `emiInt` (:232) | 6 | R |
| `e_dCarbon` | 3438 | `dCarbon` (:237) | 19 | R |
| `e_lCovArea` | 3447 | `lCovArea` (:238) | 19 | R; 2/19 rows have a single variable |
| `e_builtUpArea` | 3486 | `lCovArea` (:239) | 4 | R |
| `e_nrgInt` | 3599 | `nrgInt` (:158) | 1 | R |
| `e_elyInt` | 3604 | `elyint` (:159) | 1 | R |

#### M. Reporting identities

| equation | model.inc | paired with (model_def.inc) | rows | role |
|---|---:|---|---:|---|
| `e_rgdppc` | 3278 | `rgdppc` (:207) | 1 | R |
| `e_ggdppc` | 3283 | `ggdppc` (:278) | 1 | R |
| `e_gdpfc` | 3288 | `gdpfc` (:208) | 1 | R |
| `e_rgdpfc` | 3294 | `rgdpfc` (:209) | 1 | R |
| `e_pgdpfc` | 3299 | `pgdpfc` (:210) | 1 | R |
| `e_rshrGDP` | 3304 | `rshrGDP` (:211) | 4 | R; 1/4 rows have a single variable |
| `e_nshrGDP` | 3309 | `nShrGDP` (:215) | 3 | R |
| `e_GdpCompShr` | 3349 | `gdpCompShr` (:216) | 4 | R; 1/4 rows have a single variable |
| `e_GdpCompShrX` | 3354 | `gdpCompShr` (:217) | 1 | R |
| `e_GdpCompShrM` | 3359 | `gdpCompShr` (:218) | 1 | R |
| `e_SectTot` | 3589 | `sectTot` (:263) | 1 | R |
| `e_SectShr` | 3594 | `sectShr` (:279) | 6 | R |
| `e_dummy` | 3609 | `none` (:296) | 1 | R (no pairing in model_def.inc:296); 1/1 rows have a single variable |


### Totals

| group | blocks | rows | of which R rows |
|---|---:|---:|---:|
| A. Production nests | 10 | 5,314 | 8 |
| B. Make matrix | 5 | 205 | 1 |
| C. Trade, Armington, CET, border and agent prices | 13 | 1,840 | 0 |
| D. Final demand (CDE household, CES gov/inv, demand nests) | 13 | 139 | 15 |
| E. Institutions: income, saving, transfers | 12 | 14 | 1 |
| F. Government and taxes | 31 | 138 | 2 |
| G. Loanable funds, public debt, interest | 16 | 22 | 2 |
| H. Investment, capital, returns | 13 | 13 | 2 |
| I. Factor markets and factor income | 8 | 1,131 | 2 |
| J. Macro aggregates and closure | 15 | 15 | 6 |
| K. Demography | 5 | 12 | 4 |
| L. Emissions, energy, land cover | 13 | 583 | 583 |
| M. Reporting identities | 13 | 26 | 26 |
| **Total** | **167** | **9,452** | **652** |

By role: C 43 blocks / 4,164 rows; I 75 / 4,329; R 46 / 652; S 3 / 307
(`e_lambdaxfac` 255, `e_tVat` 21, `e_tPaf` 31). 297 rows in 30 blocks are pins.

### Why 167 equation blocks fill 136 variable blocks

43 blocks pair with a variable that is not named after them. 36 of them share
12 variables: `yg` (16 blocks, one row each: `model_def.inc:81-96`), `xak` (3),
`pxFac` (3), `xfd` (3), `totTr` (3), `gdpCompShr` (3), `xa` (2), `xfac` (2),
`transfers` (2), `pf` (2), `popAge` (2), `lCovArea` (2). Seven are renamed
one-to-one: `e_xdd.xd`, `e_xds.pd`, `e_eds.xe`, `e_xfacFp.pxfac_pretax`,
`e_atCap.atFmSup`, `e_emiProd.emiXP`, `e_rgdpmp.ChixFacL`. `e_dummy` has no pair
(`model_def.inc:296`) and meets the free variable `dummy`.
So 167 - 43 + 12 = 136.

Column checks: `pxfac` 2,046 = `e_pxfac` 570 + `e_pap` 1,134 +
`e_pxfac_preTax` 342; `xfac` 2,046 = `e_xfac` 1,995 + `e_xpNew` 51; `xa` 1,199 =
`e_xap` 1,134 + `e_xaoa` 65.

Upstream typos that are reproduced, not fixed (ADR-0008):

- `model_def.inc:254` tests `%GPDCompNetExports%`, so it takes the
  `e_botGDP.botGDP` branch.
- `opt.inc:375-377` tests `%GPDCompSavf%`. It forces `savfExoRel`, which is the
  GUI's choice anyway.

### Defined in model.inc but not in this run

`model.inc` declares 244 equation names (242 unique under case folding;
`model.inc:689-1000`). 235 are defined and 234 are in the model; 167 generate
rows, so 67 are inactive. By reason:

- **Twist and shares**: `e_twist` 1396, `e_shrxFac` 1412.
- **Productivity links off**: `e_diffLabProd` 1488, `e_LandProd` 1499,
  `e_NatresProd` 1508, `e_capProd` 1517.
- **Additive CES/CET sets empty**: `e_pIND` 1572, `e_paInd` 2544, `e_psind` 2694.
- **One household**: `e_gini` 1667.
- **corrFac fixed**: `e_corrFac` 2894.
- **Tax rates fixed** (`iterloop.inc:24-86`): `e_tFac` 2558, `e_tpsb` 2578,
  `e_tPrd` 2584, `e_tPhn` 2591, `e_tExp` 2598, `e_tImp` 2604, `e_tEmi` 2611,
  `e_rtFac` 3129, `e_rtEmi` 3425.
- **Emission taxes off**: `e_ygEmi` 1765, `e_ygEmiXP` 1779.
- **No trade margins**: `e_xmarg` 2406, `e_xarmMarg` 2412, `e_pmargInd` 2420,
  `e_pmarg` 2429.
- **Sector-specific investment off**: `e_newInv` 2356, `e_afInv` 2366,
  `e_invPrice` 2382, `e_gammaInv` 2393.
- **Damages and air pollution off**: `e_xfdInv` 3508, `e_emiIntSec` 3520,
  `e_emiIntHH` 3529, `e_ExposedAirpoll` 3538, `e_dmgAir` 3545.
- **Vintages off**: `e_atCapVint` 2911, `e_trentKaps` 2942, `e_oldCap` 2951,
  `e_xpOld` 2959, `e_pk` 2972, `e_rrat1` 2978, `e_rrat` 2984, `e_xpOldNoCap` 3006,
  `e_pfmSupCap` 3022, `e_xfacFpCap` 3031.
- **Capital income through shares**: `e_trentInst` 2925, `e_xfmSupTot` 2887.
- **Endogenous technology off**: `e_xpEnTech` 3085, `e_einn` 3044, `e_xRD` 3051,
  `e_xfdRD` 3069, `e_yfdRD` 3075, `e_pfdrd` 3080.
- **No stocks**: `e_st` 2446, `e_delst` 2452, `e_psExpRel` 2474, `e_psExp` 2479,
  `e_xsExp` 2486, `e_tdelst` 3213, `e_rtdelst` 3219, `e_pdelst` 3228.
- **Fiscal rule off**: `e_fiscalRespDrv1` 3314, `e_fiscalRespDrv` 3317,
  `e_fiscalResp` 3334.
- **Others**: `e_emiLUCF` 3392; `e_devROR` 3181 (USAGE closure only);
  `e_savfGDP` 3141 (skipped under `savfExoRel`). `e_yield` 1473 is defined but
  not in the model.

The template builds only the 167 active blocks. An inactive equation becomes a
block when a dataset or switch needs it, with its own reference result.

### Functional forms equilibria does not have yet

- **A production tree from data.** One pair of equations covers every level:
  `e_xfac` (child demand, `model.inc:1253-1301`) and `e_pxfac` (parent dual
  price, `model.inc:1307-1357`).
  - The tree is the share table `p_alphaXfac(fpChl,fpPar,a,v)` with the flag set
    `flagXFac`.
  - A child may have several parents: `labsk` sits under `cap_labsk` and
    `labsk_labus`, and `nd` under `f-top` and `nd_wtr_land`.
  - The deepest path has 7 levels: f-top > va > cap_labsk_ener > ener >
    ener_fossil > ener_gasoil > gas_gdt > c-gas.
  - Three energy nests use the "additive" (volume-preserving) CES, which adds the
    exhaustion price `pxfacInd` (`model.inc:1361-1375`).
  - The productivity shifter enters as `lambda^(sigma-1)`, or `lambda^(-sigma)`
    in additive nests (`model.inc:1280-1282`).
  - There is a Leontief branch for sigma = 0 (f-top in a-omf).
- **CDE household demand** in GTAP's form, split into rows:
  - `e_cdeInt`: alpha·b·u^(e·b)·p^b·(y/pop)^(-b) (`model.inc:2216-2223`)
  - `e_consScale` (`model.inc:2228-2232`)
  - `e_xakh`: Marshallian demand (`model.inc:2238-2240`)
  - `e_uh`: implicit utility, 1 = Σ cdeInt/b (`model.inc:2244-2245`)

  Below the CDE sit CES sub-nests down to products (`e_xak`, `e_pxak`,
  `e_xaoa`, `model.inc:2303-2352`).
- **Ad valorem taxes by (good, agent)**: `paf(i,aa) = (1+tPaf(i,aa))·pa(i)`
  (`model.inc:2644-2649`, 1,199 rows). VAT also enters agent domestic and import
  prices (`e_pda`, `e_pma`, `model.inc:2499-2507`). Revenue comes in 16
  streams `yg(g)` (`model.inc:1690-1808`).
- **A factor-supply tree**:
  - Top supply slopes upward: xfmSup = atFmSup·(pfmSup/pgdpmp)^eps
    (`model.inc:2762-2764`).
  - A CET allocation tree splits labor into agricultural, manufacturing and other
    segments (`model.inc:2787-2797`), with perfect mobility inside a segment
    (`model.inc:2810-2815`).
  - Land uses an additive CET tree (`e_pfmSupD`, `model.inc:2877-2883`).
  - Natural resources are immobile, with a sigmoid kink
    (`model.inc:2771-2777`).
- **Smooth non-smooth functions**: GAMS's intrinsic `ncpVUpow` (smooth max) in
  `e_interest`, `e_inv`, `e_rorc`, `e_FundF` and `e_xfmSup`, and `sigmoid` in
  `e_FundF` and `e_xfmSup` (`model.inc:2027, 2070, 2079, 2163, 2775`). Python
  must use GAMS's exact definitions, including the smoothing constant.
- **Benchmark-relative and lagged terms inside equations**: for example
  `p/p.l(t0)` in `e_x` (`model.inc:1530`), `xfac/xfac.l(t0)·xp.l(t0)` in `e_xpNew`
  (`model.inc:3004`), `kstock = (1-depr.l(t-1))·kstock.l(t-1) + prodInv.l(t-1)`
  (`model.inc:3558-3574`), and debt and demography lags. Blocks do not know
  periods (ADR-0006), so these become parameters (`<var>_t0`) and fixed lag
  cells (`<var>_lag`); see "Periods and recipe".
- **Row scaling**: with `scaleModel on`, almost every row is divided on both
  sides by max(1e-3, |2014 level of its variable|) (macro `s()`,
  `model.inc:1198-1217`). This changes residuals and the Jacobian, never the
  solution. See "Parity test".

## Coverage by equilibria's blocks

What exists today (`src/equilibria/blocks/`):

- **Level-form blocks.** Twelve `LevelBlock` classes written for Hosoe's stdcge
  and used by `simple_open` (`templates/simple_open/template.py:136-149`). They
  are single-level, and their set names are fixed (`I`, `J`, `F`).
- **Legacy blocks** with hard-coded Pyomo names (for example `CESValueAdded`,
  `production/__init__.py:32`, whose Pyomo expression is Cobb-Douglas only).
- **GTAP blocks** (`blocks/gtap/`), indexed by region. They are built from
  `GTAPParameters`/`GTAPSets` and need a partner region, so they cannot serve a
  single country; their forms can be copied.

New blocks follow the level-form pattern: `LevelBlock`, `.param`/`.var`, `eq()`,
`uses` (`blocks/_levels.py:34-86`). Each name is declared once (ADR-0005,
`model.py:140`), and each is replaced by name with `@overwrite` (ADR-0004,
`templates/simple_open/template.py:50`).
**Existing blocks are not edited**, so `simple_open`'s parity cannot move.

Verdicts: **exact** (an existing block has the same equations), **partly** (the
same economics, but a different form, parametrisation or index structure, so a
new block is needed), **none**.

| group | verdict | evidence (equations compared) | new blocks |
|---|---|---|---|
| A. Production nests (5,314) | **none** | `CobbDouglasValueAdded` (`production/__init__.py:741`, `eq_va` VA = b·Π FD^β) and `LeontiefComposite` (`:775`) are one fixed level each. GTAP `ProductionSupplyBlock` has a fixed two-level tree xp -> (nd, va) indexed (r,a) (`blocks/gtap/production_supply.py:270-408`, `eq_nd`, `eq_va`, `eq_pxeq`, `eq_pndeq`, `eq_pvaeq`). MANAGE's `e_xfac`/`e_pxfac` (`model.inc:1253, 1307`) walk a data tree with multiple parents, additive CES and a Leontief branch. | shared **`CESTree`**; model-specific **`ManageProduction`** (`e_pap`, `e_xap`, `e_lambdaxfac`, `e_px`, `e_pp`, `e_xpNew`, `e_yieldLnd`) |
| B. Make matrix (205) | **partly** | `CETOutput` (`production/__init__.py:817`) is a two-way E/D CET, not activity -> commodity. GTAP `eq_x`/`eq_po` (`production_supply.py:453-482`) is the right economics, but region-indexed. In GHA, omegaP = sigmaS = +Inf, so only the linear branches are active (`model.inc:1528-1568`). | model-specific **`ManageMake`**: infinite branches only; a finite elasticity raises `NotImplementedError` naming the cell |
| C. Trade and prices (1,840) | **partly** | `ArmingtonComposite` (`trade/__init__.py:1596`) is CES in primal form Q = γ(δM·M^η + δD·D^η)^(1/η), with both sources positive (simple_open's `_positive` check) and no σ = 1 branch. MANAGE uses demand-share plus dual-price form with efficiency shifters (`e_xdd`, `e_xm`, `e_pa`, `model.inc:2519-2539`) over 58 goods, 7 with no domestic source. `CETOutput` is primal with a `tau_z` wedge; MANAGE `e_xds`/`e_eds`/`e_ps` (`model.inc:2660-2690`) use share and γ^(-1-ω) forms, an ω = +Inf branch and 50 of 51 goods exported. `ExternalSector` (`institutions/__init__.py:629`) has PE = EPS·pWe only; MANAGE's border and agent prices carry tariffs, VAT, export taxes and normalisation constants (`model.inc:2499-2507, 2703-2720`). | shared **`ArmingtonDual`** (`e_xat`, `e_xdd`, `e_xm`, `e_pa`), **`CETDual`** (`e_xds`, `e_eds`, `e_ps`), **`AgentPrices`** (`e_paf`); model-specific **`ManageBorder`** (`e_pma`, `e_pda`, `e_pe`, `e_pwe` with the finite η = 10 row for c-isr, `e_pwm`) |
| D. Final demand (139) | **partly** | CDE exists only in GTAP `DemandUtilityBlock` (`blocks/gtap/demand_utility.py:453`, `eq_uh`: Σ α·pa^b·u^(e·b)·(yc/pop)^(-b) = 1), the same form as MANAGE's `e_cdeInt` + `e_uh` (`model.inc:2216-2245`) but region-indexed and built from GTAP objects. `CobbDouglasHousehold` (`demand/__init__.py:444`) is CD with a fixed saving rate. The demand sub-nests (`e_xak`, `e_pxak`, `e_xaoa`, `model.inc:2303-2352`) are a CES tree like production's, with leaves priced at `p_chiPA·paf`. | shared **`CDEDemand`** (`e_cdeInt`, `e_consScale`, `e_xakh`, `e_uh`, `e_mmuh`, `e_xakShr`); a second instance of **`CESTree`** with its own names; model-specific **`ManageFinalDemand`** (`e_xakf`, `e_pf`, `e_pfh`, `e_yf`) |
| E. Institutions (14) | **none** | `CobbDouglasHousehold` saves a fixed share of factor income. MANAGE's `e_yh` (`model.inc:1603-1618`) adds factor-supply income by institution, transfers and debt payments; saving is a CES choice (`e_savh`/`e_pindh`, `model.inc:1629-1646`); enterprises are separate (`model.inc:1890-1918`). | model-specific **`ManageIncome`** |
| F. Government and taxes (138) | **partly** | `TaxGovernment` (`institutions/__init__.py:699`) covers a direct tax, a production tax by sector and a tariff by good, with fixed saving and consumption shares. MANAGE has 16 revenue streams `yg(g)` (`model.inc:1690-1808`), rate rows with shifters (`e_tPit`, `e_tCIT`, `e_tVat`, `e_tPaf`, `model.inc:2618-2638`) and three balances (`model.inc:1844-1880`). | model-specific **`ManageGovernment`** |
| G. Funds and debt (22) | **none** | No block has loanable funds, bonds or debt stocks. MANAGE `model.inc:1950-2149`. | model-specific **`ManageFunds`** (funds, interest, debt) |
| H. Investment and capital (13) | **none** | `StdcgeInvestment` (`templates/simple_open/blocks.py:13`) splits saving by fixed shares. MANAGE `e_inv` (`model.inc:2157-2174`) is the saving-investment balance, plus capital accumulation, returns and depreciation (`model.inc:1985-2054, 2903-2920, 3186, 3558-3574`). | model-specific **`ManageCapital`** |
| I. Factor markets (1,131) | **partly** | `FactorMarket` (`equilibrium/__init__.py:684`) has a fixed endowment. GTAP `eq_xfteq` (`blocks/gtap/factor.py:296`, xft = aft·(pft/pabs)^etaf) is MANAGE's top supply curve (`model.inc:2762-2764`), but region-indexed, with no CET segments, additive CET or sigmoid. | model-specific **`ManageFactorSupply`** (`e_xfmSup`, `e_pfmSup`, `e_pfmSupD`, `e_xfacFp`, `e_pxfac_preTax`, `e_facty`, `e_kapy`, `e_unemp`). It becomes shared only if a second model needs it (YAGNI). |
| J. Macro and closure (15) | **none** | GTAP `IncomeBlock` (`eq_gdpmp`, `eq_pgdpmp`) is regional and Fisher-based. `Numeraire` (`equilibrium/__init__.py:718`) fixes a factor price, but MANAGE fixes `er`. MANAGE `model.inc:3148-3365`; `e_rgdpmp` pairs with `chixFacL`. | model-specific **`ManageMacro`** |
| K. Demography (12) | **none** | MANAGE `model.inc:1673, 2184-2206`. | model-specific **`ManageDemography`** |
| L. Emissions and land (583, all R) | **none** | MANAGE `model.inc:3375-3500, 3599-3605`. | model-specific **`ManageEmissions`** |
| M. Reporting (26, all R) | **none** | MANAGE `model.inc:3278-3360, 3589-3610`. | model-specific **`ManageReporting`** |

Totals: **5 new shared blocks** (`CESTree` used twice, `ArmingtonDual`,
`CETDual`, `AgentPrices`, `CDEDemand`) under `src/equilibria/blocks/`, and
**13 model-specific blocks** under `src/equilibria/templates/manage/blocks/`.
No group is covered exactly.

### Shapes of the shared blocks

These fix what a reviewer needs; the plan pins the exact signatures.

- **`CESTree`** (`blocks/production/ces_tree.py`; ADR-0013, a block reads its
  nest tree from data)
  - **Sets and names.** The fields name the variables: `quantity="xfac"`,
    `price="pxfac"`, `index_price="pxfacInd"`, `shifter="lambdaxfac"`, plus
    `node_set`, `owner_set` and `prefix`. So the demand tree is a second instance
    with names `xak`/`pxak`, and ADR-0005 still holds.
  - **Data.** It reads the tree from parameters: `alpha(child,parent,owner)`,
    `sigma(parent,owner)`, `additive(parent)`, `top(parent)`.
  - **Rows.** It writes one child-demand row per (child, owner), summed over that
    child's parents, as `e_xfac` does. It writes one dual-price row per parent
    (CES, Leontief or additive branch) and one exhaustion-price row per additive
    parent.
  - **Leaves.** Leaves may be another variable family, set by the fields
    `leaf_quantity` and `leaf_price` with a constant `leaf_coef`. The demand tree
    needs this: its leaves are `xa(i,oa)`, priced at `p_chiPA·paf` (`e_xaoa`,
    `e_pxak`). The production tree keeps its leaves as `xfac`/`pxfac`, tied to
    `xa`/`paf` by `e_xap`/`e_pap`.
  - **Same instance as GAMS.** Row and column sets are taken from the data
    exactly as GAMS's `$` conditions select them, so each family has the same
    instance as GAMS.
- **`ArmingtonDual` / `CETDual`.** Two-source CES or CET in share plus dual-price
  form, with the efficiency shifter as an exponent. Either source may be absent
  for a good. An infinite elasticity becomes the price-equality branch.
- **`AgentPrices`.** `p_agent(i,aa) = (1 + t(i,aa))·p(i)` over the (i, aa) cells
  of the SAM. The rate is a variable: fixed where GAMS fixes it, free (with its
  rate row in `ManageGovernment`) for the 7 goods where GAMS leaves it free.
- **`CDEDemand`.** GTAP's CDE split into MANAGE's rows. It reads income `yf`,
  bundle prices `pxak` and population `hpop` through `uses`.

## Calibration

### What MANAGE does

- **Data loading** (`inc/Base.inc`). It reads 46 bridge symbols. Calibration uses
  `SAM0`, `defPar`, `actPar`, `comPar`, `hohPar`, `subselas`, `sigmacNest0`,
  `pnest`, `fmNest`, `fmNestPar`, `mapkNestShr`, `mapis`, the sets, and
  `GDPComp('bas','GDP',t)` (`Base.inc:436-442, 625, 801-867`).
- **Data outside the bridge.** It also reads `dat/unPop.gdx` (population,
  `Base.inc:707-725`), and `dat/gtapData9.gdx` and `dat/ghg.gdx` (land and
  emissions, `Base.inc:999-1004`, `iniEmis.inc:28-86`).
- **Labor.** Labor quantities are rebuilt from the SAM with a uniform wage,
  because `labor0` is a placeholder (`Base.inc:728-734`).
- **Units.** The SAM is scaled so GDP = 1e5: `inScale` = 1e5/39,542.375 and
  `reScale` = 3.954e-4 (`Base.inc:544-555`).
- **Calibration proper** (`inc/cal/inical.inc:20-2214`), in order:
  - prices and taxes: all nine rates `.fx` over both years (lines 148-705)
  - volumes from the SAM
  - debt, income and saving (223-395)
  - final demand nests (403-445)
  - trade, margins, VAT and sales taxes (456-626)
  - factor demands and the production shares
    `p_alphaXfac = (xfac_c·pNest/xfac_p)·(pxfac_c/(λ·pxfac_p))^σ` (794-797)
  - macro, GDP and population (841-938)
  - factor supply (994-1263)
  - the make matrix (1398-1427)
  - final-demand CES (1436-1525)
  - CDE (`calCDE.inc`)
  - Armington and CET with price normalisation to 1 (1542-1638)
  - funds and saving (1731-1819)
  - the baseline closure releases (2072-2085)
- **Three NLP solves inside calibration** (CONOPT4; `baseline.log:257-428`):

  | model | where | outcome |
  |---|---|---|
  | `m_ghgInv` (emissions allocation) | `iniEmis.inc:219, 405` | model status 4, infeasible, accepted |
  | `m_balanceXfmsup` (factor supply by institution) | `inical.inc:1016-1057` | exact (objective 1.1e-24) |
  | `ehcal` (CDE `bh`, `eh` fitted to income and price elasticities 0.735 / -0.54) | `calCDE.inc:88-171` | least squares, objective 0.0964: the targets are not met |

### Decision: read GAMS's calibrated values first

The first `calibration.py` reads the calibrated parameters and the 2014 levels
from `GHA_BaU.gdx`. It does not compute them from the bridge GDX. Why:

- **The reference decides** (ADR-0008), and the calibration is part of it. Two
  of the three internal solves depend on the solver (an infeasible one and a
  least-squares fit at CONOPT's tolerance). Python could not reproduce them to
  1e-9, and a mismatch there would hide every equation mismatch behind it.
- **Every symbol is there.** `GHA_BaU.gdx` holds all 1,166 symbols, written with
  an empty `execute_unload` list (`run.gms:458-464, 514`). Most calibrated
  "parameters" are fixed variables (`lo = up = l`): tax bases and shifters, `bh`,
  `eh`, `atFmSup`, `deprA`, `chiP`, `chiTr`, `lambdaOa`, `chixFac*`.
- **Checkable at once.** With the calibration read, every block can be checked
  against the 2014 and 2015 GAMS points from its first PR.

Rules for the reader (`templates/manage/data.py`):

1. **Solve units.** The reader returns values in solve units (GDP = 1e5).
   `rescale.inc` multiplied the `.l` of a listed set of symbols by `reScale` after
   the solve (`rescale.inc:21-112`, `133-137`). The reader divides exactly those
   entries back, from a constant table transcribed with line numbers:
   - emissions also by `reEscale` (`rescale.inc:116-131`)
   - `lCovArea` by `inscale·usd` (`run.gms:402`)
   - bounds are left as they are, since they were never rescaled
   - unlisted domains are left as they are, for example
     `atFmSup(fmAll,'sumInst',...)` (`rescale.inc:74` rescales only `inst`)

   Checks:
   - `gdpmp(2015)` = 103,365.776998 and `xp(a-pdr,2015)` = 1,128.22724 equal the
     CONVERT solution levels.
   - The whole 2015 vector equals `gha2015_solution.gms` cell by cell.
2. **No leak of the 2015 solution into the calibration.** Calibration reads
   parameters, the 2014 `.l` of every variable, and fixed cells (`lo == up`).
   Free 2015 levels are read only by the parity test.
3. **2015 exogenous values are computed by the recipe**, not read: the GDP
   target, labor supply, row transfers, population and capital stock (see
   below). They are tested against the GDX's fixed 2015 bounds.

A later change ports `inical.inc` to Python from the bridge, block by block,
with each piece tested against the parameters read here. That change needs
`unPop.gdx`, `gtapData9.gdx` and `ghg.gdx` as data (they are in
`data/manage/gams/dat/`) and a Python version of the three NLP fits. It is
outside this change.

## Periods and recipe

### What changes between 2014 and 2015 in MANAGE

The GAMS loop runs only 2015 (`run.gms:112`). Per year it runs, in order:

1. `prepSolve.inc` (`ts = 2015`)
2. `iterloop.inc run` (`iterloop.inc:132-653`):
   - `p_upDateFac = p_GDP(t)/p_GDP(t-1)` = 1.0217821 (line 169)
   - `resLastYear.inc`, which builds the start point (line 170)
   - closures, dynamics and the baseline targets
3. `fixTaxesLambda` (`iterloop.inc:20-129`)
4. the solve (`run.gms:167-355`)

Exogenous changes 2014 -> 2015:

| driver | rule | source |
|---|---|---|
| real GDP | `rgdpmp.fx = p_GDP(2015)` = p_GDP(2014)·1.0217821 | `GDPComp('bas','GDP','2015')`; `inical.inc:888-890`, `iterloop.inc:470` |
| population | `pop = p_popT` (UN): P1564 16.3268 -> 16.7461, PTOTL 28.1964 -> 28.8709 | `iterloop.inc:165`, `inical.inc:904-908` |
| labor supply | `atFmSup(labor,'sumInst')·P1564 growth` (1.025679) | `iterloop.inc:490` |
| transfers from abroad | `rowtransfers·PTOTL growth` | `iterloop.inc:171` |
| capital stock | `(1-depr(2014))·kstock(2014) + prodInv(2014)` | `model.inc:3558-3574` (a pin in 2015) |
| lagged stocks | `DebtStk`, `debtNew`, `interest`, `lCovArea`, `xfd(inv/ginv)`, `kstock` fixed at 2014 | `iterloop.inc:208-215, 398-401` |
| foreign saving | `savfGDP` fixed at its 2014 ratio (0.172598) | `iterloop.inc:651-652` |
| productivity | `chixFacL` endogenous (paired with `e_rgdpmp`) | `inical.inc:2084-2085` |
| world prices, tax rates | unchanged | `inical.inc:468, 1621`; tax rates `.fx` over both years |

### Recipe `manage_one_year`

```
periods = (Period("2014", "calibrate"), Period("2015", "step"))
plans = {
  "calibrate": KindPlan(before=(freeze_other_periods,)),
  "step": KindPlan(before=(freeze_other_periods, advance_year, start_from_last_year),
                   solver=ipopt()),
}
```

- **Period names.** The periods are named after the GDX year labels, so
  `from_gdx(..., period=...)` reads them directly. The kinds are the core's
  (`core/recipe.py:20`).
- **2014 is not solved**, as in GAMS (`ts = {2015}`).
- **No `replicate` period.** GAMS's benchmark check does not replicate in GAMS
  itself: it runs with `iterlim = 0`, aborts on purpose (`run.gms:321`), and
  shows the resLastYear residuals (below). It is replaced by the block residual
  tests on the 2014 point.
- **`advance_year`** (`templates/manage/steps.py`) fixes the exogenous 2015
  cells by the rules in the table. It also fixes the lag cells (`<var>_lag`) at
  their 2014 levels. Blocks do not know periods (ADR-0006), and a value that
  changes between periods is a step (ADR-0011).
  - Every `x.l(t-1)` in `model.inc` becomes a fixed variable `<var>_lag`.
  - Every `x.l(t0)` becomes a parameter `<var>_t0`, read by the calibration.
  - The one-year recipe needs nothing more. A multi-year recipe reuses the same
    step.
- **`start_from_last_year`** builds the 2015 start point as
  `resLastYear.inc` does:
  - volumes are grown by `p_updateFac("","")`, values by `"v"`, prices by
    `"p"` = 1 (`resLastYear.inc:23-29`)
  - some families are copied flat: `xfac`, `pxfac`, `lambdaxfac`, `kstock`,
    `xfmSup`, ...
  - per-capita families are scaled (`uh`, `mmuh`, `rgdppc`; lines 160-161, 265)
  - a cell is touched only where its range is non-zero
  - **the upstream quirks are kept**:
    - line 393 copies `tPaf` only for `iExp`, so the 7 goods Ghana neither
      produces nor exports (c-coa, c-wol, c-wht, c-gas, c-sgr, c-c_b, c-gdt)
      start at `tPaf = 0`
    - line 401 does the same for `ygPaf`
    - line 275 copies `nshrGDP` only for `f` = {gov, inv, ginv}, so
      `nshrGDP(h-hhold)` starts at 0
    - `inical.inc:2088` zeroes the 2015 activity entries of `pfmSup`, and line
      144 refills only the free `fmNestFmp` entries

  The equilibrium does not depend on the start: NLPEC/IPOPT and PATH reach the
  same point from it. Python still reproduces it, because the reference decides
  and because the start point is a cell-by-cell check of the step:
  `gha2015_scalar.gms` holds it.

## Parity test

- **References.**
  - The primary reference, which has both years:
    `reference("gha_gtap9", "gams", form="mcp", task="baseline", year=2015,
    mcp_solver="nlpec-ipopt")`, reading `GHA_BaU.gdx`.
  - The second reference:
    `reference(..., mcp_solver="path")`, reading `out.gdx`. Its `x1..x9452` are
    named by `gha2015_dict.txt` in the first folder.
  - The CONVERT files (`gha2015_scalar.gms` for the start,
    `gha2015_solution.gms` for the solution) are read by a small parser of their
    `xN.l = v;` lines and the dictionary
    (`templates/manage/convert.py`, test-only helper).
- **Cells.** Every cell of every variable family, in 2014 and in 2015:
  - **2014** (calibrated, not solved): Python's base point against GDX 2014 `.l`,
    in solve units.
  - **2015**: the 9,452 free cells against the GDX 2015 `.l`, and against the
    CONVERT solution, which needs no unit conversion. The exogenous 2015 cells
    set by `advance_year` are checked against the GDX `.lo/.up`.
  - **Not compared**: `e_rrat.rrat` (vintages off; no row), and equations'
    marginals.
- **Reading the GDX.** `equilibria.compare.from_gdx(path, period_dim=-1)` reads
  the trailing `t` as the period. A new map `compare/maps/manage.yaml` names each
  family once (ADR-0009, with `compare/maps/manage_names.yaml` and its
  completeness test):
  - it drops the vintage index `v = ""`
  - it divides the rescaled families by `reScale` (`divide_by`, read with
    `from_gdx(params=...)`)
  - it lists each family that exists on one side only with its reason
- **Scaling.** Python solves in solve units with unscaled rows. GAMS's row
  scaling (divide by max(1e-3, |2014 level|)) is applied only:
  - (a) when comparing block residuals with the CONVERT rows
  - (b) as optional IPOPT user scaling (`nlp_scaling_method=user-scaling`) if the
    plain solve does not converge

  The solution does not depend on it.
- **Block residual tests (before any solve).** Each block-group PR evaluates its
  rows at two GAMS points (`templates/manage/residuals.py`):
  - At the **2015 solution**, in solve units, with the lag cells at 2014 values:
    every row of the group, scaled as GAMS scales it, must be at most the
    measured residual of the CONVERT solution itself. That residual is measured
    in Task 2 and is expected near 1e-9; IPOPT stopped at its own tolerance.
  - At the **2014 base point**: every row that does not read a lag, the
    calibration identity.
  - The row instance of each family must also equal GAMS's (count and index set,
    from `gha2015_dict.txt`).
- **Tolerance policy: measure first, the owner sets it.** For each family and
  year, the task records the largest absolute and relative difference:
  - Python against GAMS IPOPT
  - Python against PATH
  - GAMS IPOPT against PATH: 1.1e-6 absolute, 7.9e-9 relative; nothing finer is
    meaningful

  The numbers go to the maintainer, who sets the tolerance written into the map.
  No tolerance is chosen in this design, and none is loosened later to make a
  test pass (AGENTS.md rules).

## Risks and unknowns, ranked

1. **The GDX reader**, measured on this branch at origin/main de84529a:
   - `read_gdx` lists all 1,163 symbols of `GHA_BaU.gdx`, but
     `read_variable_values` returns 0 records for 276 of the 280 non-empty
     variables (for example `xp`: 0 of 102).
   - `read_parameter_values` returns 0 for 134 of 185 parameters.
   - In the bridge GDX, 8 of 22 non-empty parameters fail.
   - `gdxdump` reads all of them.

   The tests would otherwise need `needs_gdxdump` and could not use `from_gdx`.
   The first PR fixes it in `babel/gdx/reader.py`, with a test against
   `gdxdump` on every symbol.
2. **Units and the rescale gaps.** The listed `.l` entries are rescaled, but
   bounds are not, some domains are not (`atFmSup('sumInst')`), and emissions and
   land cover have their own factors. A wrong entry in the unscale table gives
   numbers off by a factor of 2,529 in one family. Check: the reader's 2015 vector
   equals `gha2015_solution.gms` in every cell.
3. **The tree data.** Multiple parents, additive nests, Leontief branches,
   `fpIdent` and the old-vintage price rules (`model.inc:1286-1294`) are data
   paths that fail silently. Check: the row instance of each family equals
   GAMS's, and the residuals at the GAMS solution.
4. **The full solve.** 8,843 rows (9,452 with reporting) on Pyomo and IPOPT, at a
   point far from the start (a-ele output x10.5, a-omf x0.53). GAMS itself
   relied on row scaling. Mitigations:
   - start from the reproduced GAMS start point
   - IPOPT user scaling with GAMS's divisors
   - for diagnosis only, PATH on NEOS (the maintainer's approval per job; the
     local PATH licence stops at about 1,000 rows)
5. **Intrinsic functions.** `ncpVUpow` and `sigmoid` must match GAMS's
   definitions and smoothing constants exactly, and so must the `1e-8`/`1e-12`
   offsets in `e_FundG`/`e_FundF`. They only show up where the kinks bind; in
   GHA 2015, check whether any does.
6. **Fixed versus free.** The model is square only if Python fixes exactly the
   cells GAMS fixes. That set is the result of `.fx` calls spread over
   `inical.inc`, `iterloop.inc` and `holdfixed`. Check: the 2015 free cells equal
   the 9,452 names in `gha2015_dict.txt`.
7. **Build time and memory** of a two-period Pyomo model with about 10k
   variables per period: unknown, measured in the solve task. If it is slow, the
   solve test is marked `slow` and runs locally, as `gtap7_20x41` does
   (decision 5).
8. **Future data.** The inactive branches (finite make-matrix elasticities,
   vintages, stocks, margins) raise rather than guess. A new dataset turns one on
   with its own reference.

## Alternatives considered

- **Translate `model.inc` into one Pyomo file.** That would be a monolith, which
  ADR-0001 allows only as an oracle; it would also duplicate the block work.
  Rejected.
- **Re-implement `inical.inc` first.** Rejected for the first pass; see
  "Calibration".
- **Reuse the GTAP blocks with one region.** Rejected: they need a partner
  region, global closures (`eq_capAcct`) and GTAP's parameter objects, and the
  functional forms differ (MANAGE's tree is data).
- **Generalise `ArmingtonComposite`/`CETOutput`.** Rejected: editing them risks
  `simple_open`'s parity; the new dual-form blocks stand beside them.
- **Solve as an MCP with PATH in Python.** Not needed: the system is square,
  equalities only and unbounded. The local PATH licence also stops at about
  1,000 rows.

## Verification

- Per block-group PR: the residual tests at the 2014 and 2015 GAMS points, and
  the row instances against `gha2015_dict.txt`. No solve is needed.
- Solve PR: the start point against `gha2015_scalar.gms`, square at 8,843 free
  cells (9,452 after reporting), the 2015 solution against both references, and
  the 2014 base against the GDX.
- `uv run pytest tests/templates/manage tests/blocks -q`, plus the existing
  `tests/templates/simple_open` unchanged.
- The `spec-check` workflow (`scripts/dev/spec_tools.py check-pr`).

## Decisions taken (maintainer, 2026-11-27)

1. **Area.** The template has its own area, `templates/manage`
   (`specs/areas.toml`: `src/equilibria/templates/manage/`,
   `src/equilibria/manage.py`), and this change lives under it.
2. **Shared blocks.** The 5 shared blocks live under `src/equilibria/blocks/`.
   "A block reads its nest tree from data" is ADR-0013 (status proposed).
3. **Reporting rows.** The 652 reporting rows are in this change, in PR 10, so
   parity covers all 9,452 cells.
4. **Units.** Python works in GAMS's solve units (GDP = 1e5).
5. **CI.** Whether the full 2015 solve runs in PR CI is decided after it is
   measured in PR 9.
6. **Calibration.** This change reads the calibrated parameters from the GAMS
   output. Porting `inical.inc` is a separate, later change.
7. **Tolerance.** The maintainer sets it after the PR 9 measurement.

Still open: nothing that blocks PR 1.
