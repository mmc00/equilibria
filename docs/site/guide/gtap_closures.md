# GTAP closures: the usual ones and how to write each

A closure says which variables the model takes as given (exogenous) and which
it solves for (endogenous). The same shock gives different answers under
different closures, so a GTAP result always comes with its closure. This page
shows the closures most used in GTAP papers and simulations, what each one
assumes, how GEMPACK writes it (the `swap` of the `.cmf` or `.exp` file) and how
equilibria writes it.

Every example runs on the Burfisher `nus333` database (2 regions, `USA` and
`ROW`, by 3 sectors) and is run by the documentation tests. The shock of the
first three is the one of the {doc}`gtap_quickstart`: +10 % on the tariff power
of the `USA` to `ROW` route for `MFG`.

| Case | What it assumes | In equilibria |
|---|---|---|
| Default | Free trade balance, investment by the rate of return | `ClosureConfig.gtap_default()` |
| Fixed trade balance | Foreign savings do not change | `gtap_default(savf_flag="capFix")` |
| Sluggish capital | Capital moves only partly between sectors | `load("nus333", prm="sluggishk.prm")` |
| Unemployment | The real wage is fixed and employment adjusts | `@overwrite`, as Burfisher TBL65B |
| Shadow tax of a regulation | A tax adjusts to reach a quantity | `@overwrite` with `b.target`, as TBL94 |
| Baseline with a given GDP | Productivity adjusts to reach a GDP | `@overwrite` with `b.target`, as ME9A |

```{note}
These examples run on the Pyomo backend (the default). The family (JAX)
backend runs the first three too (`GTAP(..., backend="family")`): `capFlex`
takes its `risk` from the data as GAMS does (`cal.gms`), with no GAMS GDX, and
the shock can go in steps or as one Johansen step. All 37 Burfisher exercises
of `scripts/gtap/run_burfisher.py` give the same numbers on both backends. The
three `@overwrite` cases are still refused there, and the family backend has no
EV yet: that work is MAR-251.
```

## Default: RunGTAP's standard closure

RunGTAP's default closure (the Burfisher `.cmf` has no `swap`): the trade
balance is free, regional investment follows its rate of return so that
expected returns move together (`savf_flag="capFlex"`), and endowments, taxes
and technology are the model's instruments, fixed unless the shock moves them.
How mobile each endowment is between sectors comes from the data, not from the
closure (see "Sluggish capital" below): with `nus333`'s `default.prm` land,
labour and capital are all mobile. It is the closure equilibria
reproduces at 100 % against GEMPACK on the Burfisher exercises.

```python
from equilibria.gtap import GTAP, ClosureConfig, Exercise, gtap_static, load, results

params = load("nus333")
SHOCK = {"imptx": {("USA", "MFG", "ROW"): 10.0}}


def run(closure, shock=SHOCK, params=params, **recipe):
    ex = Exercise(
        GTAP(params, closure, residual_region="ROW"),
        gtap_static("gtap", solve_check=True, **recipe),
        shock=shock,
    )
    codes = ex.run()
    assert codes["shock"]["code"] == "optimal", codes
    return ex


default = results(run(ClosureConfig.gtap_default()))
print(default)
#            EV      qgdp    exports    imports
# REG
# USA -0.124372 -0.059301 -11.347078 -13.811595
# ROW  0.105934 -0.015695  -1.045167  -0.013760
```

`gtap_default(**changes)` takes any field of the closure on top, and checks it
as the constructor does: a field that does not exist, or a value that is not
allowed, raises an error.

## Fixed trade balance

Foreign savings (the capital account, and so the trade balance) stay at their
base value in real terms, and investment adjusts to the savings available.
It is the usual choice for a short-run trade policy study. The household's
saving share stays fixed, as in GEMPACK's standard closure (`dpsave`
exogenous). Do not read it as the GEMPACK swap `dpsave(r) = del_tbalry(r)`:
that swap frees the saving share and fixes the trade balance as a share of
world income, a different closure that equilibria does not have yet
(`docs/archive/findings/gempack_fixture_closure_mislabeled_2026-08-02.md`).

```python
fixed = results(run(ClosureConfig.gtap_default(savf_flag="capFix")))
print(fixed)
#            EV      qgdp    exports    imports
# REG
# USA -0.107918 -0.047646 -14.879318 -12.070108
# ROW  0.088527 -0.023899  -0.568336  -0.186962
```

The savings closure is `savf_flag`:

| `savf_flag` | What it fixes |
|---|---|
| `capFlex` (default of `gtap_default`) | Nothing in the capital account: expected rates of return move together across regions (`risk * rore = rorg`) |
| `capFix` | Foreign savings in real terms (`savf = pigbl * savf_bar`), except in the residual region |
| `capSFix` | Foreign savings as a share of regional income (`chif`), except in the residual region |
| `capFixDp` | Foreign savings as with `capFix`, and the household's saving share (`dpsave`, `betas`) becomes endogenous. Not validated: against the `dpsave = del_tbalry` GEMPACK run it matches 34 % (the saving share runs away), because that swap also fixes `del_tbalry`, which the model has no equation for. Do not use it for results |
| `capShrFix` | Accepted by the closure, but the blocks do not have GAMS's `chiInv` (the share of investment it fixes): not validated, do not use it for results |

## Sluggish capital

Capital moves only partly between sectors, so its return differs by sector: a
short to medium run view. In GTAP the mobility of each endowment is a
parameter, the elasticity of transformation `ETRAE` of the `.prm` file, not a
swap: `0` is a mobile endowment, a negative value a sluggish one. `nus333` ships
`sluggishk.prm` (capital at `-0.2`) and `sectorspecifick.prm` (capital almost
fixed in each sector), and `load` picks one with `prm=`:

```python
sluggish = results(run(ClosureConfig.gtap_default(), params=load("nus333", prm="sluggishk.prm")))
print(sluggish)
#            EV      qgdp    exports    imports
# REG
# USA -0.122687 -0.056018 -11.608848 -13.741809
# ROW  0.103319 -0.019248  -1.038236  -0.046701
```

The closure's `capital_mobility` field is not read by the block model: the
elasticities file decides.

## Unemployment: a fixed real wage

The real wage of labour in `USA` is fixed and employment adjusts, a Keynesian
short-run closure. Burfisher's `TBL65B.EXP` swaps `qe("LABOR","USA")` for
`pebfactreal("LABOR","USA")` and cuts the production tax of `MFG` in `USA`. In
equilibria a closure change is one `@overwrite` per block, in the shock period
only: labour supply `aft[USA,LABOR]` becomes endogenous in the `ShockBlock`,
and the `ClosureBlock` gets a row that fixes the wage against a Törnqvist index
of household prices. One unknown freed, one row added: the system stays
square. Hooks last for the whole Python process, so clear them before and
after.

```python
from pyomo.environ import value

from equilibria.blocks.gtap import ClosureBlock, ShockBlock, ppriv_tornqvist
from equilibria.gtap import overwrite

overwrite.clear()


@overwrite(ShockBlock, period="shock")
def unemployment(b):
    b.endogenous("aft", ("USA", "LABOR"))


@overwrite(ClosureBlock, period="shock")
def real_wage(b):
    # When the row is built, value(...) is the base level (a constant).
    b.equation(
        "eq_wreal",
        ("USA", "LABOR"),
        lambda m, r, f: m.pft[r, f] == value(m.pft[r, f]) * ppriv_tornqvist(m, r),
    )


steps = {"mute_welfare": True, "shock_steps": (0.25, 0.5, 0.75, 1.0)}
ex = run(
    ClosureConfig.gtap_default(),
    shock={"prdtx_rai": {("USA", "MFG", "MFG"): -10.9414}},
    **steps,
)
overwrite.clear()
print(results(ex).tables["qes"].loc[("LABOR", slice(None), "USA")])  # employment moves
#             pct
# ACTS
# AGR    9.039491
# MFG   39.092508
# SER   38.923166
```

## The shadow tax of a regulation

A regulation cuts the output of `MFG` in `USA` by 1 %, and the model finds the
production tax that has the same effect. Burfisher's `TBL94.EXP` swaps
`qca("MFG","MFG","USA")` for `to("MFG","MFG","USA")` and shocks `qca = -1`. In
equilibria the tax becomes endogenous and `b.target` declares the target: a new
variable `qca_target`, fixed at 1, and the row `x[shock] = qca_target * x[check]`.
The shock then moves the target, in % as any instrument.

```python
CELL = ("USA", "MFG", "MFG")
overwrite.clear()


@overwrite(ShockBlock, period="shock")
def regulation(b):
    b.endogenous("prdtx_rai", CELL)
    b.target(
        "qca_target",
        CELL,
        quantity=lambda m, r, a, i: m.x[r, a, i],
        domains=("r", "a", "i"),
    )


ex = run(ClosureConfig.gtap_default(), shock={"qca_target": {CELL: -1.0}}, **steps)
overwrite.clear()
m = ex.model.pyomo
print(value(m.prdtx_rai[(*CELL, "shock")]))  # the shadow tax rate
# 0.0333...
```

## A baseline with a given GDP

A 2010 to 2050 baseline: population, endowments and real GDP grow by given
amounts and productivity adjusts to deliver that GDP. Burfisher's `ME9A.EXP`
swaps `qgdp(REG)` for `aoreg(REG)` and uses the elasticities of
`climatechange.prm`. In equilibria productivity `axp` becomes endogenous in
every activity, `b.target` fixes real GDP, and a `ClosureBlock` row keeps the
productivity change equal across the activities of a region, as `aoreg` does.
This one matches GAMS; against GEMPACK it is within 0.77 % (GEMPACK's own log
rates many of its ME9A results to 2 figures).

```python
REGS, ACTS = ("USA", "ROW"), ("AGR", "MFG", "SER")
overwrite.clear()


@overwrite(ShockBlock, period="shock")
def baseline(b):
    for r in REGS:
        for a in ACTS:
            b.endogenous("axp", (r, a))
        b.target("gdp_target", (r,), quantity=lambda m, r: m.rgdpmp[r], domains=("r",))


@overwrite(ClosureBlock, period="shock")
def uniform_productivity(b):
    for r in REGS:
        for a in ("MFG", "SER"):
            b.equation(
                "eq_aoreg",
                (r, a),
                lambda m, r, a: m.axp[r, a] / value(m.axp[r, a])
                == m.axp[r, "AGR"] / value(m.axp[r, "AGR"]),
                domains=("r", "a"),
            )


ME9A = {
    "gdp_target": {("USA",): 109.6, ("ROW",): 284.5},
    "pop": {("USA",): 32.3, ("ROW",): 37.3},
    "aft": {
        ("USA", "LABOR"): 24.1, ("ROW", "LABOR"): 38.4,
        ("USA", "CAPITAL"): 60.6, ("ROW", "CAPITAL"): 213.1,
        ("USA", "LAND"): -0.93, ("ROW", "LAND"): 4.4,
    },
}
ex = run(
    ClosureConfig.gtap_default(),
    shock=ME9A,
    params=load("nus333", prm="climatechange.prm"),
    **steps,
)
overwrite.clear()
print(results(ex).tables["qgdp"])  # the target: +109.6 % and +284.5 %
#        pct
# REG
# USA  109.6
# ROW  284.5
```

## To be added

- **Fiscal neutrality** (a tax that adjusts to keep the government's revenue):
  common in the literature, but the repository has no validated example yet,
  so this page shows no code for it.

## Sources

- Burfisher, M. E., *Introduction to Computable General Equilibrium Models*,
  3rd ed., Cambridge University Press: the exercises TBL65B, TBL94 and ME9A
  and their `.exp` files (`data/gtap7/nus333/`).
- Yuan, Z. and Burfisher, M. E. (2021), USITC working paper on GTAP closures.
- The GTAP Models pages of the Center for Global Trade Analysis, Purdue
  University (the standard closure of GTAP v7).
- McDougall, R., "GE mechanisms" (GTAP, the economics behind the standard
  closures).
- The WTO study of trade balance closures in GTAP.
