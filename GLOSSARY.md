# equilibria

equilibria builds computable general equilibrium (CGE) models in Python from
reusable blocks and checks them against their reference implementations in GAMS
and GEMPACK. This glossary fixes the words the code, specs and ADRs use.

## Model structure

**Block**:
A self-contained piece of a model that requires its sets and declares its
parameters, variables and equations for a single period, and lists the
variables it reads from other blocks.
_Avoid_: module, component

**Template**:
A complete model of a known type (GTAP, PEP, simple open economy) with its data,
closures and solve path. GTAP is assembled from blocks; PEP is still written as
a monolith and a Pyomo rebuild.
_Avoid_: model type, preset

**Monolith**:
A template's single-file implementation that writes every equation in one place;
a starting point, and what a parity gate may check against an oracle, until the
blocks replace it.
_Avoid_: legacy model, old model

**SP**:
The single-period model, built from the blocks with no time index; the
multi-period model is a copy of it per period.
_Avoid_: single model, static model

**Driver**:
The code that prepares each period of a multi-period run and solves it.
_Avoid_: runner, solver loop

**Recipe**:
The ordered list of preparation steps for one period in one mode.
_Avoid_: pipeline, procedure

## Closures and shocks

**Closure**:
The choice of which variables are exogenous and which are endogenous, including
the numeraire and the factor-market rules, that makes the model square.
_Avoid_: model setup, configuration

**Numeraire**:
The price fixed to anchor the price level (`pnum` in GTAP, the exchange rate `e`
in PEP).
_Avoid_: base price, anchor

**Instrument**:
An exogenous policy or technology variable of the shock block, fixed in every
period, that a shock moves.
_Avoid_: shock variable, lever

**Shock**:
The change of one or more instrument cells in the shock period relative to the
check period, written as the GEMPACK percentage.
_Avoid_: experiment, perturbation

**Exercise**:
One simulation, usually one notebook, that may change one block with
`@overwrite` for that exercise only.
_Avoid_: run

## Periods and solving

**Period**:
One of the time points a multi-period model solves: today `base`, `check` and
`shock`.
_Avoid_: step, stage

**Base**:
The period that holds the calibrated benchmark levels.
_Avoid_: benchmark period, t0

**Check**:
The period between base and shock, solved with the standard closure; the shock
is measured from it. It usually replicates the benchmark, but need not (a
baseline path moves it), and with base calibrated it is not solved.
_Avoid_: replication, validation period

**Seed**:
The starting values of the variables handed to the solver for a period.
_Avoid_: initial guess

**Base calibrated**:
The mode in which the base starts from a point the model itself has settled to,
rather than from the reference levels; required to match GEMPACK.
_Avoid_: settled base, self-calibrated

## Data

**SAM**:
A social accounting matrix: the square table of flows between accounts that a
model is calibrated from.
_Avoid_: IO table, MIP (an input-output table is a different object that a SAM
can be built from)

**HAR**:
GEMPACK's header array file format, which holds GTAP data and parameters.
GEMPACK solutions (`.sl4`) are a separate format with the same structure.
_Avoid_: header file

## Validation

**Oracle**:
The reference output a model is compared against: the output of GAMS or
GEMPACK.
_Avoid_: ground truth, golden file

**Parity gate**:
A test that compares a template against its oracle and fails below a fixed
floor; the GTAP gates must pass before a push that touches GTAP.
_Avoid_: parity test, regression test

## Specs

**Baseline**:
The description of what an architecture area does today, one requirement per
behavior.
_Avoid_: current spec

**Change**:
A proposed modification to one area, with its proposal and its deltas, open until
archived.
_Avoid_: proposal, RFC

**Delta**:
One requirement a change adds, modifies or removes in a baseline.
_Avoid_: diff, patch
