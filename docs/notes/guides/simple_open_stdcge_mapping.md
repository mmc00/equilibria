# SimpleOpen stdcge mapping

This guide pins how we use `stdcge.gms` as the official anchor for `SimpleOpen`.

Official reference:

- `stdcge.gms : A Standard CGE Model`
- https://www.gams.com/latest/gamslib_ml/libhtml/gamslib_stdcge.html

## Scope decision

We are not copying the whole of `stdcge`.

We are using a minimal, explicit reduction of `stdcge` for the contract:

- `simple_open_v1`

That means:

- `stdcge` is the conceptual and algebraic anchor
- `simple_open_v1` is a small, controlled reduction
- the new `.gms` is a `stdcge-reduced benchmark`, not a clone of the whole model

## Which parts of stdcge we do use

### 1. Transformation block

In `stdcge`:

- `eqpzd`
- `eqE`
- `eqDs`

In `SimpleOpen` we use the idea of a CET transformation between:

- the domestic good `D`
- exports `E`
- the exchange rate `ER/PFX`

That reduces to:

- `EQ_CET`

## 2. External prices block

In `stdcge`:

- `eqpe`
- `eqpm`
- `epsilon`

In `SimpleOpen` we do not model the whole trade block.

We take only the idea of:

- `ER`
- `PFX`

as exogenous prices of the benchmark.

## 3. Domestic / absorption block

In `stdcge`:

- `Q`
- `M`
- `D`
- `eqpqs`
- `eqM`
- `eqD`
- `eqpqd`

In `SimpleOpen` we do not keep the whole Armington block.

We keep only:

- `D`

as an exogenous input of the reduced benchmark.

## 4. Value added block

In `stdcge`:

- `Y`
- `F`
- `py`
- `pf`

In `SimpleOpen` it reduces to a minimal aggregate:

- `EQ_VA`

The idea is to keep a small CES aggregate between:

- `ER`
- `PFX`

That is not a literal copy of the production block of `stdcge`, but it does follow its spirit of an aggregate block with calibrated parameters.

## 5. Reduced intermediate identity

`EQ_INT` does not come from a single literal equation of `stdcge`.

It is a reduced identity that captures a small layer of internal consistency between:

- `INT`
- `X`
- `CAB`
- `FSAV`

It closes the minimal `SimpleOpen` benchmark without bringing in the whole absorption/investment/government machinery of `stdcge`.

## Which parts of stdcge we do not use

For now these stay out:

- household utility and demand
- government consumption
- investment
- direct and production taxes
- imports `M`
- the Armington composite good `Q`
- complete factor markets
- the complete macro balance

That is on purpose. The current goal of the epic is not to reproduce the whole of `stdcge`, but to give `SimpleOpen` a minimal, clean GAMS reference.

## Outcome of this decision

The first GAMS artifact that comes out of this anchor is:

- [simple_open_v1_benchmark.gms](../../../src/equilibria/templates/reference/simple_open/scripts/simple_open_v1_benchmark.gms)

That script:

- uses `stdcge` as the conceptual reference
- respects the canonical `SimpleOpen` contract
- solves only the reduced benchmark
- exports levels and residuals to GDX

## Why we do not copy stdcge literally

Because today `SimpleOpen` does not yet have:

- the same number of blocks
- the same macro closure
- the same set of variables

If we copied the whole of `stdcge`, it would stop being `SimpleOpen` and become another model.

The right decision here is:

- official anchor: `stdcge`
- own implementation: `simple_open_v1`
- an explicit mapping between the two

## Next cut

With this mapping pinned, the next step is:

1. run the benchmark `.gms`
2. read the GDX from Python
3. set up the first Python vs GAMS comparator for `SimpleOpen`
