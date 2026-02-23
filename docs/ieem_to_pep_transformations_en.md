# ieem to pep transformations (en)

English version. This explains the IEEM->PEP transformations in `src/equilibria/sam_tools/ieem_to_pep_transformations.py` with minimal before/after examples.

Note:
- `aggregate_mapping` stays generic (format-agnostic) in `src/equilibria/sam_tools/ieem_raw_excel.py`.
- `scale_all`, `scale_slice`, `rebalance_ipfp`, and `enforce_export_balance` remain in the workflow executor (`src/equilibria/sam_tools/executor.py`).

## context

In PEP, commodity supply is usually expected through activity accounts (`J`), and some fiscal flows must be routed to specific accounts (`AG.ti`).
An IEEM-style SAM can be fully valid from an accounting perspective, but still place flows in "doors" that PEP does not use in the same way.
These transformations re-route those flows without changing the aggregate total.

## symbols and notation glossary

- `A -> B`: flow from SAM row `A` into SAM column `B`.
- `*`: wildcard (all elements in a category), e.g. `K.*`.
- `i`: commodity placeholder (e.g. `agr`, `ser`, `food`).
- `map(i)`: commodity -> activity mapping (e.g. `food -> ind`).

Main categories:

- `I`: commodities (goods/services).
- `J`: production activities/sectors.
- `K`: capital factors.
- `L`: labor factors.
- `AG`: institutional/fiscal accounts (households, government, rest of world, taxes).
- `MARG`: trade/transport margins account.
- `OTH`: other macro accounts (investment, inventory change).

Frequent elements:

- `K.cap`: capital.
- `K.land`: land.
- `L.usk`: unskilled labor.
- `L.sk`: skilled labor.
- `AG.gvt`: government.
- `AG.row`: rest of world.
- `AG.ti`: indirect commodity taxes.
- `AG.tx`: export-related/alternate tax route not used here for `I` columns.
- `AG.td`: direct taxes.
- `AG.tm`: import tariffs/taxes.
- `MARG.MARG`: aggregate margins account.

## 1) move_k_to_ji

What it does:
- Moves commodity-column inflows coming from capital factors (`K.* -> I.i`) into activity supply rows (`J.map(i) -> I.i`).

Economic intuition:
- In PEP, goods are supplied by producing activities (`J`), not directly by the capital factor account.

Example:

Before:
- `K.cap -> I.agr = 50`
- `J.agr -> I.agr = 120`

After:
- `K.cap -> I.agr = 0`
- `J.agr -> I.agr = 170`

## 2) move_l_to_ji

What it does:
- Moves commodity-column inflows from labor factors (`L.* -> I.i`) into activity supply rows (`J.map(i) -> I.i`).

Economic intuition:
- Same logic as capital: labor does not directly "supply" commodities in PEP market-clearing channels.

Example:

Before:
- `L.usk -> I.ser = 30`
- `J.ser -> I.ser = 200`

After:
- `L.usk -> I.ser = 0`
- `J.ser -> I.ser = 230`

## 3) move_margin_to_i_margin

What it does:
- Reallocates `MARG.MARG -> I.i` into a chosen margin commodity row, typically `I.ser -> I.i`.

Economic intuition:
- Instead of leaving margins in an account not used as effective commodity supply in PEP, margins are routed through a commodity (typically services) that can enter PEP market channels.

Example:

Before:
- `MARG.MARG -> I.food = 12`
- `I.ser -> I.food = 8`

After:
- `MARG.MARG -> I.food = 0`
- `I.ser -> I.food = 20`

## 4) move_tx_to_ti_on_i

What it does:
- For commodity columns (`I.*`), moves `AG.tx -> I.i` into `AG.ti -> I.i`.

Economic intuition:
- In PEP, these commodity-related taxes are expected in the indirect tax account (`ti`) so fiscal identities close as specified by the model equations.

Example:

Before:
- `AG.tx -> I.agr = 9`
- `AG.ti -> I.agr = 4`

After:
- `AG.tx -> I.agr = 0`
- `AG.ti -> I.agr = 13`

## 5) apply_pep_structural_moves (legacy/composite)

What it does:
- Executes the four transformations above in sequence.

When to use:
- Mainly for backward compatibility.
- For full traceability, prefer explicit YAML steps with the four disaggregated operations.

## 6) balance_ras

What it does:
- Runs a RAS balancing pass over a RAW SAM to reduce row-vs-column gaps before account normalization.

Economic intuition:
- After aggregation, a RAW matrix can keep small inconsistencies.
- `balance_ras` distributes correction factors row/column-wise to recover accounting balance.

Key parameters:
- `ras_type`: `arithmetic` (default), `geometric`, `row`, `column`.
- `tol`, `max_iter`: numerical tolerance and maximum iterations.
- Implementation class: `src/equilibria/sam_tools/balancing.py` (`RASBalancer`).

Example:

Before:
- max row-column difference = `120`

After:
- max row-column difference = `0.000001` (approx, depending on tolerance)

## 7) normalize_pep_accounts

What it does:
- Converts aggregated RAW labels (`A-*`, `C-*`, `USK`, `GVT`, etc.) into the canonical PEP account structure (`J/I/L/K/AG/MARG/OTH`).

Economic intuition:
- PEP equations expect specific account doors.
- This step remaps the same value flows into those expected doors.

Example:

Before:
- `RAW.A-AGR -> RAW.C-AGR = 170`

After:
- `J.agr -> I.agr = 170`

## 8) create_x_block

What it does:
- Adds one `X.i` row/column for each commodity `I.i`.

Economic intuition:
- PEP trade logic separates domestic commodity flow (`I`) from the export channel (`X`).

Example:

Before:
- no `X.agr` account

After:
- `X.agr` row/column exists and can receive export flows

## 9) convert_exports_to_x

What it does:
- Moves export flows from `I.i -> AG.row` into `X.i -> AG.row`.
- Reallocates supply from `J -> I` to `J -> X` proportionally so output-use links stay consistent.

Economic intuition:
- This restores the expected PEP trade door: exports should leave through `X`, not directly from `I`.

## 10) align_ti_to_gvt_j

What it does:
- Moves fiscal entries `AG.ti -> J.*` into `AG.gvt -> J.*`.

Economic intuition:
- Keeps tax/government routing consistent with how the PEP calibration/identities read those flows.

## 11) scale_all and scale_slice (rescaling)

Where they live:
- `src/equilibria/sam_tools/executor.py`

What they do:
- `scale_all`: multiplies the full SAM by a factor.
- `scale_slice`: multiplies only one selected block (`row` x `col`) by a factor.

Economic intuition:
- Useful for unit conversion (e.g., millions to thousands) or controlled partial adjustments before rebalancing.

`scale_all` example:

Before:
- `J.agr -> I.agr = 100`

After with `factor=0.001`:
- `J.agr -> I.agr = 0.1`

`scale_slice` example:

Before:
- `AG.tm -> I.agr = 40`

After with `row=AG.tm`, `col=I.*`, `factor=1.1`:
- `AG.tm -> I.agr = 44`

## 12) rebalance_ipfp

Where it lives:
- `src/equilibria/sam_tools/executor.py` (calling helpers in `pep_sam_compat.py`)

What it does:
- Rebalances the matrix to reduce row/column closure gaps under a support mask (`support`) using IPFP.

Economic intuition:
- After flow moves, small row-vs-column gaps can remain.
- IPFP spreads that adjustment over allowed cells while preserving the intended structure.

Key parameters:
- `target_mode`: how row/column targets are defined (`geomean`, `average`, `original`).
- `support`: where adjustments are allowed (`pep_compat` or `full`).
- `epsilon`, `tol`, `max_iter`: numerical stability/tolerance controls.

Example:

Before:
- max row-column difference: `120`

After:
- max row-column difference: `0.000001` (approx)

## 13) enforce_export_balance

Where it lives:
- `src/equilibria/sam_tools/executor.py` (calling `enforce_export_value_balance`)

What it does:
- Enforces export value identity so the trade block remains accounting-consistent for PEP.

Economic intuition:
- If a small export-value gap remains after transformations/rebalancing, this step applies a final focused adjustment to close that identity.

Example:

Before:
- export value from supply side = `1000`
- export value from demand/price side = `997`

After:
- both sides = `1000` (within `tol`)

## conservation note

Structural transformations only reallocate values across cells; they do not create or destroy value. The SAM total is preserved; then `rebalance_ipfp` and `enforce_export_balance` are used to complete accounting and trade closure.
