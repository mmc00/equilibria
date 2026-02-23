# ieem to pep transformations (en)

English version. This explains what each transformation in `src/equilibria/sam_tools/ieem_to_pep_transformations.py` does, why it is needed, and a minimal before/after example.

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

## conservation note

Each transformation only reallocates values across cells. It does not create or destroy value. The SAM total is preserved; then `rebalance_ipfp` handles row/column closure adjustments if needed.
