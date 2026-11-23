## ADDED Requirements

### Requirement: A uniform shock on an instrument
Source: change uniform-shock, MAR-227

`shocks.uniform(pct)` may replace the cell map of an instrument in a shock
(`{"imptx": uniform(3)}`). `apply_shock` and the family `apply_shock` must
expand it, on the built model, to every `shock` cell of that instrument that
can take a shock, each at `pct` with the instrument's GEMPACK kind, all or
nothing, and must raise `ValueError` when no cell can take it. In
`mode="gtap"`, `{"imptx": uniform(10)}` must write the same cells and values
as the default tariff shock.
