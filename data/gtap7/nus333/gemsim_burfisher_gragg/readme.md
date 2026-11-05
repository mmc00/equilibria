# Burfisher's exercises with Gragg and the levels of six more variables

The same exercises as `../gempack_burfisher/`, rerun with two changes:

- **Gragg 2-4-6 in every exercise**, also in the six the book solves with
  one-step Johansen (TBL45A/B/C, TBL64, ME4A/B). A model in levels need not match
  a one-step linear extrapolation; with Gragg it is comparable.
- **The TAB with the levels patch** (`gtapuv7_levels.tab`, from
  `scripts/gtap/make_levels_tab.py`): `rore`, `y`, `pinv`, `qinv`, `pva` and `pe`
  have a level in the `.sl4`.

Each `<exp>/` folder is one record (its `source.yaml` says how and where it
ran). `sltoht_burfisher_gragg/<exp>/` holds the `sltoht` export of the same
solution: `sl4levels.har` (pre and post levels) and `sl4dump.har` (% changes).

| File | What it is |
|---|---|
| `solution.sl4` | the solution; without `PLEV`, so its levels are read from `sl4levels.har` |
| `updated.har` | the updated data; post `VDFB` comes from here |
| `run.cmf` | the command file that ran, with the method and the steps |
| `gemsim_console.txt` | the gemsim console |

`me8` and `tbl813` run their `-dir.exp` (as in `../gempack_burfisher/`).

**ME4A and ME4B have no folder** (D1, phase 3b). Gragg does not converge on the
book's shock and GEMPACK stops with "Accuracy too low"; the book's Johansen
solutions in `../gempack_burfisher/` are their oracle.

Read with `scripts/gtap/compare_burfisher_gempack.py --gempack-dir gragg`, or
`reference("nus333", "gemsim", experiment=<exp>, method="gragg")`.
