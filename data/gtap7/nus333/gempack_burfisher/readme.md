# GEMPACK solutions of Burfisher's exercises (nus333)

One run per exercise of Table 3, on `data/gtap7/nus333`, the book's package.
Each `.exp` is appended unchanged to the command file, with its `.prm`, its
closure and its method. Each `<exp>/` folder is one record (its `source.yaml`
says how and where it ran); `sltoht_burfisher/<exp>/` holds the `sltoht` dump of
the same solution.

| File | What it is |
|---|---|
| `solution.sl4` | the solution. It carries the pre (`LEVB`) and post (`LEVA`) levels of 44 variables |
| `updated.har` | the updated data; post `VDFB` comes from here |
| `run.cmf` | the command file that ran, with the method (`Method`/`Steps`) and the shocks |
| `solve_console.txt` | the GEMPACK console |
| `gtapv7/` | `gtapv7.exp` run with `gtapv7.for`, the condensed model. That model **omits 11 variables**, so it proves the arguments are read from the solution's list (`VCNA`/`VCAR`) and not the model's (`VCSP`/`VCSN`) (#148) |
| `gtapuv7/` | `gtapuv7.exp`: the same shock to the numeraire (`pfactwld` +10 %, one-step Johansen), with the uncondensed model like the rest. Its closure leaves exogenous the taxes that exist only in that model; in equilibria they are already parameters, so it is compared with the same run as GTAPV7 (homogeneity, §6.4 of the paper) |

`me8` and `tbl813` run their `-dir.exp` (see `scripts/gtap/mk_exp_directos.py`).

**Method.** The runs use Gragg, except six that use one-step Johansen because the
book solved them that way: TBL45A, TBL45B, TBL45C, TBL64, ME4A and ME4B.

**Variables without a level.** Without the patch of
`scripts/gtap/make_levels_tab.py`, `rore`, `y`, `pinv`, `qinv`, `pva` and `pe`
have no level in the `.sl4`. `gemsim_burfisher_gragg/` reruns the exercises with
that patch and with Gragg.

**Missing exercise.** TBL53 is not here: its `.exp` uses `target%`, which only
RunGTAP's GUI solves.

**Reading.** `compare_burfisher_gempack.py` and `tests/compare/_oracle.py`
(`burfisher_sl4`) find a run with
`reference("nus333", "gempack", experiment=<exp>, method="book")`.
