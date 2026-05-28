# GEMPACK Gragg-multi 2-4-6 reference — gtap6_3x3, Food/USA/EU_28 -10% tariff

Reference GEMPACK solution used to validate Python implementation
(Phase 3.36–3.38). Files committed for reproducibility of parity
comparison; intermediate GEMPACK binaries (`.sl4`, `.slc`, log)
are regenerable from the `.cmf`.

## Files

| File | Purpose |
|:-----|:--------|
| `Shock1.cmf` | GEMPACK command file — defines the experiment |
| `Shock1-upd.har` | Post-shock SAM (HAR format) — used to read VIWS reference |
| `Shock1_sol.har` | Solution variables (HAR format) — used to read full result set |

## Reproduce

```bash
uv run python scripts/gtap_v62/run_gempack_generic.py \
  --workdir runs/gempack_3x3_food_usa_eu \
  --dataset-dir datasets/gtap6_3x3 \
  --shock-comm Food --shock-src USA --shock-dst EU_28 \
  --exp-name Shock1 \
  --steps "2 4 6"
```

Requires RunGTAP installation at `C:\runGTAP375\` (Windows) — see
`scripts/gtap_v62/run_gempack_oracle.py` for the toolchain detail.

## Reference results (read via equilibria.babel.har)

```
VIWS  (qxs × pmcif, CIF/world price)  Food[USA→EU_28]:  +62.3585%
VIMS  (qxs × pms,   agent price)      Food[USA→EU_28]:  +46.1227%
VXWD  (qxs × pe,    FOB)              Food[USA→EU_28]:  +62.3670%
VXMD  (qxs × pcif,  basic price)      Food[USA→EU_28]:  +62.3670%
```

Phase 3.38 IPOPT NLP (Python) produces `+62.4001%` on VIWS (gap +0.04 pp,
0.07% relative). See `docs/findings/gtap_v62_phase338_*.md`.
