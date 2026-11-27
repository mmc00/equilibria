# templates/manage context

Vocabulary lives in `docs/glossary.md`, not here.

## Dangerous files

- `src/equilibria/templates/manage/rescale.py` (planned): the table of the
  values GAMS rescaled before writing its GDX; a wrong entry puts one family
  off by a factor of about 2,529.
- `src/equilibria/templates/manage/steps.py` (planned): the 2015 start point and
  the exogenous 2015 cells, including the upstream `resLastYear.inc` quirks
  that must be kept.

## Commands

```bash
uv run pytest tests/templates/manage -q
uv run pytest tests/babel/gdx -q   # the GDX reader the reference needs
```

## Risks

- Python works in GAMS's solve units (GDP = 1e5); a value read from the GDX
  that is not converted, or converted twice, still solves.
- The reference does not replicate its own benchmark check (iterlim 0, the
  `tPaf`/`nshrGDP` start-point quirks); that is GAMS's behavior, not a bug to fix.
