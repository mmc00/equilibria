# templates/gtap context

Vocabulary lives in the root `GLOSSARY.md`, not here.

## Dangerous files

- `src/equilibria/templates/gtap/gtap_multiperiod_driver.py`: orchestrates the
  three periods; the shock phase decides which rows are live and which values
  are patched after the solve.
- `src/equilibria/templates/gtap/instruments.py`: which rows read each
  instrument; a wrong entry lets a shock be lost with code=1.

## Commands

```bash
uv run pytest tests/templates/gtap tests/blocks/gtap -q
uv run pytest tests/templates/gtap/test_gtap7_mcp_parity.py  # needs PATH + gdxdump
```

## Risks

- A shock that never enters the solve still gives code=1: compare the moved
  cells against GAMS or GEMPACK, not only the pass rate at 1%.
