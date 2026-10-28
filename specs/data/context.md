# Data context

Vocabulary lives in the root `GLOSSARY.md`, not here.

## Dangerous files

- `src/equilibria/sources.py`: the only reader of `source.yaml`; a looser
  schema lets a mislabelled reference back in (ADR-0010).
- `scripts/dev/check_layout.py`: its allow-lists decide what may sit at the
  root and which non-English names survive; edit them on purpose only.
- `src/equilibria/datasets.py`: `load_bundled` is public API used by the
  README and the quickstarts.

## Commands

```bash
uv run pytest tests/data tests/scripts/test_write_source.py tests/scripts/test_check_layout.py -q
python scripts/dev/check_layout.py
```

## Risks

- A reference read by path instead of `reference()` skips the variant check;
  that is how the `.nl` gate compared `ifSUB=0` against `ifSUB=1` fixtures.
- Moving a file is not regenerating it: only `write_source.py` running the
  command makes a result `verified`.
