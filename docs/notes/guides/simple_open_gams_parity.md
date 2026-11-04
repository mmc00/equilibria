# SimpleOpen GAMS parity

This guide covers the comparison of the canonical `simple_open_v1` benchmark against its GAMS reference.

## What it compares

Parity checks, per closure:

- `benchmark`
- `level`
- `residual`
- `calib`

The `.gdx` is read with `equilibria.babel`, not `gdxdump`.

## Canonical closures

- `simple_open_default`
- `flexible_external_balance`

## Official reference

To generate the official reference:

```bash
uv run python scripts/parity/generate_simple_open_gams_reference.py \
  --gams-bin "$(command -v gams)" \
  --output-dir output/simple_open_gams_reference/latest
```

That leaves:

- `output/simple_open_gams_reference/latest/manifest.json`
- one `Results.gdx` per closure under `output/simple_open_gams_reference/latest/closures/<closure>/`

## Minimum prerequisite

First generate the reference `.gdx` files with GAMS:

```bash
cd src/equilibria/templates/reference/simple_open/scripts
gams simple_open_v1_benchmark.gms lo=2 --CLOSURE=simple_open_default --OUT_GDX=/abs/path/output/simple_open_v1_benchmark_default.gdx
gams simple_open_v1_benchmark.gms lo=2 --CLOSURE=flexible_external_balance --OUT_GDX=/abs/path/output/simple_open_v1_benchmark_flexible.gdx
```

## Running the parity

With the artifacts in `output/`:

```bash
uv run python scripts/parity/run_simple_open_gams_parity.py --gate --save-report output/simple_open_gams_parity.json
```

With the official manifest:

```bash
uv run python scripts/parity/run_simple_open_gams_parity.py \
  --gate \
  --reference-manifest output/simple_open_gams_reference/latest/manifest.json \
  --require-reference-manifest \
  --save-report output/simple_open_gams_parity_from_manifest.json
```

With explicit paths:

```bash
uv run python scripts/parity/run_simple_open_gams_parity.py \
  --gate \
  --gdx simple_open_default=/abs/path/default.gdx \
  --gdx flexible_external_balance=/abs/path/flexible.gdx
```

## Pass criterion

Each closure must meet:

- `active_closure_match = true`
- `modelstat = 1`
- `solvestat = 1`
- `benchmark`, `level`, `residual` and `calib` with no mismatches

## Key files

- [run_simple_open_gams_parity.py](../../../scripts/parity/run_simple_open_gams_parity.py)
- [generate_simple_open_gams_reference.py](../../../scripts/parity/generate_simple_open_gams_reference.py)
- [simple_open_parity_pipeline.py](../../../src/equilibria/templates/simple_open_parity_pipeline.py)
- [simple_open_v1_benchmark.gms](../../../src/equilibria/templates/reference/simple_open/scripts/simple_open_v1_benchmark.gms)
