# SimpleOpen GAMS parity

Esta guia cubre la comparacion del benchmark canonico `simple_open_v1` contra su referencia GAMS.

## que compara

La paridad revisa, por closure:

- `benchmark`
- `level`
- `residual`
- `calib`

La lectura del `.gdx` usa `equilibria.babel`, no `gdxdump`.

## closures canonicas

- `simple_open_default`
- `flexible_external_balance`

## referencia oficial

Para generar la referencia oficial:

```bash
uv run python scripts/parity/generate_simple_open_gams_reference.py \
  --gams-bin /Library/Frameworks/GAMS.framework/Versions/48/Resources/gams \
  --output-dir output/simple_open_gams_reference/latest
```

Eso deja:

- `output/simple_open_gams_reference/latest/manifest.json`
- un `Results.gdx` por closure bajo `output/simple_open_gams_reference/latest/closures/<closure>/`

## prerequisito minimo

Primero generar los `.gdx` de referencia con GAMS:

```bash
cd src/equilibria/templates/reference/simple_open/scripts
/Library/Frameworks/GAMS.framework/Versions/48/Resources/gams simple_open_v1_benchmark.gms lo=2 --CLOSURE=simple_open_default --OUT_GDX=/abs/path/output/simple_open_v1_benchmark_default.gdx
/Library/Frameworks/GAMS.framework/Versions/48/Resources/gams simple_open_v1_benchmark.gms lo=2 --CLOSURE=flexible_external_balance --OUT_GDX=/abs/path/output/simple_open_v1_benchmark_flexible.gdx
```

## correr la paridad

Con los artefactos en `output/`:

```bash
uv run python scripts/parity/run_simple_open_gams_parity.py --gate --save-report output/simple_open_gams_parity.json
```

Con el manifest oficial:

```bash
uv run python scripts/parity/run_simple_open_gams_parity.py \
  --gate \
  --reference-manifest output/simple_open_gams_reference/latest/manifest.json \
  --require-reference-manifest \
  --save-report output/simple_open_gams_parity_from_manifest.json
```

Con rutas explicitas:

```bash
uv run python scripts/parity/run_simple_open_gams_parity.py \
  --gate \
  --gdx simple_open_default=/abs/path/default.gdx \
  --gdx flexible_external_balance=/abs/path/flexible.gdx
```

## criterio de pase

Cada closure debe cumplir:

- `active_closure_match = true`
- `modelstat = 1`
- `solvestat = 1`
- `benchmark`, `level`, `residual` y `calib` sin mismatches

## archivos clave

- [run_simple_open_gams_parity.py](/Users/marmol/proyectos/equilibria/scripts/parity/run_simple_open_gams_parity.py)
- [generate_simple_open_gams_reference.py](/Users/marmol/proyectos/equilibria/scripts/parity/generate_simple_open_gams_reference.py)
- [simple_open_parity_pipeline.py](/Users/marmol/proyectos/equilibria/src/equilibria/templates/simple_open_parity_pipeline.py)
- [simple_open_v1_benchmark.gms](/Users/marmol/proyectos/equilibria/src/equilibria/templates/reference/simple_open/scripts/simple_open_v1_benchmark.gms)
