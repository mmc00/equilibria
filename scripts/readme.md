# scripts index

organizacion actual:

- `scripts/cli/`: comandos operativos principales (calibrar, resolver, ejecutar flujos).
- `scripts/parity/`: comparaciones y auditorias de paridad (gams vs python).
- `scripts/qa/`: chequeos de calidad/diagnostico y reportes.
- `scripts/sam_tools/`: utilidades de transformacion y auditoria SAM (manual pipeline y helpers sobre `Sam`).
- `scripts/dev/`: scripts de exploracion/depuracion tecnica.

compatibilidad:

- los wrappers legacy en `scripts/*.py` (raiz) fueron retirados.
- usar rutas por categoria (por ejemplo `scripts/cli/run_solver.py`).
- la raiz `scripts/` se reserva para `scripts/check_lowercase_filenames.py`.

flujo actual recomendado:

- calibracion completa:
  - `uv run python scripts/cli/run_all_calibration.py`
- resolver base (pep2):
  - `uv run python scripts/cli/run_solver.py --method ipopt --init-mode equation_consistent`
- validar equilibrio calibrado:
  - `uv run python scripts/qa/verify_calibration.py`
- paridad completa con gams:
  - `uv run python scripts/parity/verify_pep2_full_parity.py --tol 1e-9 --presolve-gdx src/equilibria/templates/reference/pep2/scripts/PreSolveLevels.gdx`
- generar referencia oficial `GAMS + IPOPT + NLP`:
  - base sola: `uv run python scripts/parity/generate_pep_gams_nlp_reference.py --scenario base`
  - paquete core: `uv run python scripts/parity/generate_pep_gams_nlp_reference.py --core-scenarios`
  - modo `--skip-gams`: dejar `Results.gdx` por escenario en `output/gams_nlp_reference/latest/scenarios/<scenario>/scripts/`
  - el manifest oficial guarda referencias por escenario (`base`, `export_tax`, `import_price_agr`, `import_shock`, `government_spending`)
  - los `.gms` de referencia aceptan `--PEP_SOLVE_MODE=NLP` y `--PEP_SCENARIO=<...>`
- pipeline sistemico (gates y clasificacion):
  - `uv run python scripts/parity/run_pep_systemic_parity.py --sam-file src/equilibria/templates/reference/pep2/data/SAM-V2_0.gdx --val-par-file src/equilibria/templates/reference/pep2/data/VAL_PAR.xlsx --init-mode gams --method none --save-report output/pep_systemic_parity_report.json`
- escenarios pep2 base + export_tax con comparacion a gams:
  - `uv run python scripts/cli/run_pep_base_export_tax_parity.py --save-report output/pep_base_export_tax_parity.json`
  - nota: este script ahora usa internamente `equilibria.simulations.PepSimulator`.
  - para uso programatico, preferir API python directa (`equilibria.simulations`) sobre CLI.
- gate del paquete principal de escenarios pep:
  - `uv run python scripts/cli/run_pep_core_scenarios_gate.py --save-report output/pep_core_scenarios.json`
  - opcional: `--reference-manifest /abs/path/pep_core_scenarios_manifest.json --require-reference-manifest`
  - corre `base + export_tax + import_price_agr + import_shock + government_spending`
  - cada escenario corre desde la misma base calibrada para mantener la paridad determinista
- benchmark analytic vs numeric para el Jacobiano PEP:
  - `uv run python scripts/parity/measure_pep_jacobian_modes.py --reference-manifest output/gams_nlp_reference/core_default_real_v2/manifest.json --save-report output/pep_jacobian_modes_default.json`
  - variante gate estructural: agregar `--gate`
  - el gate exige:
    - convergencia en ambos modos
    - que `analytic` no use diferencias finitas por encima del maximo permitido
    - que la paridad de `analytic` no sea peor que la de `numeric`
- benchmark/gate no-PEP sobre la misma capa comun de Jacobiano:
  - `uv run python scripts/parity/measure_simple_open_jacobian_modes.py --gate --save-report output/simple_open_jacobian_modes.json`
  - corre dos closures (`simple_open_default`, `flexible_external_balance`)
  - valida que `analytic` use `0` diferencias finitas y que no sea peor que `numeric` contra la referencia analitica
  - complemento documental: `docs/guides/model_jacobian_harness.md`
- paridad GAMS del benchmark canonico de SimpleOpen:
  - `uv run python scripts/parity/run_simple_open_gams_parity.py --gate --save-report output/simple_open_gams_parity.json`
  - compara las dos closures canonicas contra sus `.gdx` de GAMS en `output/`
  - la lectura del `.gdx` usa `equilibria.babel`
  - complemento documental: `docs/guides/simple_open_gams_parity.md`
- generacion de la referencia oficial GAMS de SimpleOpen:
  - `uv run python scripts/parity/generate_simple_open_gams_reference.py --gams-bin /Library/Frameworks/GAMS.framework/Versions/48/Resources/gams --output-dir output/simple_open_gams_reference/latest`
  - deja un `manifest.json` oficial con las dos closures canonicas
  - el gate puede consumirlo con `--reference-manifest`
*transformar SAM ahora usa el pipeline manual registrado en `scripts/sam_tools/run_manual_sam_pipeline.py` (ver documentos y ejemplos actualizados).* 

policy:

- por defecto, usar nombres de archivo en minusculas.
- enforcement automatizado: `python scripts/check_lowercase_filenames.py`.
