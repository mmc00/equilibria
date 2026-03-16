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
*transformar SAM ahora usa el pipeline manual registrado en `scripts/sam_tools/run_manual_sam_pipeline.py` (ver documentos y ejemplos actualizados).* 

policy:

- por defecto, usar nombres de archivo en minusculas.
- enforcement automatizado: `python scripts/check_lowercase_filenames.py`.
