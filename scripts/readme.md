# scripts index

organizacion actual:

- `scripts/cli/`: comandos operativos principales (calibrar, resolver, ejecutar flujos).
- `scripts/parity/`: comparaciones y auditorias de paridad (gams vs python).
- `scripts/qa/`: chequeos de calidad/diagnostico y reportes.
- `scripts/sam_tools/`: utilidades de transformacion y auditoria SAM (incluye CRI).
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

policy:

- por defecto, usar nombres de archivo en minusculas.
- enforcement automatizado: `python scripts/check_lowercase_filenames.py`.
