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

policy:

- por defecto, usar nombres de archivo en minusculas.
- enforcement automatizado: `python scripts/check_lowercase_filenames.py`.
