# scripts index

organizacion actual:

- `scripts/cli/`: comandos operativos principales (calibrar, resolver, ejecutar flujos).
- `scripts/parity/`: comparaciones y auditorias de paridad (gams vs python).
- `scripts/qa/`: chequeos de calidad/diagnostico y reportes.
- `scripts/sam_tools/`: utilidades de transformacion y auditoria SAM (incluye CRI).
- `scripts/dev/`: scripts de exploracion/depuracion tecnica.

compatibilidad:

- los nombres historicos en `scripts/*.py` se mantienen como wrappers para no romper
  workflows, docs o comandos previos.
- para uso nuevo, preferir rutas por categoria (por ejemplo `scripts/cli/run_solver.py`).

policy:

- por defecto, usar nombres de archivo en minusculas.
- enforcement automatizado: `python scripts/check_lowercase_filenames.py`.
