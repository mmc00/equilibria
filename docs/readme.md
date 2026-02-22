# documentation index

este directorio se organiza por tipo de contenido:

- `docs/guides/`: guias de uso y flujos operativos.
- `docs/architecture/`: contratos y matrices de arquitectura/paridad.
- `docs/findings/`: hallazgos tecnicos y diagnosticos.
- `docs/plans/`: planes de trabajo y estabilizacion.
- `docs/reference/gdx/`: referencia tecnica del lector/encoder gdx.
- `docs/archive/`: notas historicas que se mantienen por trazabilidad.

documentos de entrada recomendados:

- guia de pipeline de paridad: `docs/guides/pep_systemic_parity_pipeline.md`
- hallazgo estructural cri: `docs/findings/finding_sam_ieem_vs_sam_pep.md`
- plan de estabilizacion: `docs/plans/equilibria_structural_stabilization_plan_2026-02-19.md`

policy de nombres de archivos:

- por defecto, usar nombres en minusculas.
- excepciones legacy/contractuales estan en `scripts/check_lowercase_filenames.py`.
- verificacion local: `python scripts/check_lowercase_filenames.py`.
