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
- guia de referencia oficial `GAMS + IPOPT + NLP`: `docs/guides/pep_gams_nlp_reference.md`
- guia del gate de escenarios PEP: `docs/guides/pep_core_scenarios_gate.md`
- guia de api de simulaciones: `docs/guides/simulations_api.md`
- guia de contrato/config de PEP: `docs/guides/pep_contract_api.md`
- guia de modos de Jacobiano en PEP: `docs/guides/pep_jacobian_modes.md`
- guia de la capa comun del Jacobiano: `docs/guides/model_jacobian_harness.md`
- guia del contrato canonico de SimpleOpen para paridad: `docs/guides/simple_open_parity_contract.md`
- guia de mapeo `stdcge -> SimpleOpen`: `docs/guides/simple_open_stdcge_mapping.md`
- capa generica de contratos/runtime: importar desde `equilibria` o `equilibria.contracts`
- segundo ejemplo sobre la capa base: `equilibria.templates.SimpleOpenEconomy`
- segundo benchmark/gate sobre la capa comun de Jacobiano: `scripts/parity/measure_simple_open_jacobian_modes.py`
- hallazgo estructural cri: `docs/findings/finding_sam_ieem_vs_sam_pep.md`
- plan de estabilizacion: `docs/plans/equilibria_structural_stabilization_plan_2026-02-19.md`

policy de nombres de archivos:

- por defecto, usar nombres en minusculas.
- excepciones legacy/contractuales estan en `scripts/check_lowercase_filenames.py`.
- verificacion local: `python scripts/check_lowercase_filenames.py`.
