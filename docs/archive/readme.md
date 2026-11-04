# Archive

Documents kept for the record. Each one was written at the date shown, about the
code as it was then. They are **not maintained**: paths, APIs and numbers in them
may no longer match the repository. They are **not translated**: several are in
Spanish, as written. The living documents are in [`../notes/`](../notes/), the
decisions in [`../adr/`](../adr/).

Dates are the day the file entered the repository.

## Findings

| File | Date | Summary |
|---|---|---|
| `findings/gtap_baseline_inventory_2026-04-07.md` | 2026-04-21 | Inventory of the GTAP baseline before the GAMS parity work. |
| `findings/gtap_mcp_structural_gap_2026-04-10.md` | 2026-04-21 | The structural gap between the Python GTAP MCP and the GAMS one. |
| `findings/rungtap_welfare_parity_2026-05-15.md` | 2026-05-20 | Welfare parity against RunGTAP for NUS333 and 9x10. |
| `findings/gempack_welfare_not_cellwise_2026-07-23.md` | 2026-07-23 | Raw welfare `u` is not comparable cell by cell across engines. |
| `findings/gempack_linearization_study_2026-07-24.md` | 2026-07-24 | Status of the study of the residual against GEMPACK. |
| `findings/gempack_linearization_study_spec_2026-07-24.md` | 2026-07-24 | Design of that study. |
| `findings/gempack_linearization_study_plan_2026-07-24.md` | 2026-07-24 | Implementation plan of that study. |
| `findings/gempack_residual_is_linearization_2026-07-24.md` | 2026-07-24 | The residual against GEMPACK is linearization, not model scope or configuration. |
| `findings/gtap_ifsub_is_faithful_not_inverted_2026-07-24.md` | 2026-07-24 | `ifSUB` is faithful; the apparent inversion was a report-variable read. |
| `findings/f3_blocks_extraction_spec_2026-07-25.md` | 2026-07-29 | Design of F3: GTAP as composed symbolic blocks. |
| `findings/f3_blocks_extraction_plan_2026-07-25.md` | 2026-07-29 | Implementation plan of F3. |
| `findings/f3_blocks_done_2026-07-29.md` | 2026-07-29 | F3 done: the full parity matrix holds on blocks. |
| `findings/f3_5_base_calibrado_done_2026-07-30.md` | 2026-07-30 | F3.5 done: the settled point is adopted as the base, without the check. |
| `findings/gempack_fixture_closure_mislabeled_2026-08-02.md` | 2026-08-02 | The GEMPACK fixture closure was mislabeled; the GAMS-GEMPACK gap was a closure mismatch. |
| `findings/repo_cleanup_spec_2026-09-16.md` | 2026-09-16 | Design of the repository cleanup (Spanish). |
| `findings/repo_cleanup_plan_2026-09-16.md` | 2026-09-16 | Implementation plan of that cleanup (Spanish). |
| `findings/gempack_altertax_reference_howto_2026-09-21.md` | 2026-09-20 | How to produce the GEMPACK altertax reference in RunGTAP on Windows (Spanish). |
| `findings/finding_sam_ieem_vs_sam_pep.md` | 2026-02-22 | Differences between the IEEM and PEP SAMs for Costa Rica. |

## Plans

| File | Date | Summary |
|---|---|---|
| `plans/equilibria_gams_exact_replication_plan.md` | 2026-02-22 | Plan to replicate the GAMS models exactly. |
| `plans/equilibria_structural_stabilization_plan_2026-02-19.md` | 2026-02-22 | Plan to stabilize the structure and align the CRI baseline. |
| `plans/gtap_gams_parity_plan_2026-04-07.md` | 2026-04-21 | GTAP parity plan against GAMS. |
| `plans/mip_to_sam_implementation_plan.md` | 2026-04-10 | Implementation plan of the MIP-to-SAM conversion (Spanish). |
| `plans/gtap_calibration_mirror_plan_2026-04-17.md` | 2026-04-21 | Plan to mirror GTAP calibration against GAMS (Spanish). |
| `plans/gtap_altertax_implementation_plan_2026-05-13.md` | 2026-05-13 | Implementation plan of GTAP altertax. |
| `plans/gtap_welfare_decomp_cgebox_implementation_plan_2026-05-13.md` | 2026-05-13 | Implementation plan of the cgebox welfare decomposition extras. |
| `gtap_gams_mcp_parity_plan.md` | 2026-04-09 | Earlier GTAP parity plan against the GAMS MCP. |

## Analysis (Bolivia MIP)

| File | Date | Summary |
|---|---|---|
| `analysis/bolivia_mip_data_analysis.md` | 2026-04-16 | Analysis of the Bolivia input-output data. |
| `analysis/bolivia_mip_implementation_guide.md` | 2026-04-16 | Implementation guide for the Bolivia MIP. |
| `analysis/bolivia_mip_technical_report.md` | 2026-04-16 | Technical report on the Bolivia MIP. |
| `analysis/archive/bolivia_mip_final_solution.md` | 2026-04-16 | Final solution meeting the three constraints (Spanish). |
| `analysis/archive/mip_bolivia_analysis.md` | 2026-04-16 | Balance and conversion to a SAM (Spanish). |
| `analysis/archive/mip_bolivia_balanceo_final.md` | 2026-04-16 | Full balancing with GRAS (Spanish). |
| `analysis/archive/mip_bolivia_balancing_comparison.md` | 2026-04-16 | Comparison of balancing runs. |
| `analysis/archive/mip_bolivia_balancing_final_summary.md` | 2026-04-16 | Final balancing summary (Spanish). |
| `analysis/archive/mip_bolivia_balancing_issue.md` | 2026-04-16 | The targets problem and its fix (Spanish). |
| `analysis/archive/mip_bolivia_balancing_methods_comparison.md` | 2026-04-16 | Comparison of balancing methods (Spanish). |
| `analysis/archive/mip_bolivia_balancing_summary.md` | 2026-04-16 | Summary of the balancing problem (Spanish). |
| `analysis/archive/mip_bolivia_constraint_incompatibility.md` | 2026-04-16 | The three constraints are incompatible (Spanish). |
| `analysis/archive/mip_bolivia_df_analysis_final.md` | 2026-04-16 | Final-demand analysis (Spanish). |
| `analysis/archive/mip_bolivia_error_estructural.md` | 2026-04-16 | Diagnosis of a structural error (Spanish). |
| `analysis/archive/mip_bolivia_formulas_analysis.md` | 2026-04-16 | The Excel formulas behind the table (Spanish). |
| `analysis/archive/mip_bolivia_impuestos_analisis.md` | 2026-04-16 | Taxes, value added and GDP (Spanish). |
| `analysis/archive/mip_bolivia_reconstruction_analysis.md` | 2026-04-16 | Reconstruction and balancing analysis. |
| `analysis/archive/mip_bolivia_resumen_final.md` | 2026-04-16 | Executive summary (Spanish). |
| `analysis/archive/supply_demand_constraint_final_report.md` | 2026-04-16 | The supply-demand balance constraint (Spanish). |

## GDX reader

| File | Date | Summary |
|---|---|---|
| `gdx_decoder_analysis_summary.md` | 2026-02-22 | Summary of the GDX decoder analysis. |
| `gdx_parameter_delta_encoding_issue.md` | 2026-02-22 | The delta encoding of GDX parameters. |
| `gdx/delta_gdx.md` | 2026-02-22 | Notes on decoding delta-encoded records (Spanish). |
| `gdx/delta_gdx2.md` | 2026-02-22 | More notes on delta decoding. |
| `gdx/gdx_params.md` | 2026-02-22 | Notes on reading GDX parameters. |
| `gdx/6d_implementation.md` | 2026-02-22 | Support for 6-dimensional parameters. |
| `gdx/gdx_geometric_interpolation.md` | 2026-02-22 | Geometric interpolation of GDX values. |
| `gdx/multidim_implementation.md` | 2026-02-22 | Support for 3- to 5-dimensional parameters (Spanish). |
| `architecture/gams_parity_matrix.md` | 2026-02-22 | Early matrix of parity checks against GAMS. |

## Reports

| File | Date | Summary |
|---|---|---|
| `reports/gdx_comparison_report.md` | 2026-02-09 | GDX comparison between equilibria and cge_babel. |
| `reports/gdx_comparison_final_report.md` | 2026-02-09 | GDX value comparison, final. |
| `reports/validation_report_final.md` | 2026-02-09 | Validation of the GDX reader against cge_babel (Spanish). |
