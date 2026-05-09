# User guide

The guide is organized around the three main workflows in `equilibria`:

1. **Data preparation** — turning raw IO/MIP tables into a balanced SAM
   (`equilibria.sam_tools`).
2. **Model calibration & solution** — fitting a model template (PEP, GTAP)
   to a SAM and solving baselines or shocks.
3. **Validation** — comparing Python results against reference GAMS
   solutions to verify parity.

```{toctree}
:maxdepth: 1

installation
mip_to_sam
pep_quickstart
gtap_quickstart
path_capi
```

The {doc}`example gallery <../gallery/index>` complements these chapters with
short, runnable scripts that are re-executed on every documentation build.
