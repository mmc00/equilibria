---
sd_hide_title: true
---

# equilibria

```{rubric} A modern Python framework for Computable General Equilibrium (CGE) modeling.
```

`equilibria` provides modular equation blocks, universal data I/O for SAM/MIP/GDX,
and an auto-calibration pipeline. It targets parity with reference GAMS models
(GTAP Standard 7, PEP 1-1) while keeping the workflow Pythonic.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} 🚀 Get started
:link: guide/installation
:link-type: doc

Install `equilibria` and run your first model in five minutes.
:::

:::{grid-item-card} 📖 User guide
:link: guide/index
:link-type: doc

MIP → SAM closure, calibration, and shock analysis walk-throughs.
:::

:::{grid-item-card} 🧪 Gallery of examples
:link: gallery/index
:link-type: doc

Runnable scripts that produce plots and tables — re-executed on every build.
:::

:::{grid-item-card} 🔍 API reference
:link: api/index
:link-type: doc

Auto-generated reference for the public modules.
:::
::::

```{toctree}
:hidden:
:caption: Start here

guide/installation
```

```{toctree}
:hidden:
:caption: Templates

guide/gtap_quickstart
guide/pep_quickstart
guide/welfare_decomposition
```

```{toctree}
:hidden:
:caption: Data & solvers

guide/mip_to_sam
guide/har_io
guide/path_capi
```

```{toctree}
:hidden:
:caption: Validation & parity

guide/benchmarks
guide/gtap7_coverage_matrix
guide/pep_coverage_matrix
```

```{toctree}
:hidden:
:caption: Examples

gallery/index
```

```{toctree}
:hidden:
:caption: Reference

api/index
```
