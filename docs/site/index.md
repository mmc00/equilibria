---
sd_hide_title: true
---

# equilibria

```{rubric} A modern Python framework for Computable General Equilibrium (CGE) modeling.
```

`equilibria` builds CGE models from modular equation blocks and checks every
template cell by cell against its reference implementation in GAMS and GEMPACK
(GTAP Standard 7, PEP-1-1). It reads and writes GEMPACK HAR files, reads GDX,
and turns national-accounts MIPs into balanced SAMs.

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

sections/getting_started
sections/templates
sections/data_solvers
sections/validation
gallery/index
sections/reference
```
