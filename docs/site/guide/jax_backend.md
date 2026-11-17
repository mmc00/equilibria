# How the JAX backend works

GTAP is built from blocks, like a Lego set. With those same blocks the model can be
solved down two roads: the usual one (Pyomo) and a newer, faster one (JAX, called
"family" in the code, under `src/equilibria/family/`). This page tells how the JAX
road works, first as a story and then with the code.

## The story in seven pictures

### 1. The blocks are bricks

```{raw} html
:file: ../../assets/jax-backend/stack.svg
```

GTAP is assembled from blocks, and each block is one part of the economy:
households, production, trade, government. Think of each block as a Lego brick,
with its equations inside.

### 2. The same bricks, two roads

```{raw} html
:file: ../../assets/jax-backend/same-brick.svg
```

With the same bricks the model can be built in two ways: with Pyomo, as always, or
with JAX, the new road. No brick was touched to add JAX. The only thing that changes
is what is done with them.

### 3. Clay and a mold

```{raw} html
:file: ../../assets/jax-backend/press.svg
```

JAX does not use the brick directly. It presses it into clay, and what is left is a
mold with the exact shape of each equation. This happens once, at the start.

### 4. One mold, a thousand equal pieces

```{raw} html
:file: ../../assets/jax-backend/one-mold.svg
```

```{epigraph}
"The equations are the same, but they depend on the sets. So instead of writing the
equations for every combination of sets, one mold is used for all those
combinations. That is how JAX does it."

-- Marlon Molina, 2026-11-17
```

Many equations have the same shape: the same calculation for every region and every
product, only with other numbers. With one mold they all come out at once, instead of
one by one. That is why JAX is fast. In the 20x41 database there are **210,256
single equations**, but only **202 molds**.

**Real examples: the equations that repeat the most.** GTAP database 20x41: 41
regions and 20 sectors. Counted on 2026-11-17 on the molds saved in the storage box
(`records.pkl`), with the code of `main`.

| Equation | What it says, in words | Copies | Molds |
|---|---|---:|---:|
| `eq_peeq` | The price at which each region sells each product to each other region | 33,620 | 1 |
| `eq_xweq` | How much of each product goes from each region to each other region | 33,620 | 5 |
| `eq_xda` | How much domestic product each buyer in each region purchases (the 20 sectors and the final buyers, such as households and government) | 19,680 | 3 |
| `eq_xma` | The same, for the imported product | 19,680 | 2 |
| `eq_paa` | The price each buyer pays for each product | 18,901 | 3 |
| `eq_xaa_activity` | How much of each input each sector uses to produce | 16,400 | 2 |
| **The whole model** (91 equations) | | **210,256** | **202** |

*Why some equations have more than one mold.* When a number is zero in the data (a
region that sells nothing to another one, for example), that part of the
calculation disappears and the shape changes a little. Each different shape is one
more mold.

*Where JAX wins.* Pyomo has 210,256 objects and, at every step of the engine, walks
them one by one. JAX has 202 molds, and at every step it pours each mold once
through its table: the 33,620 copies of `eq_peeq` come out of a single operation.
The derivative of each mold (for J) is also worked out once and holds for all its
copies. On the full 20x41 run that gives 1.35 min with JAX against 7.2 min with
Pyomo (measured 2026-11-06).

### 5. The same engine

```{raw} html
:file: ../../assets/jax-backend/engine.svg
```

Both roads hand the same thing to the same engine: how far each equation is from
holding, and which way to move. The engine (Newton with MUMPS) takes steps until
every equation holds. It does not know which road the information came from.

### 6. The storage box

```{raw} html
:file: ../../assets/jax-backend/storage-box.svg
```

Making the molds takes time; using them is fast. So they are kept in a box on disk,
and the next run takes them from there. Each drawer has a label. If the data or the
code change, the label no longer matches and the mold is made again.

### 7. The photo on the box

```{raw} html
:file: ../../assets/jax-backend/box-photo.svg
```

A Lego set comes with a photo of the finished model on the box. Here the photo is a
Pyomo run saved in the repo. JAX builds the model and compares it with the photo,
number by number. They have to match to the ninth decimal.

### What it does, and what not yet

What it already does:

- Full GTAP 7, from 3x3 to 20x41, with all its closures.
- Gives the same results as Pyomo and as GEMPACK.
- On 20x41 it takes 1.35 min, against 7.2 min with Pyomo (measured 2026-11-06).

What it does not do yet:

- Other models, such as PEP: they still go through Pyomo.
- Some GTAP options. If you ask for one it does not have, it says so before it
  starts.
- Use the graphics card (GPU): the mode exists, but it has not been measured.

## For developers

From here down is the detail: the code, the files and the tables.

### The pieces and their names in the code

Every piece in the drawings always means the same thing. The first seven come from
the guide to the blocks; the five with a dashed border belong to the JAX road.

```{raw} html
:file: ../../assets/jax-backend/parts.svg
```

### The big picture

```{raw} html
:file: ../../assets/jax-backend/big-picture.svg
```

At the top are the blocks, which do not change. At the bottom is the engine, which
does not change either. In between there are two ways to get the only two things the
engine needs: **F**, how far each equation is from holding, and **J**, how F changes
when each variable moves. Grey: shared by both roads. Blue: Pyomo only. Orange: JAX
only.

### The rule does not know who is calling it

Each equation of a block is a class with three things: a name, the sets it runs over
(`("r", "i", "a")`) and a function `build_expression(pyomo_model, indices)`. That
function asks an object `m` for variables and parameters and returns
`left == right`. The rule only cares that `m` answers the same questions. The Pyomo
road passes a Pyomo model; the JAX road passes a `RecordingModel` (the clay).

<!-- fragmento -->
```python
# src/equilibria/blocks/gtap/trade_armington_bilateral.py
# Demand for good i by activity a (CES of inputs)
class EqXaaActivity(SymbolicEquation):
    name: str = "eq_xaa_activity"
    domains: tuple = ("r", "i", "a")

    def build_expression(self, pyomo_model, indices):
        model = pyomo_model
        r, i, a = indices
        io_val = value(model.p_io[r, i, a])               # parameter -> number
        if io_val <= 0.0:
            return model.xaa[r, i, a] == 0.0
        sigmand = _get_sigmand(r, a)                      # elasticity -> number
        lambdaio = max(value(model.lambdaio[r, i, a]), 1e-8)
        return model.xaa[r, i, a] == (
            io_val
            * model.nd[r, a]                              # variable
            * (model.pnd[r, a] / model.pa[r, i, a]) ** sigmand
            * (lambdaio ** (sigmand - 1.0))
        )
```

```{raw} html
:file: ../../assets/jax-backend/two-objects.svg
```

Both backends walk the equations the same way. The call to `build_expression` is
identical in both; Pyomo keeps the result as a constraint, JAX keeps it as a tree
(the mold).

<!-- fragmento -->
```python
# Pyomo: backends/pyomo_backend.py, PyomoBackend._build_constraints (trimmed)
for eq_name in model.equation_manager.list_equations():
    eq = model.equation_manager.get(eq_name)
    constraint_dict = {}
    for indices in eq.get_indices(model.set_manager):
        expr = eq.build_expression(self.pyomo_model, indices)
        if expr is not None:
            constraint_dict[indices] = expr
```

<!-- fragmento -->
```python
# JAX: backends/family_backend.py, FamilyBackend.build (trimmed)
rec = RecordingModel.from_model(model)
for name in em.list_equations():
    eq = em.get(name)
    cells = []
    for idx in eq.get_indices(model.set_manager):
        res = eq.build_expression(rec, idx)
        if res is not None:
            cells.append((tuple(idx), res))
    if cells:
        self.records[name] = cells   # saved to disk
```

### Same questions, different answers

| The rule asks for | Pyomo gives it | The clay gives it |
|---|---|---|
| A set: `m.r` | a Pyomo Set | a list of names: `["USA", "CHN", ...]` |
| A parameter: `m.p_io[r,i,a]` | a Pyomo Param | the number itself |
| A variable: `m.nd[r,a]` | a Pyomo variable, with its value, bounds and fixed flag | a **leaf**: `Node("v", ("nd", 37, 1.42, ...))`, "I am cell 37 of `nd`" |
| Operations: `*  /  **  log  exp` | a Pyomo expression | a tree node (`mul`, `div`, `pow`...) |
| The result: `left == right` | an equality constraint | a `Residual(left, right)`, that is left minus right |

<!-- fragmento -->
```python
# family/recorder.py, RecordingModel.__getattr__ (trimmed): when the rule asks for m.something
def __getattr__(self, name):
    if name in self._sets:                 # set -> list
        return self._sets[name]
    if name in self._params:               # parameter -> numbers
        return _ParamView(name, ...)
    if name in self._vars:                 # variable -> leaves Node("v", ...)
        return _VarView(name, ...)

# Node overloads the operators to build the tree instead of computing:
class Node:
    def __mul__(self, o):  return _mul(self, o)     # a * b -> Node("mul", [a, b])
    def __truediv__(self, o): return _div(self, o)  # a / b -> Node("div", [a, b])
```

```{raw} html
:file: ../../assets/jax-backend/tree.svg
```

This is `eq_xaa_activity` for one cell. Orange: an operation. Solid box: a variable.
Dashed box: a fixed number. The numbers are illustrative. On 20x41 the recorder
writes about **5.9 million nodes** like these, from 89 equations (measured
2026-11-12). It happens once and is saved to disk, about 250 MB.

### A thousand equal trees, one mold

Cells with the same shape are grouped into one mold and a table with what changes in
each row. That group is a **family**, which gives the backend its name.

```{raw} html
:file: ../../assets/jax-backend/families.svg
```

The formula is shortened; the cells and columns are illustrative. `jacrev`
differentiates the mold exactly, with no finite differences.

<!-- fragmento -->
```python
# Pyomo: solver/path_capi.py (trimmed), pynumero reads the Pyomo model
nlp = PyomoNLP(model)        # writes .nl, opens ASL
F = nlp.evaluate_eq_constraints()
J = nlp.evaluate_jacobian_eq()   # sparse matrix
```

<!-- fragmento -->
```python
# JAX: family/core.py, one mold applied to all its rows
def _eval_part_impl(fn, x_all, cols, args):
    return jax.vmap(lambda c, a: fn(x_all[c], **a))(cols, args)

# its exact derivative, the same way
def _jac_part_impl(fn, x_all, cols, args):
    return jax.vmap(lambda c, a:
        jax.jacrev(lambda xs: fn(xs, **a))(x_all[c]))(cols, args)
```

<!-- fragmento -->
```python
# The same in both: solver/newton_tr.py
res = newton_tr(F, J, x0, linsolve="mumps")
# each step: MUMPS factors J, solves J dx = -F, keeps the step if ||F|| goes down
```

### A run from start to finish

```{raw} html
:file: ../../assets/jax-backend/flight.svg
```

One call, `solve_gtap_family(params, closure, ...)` in
`src/equilibria/family/driver.py`, runs the whole flight plan: base, check and shock.
It never builds a Pyomo model. The base is not solved; check and shock are. The
stages, in order (JAX only unless marked "shared"):

1. **Refuse what it cannot do.** Before any work, `check_supported` stops on the
   first setting the road does not run. (`src/equilibria/family/driver.py`)
2. **Read the data and the closure** (shared). The same `GTAPParameters` and
   `GTAPClosureConfig` as Pyomo. For capFlex, the risk factor is read first from the
   GAMS reference GDX. (`src/equilibria/templates/gtap`,
   `src/equilibria/family/capflex.py`)
3. **Make the molds, or take them from the box.** If the box already has them for
   these exact data and code, they are loaded. (`src/equilibria/backends/family_backend.py`,
   `src/equilibria/family/recorder.py`, `src/equilibria/family/disk_cache.py`)
4. **Compose the starting state** (shared). The starting value of every variable,
   with the same code as Pyomo, kept in the box. (`src/equilibria/family/state.py`)
5. **Record the rows that join periods.** The Fisher and Tornqvist indices between
   one period and the next, as one more family. (`src/equilibria/family/cross_period.py`)
6. **Prepare each period.** On arrays: seed values, freeze what the period keeps
   fixed, set bounds, mute welfare and, on the shock, apply the shock.
   (`src/equilibria/family/prep_recipe.py`, `src/equilibria/family/shock.py`)
7. **Square the system.** The closure fixes some variables and each equation is
   paired with one unknown, from the pattern of J. The pairing is kept in the box too.
   (`src/equilibria/family/prep_square.py`, `src/equilibria/family/squared_system.py`,
   `src/equilibria/solver/_matching.py`)
8. **Compute F and J.** Each mold goes through its table with `vmap`, and `jacrev`
   gives the exact J. The compiled programs are kept in the box.
   (`src/equilibria/family/core.py`, `src/equilibria/family/aot_cache.py`)
9. **Start the engine** (shared). Newton with MUMPS, ICNTL(6)=7, fixed for this road.
   A period counts as solved (code 1) only if Newton succeeds and the largest residual
   is below 1e-5. (`src/equilibria/solver/newton_tr.py`)
10. **Write the results back.** Write the solution, recompute the report variables,
    finish welfare and freeze the period for the next one.
    (`src/equilibria/family/post_solve.py`)

The settings the road needs:

```text
EQUILIBRIA_GTAP_SOLVER=scipy_newton_tr   # the Newton engine
EQUILIBRIA_GTAP_SOLVE_NLP=1              # the square-system road
EQUILIBRIA_GTAP_TR_LINSOLVE=mumps        # or "direct" (the default); MUMPS needs pymumps
EQUILIBRIA_FAMILY_CACHE=~/.cache/equilibria/family   # optional: where the box lives
```

### Shared bricks, JAX-only tools

The economy is shared. What JAX adds are the tools around it. The split is written
in the code: `src/equilibria/blocks/gtap/fingerprint.py` lists the Pyomo-only files
and treats all of `src/equilibria/family/` as the JAX road.

| Shared | Pyomo only | JAX only |
|---|---|---|
| The block equations, `src/equilibria/blocks/gtap` | `src/equilibria/backends/pyomo_backend.py`, `src/equilibria/backends/pyomo_equations.py` | `src/equilibria/family/recorder.py`, `src/equilibria/backends/family_backend.py` |
| Data, closure and starting state, `src/equilibria/templates/gtap` | The multiperiod driver, `src/equilibria/templates/gtap/gtap_model_multiperiod.py`, `src/equilibria/templates/gtap/period_prep.py` | The period steps on arrays: `src/equilibria/family/prep_recipe.py`, `src/equilibria/family/prep_square.py`, `src/equilibria/family/post_solve.py` |
| The engine, `src/equilibria/solver/newton_tr.py`, `src/equilibria/solver/_matching.py` | PATH and ASL, `src/equilibria/solver/path_capi.py`, `src/equilibria/solver/jacobians` | F and J, `src/equilibria/family/core.py`, `src/equilibria/family/aot_cache.py` |
| The shock instruments, `src/equilibria/blocks/gtap/shock.py` | ALTERTAX on Pyomo, `src/equilibria/templates/gtap/altertax` | ALTERTAX on JAX, `src/equilibria/family/altertax` |

| Step around the math | In Pyomo | In JAX |
|---|---|---|
| Prepare each period (fix, seed, bounds) | `PeriodPreparer` | `src/equilibria/family/prep_recipe.py`, with NumPy masks |
| Pair each equation with an unknown | from the Pyomo expressions | `src/equilibria/family/squared_system.py`, from the pattern of J |
| Row scale (when it has converged) | `src/equilibria/solver/path_capi.py` | `src/equilibria/family/fscale.py`, with the same number |
| After solving | the multiperiod driver | `src/equilibria/family/post_solve.py` |
| Fisher indices between periods | `build_equations_fisher` | `src/equilibria/family/cross_period.py` |

`tests/family/test_cross_period.py` compares the JAX cross-period rows with the Pyomo
ones, row by row.

```{raw} html
:file: ../../assets/jax-backend/time-bar.svg
```

### The drawers of the box

The box lives in `~/.cache/equilibria/family`, or wherever `EQUILIBRIA_FAMILY_CACHE`
points. A drawer with the wrong label is never used: it is filled again.

| Drawer | Where, inside the box | Made again when... |
|---|---|---|
| The molds (recorded equations) | `<key>/records.pkl`, `cross_records.pkl` | the data, the closure, the residual region, the code of the blocks, templates, solver or JAX road change, or the versions of Python, JAX, NumPy, SciPy, Pyomo or pymumps. With an `@overwrite` hook active, nothing is stored. |
| The starting state | `<key>/state.pkl` | the same as the molds (same label). |
| The pairing | `<key>/pairing-<period>-<digest>.npy` | the molds, the rows, the columns or the matcher code change. It is checked before use. |
| Compiled programs (ours) | `jax_aot/<kind>-<hash>.aot` | the mold's shape, JAX, the machine or its CPU, the XLA flags, or the code of `src/equilibria/family/recorder.py`, `src/equilibria/family/core.py` or `src/equilibria/family/aot_cache.py` change. A damaged file is compiled again with a warning. |
| Compiled programs (JAX's own) | `jax/` | JAX decides, with its own label. |
| Unpacked photos | `photos/<dataset>_<folder>_<sha>/` | the photo saved in the repo changes. |
| The settled starting point (optional) | `~/.cache/equilibria/settled_seed`, only with `EQUILIBRIA_FAMILY_SETTLE_CACHE=1` | the data, the closure, the reference GDX, the code, the Newton tolerances or any other `EQUILIBRIA_*` setting change. |

In CI the box lives in `$RUNNER_TEMP/family-cache` and goes from one run to the next
with the key `family-python-<os>-<py and jax versions>-<group>-<run id>`
(`.github/workflows/tests.yml`). Since every drawer checks its own label, using an
older box is safe.

### The photos on the box

A photo is a Pyomo run saved in `data/gtap7/<dataset>/equilibria_pyomo-photo_<tag>/`:
the final value of every variable, with its bounds and fixed flag. It is read by
`src/equilibria/family/photos.py`. The tests do not run Pyomo: they compare against
the photo. When the model changes, the photo is taken again with
`scripts/family/regen_photos.py`, which also checks it against GAMS and GEMPACK.

| Dataset | Photos | What is saved | Where it runs |
|---|---|---|---|
| gtap7_3x3 | ifSUB=1 capFix, ifSUB=0 capFix, ifSUB=1 capFlex | every step | every PR, and before every push |
| gtap7_10x7 | ifSUB=1 capFix | every step | every PR |
| gtap7_15x10 | ifSUB=1 capFix | final state only | every PR: 1.44 min cold, 0.29 min warm, 2.8 GB |
| gtap7_20x41 | ifSUB=1 capFix | final state only | on the Mac only: 6.6 min, 3.9 GB |

Every photo is a MUMPS run; a run with another linear solver is refused.
`tests/family/test_photo_freshness.py` fails when a photo is older than the code it
describes. 20x41 stays on the Mac because on a GitHub runner it went past 11 GB of
memory in 5 min and was stopped (run 44001448484, 2026-11-16). Measurements are from
the phase 3 work and PR #257 (2026-11-16) unless they give another date.

| Comparison | Result | Time |
|---|---|---|
| JAX against Pyomo, GTAP 20x41 (capFix closure) | every cell within 1e-9 | 1.35 min against 7.2 min (2026-11-06) |
| ALTERTAX against GEMPACK/RunGTAP: 8 datasets, 6 outputs | 8 of 8 pass | 20x41: 95 to 119 s, GEMPACK 116 s (2026-11-12) |
| The whole CI, with the JAX tests in it | green | 6.4 min, was 8.2 (PR #257) |

### What it does not do yet, in detail

`check_supported` in `src/equilibria/family/driver.py` refuses these before it
starts, with a message that names the setting:

- capFlex with ifSUB=0, or a closure other than capFix and capFlex.
- Models other than GTAP.
- Solving the base, seeding from a previous run, skipping the check, or leaving
  welfare on.
- Shocks in steps, or a shock method other than "nonlinear".
- A solver other than the Newton engine, or a linear solver other than MUMPS or
  "direct".
- Choosing MUMPS's ICNTL(6): the road fixes it at 7.
- Debug and probe settings (`EQUILIBRIA_DEBUG_*`, warm starts, NEOS and others) and
  `@overwrite` hooks.

### Ideas for later

Ideas written down to study. None is started or approved.

**Compile once for any dataset, like GEMPACK's gtap.exe.** GEMPACK builds its
program once and reads the sizes at run time. We compile one version per dataset. We
could compile by "size class" instead.

- *What it would gain:* only the first run with a new dataset. On 15x10 the first
  run took 42 s and the second 8.9 s.
- *Obstacle 1:* what is compiled depends on the data, not only on the size, because
  terms with cells at 0 are dropped. gtap7_3x3 and nus333 share only 14 of about 330
  compiled pieces.
- *Obstacle 2:* padding up to GTAP's maximum does not work: a 3x3 would load a model
  thousands of times larger.
- *First step:* measure how much slower 15x10 gets if the zeros are not dropped.

(Measured 2026-11-12 on the maintainer's Mac, with an empty compilation cache.)

The figures are drawn by `scripts/dev/export_jax_doc_svgs.mjs`; the numbers inside
the drawings are illustrative, the numbers in the tables are measurements with their
dates. Code excerpts are trimmed from `main` (abc6f500, 2026-11-16).
