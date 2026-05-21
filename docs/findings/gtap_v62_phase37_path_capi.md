# GTAP v6.2 Phase 3.7 — PATH C-API integration + reconciliation rollback

**Date:** 2026-05-21
**Branch:** `gtap/v62-rollback`
**Builds on:** Phase 3.6 (square MCP closure, IPOPT shock pipeline)

## Summary

- Wired the GAMS PATH C-API (via `path_capi_python`) into
  `validate_v62_parity.py` as `--solver path-capi`. A user-supplied
  PATH evaluation license unlocks the full BOOK3X3 size (582 free
  vars vs the unlicensed 300-var demo limit).
- Rolled back the Phase 3.6 ``to = 0`` reconciliation. Empirically it
  inflates the dominant benchmark residual (eq_market) by ~20×, from
  ~2.3e4 to ~4.7e5, because the SAM calibrates the implicit output
  tax ``to`` to absorb the vom/vop wedge. Removing this fix restores
  the natural calibration.
- Documented why PATH terminates at term_code=2 (stationary point of
  the merit function) rather than code=1 on this model.

## What was added

- `scripts/gtap_v62/_path_capi_solver.py` — thin helper that loads the
  PATH runtime, builds Pyomo MCP callbacks via `PyomoMCPAdapter`, and
  writes the solution back into the Var objects.
- `validate_v62_parity.py shock --solver path-capi` — new solver branch
  with `--path-license`, `--path-lib`, `--lusol-lib` CLI args (also
  honoring `PATH_LICENSE_STRING`, `PATH_CAPI_LIBPATH`,
  `PATH_CAPI_LIBLUSOL` env vars). Adds a top-10 equation-residual
  diagnostic after baseline solve.

## Why ``to = 0`` was wrong

The Phase 3.6 hypothesis was that the SAM-implicit ``to = (vom/vop -
1)`` (a ~1% wedge in BOOK3X3) was a "fake" output tax that should be
zeroed out to absorb the SAM gap into the walras slack. The
diagnostic table tells a different story:

```
                       eq_qtm   eq_market   eq_qo   eq_pds
BEFORE closure:        6.6e4    2.3e4       6e-2    ≤1e-3
AFTER closure (raw):   6.6e4    2.3e4       6e-2    ≤1e-3
AFTER to=0 applied:    6.6e4    4.7e5       6e-2    7e-2
```

The ``to`` parameter participates in eq_market: zeroing it does not
"absorb" the wedge — it relocates the wedge from eq_qo (where it is
the documented imperfection ~6e-2) to eq_market (where it inflates the
already-present 1% SAM imbalance ~20×). The Phase 3.7 closure leaves
``to`` at its calibrated value.

## Why PATH terminates at term_code=2

After closure and squaring, the BOOK3X3 model has the following
dominant residuals at the SAM benchmark:

```
  eq_qtm                 max_abs = 6.58e+04  (intra-region VTWR, s==d)
  eq_market              max_abs = 2.28e+04  (~1% SAM imbalance)
  eq_qo                  max_abs = 6.18e-02  (output tax wedge)
  all others             max_abs ≤ 1e-3      (machine epsilon)
```

eq_qtm and eq_market are STRUCTURAL SAM imperfections — they cannot
be driven to zero without changing the dataset. PATH evaluates its
merit function and finds the gradient is locally zero at the SAM
benchmark (because the Jacobian rows for eq_qtm and eq_market dominate
the gradient and there is no nearby point that reduces them). PATH
declares termination code 2 (stationary point of the merit function),
even though residual norm is ~6e4.

The shocked solve inherits the warm start from the baseline and PATH
stops in ≤20 major iterations without moving the system. The Python
percent-changes come out as ~0% everywhere — a non-result rather
than a wrong result.

## IPOPT baseline without the ``to = 0`` fix

For reference, IPOPT (with the regularizer objective, no ``to``
override) does move the system, producing:

```
Cell                          GEMPACK %   Python %    pp diff
VIMS food USA->EU             +31.536    +44.681    +13.144
VIWS food USA->EU             +41.536    +60.756    +19.220
VXMD food USA->EU             +41.552    +60.749    +19.197
VDPM food EU                   -0.264     -0.185     +0.079
VIPM food EU                   +1.848     +2.457     +0.610
```

Bilateral US→EU food trade overshoots GEMPACK by ~19pp on world- and
market-price aggregates while domestic/imported food consumption in
EU (VDPM, VIPM) matches within 0.6pp. The overshoot in bilateral
trade likely reflects an Armington-elasticity or upper-bound
calibration that is too elastic relative to GEMPACK's parameterization;
this is a tuning concern beyond the IPOPT-vs-PATH choice and is
expected to be the focus of Phase 3.8.

## Reproduce

PATH C-API (requires user-supplied license + GAMS PATH dll/lusol):

```powershell
$env:PATH_LICENSE_STRING = "<your-license>"
$env:PATH_CAPI_LIBPATH = "C:/GAMS/53/path52.dll"
$env:PATH_CAPI_LIBLUSOL = "C:/GAMS/53/lusol.dll"
$env:PYTHONIOENCODING = "utf-8"
python scripts/gtap_v62/validate_v62_parity.py shock `
    --experiment Exp1a --solver path-capi `
    --workdir runs/gtap_v62_parity/BOOK3X3_Exp1a
```

IPOPT (works without GAMS license):

```powershell
python scripts/gtap_v62/validate_v62_parity.py shock `
    --experiment Exp1a --solver ipopt `
    --workdir runs/gtap_v62_parity/BOOK3X3_Exp1a
```

## Phase 3.8 — next

- Investigate eq_qtm and eq_market SAM imperfections — can a
  pre-balancing step (RAS / least-squares correction) clean them up
  so PATH can detect a feasible direction at term_code=1?
- Audit Armington upper-nest elasticity for the bilateral overshoot
  in VIMS/VIWS/VXMD (currently +13 to +19pp above GEMPACK).
- Consider applying GTAP v7's SAM-pre-balancing routine to BOOK3X3
  before feeding it into the v6.2 model.
