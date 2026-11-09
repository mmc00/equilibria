"""Tolerances of the ALTERTAX gates against GEMPACK, approved by the user 2026-11-09.

Relative ones are ``rel_diff`` with the gtap7 map's ``abs_floor`` of 1e-6:
``|a - b| / max(|b|, 1e-6)``. Every Phase B
test imports them from here. ``TOL_DECOMP`` is deliberately not defined: the decomposition
is measured first (Task 7) and the user sets it.
"""

TOL_LEVELS = 1e-4  # levels of sl4levels vs the family solution
TOL_UPDATED = 1e-4  # updated data (.upd)
TOL_SUMMARY = 1e-4  # summary tables
TOL_VOLUME = (
    1e-4  # SUPERSEDED for volume.har by TOL_CHANGE (kept so imports do not break)
)
# user 2026-11-09: keep the 1.01 stand-in for sigma=1; ~99% on changes is enough; worst measured
# 1.26% on nus333 volume. Relative on the change, for cells above 8 float32 ulps of the level.
TOL_CHANGE = 2e-2
# ABSOLUTE, in percentage points: GEMPACK % read straight from altax.sl4 (Task 8)
TOL_PCT = 1e-6
# check reproduces base: |check - base| <= CHECK_TOL * max(1, |base|)
CHECK_TOL = 5e-6
# floor of the relative metric ``rel_diff`` (every variable of compare/maps/gtap7.yaml)
ABS_FLOOR = 1.0e-6
