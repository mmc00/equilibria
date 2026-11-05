"""Family backend driver: the multiperiod GTAP run on arrays, without a Pyomo model.

Opt-in (``_solve_shock(..., backend="family")`` or ``EQUILIBRIA_GTAP_BACKEND=family``).
The route never falls back to Pyomo: a configuration it cannot run raises
``NotImplementedError`` naming it, before any work is done.

User shocks are not refused here: every ShockBlock instrument is applied through the
same ``apply_shock`` semantics as the Pyomo route.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

# Settings that change the Newton-TR system or add debug side effects the family route
# does not reproduce. Any of them set (non-empty) is refused.
_UNSUPPORTED_ENV: tuple[str, ...] = (
    "EQUILIBRIA_GTAP_REF_MODEL",
    "EQUILIBRIA_GTAP_JAC_DUMP",
    "EQUILIBRIA_GTAP_WARM_START",
    "EQUILIBRIA_GTAP_DIRECT_DEGEN_FIX",
    "EQUILIBRIA_GTAP_BTF_PROBE",
)


def check_supported(
    *,
    if_sub: bool,
    savf_flag: str,
    mode: str,
    skip_base_solve: bool,
    mute_welfare: bool,
    seed_from_prior: bool,
    solve_check: bool,
    shock_steps: tuple[float, ...] | None,
    shock_method: str,
    env: Mapping[str, str] | None = None,
) -> None:
    """Raise ``NotImplementedError`` naming the first setting the family route cannot run."""
    env = os.environ if env is None else env

    def refuse(what: str) -> None:
        raise NotImplementedError(f"family backend: {what} is not supported")

    if not if_sub:
        refuse("ifSUB=0 (closure.if_sub must be True)")
    if savf_flag != "capFix":
        refuse(f"savf_flag={savf_flag!r} (only 'capFix')")
    if mode != "gtap":
        refuse(f"mode={mode!r} (only 'gtap')")
    if not skip_base_solve:
        refuse("skip_base_solve=False (the base is not solved)")
    if seed_from_prior:
        refuse("seed_from_prior=True")
    if not mute_welfare:
        refuse("mute_welfare=False")
    if not solve_check:
        refuse("solve_check=False (the route runs base, check and shock)")
    if shock_steps:
        refuse("shock_steps")
    if shock_method != "nonlinear":
        refuse(f"shock_method={shock_method!r} (only 'nonlinear')")
    if env.get("EQUILIBRIA_GTAP_SOLVER") != "scipy_newton_tr":
        refuse(
            f"EQUILIBRIA_GTAP_SOLVER={env.get('EQUILIBRIA_GTAP_SOLVER')!r} "
            "(only 'scipy_newton_tr')"
        )
    linsolve = env.get("EQUILIBRIA_GTAP_TR_LINSOLVE", "direct").lower()
    if linsolve not in ("mumps", "direct"):
        refuse(f"EQUILIBRIA_GTAP_TR_LINSOLVE={linsolve!r} (only 'mumps' or 'direct')")
    if not env.get("EQUILIBRIA_GTAP_SOLVE_NLP"):
        refuse("EQUILIBRIA_GTAP_SOLVE_NLP unset (the NLP-square route is required)")
    for name in _UNSUPPORTED_ENV:
        if env.get(name):
            refuse(f"{name}")
    for name in sorted(env):
        if name.startswith("EQUILIBRIA_DEBUG_") and env.get(name):
            refuse(f"{name}")

    from equilibria.blocks.gtap.overwrite import overwrite

    if overwrite.active():
        refuse("@overwrite hooks")


def solve_gtap_family(
    params: Any, closure: Any, residual_region: str, *, ref_gdx: Any = None
) -> Any:
    """Run base, check and shock on arrays. Filled in by the later tasks."""
    raise NotImplementedError(
        "family backend: solve_gtap_family is not implemented yet"
    )
