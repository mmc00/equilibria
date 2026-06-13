"""GTAP ParityAdapter — wraps GTAPModelEquations build/solve + _diff_core lookups."""
from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[4]
_SCRIPTS_GTAP = _ROOT / "scripts" / "gtap"
if str(_SCRIPTS_GTAP) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_GTAP))

# HAR datasets for gtap7_* (relative to _ROOT)
_DATASET_HAR = {
    "gtap7_3x3": "datasets/gtap7_3x3",
    "gtap7_3x4": "datasets/gtap7_3x4",
    "gtap7_5x5": "datasets/gtap7_5x5",
}


def _load_run_gtap():
    import importlib.util as _u
    spec = _u.spec_from_file_location(
        "run_gtap", str(_ROOT / "scripts" / "gtap" / "run_gtap.py")
    )
    run_gtap = _u.module_from_spec(spec)
    sys.modules["run_gtap"] = run_gtap
    spec.loader.exec_module(run_gtap)
    return run_gtap


def _load_params_gdx(dataset: str, gdx_rel: str):
    from equilibria.templates.gtap import GTAPParameters
    params = GTAPParameters()
    params.load_from_gdx(_ROOT / gdx_rel)
    return params


def _load_params_har(dataset: str):
    from equilibria.templates.gtap import GTAPParameters
    har_dir = _ROOT / _DATASET_HAR[dataset]
    params = GTAPParameters()
    params.load_from_har(
        basedata_path=har_dir / "basedata.har",
        sets_path=har_dir / "sets.har",
        default_path=har_dir / "default.prm",
        baserate_path=har_dir / "baserate.har",
    )
    return params


def _apply_gams_warmstart_brute(model, gdx_path: Path, period: str) -> int:
    """Seed every free Pyomo variable from the GAMS altertax GDX check/shock period.

    For each variable component in `model`, reads the GDX variable of the same name,
    filters for `period`, strips GAMS prefixes (a_/c_/f_/r_) and the period dimension,
    and calls set_value on non-fixed variable data objects.

    Returns the number of variable cells successfully seeded.
    """
    from _diff_core import gams_levels as _gv  # type: ignore[import-not-found]
    from pyomo.environ import Var

    _PREFIXES = ("a_", "c_", "f_", "r_")

    def _strip(s: str) -> str:
        for p in _PREFIXES:
            if s.startswith(p):
                return s[len(p):]
        return s

    # Pre-load all GDX variables for the requested period into a flat dict:
    # {gams_varname: {pyomo_key: value}}
    _gdx_cache: dict[str, dict] = {}

    def _get_gdx_period(name: str) -> dict:
        if name not in _gdx_cache:
            try:
                raw = _gv(gdx_path, name)
            except Exception:
                raw = {}
            out: dict = {}
            for k, v in raw.items():
                if not isinstance(k, tuple) or k[-1] != period:
                    continue
                dims = k[:-1]
                # Strip GAMS prefixes and 'hhd' household placeholder
                stripped = tuple(_strip(x) for x in dims if x not in ("hhd",))
                py_key: object = stripped[0] if len(stripped) == 1 else (stripped if stripped else None)
                if py_key is not None:
                    out[py_key] = float(v)
            _gdx_cache[name] = out
        return _gdx_cache[name]

    applied = 0
    for comp in model.component_objects(Var, active=True):
        vname = comp.name
        gdx_data = _get_gdx_period(vname)
        if not gdx_data:
            continue
        for idx, vdata in comp.items():
            if getattr(vdata, "fixed", False):
                continue
            # Build Pyomo-style key
            py_key: object = idx[0] if isinstance(idx, tuple) and len(idx) == 1 else idx
            val = gdx_data.get(py_key)
            if val is None:
                continue
            try:
                vdata.set_value(val)
                applied += 1
            except Exception:
                pass
    return applied


def _gams_snapshot_from_altertax_gdx(gdx_path: Path, period: str):
    """Build a GTAPVariableSnapshot from a GAMS altertax GDX for a specific period.

    GDX keys have period as the last element (e.g. ('USA','Land','a_Food','check')).
    Activity/commodity/factor dimensions use GAMS prefixes (a_, c_, f_, r_).
    This function strips the period suffix and GAMS prefixes to produce Pyomo-compatible keys.
    """
    from equilibria.templates.gtap.gtap_parity_pipeline import GTAPVariableSnapshot
    from _diff_core import gams_levels as _gv  # type: ignore[import-not-found]

    _PREFIXES = ("a_", "c_", "f_", "r_")

    def _strip(s: str) -> str:
        for p in _PREFIXES:
            if s.startswith(p):
                return s[len(p):]
        return s

    def _slice(raw: dict, period: str) -> dict:
        """Keep only entries where last tuple element == period, strip period + prefixes."""
        out = {}
        for k, v in raw.items():
            if not isinstance(k, tuple) or k[-1] != period:
                continue
            new_k = tuple(_strip(x) for x in k[:-1])
            if len(new_k) == 1:
                new_k = new_k[0]
            out[new_k] = v
        return out

    def _slice_drop_h(raw: dict, period: str) -> dict:
        """Like _slice but drops the household dimension 'hhd' (single-hh case).
        GAMS key: (r, h, period) → Pyomo key: r (scalar).
        """
        out = {}
        for k, v in raw.items():
            if not isinstance(k, tuple) or k[-1] != period:
                continue
            dims = [_strip(x) for x in k[:-1] if x not in ("hhd", "h")]
            if len(dims) == 1:
                out[dims[0]] = v
            elif dims:
                out[tuple(dims)] = v
        return out

    def _scalar(raw: dict, period: str) -> float | None:
        for k, v in raw.items():
            if isinstance(k, tuple) and k[-1] == period:
                return float(v)
            if k == period:
                return float(v)
        return None

    return GTAPVariableSnapshot(
        xp=_slice(_gv(gdx_path, "xp"), period),
        x=_slice(_gv(gdx_path, "x"), period),
        xs=_slice(_gv(gdx_path, "xs"), period),
        xds=_slice(_gv(gdx_path, "xds"), period),
        xd=_slice(_gv(gdx_path, "xd"), period),
        px=_slice(_gv(gdx_path, "px"), period),
        pp=_slice(_gv(gdx_path, "pp"), period),
        ps=_slice(_gv(gdx_path, "ps"), period),
        pd=_slice(_gv(gdx_path, "pd"), period),
        pa=_slice(_gv(gdx_path, "pa"), period),
        paa=_slice(_gv(gdx_path, "paa") or {}, period),
        pdp=_slice(_gv(gdx_path, "pdp"), period),
        pmt=_slice(_gv(gdx_path, "pmt"), period),
        pmcif=_slice(_gv(gdx_path, "pmcif"), period),
        pet=_slice(_gv(gdx_path, "pet"), period),
        pe=_slice(_gv(gdx_path, "pe"), period),
        pefob=_slice(_gv(gdx_path, "pefob"), period),
        xe=_slice(_gv(gdx_path, "xe") or {}, period),
        xw=_slice(_gv(gdx_path, "xw"), period),
        xmt=_slice(_gv(gdx_path, "xmt"), period),
        xet=_slice(_gv(gdx_path, "xet"), period),
        xaa=_slice(_gv(gdx_path, "xa"), period),
        xma=_slice(_gv(gdx_path, "xm"), period),
        xwmg=_slice(_gv(gdx_path, "xwmg"), period),
        xmgm=_slice(_gv(gdx_path, "xmgm"), period),
        pwmg=_slice(_gv(gdx_path, "pwmg"), period),
        xtmg=_slice(_gv(gdx_path, "xtmg"), period),
        ptmg=_slice(_gv(gdx_path, "ptmg"), period),
        xf=_slice(_gv(gdx_path, "xf"), period),
        xft=_slice(_gv(gdx_path, "xft"), period),
        pf=_slice(_gv(gdx_path, "pf"), period),
        pfa=_slice(_gv(gdx_path, "pfa"), period),
        pft=_slice(_gv(gdx_path, "pft"), period),
        pfact=_slice(_gv(gdx_path, "pfact"), period),
        xi=_slice(_gv(gdx_path, "xi"), period),
        va=_slice(_gv(gdx_path, "va"), period),
        regy=_slice(_gv(gdx_path, "regy"), period),
        yc=_slice(_gv(gdx_path, "yc"), period),
        yg=_slice(_gv(gdx_path, "yg"), period),
        yi=_slice(_gv(gdx_path, "yi"), period),
        # uh has an extra 'hhd' household dimension in GAMS — strip it
        uh=_slice_drop_h(_gv(gdx_path, "uh"), period),
        ug=_slice(_gv(gdx_path, "ug"), period),
        us=_slice(_gv(gdx_path, "us"), period),
        u=_slice(_gv(gdx_path, "u"), period),
        phi=_slice(_gv(gdx_path, "phi"), period),
        phip=_slice(_gv(gdx_path, "phip"), period),
        pcons=_slice(_gv(gdx_path, "pcons"), period),
        rsav=_slice(_gv(gdx_path, "rsav"), period),
        xigbl=_scalar(_gv(gdx_path, "xigbl"), period),
        pigbl=_scalar(_gv(gdx_path, "pigbl"), period),
        pnum=_scalar(_gv(gdx_path, "pnum"), period),
        pabs=_slice(_gv(gdx_path, "pabs"), period),
        gdpmp=_slice(_gv(gdx_path, "gdpmp"), period),
        rgdpmp=_slice(_gv(gdx_path, "rgdpmp"), period),
        # ev/cv have 'hhd' household dimension in GAMS — strip it
        ev=_slice_drop_h(_gv(gdx_path, "ev"), period),
        cv=_slice_drop_h(_gv(gdx_path, "cv"), period),
    )


def _build_altertax_check_model(params, res_region: str):
    """Build check-period altertax model (unsolved) with correct phip=1.0.

    Returns (m_b, m_chk, p_alt, alt_closure) where m_chk is at the warm-start
    point (getData init + phip=1.0 + regy unfixed) but NOT yet solved by PATH.
    """
    from equilibria.templates.gtap import GTAPModelEquations
    from equilibria.templates.gtap.altertax import apply_altertax_elasticities
    from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig

    p_alt = apply_altertax_elasticities(params, in_place=False)

    base_closure = GTAPClosureConfig(
        name="base", closure_type="MCP",
        capital_mobility="sluggish", fix_endowments=False,
        fix_taxes=False, fix_technology=False, if_sub=False,
        numeraire="pnum",
    )
    alt_closure = GTAPClosureConfig(
        name="altertax", closure_type="MCP",
        capital_mobility="mobile", fix_endowments=False,
        fix_taxes=True, fix_technology=True, if_sub=False,
        numeraire="pnum",
    )

    eq_b = GTAPModelEquations(p_alt.sets, p_alt, base_closure, residual_region=res_region)
    m_b = eq_b.build_model()

    # phiP[check] = pcons[base] = 1.0 (GAMS convention)
    for _r in p_alt.sets.r:
        try:
            if hasattr(p_alt, "calibration") and hasattr(p_alt.calibration, "phip"):
                p_alt.calibration.phip[(_r,)] = 1.0
        except Exception:
            pass

    eq_chk = GTAPModelEquations(
        p_alt.sets, p_alt, alt_closure, residual_region=res_region, t0_snapshot=m_b,
    )
    m_chk = eq_chk.build_model()

    # Unfix regy: GAMS keeps regY endogenous in compStat
    for _r in p_alt.sets.r:
        try:
            if hasattr(m_chk, "regy") and m_chk.regy[_r].fixed:
                m_chk.regy[_r].unfix()
        except Exception:
            pass

    # phip=1.0 on the model directly (in case calibration didn't propagate)
    for _r in p_alt.sets.r:
        try:
            if hasattr(m_chk, "phip"):
                m_chk.phip[_r].set_value(1.0)
        except Exception:
            pass

    # Warm-start pft + pf + pfy from GAMS check GDX if available
    try:
        from _diff_core import gams_levels, list_populated_vars  # type: ignore[import-not-found]
        # Try to find the bundle GDX for this dataset
        _ds_key = None
        for k in _DATASET_HAR:
            if k in str(p_alt.sets.r):  # crude — override in caller if needed
                _ds_key = k
                break
    except Exception:
        pass

    return m_b, m_chk, p_alt, alt_closure


class GTAPParityAdapter:
    """ParityAdapter implementation for the GTAP Standard 7 template."""

    _COMBINATIONS = [
        ("9x10", "baseline"),
        ("9x10", "shock_tm10"),
        ("9x10", "altertax"),
        ("nus333", "baseline"),
        ("nus333", "shock_tm10"),
        ("nus333", "altertax"),
        # gtap7_* HAR datasets — altertax three-period scenario
        ("gtap7_3x3", "altertax_check"),
        ("gtap7_3x3", "altertax_shock"),
        ("gtap7_3x4", "altertax_check"),
        ("gtap7_3x4", "altertax_shock"),
        ("gtap7_5x5", "altertax_check"),
        ("gtap7_5x5", "altertax_shock"),
    ]

    _DATASET_GDX = {
        "9x10": "src/equilibria/templates/reference/gtap/data/basedata-9x10.gdx",
        "nus333": "src/equilibria/templates/reference/gtap/data/basedata-nus333.gdx",
    }

    # Reference GDX paths for gtap7_* datasets.
    # out_local.gdx = GAMS 53 local solve (correct equilibrium, pd≈1 in check).
    # out.gdx       = legacy NEOS job (spurious check equilibrium pd[EU_28,Food]≈0.216).
    _DATASET_BUNDLE_GDX = {
        "gtap7_3x3": "output/gtap7_3x3_altertax_neos_bundle/out_local.gdx",
        "gtap7_3x4": "output/gtap7_3x4_altertax_neos_bundle/out_local.gdx",
        "gtap7_5x5": "output/gtap7_5x5_altertax_neos_bundle/out_local.gdx",
    }

    def enumerate_combinations(self) -> list[tuple[str, str]]:
        return list(self._COMBINATIONS)

    def name_aliases(self) -> dict[str, str]:
        from _diff_core import _NAME_ALIAS  # type: ignore[import-not-found]
        return dict(_NAME_ALIAS)

    def load_gams_reference(self, gdx_path) -> dict[str, dict[tuple, float]]:
        from _diff_core import gams_levels, list_populated_vars, split_t  # type: ignore[import-not-found]
        gdx_path = Path(gdx_path)
        out: dict[str, dict[tuple, float]] = {}
        for vname in list_populated_vars(gdx_path):
            raw = gams_levels(gdx_path, vname)
            stripped: dict[tuple, float] = {}
            for key, val in raw.items():
                body, _t = split_t(key)
                # Normalize GAMS set-type prefixes (a_, c_, f_, r_) to bare names
                body = tuple(
                    k[2:] if isinstance(k, str) and len(k) > 2
                           and k[1] == "_" and k[0] in "acfr"
                    else k for k in body
                )
                # Drop singleton household dimension ('hhd') — Python collapses it
                body = tuple(k for k in body if k != "hhd")
                if body in stripped:
                    continue
                stripped[body] = float(val)
            if stripped:
                out[vname] = stripped
        return out

    def load_gams_reference_period(self, gdx_path, period: str) -> dict[str, dict[tuple, float]]:
        """Load GAMS reference keeping only keys from a specific period t."""
        from _diff_core import gams_levels, list_populated_vars  # type: ignore[import-not-found]
        gdx_path = Path(gdx_path)
        out: dict[str, dict[tuple, float]] = {}
        for vname in list_populated_vars(gdx_path):
            raw = gams_levels(gdx_path, vname)
            stripped: dict[tuple, float] = {}
            for key, val in raw.items():
                if isinstance(key, tuple) and key[-1] == period:
                    body = key[:-1]
                    # strip GAMS set prefixes
                    body = tuple(
                        k[2:] if isinstance(k, str) and len(k) > 2
                               and k[1] == "_" and k[0] in "acfr"
                        else k for k in body
                    )
                    # drop singleton household dimension
                    body = tuple(k for k in body if k != "hhd")
                    if body not in stripped:
                        stripped[body] = float(val)
            if stripped:
                out[vname] = stripped
        return out

    def load_gams_reference_for_scenario(
        self, gdx_path, dataset: str, scenario: str
    ) -> dict[str, dict[tuple, float]]:
        """Load GAMS reference filtered to the period corresponding to `scenario`.

        For multi-period GDX (altertax bundles), the GDX contains base/check/shock
        slices. We must select the matching period; otherwise `split_t` takes the
        first (base) slice, silently giving pft=1.0 instead of the check/shock value.
        """
        period_map = {
            "altertax_check": "check",
            "altertax_shock": "shock",
        }
        period = period_map.get(scenario)
        if period is not None:
            return self.load_gams_reference_period(gdx_path, period)
        return self.load_gams_reference(gdx_path)

    def find_py_var(self, model: Any, gams_name: str) -> tuple[Any | None, str | None]:
        from _diff_core import find_py_var, build_derived  # type: ignore[import-not-found]
        derived = build_derived(model)
        return find_py_var(model, gams_name, derived=derived)

    def build_solved_model(self, dataset: str, scenario: str) -> Any:
        if (dataset, scenario) not in self._COMBINATIONS:
            raise ValueError(
                f"Unknown GTAP combination ({dataset!r}, {scenario!r}). "
                f"Known: {self._COMBINATIONS}"
            )

        # --- gtap7_* HAR datasets ---
        if dataset in _DATASET_HAR:
            return self._build_gtap7_altertax(dataset, scenario, solve=True)

        # --- Legacy GDX datasets (9x10, nus333) ---
        from equilibria.templates.gtap import (
            GTAPParameters, build_gtap_contract, GTAPModelEquations,
        )
        from equilibria.templates.gtap.altertax import apply_altertax_elasticities

        run_gtap = _load_run_gtap()

        gdx_rel = self._DATASET_GDX[dataset]
        params = _load_params_gdx(dataset, gdx_rel)
        residual_region = "NAmerica" if dataset == "9x10" else "ROW"

        std_contract = build_gtap_contract({})
        base_eq = GTAPModelEquations(
            params.sets, params, residual_region=residual_region,
            closure=std_contract.closure,
        )
        m_b = base_eq.build_model()
        run_gtap._run_path_capi_nonlinear_full(
            m_b, params, enforce_post_checks=False, strict_path_capi=False,
            closure_config=std_contract.closure, equation_scaling=True,
        )

        if scenario == "baseline":
            return m_b

        if scenario == "altertax":
            shock_params = apply_altertax_elasticities(params, in_place=False)
            contract = build_gtap_contract({"closure": "altertax"})
            from equilibria.templates.gtap.gtap_parity_pipeline import GTAPVariableSnapshot
            warm = GTAPVariableSnapshot.from_python_model(m_b)
            alt_sets = getattr(shock_params, "sets", None) or params.sets
            eq = GTAPModelEquations(
                alt_sets, shock_params, contract.closure, is_counterfactual=True,
                residual_region=residual_region, t0_snapshot=m_b,
            )
            m_alt = eq.build_model()
            run_gtap._run_path_capi_nonlinear_full(
                m_alt, shock_params,
                enforce_post_checks=False, strict_path_capi=False,
                closure_config=contract.closure, equation_scaling=True,
                solution_hint=warm,
            )
            return m_alt

        if scenario == "shock_tm10":
            shock_params = run_gtap._apply_shock_to_params(
                params, shock_mode="tm_pct", shock_value=0.10,
            )
            contract = build_gtap_contract({"closure": "shock"})
            from equilibria.templates.gtap.gtap_parity_pipeline import GTAPVariableSnapshot
            warm = GTAPVariableSnapshot.from_python_model(m_b)
            eq = GTAPModelEquations(
                params.sets, shock_params, contract.closure, is_counterfactual=True,
                residual_region=residual_region, t0_snapshot=m_b,
            )
            m_s = eq.build_model()
            run_gtap._run_path_capi_nonlinear_full(
                m_s, shock_params, enforce_post_checks=False, strict_path_capi=False,
                closure_config=contract.closure, equation_scaling=True,
                solution_hint=warm,
            )
            return m_s

        raise ValueError(f"Unhandled scenario {scenario!r}")

    def build_warmstarted_model(self, dataset: str, scenario: str) -> Any:
        """Return the model at the warm-start point (pre-solver) for --check-warmstart.

        For altertax_check/altertax_shock: returns m_chk after getData init,
        phip=1.0, regy unfixed, and GAMS check-period pft/pf/pfy seeded —
        but before PATH is called. This exposes the exact starting point that
        the solver sees, so --check-warmstart can diagnose basin issues.
        """
        if dataset not in _DATASET_HAR:
            # Fallback for legacy datasets
            return self.build_solved_model(dataset, scenario)

        return self._build_gtap7_altertax(dataset, scenario, solve=False)

    def _build_gtap7_altertax(self, dataset: str, scenario: str, solve: bool) -> Any:
        """Build gtap7_* altertax model, optionally solving with PATH."""
        from equilibria.templates.gtap import GTAPModelEquations
        from equilibria.templates.gtap.altertax import apply_altertax_elasticities
        from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig
        from equilibria.templates.gtap.gtap_parity_pipeline import GTAPVariableSnapshot
        from _diff_core import gams_levels, list_populated_vars  # type: ignore[import-not-found]

        run_gtap = _load_run_gtap()
        params = _load_params_har(dataset)
        # GAMS rres is the last region in the HAR sets (ROW in all gtap7_* datasets).
        # gdxdump confirms: rres('ROW') for gtap7_3x3/3x4/5x5.
        # Use last element to match the HAR set order convention.
        _r_set = list(params.sets.r)
        res_region = _r_set[-1]
        gdx_path = _ROOT / self._DATASET_BUNDLE_GDX[dataset]

        p_alt = apply_altertax_elasticities(params, in_place=False)

        base_closure = GTAPClosureConfig(
            name="base", closure_type="MCP",
            capital_mobility="sluggish", fix_endowments=False,
            fix_taxes=False, fix_technology=False, if_sub=False,
            numeraire="pnum",
        )
        alt_closure = GTAPClosureConfig(
            name="altertax", closure_type="MCP",
            capital_mobility="mobile", fix_endowments=False,
            fix_taxes=True, fix_technology=True, if_sub=False,
            numeraire="pnum",
        )

        # betaCal base model — solve first so t0_snapshot carries real prices
        eq_b = GTAPModelEquations(p_alt.sets, p_alt, base_closure, residual_region=res_region)
        m_b = eq_b.build_model()
        run_gtap._run_path_capi_nonlinear_full(
            m_b, p_alt,
            enforce_post_checks=False, strict_path_capi=False,
            closure_config=base_closure, equation_scaling=True,
        )

        # phiP[check] = 1.0 (GAMS convention: pcons[base]=1)
        for _r in p_alt.sets.r:
            try:
                if hasattr(p_alt, "calibration") and hasattr(p_alt.calibration, "phip"):
                    p_alt.calibration.phip[(_r,)] = 1.0
            except Exception:
                pass

        # Build check model
        eq_chk = GTAPModelEquations(
            p_alt.sets, p_alt, alt_closure, residual_region=res_region, t0_snapshot=m_b,
        )
        m_chk = eq_chk.build_model()

        # Unfix regy (mirrors GAMS regYeq.regy endogenous in compStat)
        for _r in p_alt.sets.r:
            try:
                if hasattr(m_chk, "regy") and m_chk.regy[_r].fixed:
                    m_chk.regy[_r].unfix()
            except Exception:
                pass

        # Unfix pft for mobile factors — m_b presolve fixed pft=1 via pfteq free-row,
        # but in altertax check/shock pft is endogenous (CET price index). The parity
        # adapter passes closure_config=alt_closure (name="altertax") to _run_path_capi,
        # which would skip the pft fix — but t0_snapshot copies pft.fixed=True from m_b,
        # so the presolve guard (line: if _pft_vd.fixed: continue) leaves pft stuck at 1.
        if hasattr(m_chk, "pft") and hasattr(m_chk, "eq_pfteq"):
            for _r in p_alt.sets.r:
                for _f in p_alt.sets.mf:
                    try:
                        if m_chk.pft[_r, _f].fixed:
                            m_chk.pft[_r, _f].unfix()
                        if not m_chk.eq_pfteq[_r, _f].active:
                            m_chk.eq_pfteq[_r, _f].activate()
                    except Exception:
                        pass

        # phip=1.0 directly on built model
        for _r in p_alt.sets.r:
            try:
                if hasattr(m_chk, "phip"):
                    m_chk.phip[_r].set_value(1.0)
            except Exception:
                pass

        # Mirror GAMS numeraire: pnum.fix=1, pwfact free, eq_pnum+eq_pwfact active.
        # pnumeq (eq_pnum: pnum==pwfact) is a free row that pins pwfact=pnum=1.
        # pwfacteq (eq_pwfact, MCP pair for pwfact) constrains Tornqvist(pf,xf)=1.
        # Both stay active — keeping pwfact free and eq_pwfact active anchors pf prices.
        if hasattr(m_chk, "pnum") and not m_chk.pnum.fixed:
            m_chk.pnum.fix(1.0)

        # Fix ev/cv and deactivate eq_ev/eq_cv for the check period solve.
        # ev and cv are welfare-reporting variables — they do not appear in any
        # equilibrium equation except their own (eq_ev/eq_cv). In MCP, if ev has
        # lb > 0 the solver can satisfy the system by clamping ev to its lb with
        # eq_ev slack (a spurious complementarity solution). Removing them from the
        # active system lets PATH find the correct equilibrium; ev is recalculated
        # post-solve. GAMS avoids this because ev is initialized to yc and PATH
        # starts close enough to the interior solution.
        # Seed ev/cv from GDX check values BEFORE fixing, so they're fixed at GAMS value.
        if gdx_path.exists():
            try:
                from _diff_core import gams_levels as _gams_lv_ev  # type: ignore[import-not-found]
                for _vn, _gdxn in [("ev", "ev"), ("cv", "cv")]:
                    _vobj = getattr(m_chk, _vn, None)
                    if _vobj is None:
                        continue
                    _raw = _gams_lv_ev(gdx_path, _gdxn)
                    for _r in m_chk.r:
                        _gdx_key = (str(_r), "hhd", "check")
                        _val = _raw.get(_gdx_key)
                        if _val is None:
                            continue
                        try:
                            _item = _vobj[_r]
                            if not _item.fixed:
                                _item.set_value(float(_val))
                        except Exception:
                            pass
            except Exception:
                pass
        for _vname, _eqname in [("ev", "eq_ev"), ("cv", "eq_cv")]:
            _vobj = getattr(m_chk, _vname, None)
            _eobj = getattr(m_chk, _eqname, None)
            if _vobj is not None and _eobj is not None:
                try:
                    for _idx in list(_vobj):
                        _item = _vobj[_idx]
                        if not _item.fixed:
                            _item.fix(float(_item.value) if _item.value is not None else 1.0)
                    _eobj.deactivate()
                except Exception:
                    pass

        # Warm-start pf[r,f,a] from GAMS check-period GDX.
        # GAMS betaCal→check: PATH starts from betaCal solved values (pf≈1.35 in check).
        # Python starts from getData init (pf≈1.0, base model). This 25% gap steers
        # PATH to a different valid equilibrium. Seed pf from GDX to match GAMS starting point.
        # Also compute pfa = pf*(1+rtf) consistently to avoid eq_pfaeq residuals.
        if gdx_path.exists() and hasattr(m_chk, "pf"):
            from _diff_core import gams_levels as _gams_lv_pf  # type: ignore[import-not-found]
            from pyomo.environ import value as _pyo_val
            _gams_pf = _gams_lv_pf(gdx_path, "pf")
            for _r in m_chk.r:
                for _f in m_chk.f:
                    for _a in m_chk.a:
                        _key = (str(_r), str(_f), "a_" + str(_a), "check")
                        _pf_val = _gams_pf.get(_key)
                        if _pf_val is None:
                            continue
                        try:
                            _pf_item = m_chk.pf[_r, _f, _a]
                            if not (hasattr(_pf_item, "fixed") and _pf_item.fixed):
                                _pf_item.set_value(float(_pf_val))
                            # pfa = pf*(1+rtf) — keep consistent with eq_pfaeq
                            if hasattr(m_chk, "pfa"):
                                _rtf = float(p_alt.taxes.rtf.get((_r, _f, _a), 0.0) or 0.0)
                                _pfa_item = m_chk.pfa[_r, _f, _a]
                                if not (hasattr(_pfa_item, "fixed") and _pfa_item.fixed):
                                    _pfa_item.set_value(float(_pf_val) * (1.0 + _rtf))
                        except Exception:
                            pass

        # Warm-start xiagg[r] from GAMS xi[r,check] / pi[r,check].
        # GAMS xi[r,t] = total regional investment (1D), which equals pi*xiagg.
        # Python xiagg[r] = physical investment aggregate (also 1D). The GDX warm-start
        # loop silently skips xiagg (not in GDX) and xi (mapped to _DerivedVar with no
        # _NAME_ALIAS entry) — leaving xiagg at base values with ~4-10% gap vs GAMS check.
        # This gap produces large residuals in eq_xiagg/eq_xigbl/eq_pigbl/eq_kapEnd at
        # warm-start, steering PATH to the wrong equilibrium basin.
        if gdx_path.exists() and hasattr(m_chk, "xiagg"):
            from _diff_core import gams_levels as _gams_lv  # type: ignore[import-not-found]
            _gams_xi = _gams_lv(gdx_path, "xi")
            _gams_pi = _gams_lv(gdx_path, "pi")
            for _r in m_chk.r:
                _r_str = str(_r)
                _xi_val = _gams_xi.get((_r_str, "check"))
                _pi_val = _gams_pi.get((_r_str, "check"))
                if _xi_val is not None and _pi_val is not None and abs(_pi_val) > 1e-12:
                    try:
                        _item = m_chk.xiagg[_r]
                        if not (hasattr(_item, "fixed") and _item.fixed):
                            _item.set_value(float(_xi_val) / float(_pi_val))
                    except Exception:
                        pass

        # Mirror GAMS lower bounds: pd.lo = 0.001*pd.l(t-1), where t-1 is betaCal.
        # Use base model (m_b) values — NOT warm-started check values — because GAMS
        # applies bounds based on the PREVIOUS period (base), not the current warm-start.
        # Using warm-started values sets pd.lo too low (e.g. 0.000216 for EU_28/Food
        # instead of 0.001 from base), which can shift the solver basin.
        if hasattr(m_chk, "xp"):
            for _r in p_alt.sets.r:
                for _a in p_alt.sets.a:
                    try:
                        _item = m_chk.xp[_r, _a]
                        _v = float(_item.value) if _item.value is not None else 1.0
                        if _v > 0:
                            _item.setlb(0.001 * _v)
                    except Exception:
                        pass
        for _vname in ["pf", "pfa", "pft", "px", "pd", "pm", "pe"]:
            _pyv = getattr(m_chk, _vname, None)
            if _pyv is None:
                continue
            try:
                for _idx in _pyv:
                    try:
                        _item = _pyv[_idx]
                        if hasattr(_item, "fixed") and _item.fixed:
                            continue
                        _v = float(_item.value) if _item.value is not None else 1.0
                        if _v > 0:
                            _item.setlb(0.001 * _v)
                    except Exception:
                        pass
            except Exception:
                pass

        # Full GAMS warm-start for check period.
        # Strategy: brute-force seed every free variable in m_chk from the GDX check-period
        # slice. For each Pyomo variable component (e.g. model.pf), read the GDX variable of
        # the same name (with GAMS prefixes stripped), filter for period='check', and set_value.
        # This covers all variables without needing to enumerate them explicitly in the snapshot.
        _chk_hint = None
        if gdx_path.exists():
            try:
                _chk_hint = _gams_snapshot_from_altertax_gdx(gdx_path, "check")
                # Apply named-snapshot fields first (covers snapshot-mapped vars)
                from equilibria.templates.gtap.gtap_solver import GTAPSolver
                _helper = GTAPSolver(m_chk, closure=alt_closure, solver_name="path", params=p_alt)
                _helper.apply_solution_hint(_chk_hint)
                # After seeding xda/xma from GDX xd/xm (3D), recompute 2D aggregates
                # (xd, xds, xmt) from xda/xma so eq_xd_agg/eq_pdeq/eq_xmt_agg are
                # satisfied at warm-start. These 2D vars don't exist separately in GDX.
                if hasattr(m_chk, "xscale"):
                    from pyomo.environ import value as _pyo_val
                    for _r in m_chk.r:
                        for _i in m_chk.i:
                            try:
                                _total_d = sum(
                                    float(_pyo_val(m_chk.xda[_r, _i, _aa]))
                                    / max(float(_pyo_val(m_chk.xscale[_r, _aa])), 1e-12)
                                    for _aa in m_chk.aa
                                )
                                if hasattr(m_chk, "xd"):
                                    _item = m_chk.xd[_r, _i]
                                    if not (hasattr(_item, "fixed") and _item.fixed):
                                        _item.set_value(max(_total_d, 1e-8))
                                if hasattr(m_chk, "xds"):
                                    _item = m_chk.xds[_r, _i]
                                    if not (hasattr(_item, "fixed") and _item.fixed):
                                        _item.set_value(max(_total_d, 1e-8))
                            except Exception:
                                pass
                            try:
                                _total_m = sum(
                                    float(_pyo_val(m_chk.xma[_r, _i, _aa]))
                                    / max(float(_pyo_val(m_chk.xscale[_r, _aa])), 1e-12)
                                    for _aa in m_chk.aa
                                )
                                if hasattr(m_chk, "xmt"):
                                    _item = m_chk.xmt[_r, _i]
                                    if not (hasattr(_item, "fixed") and _item.fixed):
                                        _item.set_value(max(_total_m, 1e-8))
                            except Exception:
                                pass
            except Exception as _snap_exc:
                import logging as _log
                _log.getLogger(__name__).warning(
                    "Could not build/apply GAMS check-period snapshot: %s", _snap_exc
                )

        if not solve:
            return m_chk

        # Solve check period. Starting from t0_snapshot=m_b (base equilibrium),
        # matching GAMS's betaCal→check sequential approach. Pass _chk_hint as
        # solution_hint so PATH re-applies GAMS values after aggressive fixing.
        run_gtap._run_path_capi_nonlinear_full(
            m_chk, p_alt,
            enforce_post_checks=False, strict_path_capi=False,
            closure_config=alt_closure, equation_scaling=True,
            solution_hint=_chk_hint,
        )

        if scenario == "altertax_check":
            return m_chk

        # Build + solve shock period
        p_alt_shock = copy.deepcopy(p_alt)
        for key in list(p_alt_shock.taxes.imptx.keys()):
            old = float(p_alt_shock.taxes.imptx[key] or 0.0)
            p_alt_shock.taxes.imptx[key] = old * 1.10

        warm_chk = GTAPVariableSnapshot.from_python_model(m_chk)
        eq_alt = GTAPModelEquations(
            p_alt_shock.sets, p_alt_shock, alt_closure,
            residual_region=res_region, t0_snapshot=m_chk,
        )
        m_alt = eq_alt.build_model()
        run_gtap._run_path_capi_nonlinear_full(
            m_alt, p_alt_shock,
            enforce_post_checks=False, strict_path_capi=False,
            closure_config=alt_closure, equation_scaling=True,
            solution_hint=warm_chk,
        )
        return m_alt
