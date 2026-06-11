"""Shared, sets-agnostic dataset registry for the three parity tools.

`diff_closure.py`, `residual_at_gams_point.py`, and `nl_compare.py` all need the
same thing: given a dataset NAME, build the Python base + shock models and locate
the GAMS reference. Centralising that here means none of the tools hardcode
region/commodity names — they parameterise on `--dataset` and read sets from the
data. Adding a dataset is a single `DATASETS` entry.

A dataset is either:
  * **compstat** — single-period models; the shock model is a counterfactual
    flagged by `t0_snapshot` (e.g. NUS333, built from HAR files).
  * **multiperiod** — built via `run_gtap.solve_sequential`, which constructs a
    fresh single-period (`t_set=("base",)`) model per period with
    `is_counterfactual=True` + `t0_snapshot` for the shock (e.g. 9x10 from GDX).
    The per-period closure is solve-dependent, so this path SOLVES.

All builders return `(base_model, shock_model)` with the MCP closure applied, and
both models label their period `"base"` (run_gtap builds every period that way),
so `cf_period="base"` uniformly.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "src/equilibria/templates/reference/gtap/data"


def _closure_stack(model, params, label, closure):
    """Standard MCP closure stack (matches the solver pipeline)."""
    from equilibria.templates.gtap.gtap_solver import GTAPSolver
    from _closure_patches import apply_squareness_patches

    helper = GTAPSolver(model, solver_name="path", params=params)
    helper.apply_closure(closure)
    apply_squareness_patches(model, params, label=label)
    helper.apply_aggressive_fixing_for_mcp()


def _build_nus333(close: bool = True):
    from equilibria.templates.gtap import GTAPParameters
    from equilibria.templates.gtap.gtap_model_equations import GTAPModelEquations
    from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig
    from compare_nus333_vs_neos import _apply_tariff_shock, _copy_var_levels
    from diff_nus333_full import NUS333_HAR

    p = GTAPParameters()
    p.load_from_har(
        basedata_path=NUS333_HAR / "basedata.har",
        sets_path=NUS333_HAR / "sets.har",
        default_path=NUS333_HAR / "default.prm",
        baserate_path=NUS333_HAR / "baserate.har",
    )
    cl = GTAPClosureConfig(if_sub=False, rmuv=("ROW",), imuv=("MFG",))
    m_b = GTAPModelEquations(p.sets, p, residual_region="ROW", closure=cl).build_model()
    if close:
        _closure_stack(m_b, p, "base", cl)
    _apply_tariff_shock(p, factor=1.10)
    m_s = GTAPModelEquations(
        p.sets, p, residual_region="ROW", closure=cl, t0_snapshot=m_b
    ).build_model()
    _copy_var_levels(m_b, m_s)
    if close:
        _closure_stack(m_s, p, "shock", cl)
    return m_b, m_s


def _build_compstat_har(har_dir: Path, close: bool = True):
    """Generic comp-stat build for any GTAPAgg dataset with HAR files.

    Mirrors _build_nus333 but parameterised by the HAR directory; the residual
    region is auto-picked as the last region. NOTE: the comp-stat closure is not
    guaranteed square for arbitrary aggregations (it is hand-tuned for nus333);
    diff_closure reports the DOF gap so non-square datasets are visible.
    """
    from equilibria.templates.gtap import GTAPParameters
    from equilibria.templates.gtap.gtap_model_equations import GTAPModelEquations
    from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig
    from compare_nus333_vs_neos import _apply_tariff_shock, _copy_var_levels

    p = GTAPParameters()
    p.load_from_har(
        basedata_path=har_dir / "basedata.har",
        sets_path=har_dir / "sets.har",
        default_path=har_dir / "default.prm",
        baserate_path=har_dir / "baserate.har",
    )
    rres = list(p.sets.r)[-1]
    # fix_endowments=False: leave xft FREE with its supply eq (eq_xfteq) active, to
    # mirror GAMS exactly (xfteq.xft is an MCP pair; with etaf=0 it reads xft=aft,
    # i.e. the fixed endowment). GAMS does not fix xft — it pins it via the equation.
    cl = GTAPClosureConfig(if_sub=False, fix_endowments=False)
    m_b = GTAPModelEquations(p.sets, p, residual_region=rres, closure=cl).build_model()
    if close:
        _closure_stack(m_b, p, "base", cl)
    _apply_tariff_shock(p, factor=1.10)
    m_s = GTAPModelEquations(
        p.sets, p, residual_region=rres, closure=cl, t0_snapshot=m_b
    ).build_model()
    _copy_var_levels(m_b, m_s)
    if close:
        _closure_stack(m_s, p, "shock", cl)
    return m_b, m_s


def _build_9x10(close: bool = True):
    """Faithful multi-period build via run_gtap.solve_sequential (SOLVES)."""
    from equilibria.templates.gtap import GTAPParameters
    from run_gtap import _build_gtap_contract_with_calibration, solve_sequential

    contract = _build_gtap_contract_with_calibration("gtap_standard7_9x10")
    cl = contract.closure.model_copy(update={"if_sub": False})

    p = GTAPParameters()
    p.load_from_gdx(DATA_DIR / "basedata-9x10.gdx")

    def _shock(params_obj, tsim, prev_model):
        if tsim != "shock":
            return
        for k in list(params_obj.taxes.imptx.keys()):
            params_obj.taxes.imptx[k] = float(params_obj.taxes.imptx[k]) * 1.10

    seq = solve_sequential(
        None, p, closure_config=cl, t_set=("base", "shock"),
        equation_scaling=True, tol=1e-8, params_setup_fn=_shock,
    )
    models = seq.get("_models", {})
    return models.get("base"), models.get("shock")


@dataclass
class ParityDataset:
    name: str
    mode: str                                   # "compstat" | "multiperiod"
    build: Callable[..., tuple]                 # (close: bool) -> (m_base, m_shock)
    cf_period: str = "base"                      # period label of the Python shock model
    base_period: str = "base"
    agg_gdx: Optional[Path] = None              # single aggregated GDX (nl_compare _agg)
    gams_ref_gdx: Optional[Path] = None         # GAMS solution GDX (value/residual)
    gams_period: str = "shock"                   # period label inside the GAMS GDX
    key_remap: str = "none"                      # "none" | "nus333" (residual injection)
    # nl_compare config (GDX-based datasets only): run_gtap calibration contract
    # and the GAMS comp.gms MUV basket (rmuv, imuv) for the full dataset. These are
    # per-dataset GAMS modelling config, centralised here instead of hardcoded in
    # nl_compare.main so adding a dataset is a registry edit.
    nl_contract: Optional[str] = None
    nl_full_muv: Optional[tuple[tuple[str, ...], tuple[str, ...]]] = None


DATASETS: dict[str, ParityDataset] = {
    "nus333": ParityDataset(
        name="nus333", mode="compstat", build=_build_nus333,
        gams_ref_gdx=ROOT / "output/nus333_neos/out.gdx",
        gams_period="shock", key_remap="nus333",
    ),
    "9x10": ParityDataset(
        name="9x10", mode="multiperiod", build=_build_9x10,
        agg_gdx=DATA_DIR / "basedata-9x10.gdx",
        gams_ref_gdx=ROOT / "src/equilibria/templates/reference/gtap/output/COMP.gdx",
        gams_period="shock", key_remap="none",
        nl_contract="gtap_standard7_9x10",
        nl_full_muv=(("Oceania", "NAmerica", "EU_28"),
                     ("c_ProcFood", "c_TextWapp", "c_LightMnfc", "c_HeavyMnfc")),
    ),
}

# GTAPAgg v7 aggregations under datasets/ — registered as comp-stat (HAR loader).
# No GAMS reference GDX, so residual / nl_compare are N/A; only the closure diff
# (partition inspection) runs. agg_gdx points at the GTAPAgg basedata.gdx for any
# future nl_compare _agg use. The comp-stat closure may not be square for these
# (diff_closure prints the DOF gap).
_GTAPAGG_DIR = ROOT / "datasets"
for _name in ("gtap7_3x3", "gtap7_3x4", "gtap7_5x5", "gtap7_10x7",
              "gtap7_15x10", "gtap7_20x41"):
    _dir = _GTAPAGG_DIR / _name
    if (_dir / "basedata.har").exists():
        # nl_compare's GAMS getData (_agg) needs a CONSOLIDATED GDX (sets + all
        # params in one file). Use v7_consolidated.gdx when present; otherwise
        # agg_gdx is None and the .nl tool is unavailable for that dataset.
        _cons = _dir / "v7_consolidated.gdx"
        # GAMS reference (local v53 solve via build_gtap7_<name>_local.py) when
        # present. Python (HAR) sets are unprefixed (Food); GAMS (consolidated) are
        # c_/a_ prefixed → strip_cap remap for residual injection.
        _ref = ROOT / "output" / f"{_name}_local" / "out.gdx"
        DATASETS[_name] = ParityDataset(
            name=_name, mode="compstat",
            build=(lambda close=True, d=_dir: _build_compstat_har(d, close)),
            agg_gdx=(_cons if _cons.exists() else None),
            gams_ref_gdx=(_ref if _ref.exists() else None),
            gams_period="shock",
            key_remap=("strip_cap" if _ref.exists() else "none"),
        )



def build_models(name: str, close: bool = True):
    """Return (base_model, shock_model) for a registered dataset."""
    return DATASETS[name].build(close=close)
