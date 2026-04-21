"""Complete GTAP Model Equations (Functional Implementation)

This module implements a fully functional GTAP CGE model.
All equations are implemented to create a solvable square system.
"""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

from equilibria.templates.gtap.gtap_parameters import (
    GAMSCalibrationDump,
    GTAP_GOVERNMENT_AGENT,
    GTAP_HOUSEHOLD_AGENT,
    GTAP_INVESTMENT_AGENT,
    GTAP_MARGIN_AGENT,
)

if TYPE_CHECKING:
    from pyomo.environ import ConcreteModel

logger = logging.getLogger(__name__)


@dataclass
class _InlineReferenceSnapshot:
    """Lightweight snapshot used only for benchmark-aligned initialization.

    This avoids importing the heavy parity pipeline from inside model
    construction while still letting us seed key GTAP variables from a GAMS
    COMP CSV snapshot.
    """

    xp: Dict[Tuple[str, str], float] = field(default_factory=dict)
    xf: Dict[Tuple[str, str, str], float] = field(default_factory=dict)
    pf: Dict[Tuple[str, str, str], float] = field(default_factory=dict)
    xd: Dict[Tuple[str, str, str], float] = field(default_factory=dict)
    xaa: Dict[Tuple[str, str, str], float] = field(default_factory=dict)
    xds: Dict[Tuple[str, str], float] = field(default_factory=dict)
    xet: Dict[Tuple[str, str], float] = field(default_factory=dict)
    xw: Dict[Tuple[str, str, str], float] = field(default_factory=dict)
    pe: Dict[Tuple[str, str, str], float] = field(default_factory=dict)
    pet: Dict[Tuple[str, str], float] = field(default_factory=dict)
    pwmg: Dict[Tuple[str, str, str], float] = field(default_factory=dict)
    yi: Dict[str, float] = field(default_factory=dict)
    yc: Dict[str, float] = field(default_factory=dict)
    yg: Dict[str, float] = field(default_factory=dict)


class GTAPModelEquations:
    """Complete GTAP CGE model with all equations."""
    
    def __init__(
        self,
        sets: "GTAPSets",
        params: "GTAPParameters",
        closure: Optional["GTAPClosureConfig"] = None,
        reference_snapshot: Optional[Any] = None,
        is_counterfactual: bool = False,
    ):
        self.sets = sets
        self.params = params
        self.closure = closure
        self.reference_snapshot = reference_snapshot or self._load_reference_snapshot_from_env()
        self.gams_calibration_dump: Optional[GAMSCalibrationDump] = None
        # When False (baseline), rgdpmp == gdpmp (GAMS assignment `rgdpmp.l = gdpmp.l`).
        # When True (counterfactual/shock), the Fisher chain-volume index is active.
        self.is_counterfactual: bool = is_counterfactual
        self._configure_calibration_source()

    def _vst_value(self, region: str, commodity: str) -> float:
        """Return benchmark inventory change VST(region, commodity).

        GTAP GDX ordering for VST can arrive as either (commodity, region) or
        (region, commodity) depending on loader path. Accept both.
        """
        val = self.params.benchmark.vst.get((region, commodity))
        if val is None:
            val = self.params.benchmark.vst.get((commodity, region), 0.0)
        return float(val or 0.0)

    def _load_reference_snapshot_from_env(self) -> Optional[_InlineReferenceSnapshot]:
        """Optionally load a GAMS/COMP snapshot for benchmark-aligned initialization.

        This is intentionally opt-in so normal model construction is unchanged.
        Set `EQUILIBRIA_GTAP_REFERENCE_SNAPSHOT` to:
        - an explicit CSV/GDX path, or
        - `auto` to look for `reference/gtap/comp/COMP_generated.csv`
        - `off` / `none` / `false` / `0` to disable snapshot-based seeding
        """
        snapshot_hint = os.environ.get("EQUILIBRIA_GTAP_REFERENCE_SNAPSHOT", "").strip()
        if not snapshot_hint:
            return None
        if snapshot_hint.lower() in {"off", "none", "false", "0"}:
            return None

        if snapshot_hint.lower() == "auto":
            candidate = (
                Path(__file__).resolve().parents[1]
                / "reference"
                / "gtap"
                / "comp"
                / "COMP_generated.csv"
            )
        else:
            candidate = Path(snapshot_hint).expanduser().resolve()

        if not candidate.exists():
            return None

        if candidate.suffix.lower() != ".csv":
            return None

        try:
            from equilibria.templates.gtap.gtap_equilibrium import GTAPEquilibriumSnapshot

            raw = GTAPEquilibriumSnapshot.from_csv(candidate, year=1)

            def _pairs(name: str) -> Dict[Tuple[str, str], float]:
                data: Dict[Tuple[str, str], float] = {}
                for key, value in raw.get(name).items():
                    if len(key) == 2:
                        data[(str(key[0]), str(key[1]))] = float(value)
                return data

            def _triples(name: str) -> Dict[Tuple[str, str, str], float]:
                data: Dict[Tuple[str, str, str], float] = {}
                for key, value in raw.get(name).items():
                    if len(key) == 3:
                        data[(str(key[0]), str(key[1]), str(key[2]))] = float(value)
                return data

            def _regions(name: str) -> Dict[str, float]:
                data: Dict[str, float] = {}
                for key, value in raw.get(name).items():
                    if len(key) >= 1:
                        data[str(key[0])] = float(value)
                return data

            return _InlineReferenceSnapshot(
                # Limit env-driven CSV seeding to blocks that map cleanly from
                # COMP outputs. Raw macro-income rows from the CSV are not yet
                # normalized the same way as the Pyomo benchmark and can distort
                # rgdpmp/yc/yg when loaded directly here.
                # yi IS loaded because it is directly comparable: it drives the
                # xiagg initialization and its COMP value (year=1, income-side
                # identity) prevents the yi → xiagg → xi → xds → xet cascade
                # that causes large eq_xweq residuals and PATH no_progress.
                xp={},
                xf={},
                pf={},
                xd={},
                xaa={},
                xds={},
                xet=_pairs("xet"),
                xw=_triples("xw"),
                pe=_triples("pe"),
                pet=_pairs("pet"),
                pwmg=_triples("pwmg"),
                yi=_regions("yi"),
                yc={},
                yg={},
            )
        except Exception:
            return None

    def _parse_calibration_source(self) -> tuple[str, set[str]]:
        raw = str(getattr(self.closure, "calibration_source", "python") or "python").strip().lower()
        if raw.startswith("mixed:"):
            targets = {
                token.strip().lower()
                for token in raw.split(":", 1)[1].split(",")
                if token.strip()
            }
            return "mixed", targets
        if raw in {"python", "gams"}:
            return raw, set()
        return "python", set()

    def _resolve_calibration_dump_path(self) -> Optional[Path]:
        explicit = str(getattr(self.closure, "calibration_dump", "") or "").strip()
        if explicit:
            candidate = Path(explicit).expanduser().resolve()
            return candidate if candidate.exists() else None

        env_hint = os.environ.get("EQUILIBRIA_GTAP_CAL_DUMP", "").strip()
        if env_hint:
            candidate = Path(env_hint).expanduser().resolve()
            return candidate if candidate.exists() else None

        default_path = (
            Path(__file__).resolve().parents[1]
            / "reference"
            / "gtap"
            / "data"
            / "gams_cal_dump_9x10.gdx"
        )
        if default_path.exists():
            return default_path
        return None

    @staticmethod
    def _symbol_enabled(symbol: str, mode: str, targets: set[str], *, level: bool = False) -> bool:
        if mode == "gams":
            return True
        if mode != "mixed":
            return False
        if "all" in targets:
            return True
        if level and "levels" in targets:
            return True
        aliases = {
            "xa": "xaa",
            "xd": "xda",
        }
        symbol_lc = symbol.lower()
        if symbol_lc in targets:
            return True
        alias = aliases.get(symbol_lc)
        if alias is not None and alias in targets:
            return True
        return False

    @staticmethod
    def _normalize_override_map(
        data: Dict[Tuple[str, ...], float],
        expected_dim: int,
    ) -> Dict[Tuple[str, ...], float]:
        normalized: Dict[Tuple[str, ...], float] = {}
        for key, value in data.items():
            if len(key) != expected_dim:
                continue
            normalized[tuple(str(part) for part in key)] = float(value)
        return normalized

    def _build_snapshot_from_dump(self, mode: str, targets: set[str]) -> Optional[_InlineReferenceSnapshot]:
        if self.gams_calibration_dump is None:
            return None

        base = self.reference_snapshot or _InlineReferenceSnapshot()
        snapshot = _InlineReferenceSnapshot(
            xp=dict(getattr(base, "xp", {}) or {}),
            xf=dict(getattr(base, "xf", {}) or {}),
            pf=dict(getattr(base, "pf", {}) or {}),
            xd=dict(getattr(base, "xd", {}) or {}),
            xaa=dict(getattr(base, "xaa", {}) or {}),
            xds=dict(getattr(base, "xds", {}) or {}),
            xet=dict(getattr(base, "xet", {}) or {}),
            xw=dict(getattr(base, "xw", {}) or {}),
            pe=dict(getattr(base, "pe", {}) or {}),
            pet=dict(getattr(base, "pet", {}) or {}),
            pwmg=dict(getattr(base, "pwmg", {}) or {}),
            yi=dict(getattr(base, "yi", {}) or {}),
            yc=dict(getattr(base, "yc", {}) or {}),
            yg=dict(getattr(base, "yg", {}) or {}),
        )

        applied = 0

        def _apply_level(symbol: str, attr: str, expected_dim: int) -> None:
            nonlocal applied
            if not self._symbol_enabled(symbol, mode, targets, level=True):
                return
            raw = self.gams_calibration_dump.get_levels(symbol)
            if not raw:
                return
            cleaned = self._normalize_override_map(raw, expected_dim)
            if not cleaned:
                return
            target_map = getattr(snapshot, attr)
            target_map.update(cleaned)
            applied += len(cleaned)

        _apply_level("xp", "xp", 2)
        _apply_level("xf", "xf", 3)
        _apply_level("pf", "pf", 3)
        _apply_level("xd", "xd", 3)
        _apply_level("xa", "xaa", 3)
        _apply_level("xds", "xds", 2)
        _apply_level("xet", "xet", 2)
        _apply_level("xw", "xw", 3)
        _apply_level("pe", "pe", 3)
        _apply_level("pet", "pet", 2)
        _apply_level("pwmg", "pwmg", 3)

        if self._symbol_enabled("yi", mode, targets, level=True):
            yi = self._normalize_override_map(self.gams_calibration_dump.get_levels("yi"), 1)
            if yi:
                snapshot.yi.update({k[0]: v for k, v in yi.items()})
                applied += len(yi)
        if self._symbol_enabled("yc", mode, targets, level=True):
            yc = self._normalize_override_map(self.gams_calibration_dump.get_levels("yc"), 1)
            if yc:
                snapshot.yc.update({k[0]: v for k, v in yc.items()})
                applied += len(yc)
        if self._symbol_enabled("yg", mode, targets, level=True):
            yg = self._normalize_override_map(self.gams_calibration_dump.get_levels("yg"), 1)
            if yg:
                snapshot.yg.update({k[0]: v for k, v in yg.items()})
                applied += len(yg)

        if applied <= 0:
            return None
        logger.info("Applied %s benchmark level overrides from GAMS calibration dump", applied)
        return snapshot

    def _apply_parameter_overrides_from_dump(self, mode: str, targets: set[str]) -> None:
        if self.gams_calibration_dump is None:
            return

        applied_counts: Dict[str, int] = {}

        def _apply(symbol: str, target: Dict[Tuple[str, ...], float], expected_dim: int) -> None:
            if not self._symbol_enabled(symbol, mode, targets, level=False):
                return
            raw = self.gams_calibration_dump.get_derived(symbol)
            if not raw:
                return
            cleaned = self._normalize_override_map(raw, expected_dim)
            if not cleaned:
                return
            target.clear()
            target.update(cleaned)
            applied_counts[symbol] = len(cleaned)

        # Core calibrated shares and trade parameters.
        _apply("and", self.params.calibrated.and_param, 2)
        _apply("ava", self.params.calibrated.ava_param, 2)
        _apply("io", self.params.calibrated.io_param, 3)
        _apply("af", self.params.calibrated.af_param, 3)
        _apply("gx", self.params.calibrated.gx_param, 3)
        _apply("amw", self.params.shares.p_amw, 3)
        _apply("gw", self.params.shares.p_gw, 3)
        _apply("gd", self.params.shares.p_gd, 2)
        _apply("ge", self.params.shares.p_ge, 2)
        _apply("gf", self.params.shares.p_gf, 3)

        # Optional elasticity overrides from dump.
        _apply("esubt", self.params.elasticities.esubt, 2)
        _apply("esubc", self.params.elasticities.esubc, 2)
        _apply("esubm", self.params.elasticities.esubm, 2)
        _apply("esubva", self.params.elasticities.esubva, 2)
        _apply("sigmas", self.params.elasticities.sigmas, 2)
        _apply("omegaw", self.params.elasticities.omegaw, 2)
        _apply("omegaf", self.params.elasticities.omegaf, 2)
        _apply("etaff", self.params.elasticities.etaff, 3)

        if applied_counts:
            logger.info(
                "Applied GAMS calibration parameter overrides for symbols: %s",
                ", ".join(f"{name}({count})" for name, count in sorted(applied_counts.items())),
            )

    def _configure_calibration_source(self) -> None:
        mode, targets = self._parse_calibration_source()

        # Always attempt to load the dump when one is available.  In "python"
        # mode we still want derived params (e.g. alphaa) accessible for
        # benchmark-alignment fixes like i_share, even though we do not apply
        # full parameter overrides from the dump.
        dump_path = self._resolve_calibration_dump_path()
        if dump_path is not None:
            try:
                self.gams_calibration_dump = GAMSCalibrationDump.from_gdx(dump_path)
            except Exception as exc:
                logger.warning("Failed to load GAMS calibration dump '%s': %s", dump_path, exc)

        if mode == "python":
            return

        if self.gams_calibration_dump is None:
            logger.warning(
                "calibration_source=%s requested but no calibration dump GDX was found "
                "(checked closure.calibration_dump, EQUILIBRIA_GTAP_CAL_DUMP, default path).",
                mode,
            )
            return

        self._apply_parameter_overrides_from_dump(mode, targets)
        dumped_snapshot = self._build_snapshot_from_dump(mode, targets)
        if dumped_snapshot is not None:
            self.reference_snapshot = dumped_snapshot
        
    def build_model(self) -> "ConcreteModel":
        """Build complete functional GTAP model."""
        from pyomo.environ import ConcreteModel
        
        model = ConcreteModel(name="GTAP_Full_Model")
        
        self._add_sets(model)
        self._add_parameters(model)
        self._add_variables(model)
        # Keep benchmark seeds aligned with GAMS cal.gms scaling before
        # equation construction and residual evaluation.
        self.apply_production_scaling(model)
        # After xScale has been applied, re-sync xi[r,i] = i_share*xiagg so that
        # eq_xi AND eq_xaa_inv both start with zero residual.  This must happen
        # AFTER apply_production_scaling (which sets the final xiagg = yi/pi) and
        # BEFORE _add_equations (which captures base_mqgdp_00 from current xaa).
        self._align_xi_xaa_post_scaling(model)
        self._add_equations(model)
        self._add_objective(model)
        
        return model
    
    def _align_xi_xaa_post_scaling(self, model: "ConcreteModel") -> None:
        """Re-sync xi, xaa, xda, xma and all downstream aggregates with income-side xiagg.

        After apply_production_scaling, xiagg[r] = yi[r]/pi[r] uses the GAMS
        income-side identity (pi*depr*kstock + rsav + savf).  The xi variables
        were initialised from the demand-side SAM totals, which can differ by
        ~0.5% (e.g. EastAsia: demand 5.119 vs income 5.114).  This mismatch
        creates a ~2.6e-3 initial residual in eq_xi that PATH cannot reduce
        (code 2 / no_progress).

        Full cascade (mirrors the logic in apply_production_scaling but runs
        AFTER _refresh_macro_initial_state has set the income-side xiagg):

          xi, xaa[inv]       → eq_xi, eq_xaa_inv satisfied (residual = 0)
          xda[inv] *= k      → Armington shares preserved → eq_paa, eq_xda, eq_xma
          xma[inv] *= k      → same
          xd, xmt            → eq_xd_agg, eq_xmt_agg satisfied
          xds                → eq_pdeq satisfied
          xa                 → eq_xa satisfied
          xet (omega=inf)    → eq_xseq (supply identity xs=xds+xet) satisfied
          gw_share from xet  → eq_xw remains satisfied with old xw values
          gdpmp, rgdpmp      → eq_gdpmp, eq_pgdpmp satisfied

        The only remaining small residuals are in eq_xweq[rp,i,r] (Armington
        bilateral import demand), of order amw * Δxmt ≈ O(1e-4), easily handled
        by PATH in the first iteration.
        """
        from pyomo.environ import value as pyo_value

        # ---------- per-(r,i) xi and Armington updates -------------------------
        delta_xi: dict = {}
        for r in model.r:
            pi_val = max(float(pyo_value(model.pi[r])), 1e-8)
            xiagg = float(pyo_value(model.xiagg[r]))
            sigmai_raw = float(self.params.elasticities.esubi.get(r, 1.0))
            if abs(sigmai_raw - 1.0) < 1e-8:
                sigmai_raw = 1.01
            for i in model.i:
                share = float(pyo_value(model.i_share[r, i]))
                if share <= 0.0:
                    delta_xi[(r, i)] = 0.0
                    continue
                pa_inv = max(float(pyo_value(model.pa[r, i, GTAP_INVESTMENT_AGENT])), 1e-12)
                xi_old = float(pyo_value(model.xi[r, i]))
                xi_new = max(share * xiagg * (pi_val / pa_inv) ** sigmai_raw, 0.0)
                delta = xi_new - xi_old
                delta_xi[(r, i)] = delta

                # xi and xaa[inv]
                model.xi[r, i].set_value(xi_new)
                if hasattr(model, "xaa") and (r, i, GTAP_INVESTMENT_AGENT) in model.xaa:
                    model.xaa[r, i, GTAP_INVESTMENT_AGENT].set_value(xi_new)

                # Scale xda/xma proportionally so Armington shares are preserved.
                if xi_old > 1e-12 and abs(delta) > 1e-14:
                    k = xi_new / xi_old
                    if hasattr(model, "xda") and (r, i, GTAP_INVESTMENT_AGENT) in model.xda:
                        old = float(pyo_value(model.xda[r, i, GTAP_INVESTMENT_AGENT]))
                        model.xda[r, i, GTAP_INVESTMENT_AGENT].set_value(max(old * k, 0.0))
                    if hasattr(model, "xma") and (r, i, GTAP_INVESTMENT_AGENT) in model.xma:
                        old = float(pyo_value(model.xma[r, i, GTAP_INVESTMENT_AGENT]))
                        model.xma[r, i, GTAP_INVESTMENT_AGENT].set_value(max(old * k, 0.0))

        # ---------- recompute aggregates so all aggregation eqs are satisfied --
        for r in model.r:
            for i in model.i:
                # eq_xd_agg: xd = sum_aa(xda/xscale)
                if hasattr(model, "xd"):
                    total_xd = sum(
                        float(pyo_value(model.xda[r, i, aa]))
                        / max(float(pyo_value(model.xscale[r, aa])), 1e-12)
                        for aa in model.aa
                    )
                    model.xd[r, i].set_value(max(total_xd, 1e-8))

                # eq_xmt_agg: xmt = sum_aa(xma/xscale)
                if hasattr(model, "xmt"):
                    total_xmt = sum(
                        float(pyo_value(model.xma[r, i, aa]))
                        / max(float(pyo_value(model.xscale[r, aa])), 1e-12)
                        for aa in model.aa
                    )
                    model.xmt[r, i].set_value(max(total_xmt, 1e-8))

                # eq_pdeq: xds = sum_aa(xda/xscale for aa with domestic_share>0)
                # For simplicity use the same sum as xd (same result if all aa have share>0)
                if hasattr(model, "xds"):
                    model.xds[r, i].set_value(max(float(pyo_value(model.xd[r, i])), 1e-8))

                # eq_xa: xa = sum_aa(xaa/xscale) + vst
                if hasattr(model, "xa"):
                    total_xa = sum(
                        float(pyo_value(model.xaa[r, i, aa]))
                        / max(float(pyo_value(model.xscale[r, aa])), 1e-12)
                        for aa in model.aa
                    )
                    inventory = self._vst_value(str(r), str(i))
                    model.xa[r, i].set_value(max(total_xa + inventory, 1e-8))

                # eq_xseq (omega=inf case): xs = xds + xet  →  xet = xs - xds
                # For finite omega, eq_xseq is a price eq: skip xet update.
                omega = self.params.elasticities.omegax.get((r, i), float("inf"))
                if omega == float("inf") and hasattr(model, "xet") and hasattr(model, "xs"):
                    xs_val = float(pyo_value(model.xs[r, i]))
                    xds_val = float(pyo_value(model.xds[r, i]))
                    xet_old = max(float(pyo_value(model.xet[r, i])), 1e-12)
                    xet_new = max(xs_val - xds_val, 0.0)
                    if xet_new > 1e-12 and abs(xet_new - xet_old) > 1e-14:
                        model.xet[r, i].set_value(xet_new)
                        scale = xet_new / xet_old
                        # For omegaw=inf: eq_peteq says xet = sum(xw), eq_peeq says pe = pet.
                        # Scaling xw proportionally satisfies eq_peteq with xet_new.
                        # gw_share is NOT used in omegaw=inf equations, so leave unchanged.
                        if hasattr(model, "xw"):
                            omegaw = self.params.elasticities.omegaw.get((r, i), float("inf"))
                            if omegaw == float("inf"):
                                for rp in model.rp:
                                    if (r, i, rp) in model.xw:
                                        xw_old = float(pyo_value(model.xw[r, i, rp]))
                                        if xw_old > 0.0:
                                            model.xw[r, i, rp].set_value(xw_old * scale)

        # ---------- update gdpmp / rgdpmp: Δgdpmp = Σ_i Δxi per region ----------
        for r in model.r:
            delta_r = sum(delta_xi.get((r, i), 0.0) for i in model.i)
            if abs(delta_r) < 1e-14:
                continue
            old_gdpmp = float(pyo_value(model.gdpmp[r]))
            model.gdpmp[r].set_value(max(old_gdpmp + delta_r, 1e-8))
            old_rgdpmp = float(pyo_value(model.rgdpmp[r]))
            model.rgdpmp[r].set_value(max(old_rgdpmp + delta_r, 1e-8))

        updated = sum(1 for v in delta_xi.values() if abs(v) > 1e-14)
        if updated:
            logger.info(
                "_align_xi_xaa_post_scaling: updated %d (r,i) investment pairs "
                "(xi/xaa/xda/xma/xd/xmt/xds/xa/xet/gw_share/gdpmp) to income-side xiagg",
                updated,
            )

    def apply_production_scaling(self, model: "ConcreteModel") -> None:
        """Apply xScale to production variables after initialization.
        
        Following GAMS pattern (cal.gms lines 905-911):
        - Variables are initialized at benchmark (unscaled) values
        - After initialization, production-side variables are multiplied by xScale
        - This improves numerical conditioning for the solver
        
        Note: Python model has different indexing than GAMS:
        - GAMS: xd(r,i,a,t), xa(r,i,a,t), xm(r,i,a,t) - indexed by activity
        - Python: xd(r,i), xa(r,i) - NOT indexed by activity (aggregated)
        So we only scale variables indexed by activity: xf, xp, va, nd
        """
        from pyomo.environ import value

        final_demand_agents = (
            GTAP_HOUSEHOLD_AGENT,
            GTAP_GOVERNMENT_AGENT,
            GTAP_INVESTMENT_AGENT,
        )
        # Capture the base-year levels used by the compStat Fisher indices.
        # GAMS formulas mix current prices/quantities with the original t0 levels.
        base_pa = {
            (r, i, agent): float(value(model.pa[r, i, agent]))
            for r in model.r
            for i in model.i
            for agent in final_demand_agents
        }
        base_xaa = {
            (r, i, agent): float(value(model.xaa[r, i, agent]))
            for r in model.r
            for i in model.i
            for agent in final_demand_agents
        }
        base_pefob = {
            (r, i, rp): float(value(model.pefob[r, i, rp]))
            for r in model.r
            for i in model.i
            for rp in model.rp
        }
        base_pmcif = {
            (rp, i, r): float(value(model.pmcif[rp, i, r]))
            for rp in model.rp
            for i in model.i
            for r in model.r
        }
        base_xw = {
            (r, i, rp): float(value(model.xw[r, i, rp]))
            for r in model.r
            for i in model.i
            for rp in model.rp
        }
        base_pabs = {r: max(float(value(model.pabs[r])), 1e-8) for r in model.r}
        base_rgdpmp = {r: max(float(value(model.rgdpmp[r])), 1e-8) for r in model.r}
        
        # Scale factor demands (xf) - indexed by (r, f, a)
        for key in model.xf:
            r, f, a = key
            xscale_val = float(value(model.xscale[r, a]))
            if abs(xscale_val - 1.0) > 1e-12:
                xf_val = value(model.xf[key])
                if xf_val is not None:
                    model.xf[key].set_value(xf_val * xscale_val)

        # Scale production aggregates - indexed by (r, a)
        for key in model.xp:
            r, a = key
            xscale_val = float(value(model.xscale[r, a]))
            if abs(xscale_val - 1.0) > 1e-12:
                xp_val = value(model.xp[key])
                if xp_val is not None:
                    model.xp[key].set_value(xp_val * xscale_val)
        
        for key in model.va:
            r, a = key
            xscale_val = float(value(model.xscale[r, a]))
            if abs(xscale_val - 1.0) > 1e-12:
                va_val = value(model.va[key])
                if va_val is not None:
                    model.va[key].set_value(va_val * xscale_val)
        
        for key in model.nd:
            r, a = key
            xscale_val = float(value(model.xscale[r, a]))
            if abs(xscale_val - 1.0) > 1e-12:
                nd_val = value(model.nd[key])
                if nd_val is not None:
                    model.nd[key].set_value(nd_val * xscale_val)

        # GAMS also rescales activity-level Armington quantities on the
        # production side: xd(r,i,a), xm(r,i,a), xa(r,i,a).
        for var_name in ("xda", "xma", "xaa"):
            if not hasattr(model, var_name):
                continue
            var = getattr(model, var_name)
            for key in var:
                if not isinstance(key, tuple) or len(key) != 3:
                    continue
                r, _i, aa = key
                if aa not in self.sets.a:
                    continue
                xscale_val = float(value(model.xscale[r, aa]))
                if abs(xscale_val - 1.0) > 1e-12:
                    level = value(var[key])
                    if level is not None:
                        var[key].set_value(level * xscale_val)

        # Refresh Armington aggregates after scaling activity-level xda/xma/xaa.
        if hasattr(model, "xd"):
            for r in model.r:
                for i in model.i:
                    total_xd = sum(
                        value(model.xda[r, i, aa]) / max(value(model.xscale[r, aa]), 1e-12)
                        for aa in model.aa
                    )
                    model.xd[r, i].set_value(max(total_xd, 1e-8))

        if hasattr(model, "xa"):
            for r in model.r:
                for i in model.i:
                    total_xa = sum(
                        value(model.xaa[r, i, aa]) / max(value(model.xscale[r, aa]), 1e-12)
                        for aa in model.aa
                    )
                    inventory = self._vst_value(str(r), str(i))
                    model.xa[r, i].set_value(max(total_xa + inventory, 1e-8))

        if hasattr(model, "xds") and hasattr(model, "xda"):
            for r in model.r:
                for i in model.i:
                    total_xds = sum(
                        value(model.xda[r, i, aa]) / max(value(model.xscale[r, aa]), 1e-12)
                        for aa in model.aa
                    )
                    model.xds[r, i].set_value(max(total_xds, 1e-8))

        if hasattr(model, "xet") and hasattr(model, "xs") and hasattr(model, "xds"):
            for r in model.r:
                for i in model.i:
                    has_export_route = any(
                        value(model.xw_flag[r, i, rp]) > 0.0
                        for rp in model.rp
                    )
                    if not has_export_route:
                        lb = model.xet[r, i].lb
                        if lb is not None and float(lb) > 0.0:
                            model.xet[r, i].setlb(0.0)
                        model.xet[r, i].set_value(0.0)
                        if hasattr(model, "xet_flag"):
                            model.xet_flag[r, i].set_value(0.0)
                        continue

                    # Match GAMS cal.gms initialization:
                    # xet.l = (ps.l*xs.l - pd.l*xds.l) / pet.l
                    numerator = (
                        value(model.ps[r, i]) * value(model.xs[r, i])
                        - value(model.pd[r, i]) * value(model.xds[r, i])
                    )
                    pet_val = max(value(model.pet[r, i]), 1e-12)
                    xet_val = max(numerator / pet_val, 0.0)
                    if xet_val <= 0.0:
                        lb = model.xet[r, i].lb
                        if lb is not None and float(lb) > 0.0:
                            model.xet[r, i].setlb(0.0)
                    model.xet[r, i].set_value(xet_val)
                    if hasattr(model, "xet_flag"):
                        model.xet_flag[r, i].set_value(1.0 if xet_val > 1e-7 else 0.0)

        if hasattr(model, "gw_share") and hasattr(model, "xw") and hasattr(model, "xet"):
            for r in model.r:
                for i in model.i:
                    xet_val = max(value(model.xet[r, i]), 1e-12)
                    omegaw = self.params.elasticities.omegaw.get((r, i), float("inf"))
                    for rp in model.rp:
                        if value(model.xw_flag[r, i, rp]) <= 0.0:
                            model.gw_share[r, i, rp].set_value(0.0)
                            continue
                        xw_val = max(value(model.xw[r, i, rp]), 0.0)
                        pe_val = max(value(model.pe[r, i, rp]), 1e-12)
                        pet_val = max(value(model.pet[r, i]), 1e-12)
                        if omegaw == float("inf"):
                            share = (pe_val * xw_val) / max(pet_val * xet_val, 1e-12)
                        else:
                            share = (xw_val / xet_val) * (pet_val / pe_val) ** omegaw
                        model.gw_share[r, i, rp].set_value(max(share, 0.0))
                    if hasattr(model, "xet_flag"):
                        model.xet_flag[r, i].set_value(1.0 if xet_val > 1e-7 else 0.0)

        self._refresh_cet_share_state(model)

        if hasattr(model, "xmt") and hasattr(model, "xma"):
            for r in model.r:
                for i in model.i:
                    total_xm = sum(
                        value(model.xma[r, i, aa]) / max(value(model.xscale[r, aa]), 1e-12)
                        for aa in model.aa
                    )
                    model.xmt[r, i].set_value(max(total_xm, 1e-8))

        self._refresh_cet_share_state(model)

        # Enforce benchmark household demand coherence before macro refresh:
        # yc = sum_i pa*hhd * xc and xc = c_share * yc / pa.
        if hasattr(model, "xc") and hasattr(model, "c_share") and hasattr(model, "yc"):
            for r in model.r:
                yc_target = sum(
                    float(self.params.benchmark.get_private_demand(str(r), str(i))[0] or 0.0)
                    for i in model.i
                )
                yc_target = max(yc_target, 1e-8)
                model.yc[r].set_value(yc_target)
                for i in model.i:
                    share = max(float(value(model.c_share[r, i]) or 0.0), 0.0)
                    pa_hhd = max(float(value(model.pa[r, i, GTAP_HOUSEHOLD_AGENT]) or 1.0), 1e-12)
                    model.xc[r, i].set_value(max((share * yc_target) / pa_hhd, 0.0))

        if hasattr(model, "xaa") and hasattr(model, "xc"):
            for r in model.r:
                for i in model.i:
                    model.xaa[r, i, GTAP_HOUSEHOLD_AGENT].set_value(max(value(model.xc[r, i]), 0.0))
        if hasattr(model, "xaa") and hasattr(model, "xg"):
            for r in model.r:
                for i in model.i:
                    model.xaa[r, i, GTAP_GOVERNMENT_AGENT].set_value(max(value(model.xg[r, i]), 0.0))
        if hasattr(model, "xaa") and hasattr(model, "xi"):
            for r in model.r:
                for i in model.i:
                    model.xaa[r, i, GTAP_INVESTMENT_AGENT].set_value(max(value(model.xi[r, i]), 0.0))

        self._refresh_macro_initial_state(model)

    def _refresh_macro_initial_state(self, model: "ConcreteModel") -> None:
        """Refresh macro variables after any xScale-sensitive initialization changes."""
        from pyomo.environ import value

        final_demand_agents = (
            GTAP_HOUSEHOLD_AGENT,
            GTAP_GOVERNMENT_AGENT,
            GTAP_INVESTMENT_AGENT,
        )
        base_pa = {
            (r, i, agent): float(value(model.pa[r, i, agent]))
            for r in model.r
            for i in model.i
            for agent in final_demand_agents
        }
        base_xaa = {
            (r, i, agent): float(value(model.xaa[r, i, agent]))
            for r in model.r
            for i in model.i
            for agent in final_demand_agents
        }
        base_pefob = {
            (r, i, rp): float(value(model.pefob[r, i, rp]))
            for r in model.r
            for i in model.i
            for rp in model.rp
        }
        base_pmcif = {
            (rp, i, r): float(value(model.pmcif[rp, i, r]))
            for rp in model.rp
            for i in model.i
            for r in model.r
        }
        base_xw = {
            (r, i, rp): float(value(model.xw[r, i, rp]))
            for r in model.r
            for i in model.i
            for rp in model.rp
        }
        base_pabs = {r: max(float(value(model.pabs[r])), 1e-8) for r in model.r}
        base_rgdpmp = {r: max(float(value(model.rgdpmp[r])), 1e-8) for r in model.r}

        for r in model.r:
            capital_factors = [f for f in model.f if str(f).lower() in ("capital", "cap", "k", "kap")]
            for f in model.f:
                if hasattr(model, "xft") and (r, f) in model.xft and f in self.sets.mf:
                    model.xft[r, f].set_value(
                        sum(
                            value(model.xf[r, f, a]) / max(value(model.xscale[r, a]), 1e-12)
                            for a in model.a
                        )
                    )

            if hasattr(model, "kstock") and r in model.kstock:
                raw_vkb = self.params.benchmark.vkb
                benchmark_kstock_val = raw_vkb.get(r)
                if benchmark_kstock_val is None:
                    benchmark_kstock_val = raw_vkb.get((r,), 0.0)
                benchmark_kstock = float(benchmark_kstock_val or 0.0)
                if benchmark_kstock > 0.0:
                    model.kstock[r].set_value(max(benchmark_kstock, 1e-8))

            if hasattr(model, "ytax") and (r, "ft") in model.ytax:
                ft_total = 0.0
                for (rr, f, a), rtf in self.params.taxes.rtf.items():
                    if rr != r:
                        continue
                    ft_total += (
                        float(rtf)
                        * value(model.pf[r, f, a])
                        * value(model.xf[r, f, a])
                        / max(value(model.xscale[r, a]), 1e-12)
                    )
                model.ytax[r, "ft"].set_value(ft_total)

            if hasattr(model, "ytax") and (r, "dt") in model.ytax:
                dt_total = 0.0
                for f in model.f:
                    for a in model.a:
                        kappa = float(self.params.taxes.kappaf_activity.get((r, f, a), 0.0) or 0.0)
                        if kappa == 0.0:
                            kappa = float(self.params.taxes.kappaf.get((r, f), 0.0) or 0.0)
                        if kappa == 0.0:
                            continue
                        dt_total += (
                            kappa
                            * value(model.pf[r, f, a])
                            * value(model.xf[r, f, a])
                            / max(value(model.xscale[r, a]), 1e-12)
                        )
                model.ytax[r, "dt"].set_value(dt_total)

            # Recalibrate ytax[pt] (production/output tax) from eq_ytax rule: 
            # sum_a sum_i prdtx_rai * p_rai * x (where xflag > 0)
            if hasattr(model, "ytax") and (r, "pt") in model.ytax:
                pt_total = 0.0
                for a in model.a:
                    outputs = self.sets.activity_commodities.get(str(a), list(self.sets.i))
                    for i in outputs:
                        if (r, a, i) not in model.xflag or value(model.xflag[r, a, i]) <= 0.0:
                            continue
                        prdtx = value(model.prdtx_rai[r, a, i]) if (r, a, i) in model.prdtx_rai else 0.0
                        if prdtx == 0.0:
                            continue
                        p_rai_v = value(model.p_rai[r, a, i]) if (r, a, i) in model.p_rai else 1.0
                        x_v = value(model.x[r, a, i]) if (r, a, i) in model.x else 0.0
                        pt_total += prdtx * p_rai_v * x_v
                model.ytax[r, "pt"].set_value(pt_total)

            # Recalibrate ytax[pc/gc/ic] (commodity taxes on private/gov/invest) from eq_ytax rule
            _ctax_agents = {
                "pc": [GTAP_HOUSEHOLD_AGENT],
                "gc": [GTAP_GOVERNMENT_AGENT],
                "ic": [GTAP_INVESTMENT_AGENT],
                "fc": list(model.a),
            }
            for gy, agents in _ctax_agents.items():
                if not (hasattr(model, "ytax") and (r, gy) in model.ytax):
                    continue
                ctax_total = 0.0
                for aa in agents:
                    for i in model.i:
                        dintx = float(self.params.taxes.dintx0.get((r, i, aa), 0.0) or 0.0)
                        mintx = float(self.params.taxes.mintx0.get((r, i, aa), 0.0) or 0.0)
                        scale = value(model.xscale[r, aa]) if aa in model.a and (r, aa) in model.xscale else 1.0
                        if dintx != 0.0 and (r, i, aa) in model.xda:
                            ctax_total += dintx * value(model.pd[r, i]) * value(model.xda[r, i, aa]) / max(scale, 1e-12)
                        if mintx != 0.0 and (r, i, aa) in model.xma:
                            ctax_total += mintx * value(model.pmt[r, i]) * value(model.xma[r, i, aa]) / max(scale, 1e-12)
                model.ytax[r, gy].set_value(ctax_total)

            if hasattr(model, "facty") and r in model.facty:
                gross_factor_income = sum(
                    value(model.pf[r, f, a]) * value(model.xf[r, f, a]) / max(value(model.xscale[r, a]), 1e-12)
                    for f in model.f
                    for a in model.a
                )
                model.facty[r].set_value(
                    gross_factor_income - value(model.fdepr[r]) * value(model.pi[r]) * value(model.kstock[r])
                )

            if hasattr(model, "ytaxTot") and r in model.ytaxTot:
                model.ytaxTot[r].set_value(sum(value(model.ytax[r, gy]) for gy in model.gy))
            if hasattr(model, "ytax_ind") and r in model.ytax_ind:
                model.ytax_ind[r].set_value(value(model.ytaxTot[r]) - value(model.ytax[r, "dt"]))
            if hasattr(model, "regy") and r in model.regy:
                model.regy[r].set_value(value(model.facty[r]) + value(model.ytax_ind[r]))
            regy_raw = value(model.regy[r])
            regy_val = max(abs(regy_raw), 1e-8)
            if hasattr(model, "ytaxshr"):
                for gy in model.gy:
                    model.ytaxshr[r, gy].set_value(value(model.ytax[r, gy]) / regy_val)
            if hasattr(model, "yc") and r in model.yc:
                # GAMS benchmark identity (cal.gms): yc = sum_i pa(r,i,hhd) * xa(r,i,hhd)
                yc_demand = sum(
                    value(model.pa[r, i, GTAP_HOUSEHOLD_AGENT]) * value(model.xaa[r, i, GTAP_HOUSEHOLD_AGENT])
                    for i in model.i
                    if (r, i, GTAP_HOUSEHOLD_AGENT) in model.pa and (r, i, GTAP_HOUSEHOLD_AGENT) in model.xaa
                )
                if yc_demand > 0.0:
                    model.yc[r].set_value(yc_demand)
                else:
                    model.yc[r].set_value(
                        value(model.betap[r]) * (value(model.phi[r]) / max(value(model.phip[r]), 1e-8)) * regy_raw
                    )
            if hasattr(model, "yg") and r in model.yg:
                # GAMS benchmark identity (cal.gms): yg = sum_i pa(r,i,gov) * xa(r,i,gov)
                yg_demand = sum(
                    value(model.pa[r, i, GTAP_GOVERNMENT_AGENT]) * value(model.xaa[r, i, GTAP_GOVERNMENT_AGENT])
                    for i in model.i
                    if (r, i, GTAP_GOVERNMENT_AGENT) in model.pa and (r, i, GTAP_GOVERNMENT_AGENT) in model.xaa
                )
                if yg_demand > 0.0:
                    model.yg[r].set_value(yg_demand)
                else:
                    model.yg[r].set_value(value(model.betag[r]) * value(model.phi[r]) * regy_raw)
            if hasattr(model, "rsav") and r in model.rsav:
                # Prefer SAM benchmark save directly so that eq_yi holds at init.
                # Using betas*phi*regy_raw can drift if regy_raw ≠ regy_gams.
                # GAMS cal.gms sets rsav.l(r) = save(r) (benchmark identity).
                save_bench = float(self.params.benchmark.save.get(str(r), 0.0))
                if save_bench > 0.0:
                    model.rsav[r].set_value(save_bench)
                else:
                    model.rsav[r].set_value(value(model.betas[r]) * value(model.phi[r]) * regy_raw)
            # Recalibrate aus consistently with final rsav.
            # GAMS identity: aus.l(r) = pop.l(r) / (psave.l(r) * rsav.l(r))
            # This ensures eq_us: us = aus*rsav/(psave*pop) gives us=1 at benchmark.
            if hasattr(model, "aus") and r in model.aus:
                rsav_val = value(model.rsav[r]) if hasattr(model, "rsav") and r in model.rsav else 0.0
                pop_val = value(model.pop[r])
                psave_val = (
                    value(model.psave[r])
                    if hasattr(model, "psave") and r in model.psave
                    else 1.0
                )
                if rsav_val > 1e-12 and pop_val > 1e-12:
                    model.aus[r].set_value(pop_val / (psave_val * rsav_val))
            # Re-calibrate betap/betag/betas from post-init yc/yg/rsav so that
            # eq_yc/yg/rsav hold at the benchmark initial point (GAMS cal.gms:619).
            if regy_val > 1e-8 and hasattr(model, "betap") and r in model.betap:
                yc_final = value(model.yc[r]) if hasattr(model, "yc") and r in model.yc else 0.0
                yg_final = value(model.yg[r]) if hasattr(model, "yg") and r in model.yg else 0.0
                rsav_final = value(model.rsav[r]) if hasattr(model, "rsav") and r in model.rsav else 0.0
                phi_val = max(value(model.phi[r]), 1e-12)
                phip_val = max(value(model.phip[r]), 1e-12)
                model.betap[r].set_value(yc_final * phip_val / (phi_val * max(regy_raw, 1e-12)))
                model.betag[r].set_value(yg_final / (phi_val * max(regy_raw, 1e-12)))
                model.betas[r].set_value(rsav_final / (phi_val * max(regy_raw, 1e-12)))
            if hasattr(model, "chif") and r in model.chif:
                model.chif[r].set_value(value(model.savf[r]) / regy_val)
            if hasattr(model, "yi") and r in model.yi:
                # Compute yi from the income identity so eq_yi = 0 at init.
                # rsav is already initialized to save_param (the GAMS benchmark),
                # so yi_formula ≈ yi_gams within numerical precision (~2-3e-6).
                # This gives a strictly feasible starting point for eq_yi.
                model.yi[r].set_value(
                    value(model.pi[r]) * value(model.depr[r]) * value(model.kstock[r])
                    + value(model.rsav[r])
                    + value(model.savf[r])
                )
            if hasattr(model, "us") and r in model.us:
                model.us[r].set_value(
                    value(model.aus[r]) * value(model.rsav[r])
                    / max(value(model.psave[r]) * value(model.pop[r]), 1e-8)
                )

            if hasattr(model, "pmt"):
                for i in model.i:
                    esubm = self.params.elasticities.esubm.get((r, i), 5.0)
                    expo = 1.0 - esubm
                    terms = []
                    for rp in model.r:
                        amw = float(self.params.shares.normalized.import_source_share.get((r, i, rp), 0.0) or 0.0)
                        if amw <= 0.0:
                            continue
                        bilateral_exports = float(self.params.benchmark.vxmd.get((rp, i, r), 0.0) or 0.0)
                        bilateral_imports = float(self.params.benchmark.vcif.get((rp, i, r), 0.0) or 0.0)
                        if bilateral_exports <= 0.0 and bilateral_imports <= 0.0:
                            continue
                        if bilateral_exports > 0.0 and bilateral_imports > 0.0:
                            pmcif = max(bilateral_imports / bilateral_exports, 1e-8)
                        elif bilateral_imports > 0.0:
                            pmcif = max(bilateral_imports, 1e-8)
                        else:
                            export_tax = float(self.params.taxes.rtxs.get((rp, i, r), 0.0) or 0.0)
                            tmarg = sum(self.params.benchmark.vtwr.get((rp, i, r, margin), 0.0) for margin in self.sets.m)
                            tmarg = tmarg / max(bilateral_exports, 1e-12) if bilateral_exports > 0.0 else 0.0
                            pmcif = max(1.0 + export_tax + tmarg, 1e-8)
                        imptx = float(self.params.taxes.imptx.get((rp, i, r), 0.0) or 0.0)
                        pm = max((1.0 + imptx) * pmcif, 1e-8)
                        terms.append(amw * (pm ** expo))
                    if terms:
                        rhs = sum(terms)
                        if rhs > 0.0:
                            model.pmt[r, i].set_value(max(rhs ** (1.0 / expo), 1e-8))

            mqabs_tt = 0.0
            mqabs_t0 = 0.0
            mqabs_0t = 0.0
            mqabs_00 = 0.0
            for i in model.i:
                for agent in final_demand_agents:
                    pa_t = float(value(model.pa[r, i, agent]))
                    xa_t = float(value(model.xaa[r, i, agent]))
                    pa_0 = base_pa[(r, i, agent)]
                    xa_0 = base_xaa[(r, i, agent)]
                    mqabs_tt += pa_t * xa_t
                    mqabs_t0 += pa_t * xa_0
                    mqabs_0t += pa_0 * xa_t
                    mqabs_00 += pa_0 * xa_0

            mqtrade_tt = 0.0
            mqtrade_t0 = 0.0
            mqtrade_0t = 0.0
            mqtrade_00 = 0.0
            for i in model.i:
                for rp in model.rp:
                    pexp_t = float(value(model.pefob[r, i, rp]))
                    pexp_0 = base_pefob[(r, i, rp)]
                    xexp_t = float(value(model.xw[r, i, rp]))
                    xexp_0 = base_xw[(r, i, rp)]
                    pimp_t = float(value(model.pmcif[rp, i, r]))
                    pimp_0 = base_pmcif[(rp, i, r)]
                    ximp_t = float(value(model.xw[rp, i, r]))
                    ximp_0 = base_xw[(rp, i, r)]

                    mqtrade_tt += pexp_t * xexp_t - pimp_t * ximp_t
                    mqtrade_t0 += pexp_t * xexp_0 - pimp_t * ximp_0
                    mqtrade_0t += pexp_0 * xexp_t - pimp_0 * ximp_t
                    mqtrade_00 += pexp_0 * xexp_0 - pimp_0 * ximp_0

            gdp_current = max(mqabs_tt + mqtrade_tt, 1e-8)
            model.gdpmp[r].set_value(gdp_current)

            if mqabs_00 > 1e-12 and mqabs_0t > 1e-12:
                pabs_fisher = base_pabs[r] * math.sqrt((mqabs_t0 / mqabs_00) * (mqabs_tt / mqabs_0t))
                model.pabs[r].set_value(max(pabs_fisher, 1e-8))

            mqgdp_00 = mqabs_00 + mqtrade_00
            mqgdp_t0 = mqabs_t0 + mqtrade_t0
            mqgdp_0t = mqabs_0t + mqtrade_0t
            if self.is_counterfactual:
                if mqgdp_00 > 1e-12 and mqgdp_t0 > 1e-12 and mqgdp_0t > 1e-12:
                    rgdp_fisher = base_rgdpmp[r] * math.sqrt((gdp_current / mqgdp_00) * (mqgdp_0t / mqgdp_t0))
                    model.rgdpmp[r].set_value(max(rgdp_fisher, 1e-8))
            else:
                # Baseline: rgdpmp = gdpmp (replicates GAMS rgdpmp.l = gdpmp.l assignment).
                model.rgdpmp[r].set_value(max(gdp_current, 1e-8))
            model.pgdpmp[r].set_value(max(gdp_current / max(value(model.rgdpmp[r]), 1e-8), 1e-8))
            pi_val = max(value(model.pi[r]), 1e-8)
            model.xiagg[r].set_value(max(value(model.yi[r]) / pi_val, 1e-8))
            model.kapEnd[r].set_value(
                max((1.0 - value(model.depr[r])) * value(model.kstock[r]) + value(model.xiagg[r]), 1e-8)
            )

            cap_return = 0.0
            for f in capital_factors:
                for a in model.a:
                    kappa = float(self.params.taxes.kappaf_activity.get((r, f, a), 0.0))
                    cap_return += (
                        (1.0 - kappa)
                        * value(model.pf[r, f, a])
                        * value(model.xf[r, f, a])
                        / max(value(model.xscale[r, a]), 1e-12)
                    )
            arent_val = cap_return / max(value(model.kstock[r]), 1e-8) if capital_factors else 0.0
            model.arent[r].set_value(max(arent_val, 1e-8))
            rorc_val = arent_val / pi_val - value(model.fdepr[r])
            model.rorc[r].set_value(rorc_val)
            rore_val = rorc_val * (value(model.kstock[r]) / max(value(model.kapEnd[r]), 1e-8)) ** value(model.rorflex[r])
            model.rore[r].set_value(rore_val)

        xigbl_current = sum(
            value(model.xiagg[r]) - value(model.depr[r]) * value(model.kstock[r])
            for r in model.r
        )
        model.xigbl.set_value(max(xigbl_current, 1e-8))
        pigbl_numer = sum(
            value(model.pi[r]) * (value(model.xiagg[r]) - value(model.depr[r]) * value(model.kstock[r]))
            for r in model.r
        )
        model.pigbl.set_value(max(pigbl_numer / max(value(model.xigbl), 1e-8), 1e-8))
        rorg_numer = sum(
            value(model.rore[r]) * value(model.pi[r]) * (value(model.xiagg[r]) - value(model.depr[r]) * value(model.kstock[r]))
            for r in model.r
        )
        model.rorg.set_value(rorg_numer / max(pigbl_numer, 1e-8))
        residual_gap = sum(
            value(model.yi[r]) - (value(model.pi[r]) * value(model.depr[r]) * value(model.kstock[r]) + value(model.rsav[r]) + value(model.savf[r]))
            for r in model.r
            if str(r) == "RestofWorld"
        )
        model.walras.set_value(residual_gap)

    def _refresh_cet_share_state(self, model: "ConcreteModel") -> None:
        """Recalibrate top-level CET shares from the current benchmark-consistent state.

        GAMS calibrates `gd` and `ge` after levels/prices are in place:
        gd = (xds/xs) * (ps/pd)^omegax
        ge = (xet/xs) * (ps/pet)^omegax
        """
        from pyomo.environ import value

        if not hasattr(model, "gd_share") or not hasattr(model, "ge_share"):
            return

        for r in model.r:
            for i in model.i:
                xs_val = max(value(model.xs[r, i]), 1e-12)
                xds_val = max(value(model.xds[r, i]), 0.0)
                xet_val = max(value(model.xet[r, i]), 0.0)
                pd_val = max(value(model.pd[r, i]), 1e-12)
                ps_val = max(value(model.ps[r, i]), 1e-12)
                pet_val = max(value(model.pet[r, i]), 1e-12)
                omega = self.params.elasticities.omegax.get((r, i), float("inf"))

                if omega == float("inf"):
                    gd_val = (pd_val * xds_val) / max(ps_val * xs_val, 1e-12)
                    ge_val = (pet_val * xet_val) / max(ps_val * xs_val, 1e-12)
                else:
                    gd_val = (xds_val / xs_val) * (ps_val / pd_val) ** omega
                    ge_val = (xet_val / xs_val) * (ps_val / pet_val) ** omega

                model.gd_share[r, i].set_value(max(gd_val, 0.0))
                model.ge_share[r, i].set_value(max(ge_val, 0.0))

    def _compute_ytax_ind_bench(self, region: str) -> float:
        """Compute total indirect tax revenue for a region at benchmark.

        Uses the same formula as get_ytax_ind_init so that betap/regy calibration
        is consistent with variable initialization.
        """
        bm = self.params.benchmark
        taxes = self.params.taxes
        sets = self.sets
        total = 0.0
        for a in sets.a:
            outputs = sets.activity_commodities.get(a, list(sets.i))
            for i in outputs:
                total += float(bm.makb.get((region, a, i), 0.0) or 0.0) \
                       - float(bm.maks.get((region, a, i), 0.0) or 0.0)
        for (rr, f, a), rtf in taxes.rtf.items():
            if rr == region:
                # Use evfb to match eq_ytax["ft"] = sum(rtf * pf * xf / xscale) = sum(rtf * evfb)
                evfb_val = bm.evfb.get((region, f, a), bm.vfm.get((region, f, a), 0.0))
                total += float(rtf) * float(evfb_val or 0.0)
        for (rr, i, a), rtpd in taxes.rtpd.items():
            if rr == region:
                total += float(rtpd) * float(bm.vdfb.get((region, i, a), 0.0))
        for (rr, i, a), rtpi in taxes.rtpi.items():
            if rr == region:
                total += float(rtpi) * float(bm.vmfb.get((region, i, a), 0.0))
        for (rr, i), rtgd in taxes.rtgd.items():
            if rr == region:
                total += float(rtgd) * float(bm.vdgb.get((region, i), 0.0))
        for (rr, i), rtgi in taxes.rtgi.items():
            if rr == region:
                total += float(rtgi) * float(bm.vmgb.get((region, i), 0.0))
        for i in sets.i:
            total += float(bm.vdpp.get((region, i), 0.0) or 0.0) - float(bm.vdpb.get((region, i), 0.0) or 0.0)
            total += float(bm.vmpp.get((region, i), 0.0) or 0.0) - float(bm.vmpb.get((region, i), 0.0) or 0.0)
            total += float(bm.vdip.get((region, i), 0.0) or 0.0) - float(bm.vdib.get((region, i), 0.0) or 0.0)
            total += float(bm.vmip.get((region, i), 0.0) or 0.0) - float(bm.vmib.get((region, i), 0.0) or 0.0)
        for (rr, i, rp), rtxs in taxes.rtxs.items():
            if rr == region:
                total += float(rtxs) * float(bm.vxsb.get((region, i, rp), 0.0))
        for (exporter, i, importer), rate in taxes.imptx.items():
            if importer == region:
                vmsb_val = bm.vmsb.get((exporter, i, region), 0.0)
                total += float(rate) * float(vmsb_val or 0.0)
        return total

    def _add_sets(self, model: "ConcreteModel") -> None:
        """Add sets."""
        from pyomo.environ import Set

        agent_labels = list(self.sets.a) + [
            GTAP_HOUSEHOLD_AGENT,
            GTAP_GOVERNMENT_AGENT,
            GTAP_INVESTMENT_AGENT,
            GTAP_MARGIN_AGENT,
        ]
        tax_streams = ["pt", "fc", "pc", "gc", "ic", "dt", "mt", "et", "ft", "fs"]
        
        model.r = Set(initialize=self.sets.r, doc="Regions")
        model.i = Set(initialize=self.sets.i, doc="Commodities")
        model.a = Set(initialize=self.sets.a, doc="Activities")
        model.f = Set(initialize=self.sets.f, doc="Factors")
        model.mf = Set(initialize=self.sets.mf, doc="Mobile factors")
        model.sf = Set(initialize=self.sets.sf, doc="Specific factors")
        model.m = Set(initialize=self.sets.m, doc="Margin commodities")
        model.aa = Set(initialize=agent_labels, doc="Absorption agents and activities")
        model.gy = Set(initialize=tax_streams, doc="Government tax streams")
        
        # Aliases for trade
        model.rp = Set(initialize=self.sets.r, doc="Regions (alias)")
    
    def _add_parameters(self, model: "ConcreteModel") -> None:
        """Add all parameters."""
        from pyomo.environ import Param

        # Match GAMS cal.gms scaling:
        #   xScale(r,aa) = 1
        #   xScale(r,a)$xpFlag(r,a) = xpScale*10**(-round(log10(xp.l(r,a,t0))))
        # where xpScale=1 and only activity columns are rescaled.
        xscale_data: Dict[tuple[str, str], float] = {}
        for r in self.sets.r:
            for aa in [
                GTAP_HOUSEHOLD_AGENT,
                GTAP_GOVERNMENT_AGENT,
                GTAP_INVESTMENT_AGENT,
                GTAP_MARGIN_AGENT,
            ]:
                xscale_data[(r, aa)] = 1.0

        def _gams_round(value: float) -> int:
            if value >= 0.0:
                return int(math.floor(value + 0.5))
            return int(math.ceil(value - 0.5))

        for r in self.sets.r:
            for a in self.sets.a:
                # Keep xscale calibration on the same benchmark identity used by
                # production calibration (xp = nd + va at purchaser value).
                nd_level = sum(
                    float(self.params.benchmark.vdfp.get((r, i, a), 0.0) or 0.0)
                    + float(self.params.benchmark.vmfp.get((r, i, a), 0.0) or 0.0)
                    for i in self.sets.i
                )
                if nd_level <= 0.0:
                    nd_level = sum(
                        float(self.params.benchmark.vdfm.get((r, i, a), 0.0) or 0.0)
                        + float(self.params.benchmark.vifm.get((r, i, a), 0.0) or 0.0)
                        for i in self.sets.i
                    )

                va_level = 0.0
                for f in self.sets.f:
                    evfb_val = float(
                        self.params.benchmark.evfb.get(
                            (r, f, a),
                            self.params.benchmark.vfm.get((r, f, a), 0.0),
                        )
                        or 0.0
                    )
                    if evfb_val <= 0.0:
                        continue
                    factor_tax = float(self.params.taxes.rtf.get((r, f, a), 0.0) or 0.0)
                    va_level += evfb_val * (1.0 + factor_tax)

                xp_level = nd_level + va_level
                if xp_level <= 0.0:
                    xp_level = float(self.params.benchmark.vom.get((r, a), 0.0) or 0.0)
                if xp_level <= 0.0:
                    xp_level = sum(
                        float(self.params.benchmark.makb.get((r, a, i), 0.0) or 0.0)
                        for i in self.sets.i
                    )
                if self.reference_snapshot is not None:
                    ref_xp = self.reference_snapshot.xp.get((r, a))
                    if ref_xp is not None and float(ref_xp) > 0.0:
                        xp_level = float(ref_xp)
                if xp_level > 0.0:
                    exponent = _gams_round(math.log10(xp_level))
                    xscale_data[(r, a)] = float(10.0 ** (-exponent))
                else:
                    xscale_data[(r, a)] = 1.0
        
        # Helper to create indexed parameters
        def create_indexed_param(
            name: str,
            index_sets,
            data: Dict,
            default: float = 0.0,
            *,
            mutable: bool = False,
        ):
            if not data:
                return
            # Build index values
            values = {}
            for key, value in data.items():
                if isinstance(key, tuple):
                    values[key] = value
                else:
                    values[(key,)] = value
            
            # Get pyomo sets for indexing
            if len(index_sets) == 1:
                idx_set = getattr(model, index_sets[0])
                setattr(model, name, Param(idx_set, initialize=values, default=default, doc=name, mutable=mutable))
            elif len(index_sets) == 2:
                idx_set1 = getattr(model, index_sets[0])
                idx_set2 = getattr(model, index_sets[1])
                setattr(
                    model,
                    name,
                    Param(idx_set1, idx_set2, initialize=values, default=default, doc=name, mutable=mutable),
                )
            elif len(index_sets) == 3:
                idx_set1 = getattr(model, index_sets[0])
                idx_set2 = getattr(model, index_sets[1])
                idx_set3 = getattr(model, index_sets[2])
                setattr(
                    model,
                    name,
                    Param(
                        idx_set1,
                        idx_set2,
                        idx_set3,
                        initialize=values,
                        default=default,
                        doc=name,
                        mutable=mutable,
                    ),
                )
            elif len(index_sets) == 4:
                idx_set1 = getattr(model, index_sets[0])
                idx_set2 = getattr(model, index_sets[1])
                idx_set3 = getattr(model, index_sets[2])
                idx_set4 = getattr(model, index_sets[3])
                setattr(
                    model,
                    name,
                    Param(
                        idx_set1,
                        idx_set2,
                        idx_set3,
                        idx_set4,
                        initialize=values,
                        default=default,
                        doc=name,
                        mutable=mutable,
                    ),
                )
        
        # Elasticities
        create_indexed_param("esubva", ["r", "a"], self.params.elasticities.esubva, 1.0)
        create_indexed_param("esubd", ["r", "i"], self.params.elasticities.esubd, 2.0)
        create_indexed_param("esubm", ["r", "i"], self.params.elasticities.esubm, 4.0)
        create_indexed_param("omegax", ["r", "i"], self.params.elasticities.omegax, float("inf"))
        
        # Benchmark values
        create_indexed_param("vom", ["r", "a"], self.params.benchmark.vom, 0.0)
        create_indexed_param("vfm", ["r", "f", "a"], self.params.benchmark.vfm, 0.0)

        # Flags (GAMS-style) derived from benchmark flows
        xflag_data: Dict[tuple[str, str, str], float] = {}
        xfflag_data: Dict[tuple[str, str, str], float] = {}
        xftflag_data: Dict[tuple[str, str], float] = {}
        for r in self.sets.r:
            for a in self.sets.a:
                for i in self.sets.i:
                    val = self.params.benchmark.makb.get((r, a, i), 0.0)
                    xflag_data[(r, a, i)] = 1.0 if abs(val) > 1e-12 else 0.0
            for f in self.sets.f:
                any_flow = False
                for a in self.sets.a:
                    val = self.params.benchmark.evfb.get((r, f, a), self.params.benchmark.vfm.get((r, f, a), 0.0))
                    if abs(val) > 1e-12:
                        xfflag_data[(r, f, a)] = 1.0
                        any_flow = True
                    else:
                        xfflag_data[(r, f, a)] = 0.0
                xftflag_data[(r, f)] = 1.0 if (any_flow and f in self.sets.mf) else 0.0
        create_indexed_param("xflag", ["r", "a", "i"], xflag_data, 0.0)
        create_indexed_param("xfflag", ["r", "f", "a"], xfflag_data, 0.0)
        create_indexed_param("xftflag", ["r", "f"], xftflag_data, 0.0)
        
        adjusted_and_param: Dict[tuple[str, str], float] = {}
        adjusted_ava_param: Dict[tuple[str, str], float] = {}
        adjusted_nd_share: Dict[tuple[str, str], float] = {}
        adjusted_p_io: Dict[tuple[str, str, str], float] = {}
        import_scale_by_commodity: Dict[tuple[str, str], float] = {}
        adjusted_nd_total_by_activity: Dict[tuple[str, str], float] = {}

        for r in self.sets.r:
            for i in self.sets.i:
                total_raw_import = 0.0
                for a in self.sets.a:
                    total_raw_import += max(float(self.params.benchmark.vifm.get((r, i, a), 0.0) or 0.0), 0.0)
                total_raw_import += max(float(self.params.benchmark.vmpb.get((r, i), 0.0) or 0.0), 0.0)
                total_raw_import += max(float(self.params.benchmark.vmgb.get((r, i), 0.0) or 0.0), 0.0)
                total_raw_import += max(float(self.params.benchmark.vmib.get((r, i), 0.0) or 0.0), 0.0)

                target_import_total = sum(
                    float(self.params.benchmark.vmsb.get((rp, i, r), 0.0) or 0.0)
                    for rp in self.sets.r
                )
                import_scale_by_commodity[(r, i)] = (
                    target_import_total / total_raw_import if total_raw_import > 0.0 else 1.0
                )

        for r in self.sets.r:
            for a in self.sets.a:
                adjusted_total_intermediate = 0.0
                adjusted_values: Dict[str, float] = {}
                for i in self.sets.i:
                    # Use purchaser prices (vdfp + vmfp) to match get_nd_init and get_xaa_init
                    # so that p_io * nd = xaa_init exactly at benchmark (eq_xaa_activity).
                    domestic_val = float(self.params.benchmark.vdfp.get((r, i, a), 0.0) or 0.0)
                    import_val = float(self.params.benchmark.vmfp.get((r, i, a), 0.0) or 0.0)
                    adjusted_val = max(domestic_val, 0.0) + max(import_val, 0.0)
                    adjusted_values[i] = adjusted_val
                    adjusted_total_intermediate += adjusted_val
                adjusted_nd_total_by_activity[(r, a)] = adjusted_total_intermediate

                # Use purchaser-price ND to match get_nd_init = sum(vdfp+vmfp).
                # and_param must satisfy: and_param * xp_model = nd_model at benchmark.
                # xp_model = (nd_p + va_p) * xscale; nd_model = nd_p * xscale.
                # => and_param = nd_p / (nd_p + va_p).
                nd_p = sum(
                    float(self.params.benchmark.vdfp.get((r, i, a), 0.0) or 0.0)
                    + float(self.params.benchmark.vmfp.get((r, i, a), 0.0) or 0.0)
                    for i in self.sets.i
                )
                if nd_p <= 0.0:
                    nd_p = adjusted_total_intermediate
                # VA at purchaser prices: sum(pfa * xf_phys) = sum(evfb * (1+rtf))
                va_p = sum(
                    float(self.params.benchmark.evfb.get((r, f, a), self.params.benchmark.vfm.get((r, f, a), 0.0)) or 0.0)
                    * (1.0 + float(self.params.taxes.rtf.get((r, f, a), 0.0) or 0.0))
                    for f in self.sets.f
                )
                xp_model_equiv = nd_p + va_p
                adjusted_and_param[(r, a)] = nd_p / xp_model_equiv if xp_model_equiv > 0.0 else 0.0
                adjusted_ava_param[(r, a)] = float(self.params.calibrated.ava_param.get((r, a), 0.0) or 0.0)

                # nd_share for eq_pxeq: use same nd_p / xp_model_equiv so and+ava=1
                adjusted_nd_share[(r, a)] = adjusted_and_param[(r, a)]

                if adjusted_total_intermediate > 0.0:
                    for i, adjusted_val in adjusted_values.items():
                        adjusted_p_io[(r, i, a)] = adjusted_val / adjusted_total_intermediate
                else:
                    for i in self.sets.i:
                        adjusted_p_io[(r, i, a)] = 0.0

        # GAMS-style calibrated parameters (and, ava, io, af, gx)
        scaled_af_xf_param = {
            key: value / max(xscale_data[(key[0], key[2])], 1e-12)
            for key, value in self.params.calibrated.af_param.items()
        }
        create_indexed_param("and_param", ["r", "a"], adjusted_and_param, 0.0)
        create_indexed_param("ava_param", ["r", "a"], adjusted_ava_param, 0.0)
        create_indexed_param("io_param", ["r", "i", "a"], self.params.calibrated.io_param, 0.0)
        if self.params.shifts.lambdaio:
            create_indexed_param("lambdaio", ["r", "i", "a"], self.params.shifts.lambdaio, 1.0)
        else:
            model.lambdaio = Param(model.r, model.i, model.a, initialize={}, default=1.0, doc="lambdaio")
        create_indexed_param("af_param", ["r", "f", "a"], self.params.calibrated.af_param, 0.0)
        create_indexed_param("af_xf_param", ["r", "f", "a"], scaled_af_xf_param, 0.0)
        create_indexed_param("gx_param", ["r", "a", "i"], self.params.calibrated.gx_param, 0.0)
        create_indexed_param("xscale", ["r", "aa"], xscale_data, 1.0)

        create_indexed_param("p_io", ["r", "i", "a"], adjusted_p_io, 0.0)
        create_indexed_param("p_ax", ["r", "a", "i"], self.params.shares.p_ax, 0.0)
        create_indexed_param("gd_share", ["r", "i"], self.params.shares.p_gd, 0.0, mutable=True)
        create_indexed_param("ge_share", ["r", "i"], self.params.shares.p_ge, 0.0, mutable=True)
        create_indexed_param("gw_share", ["r", "i", "rp"], self.params.shares.p_gw, 0.0, mutable=True)
        xw_flag_data: Dict[tuple[str, str, str], float] = {}
        for r in self.sets.r:
            for i in self.sets.i:
                for rp in self.sets.r:
                    vxsb = float(self.params.benchmark.vxsb.get((r, i, rp), 0.0) or 0.0)
                    xw_flag_data[(r, i, rp)] = 1.0 if vxsb > 0.0 else 0.0
        create_indexed_param("xw_flag", ["r", "i", "rp"], xw_flag_data, 0.0, mutable=True)
        create_indexed_param("xet_flag", ["r", "i"], {(r, i): 1.0 for r in self.sets.r for i in self.sets.i}, 0.0, mutable=True)
        
        # Simple shares (kept for compatibility)
        create_indexed_param("va_share", ["r", "a"], self.params.shares.p_va, 0.0)
        create_indexed_param("nd_share", ["r", "a"], adjusted_nd_share, 0.0)
        # GAMS-consistent calibration for factor allocation shares (cal.gms:874-888):
        #   - Mobile factors (omegaf=inf): gf(r,fm,a) = xf.l / xft.l  (share)
        #   - Sector-specific factors   : gf(r,fnm,a) = xf.l * (pabs.l/pfy.l)^etaff
        #                                  (absolute level at benchmark prices)
        # Both are stored in model.gf_share but have different semantics by factor
        # type; eq_pfeq consumes them accordingly.
        gf_share_data: Dict[tuple[str, str, str], float] = dict(self.params.shares.p_gf)
        for r in self.sets.r:
            for f in self.sets.mf:
                xf_by_activity: Dict[str, float] = {}
                total_xf = 0.0
                for a in self.sets.a:
                    factor_flow = float(
                        self.params.benchmark.evfb.get((r, f, a), self.params.benchmark.vfm.get((r, f, a), 0.0)) or 0.0
                    )
                    if factor_flow <= 0.0:
                        continue
                    kappa = float(self.params.taxes.kappaf_activity.get((r, f, a), 0.0) or 0.0)
                    if kappa == 0.0:
                        kappa = float(self.params.taxes.kappaf.get((r, f), 0.0) or 0.0)
                    pf_val = max(1.0 / max(1.0 - kappa, 1e-8), 1e-8)
                    xf_val = max(factor_flow / pf_val, 0.0)
                    if xf_val <= 0.0:
                        continue
                    xf_by_activity[a] = xf_val
                    total_xf += xf_val
                if total_xf <= 0.0:
                    continue
                for a in self.sets.a:
                    gf_share_data[(r, f, a)] = xf_by_activity.get(a, 0.0) / total_xf
            for f in self.sets.sf:
                for a in self.sets.a:
                    factor_flow = float(
                        self.params.benchmark.evfb.get((r, f, a), self.params.benchmark.vfm.get((r, f, a), 0.0)) or 0.0
                    )
                    if factor_flow <= 0.0:
                        gf_share_data[(r, f, a)] = 0.0
                        continue
                    kappa = float(self.params.taxes.kappaf_activity.get((r, f, a), 0.0) or 0.0)
                    if kappa == 0.0:
                        kappa = float(self.params.taxes.kappaf.get((r, f), 0.0) or 0.0)
                    pf_val = max(1.0 / max(1.0 - kappa, 1e-8), 1e-8)
                    xf_val = max(factor_flow / pf_val, 0.0)
                    # At benchmark pabs=pfy=1, so gf = xf.l * (1/1)^etaff = xf.l.
                    gf_share_data[(r, f, a)] = xf_val

        create_indexed_param("gf_share", ["r", "f", "a"], gf_share_data, 0.0)
        create_indexed_param("af_share", ["r", "f", "a"], self.params.shares.p_af, 0.0)
        create_indexed_param("p_gx", ["r", "a", "i"], self.params.shares.p_gx, 0.0)
        
        # Tax rates
        create_indexed_param("rto", ["r", "a"], self.params.taxes.rto, 0.0)
        create_indexed_param("rtf", ["r", "f", "a"], self.params.taxes.rtf, 0.0)

        # Commodity-level output tax wedge used in ppeq-style mapping:
        # pp(r,a,i) = (1 + prdtx(r,a,i)) * p(r,a,i).
        # GAMS cal.gms (lines 19391-19392): prdtx = makb/maks - 1 per commodity.
        prdtx_rai_data: Dict[tuple[str, str, str], float] = {}
        for r in self.sets.r:
            for a in self.sets.a:
                outputs = self.sets.activity_commodities.get(a, [])
                if not outputs:
                    outputs = list(self.sets.i)
                for i in outputs:
                    makb_val = float(self.params.benchmark.makb.get((r, a, i), 0.0) or 0.0)
                    maks_val = float(self.params.benchmark.maks.get((r, a, i), 0.0) or 0.0)
                    if maks_val > 0 and makb_val > 0:
                        prdtx_rai_data[(r, a, i)] = makb_val / maks_val - 1.0
                    else:
                        prdtx_rai_data[(r, a, i)] = float(self.params.taxes.rto.get((r, a), 0.0))
        create_indexed_param("prdtx_rai", ["r", "a", "i"], prdtx_rai_data, 0.0)

        # Trade margin parameters (tmarg, amgm, lambdamg)
        tmarg_data: Dict[tuple[str, str, str], float] = {}
        amgm_data: Dict[tuple[str, str, str, str], float] = {}
        lambdamg_data: Dict[tuple[str, str, str, str], float] = {}
        for r in self.sets.r:
            for i in self.sets.i:
                for rp in self.sets.r:
                    margin_flow = sum(
                        self.params.benchmark.vtwr.get((r, i, rp, m), 0.0)
                        for m in self.sets.m
                    )
                    xw_bench = float(self.params.benchmark.vxsb.get((r, i, rp), 0.0) or 0.0)
                    vcif = float(self.params.benchmark.vcif.get((r, i, rp), 0.0) or 0.0)
                    vfob = float(self.params.benchmark.vfob.get((r, i, rp), 0.0) or 0.0)
                    tmarg = max(vcif - vfob, 0.0) / max(xw_bench, 1e-12) if xw_bench > 0.0 else 0.0
                    tmarg_data[(r, i, rp)] = tmarg

                    for m in self.sets.m:
                        flow = self.params.benchmark.vtwr.get((r, i, rp, m), 0.0)
                        share = flow / margin_flow if margin_flow > 0.0 else 0.0
                        amgm_data[(m, r, i, rp)] = share
                        lambdamg_data[(m, r, i, rp)] = 1.0

        create_indexed_param("tmarg", ["r", "i", "rp"], tmarg_data, 0.0)
        create_indexed_param("amgm", ["m", "r", "i", "rp"], amgm_data, 0.0)
        create_indexed_param("lambdamg", ["m", "r", "i", "rp"], lambdamg_data, 1.0)
        if not hasattr(model, "tmarg"):
            model.tmarg = Param(model.r, model.i, model.rp, initialize={}, default=0.0, doc="tmarg")
        if not hasattr(model, "amgm"):
            model.amgm = Param(model.m, model.r, model.i, model.rp, initialize={}, default=0.0, doc="amgm")
        if not hasattr(model, "lambdamg"):
            model.lambdamg = Param(model.m, model.r, model.i, model.rp, initialize={}, default=1.0, doc="lambdamg")

        # GAMS trade calibration objects for the lower Armington nest.
        # `chipm` is normalized back to 1 after benchmark initialization in cal.gms,
        # while `lambdam`, `mtax`, and `etax` remain fixed benchmark shifters.
        chipm_data: Dict[tuple[str, str, str], float] = {}
        for exporter in self.sets.r:
            for commodity in self.sets.i:
                for importer in self.sets.r:
                    vxmd = float(self.params.benchmark.vxmd.get((exporter, commodity, importer), 0.0) or 0.0)
                    vxsb = float(self.params.benchmark.vxsb.get((exporter, commodity, importer), 0.0) or 0.0)
                    vcif = float(self.params.benchmark.vcif.get((exporter, commodity, importer), 0.0) or 0.0)
                    if vxmd <= 0.0 and vxsb <= 0.0 and vcif <= 0.0:
                        continue
                    chipm_data[(exporter, commodity, importer)] = 1.0
        create_indexed_param("chipm", ["r", "i", "rp"], chipm_data, 1.0)
        if not hasattr(model, "chipm"):
            model.chipm = Param(model.r, model.i, model.rp, initialize={}, default=1.0, doc="chipm")

        # Regional income shares calibrated from benchmark absorption totals.
        regional_income_share_data: Dict[tuple[str], float] = {}
        regional_government_share_data: Dict[tuple[str], float] = {}
        regional_investment_share_data: Dict[tuple[str], float] = {}
        private_share_data: Dict[tuple[str, str], float] = {}
        government_share_data: Dict[tuple[str, str], float] = {}
        investment_share_data: Dict[tuple[str, str], float] = {}
        axi_data: Dict[tuple[str], float] = {}
        invwgt_data: Dict[tuple[str], float] = {}
        savwgt_data: Dict[tuple[str], float] = {}
        auh_data: Dict[tuple[str], float] = {}
        aug_data: Dict[tuple[str], float] = {}
        aus_data: Dict[tuple[str], float] = {}
        au_data: Dict[tuple[str], float] = {}
        regional_savings_data: Dict[tuple[str], float] = {}
        regy_bench_data: Dict[tuple[str], float] = {}
        savf_bar_data: Dict[tuple[str], float] = {}
        betap_data: Dict[tuple[str], float] = {}
        betag_data: Dict[tuple[str], float] = {}
        betas_data: Dict[tuple[str], float] = {}
        phip_data: Dict[tuple[str], float] = {}
        phi_data: Dict[tuple[str], float] = {}
        chif_data: Dict[tuple[str], float] = {}
        eh_data: Dict[tuple[str, str], float] = {}
        bh_data: Dict[tuple[str, str], float] = {}
        alphaa_hhd_data: Dict[tuple[str, str], float] = {}
        fdepr_data: Dict[tuple[str], float] = {}
        depr_data: Dict[tuple[str], float] = {}
        rorflex_data: Dict[tuple[str], float] = {}
        pop_data: Dict[tuple[str], float] = {}
        aft_data: Dict[tuple[str, str], float] = {}
        etaf_data: Dict[tuple[str, str], float] = {}

        def _lookup_etrae(region: str, factor: str) -> float:
            # GTAP ETRAE may come keyed as (fp,r) or (r,fp), depending on source.
            raw = self.params.elasticities.etrae
            for key in ((factor, region), (region, factor), factor):
                try:
                    val = raw.get(key)  # type: ignore[arg-type]
                except Exception:
                    val = None
                if val is not None:
                    try:
                        return float(val)
                    except (TypeError, ValueError):
                        continue
            return 0.0

        def _private_total(region: str, commodity: str) -> float:
            total, _, _ = self.params.benchmark.get_private_demand(region, commodity)
            return float(total or 0.0)

        def _government_total(region: str, commodity: str) -> float:
            total, _, _ = self.params.benchmark.get_government_demand(region, commodity)
            return float(total or 0.0)

        def _investment_total(region: str, commodity: str) -> float:
            total, _, _ = self.params.benchmark.get_investment_demand(region, commodity)
            return float(total or 0.0)

        def _pop_value(region: str) -> float:
            raw = self.params.benchmark.pop
            val = raw.get(region)
            if val is None:
                val = raw.get((region,), 1.0)
            return float(val or 1.0)

        for region in self.sets.r:
            # Use evfb (buyer-price factor flows) to match model initialization:
            # xf_model = (evfb/pf) * xscale, so pf*xf/xscale = evfb (not vfm).
            factor_income = sum(
                float(
                    self.params.benchmark.evfb.get((region, factor, activity),
                        self.params.benchmark.vfm.get((region, factor, activity), 0.0)
                    ) or 0.0
                )
                for factor in self.sets.f
                for activity in self.sets.a
            )
            private_total = sum(_private_total(region, commodity) for commodity in self.sets.i)
            government_total = sum(_government_total(region, commodity) for commodity in self.sets.i)
            investment_total = sum(_investment_total(region, commodity) for commodity in self.sets.i)
            raw_savings_total = float(self.params.benchmark.save.get(region, 0.0))

            vkb = float(self.params.benchmark.vkb.get(region, 0.0))
            vdep = float(self.params.benchmark.vdep.get(region, 0.0))
            fdepr = (vdep / vkb) if vkb > 0.0 else 0.0
            fdepr_data[(region,)] = fdepr
            depr_data[(region,)] = fdepr
            rorflex_data[(region,)] = 1.0
            pop_data[(region,)] = _pop_value(region)
            for factor in self.sets.f:
                if factor in self.sets.mf:
                    benchmark_xft = 0.0
                    for activity in self.sets.a:
                        factor_flow = float(
                            self.params.benchmark.evfb.get(
                                (region, factor, activity),
                                self.params.benchmark.vfm.get((region, factor, activity), 0.0),
                            )
                            or 0.0
                        )
                        if factor_flow <= 0.0:
                            continue
                        kappa = float(self.params.taxes.kappaf_activity.get((region, factor, activity), 0.0) or 0.0)
                        if kappa == 0.0:
                            kappa = float(self.params.taxes.kappaf.get((region, factor), 0.0) or 0.0)
                        pf_val = max(1.0 / max(1.0 - kappa, 1e-8), 1e-8)
                        benchmark_xft += factor_flow / pf_val
                    aft_data[(region, factor)] = benchmark_xft
                    # In standard GTAP, etaf defaults to 0 unless explicitly overridden.
                    # Keep that default, but honor region/factor ETRAE when present.
                    etaf_data[(region, factor)] = _lookup_etrae(region, factor)

            facty_bench = max(factor_income - vdep, 0.0)
            ytax_ind_bench = self._compute_ytax_ind_bench(region)
            regy_bench = max(facty_bench + ytax_ind_bench, 1e-8)
            regy_bench_data[(region,)] = regy_bench
            savings_total = raw_savings_total if raw_savings_total > 0.0 else max(
                regy_bench - private_total - government_total,
                0.0,
            )
            regional_savings_data[(region,)] = savings_total

            vdep_bench = fdepr * vkb
            savf_bar_data[(region,)] = investment_total - vdep_bench - savings_total

            # GAMS betaCal: betaP = yc.l / regY.l where regY = factY + yTaxInd (cal.gms:619)
            # phi = phiP = 1 at benchmark (all prices normalised to 1)
            phip = 1.0
            phi = 1.0
            betap = private_total / regy_bench if regy_bench > 0.0 else 0.0
            betag = government_total / regy_bench if regy_bench > 0.0 else 0.0
            betas = savings_total / regy_bench if regy_bench > 0.0 else 0.0
            regy_base = regy_bench  # kept for downstream references below
            regional_income_share_data[(region,)] = private_total / regy_base if regy_base > 0.0 else 0.0
            regional_government_share_data[(region,)] = government_total / regy_base if regy_base > 0.0 else 0.0
            regional_investment_share_data[(region,)] = investment_total / regy_base if regy_base > 0.0 else 0.0

            sigmai = float(self.params.elasticities.esubi.get(region, 1.0))
            if abs(sigmai - 1.0) < 1e-8:
                sigmai = 1.01
            axi_data[(region,)] = 1.0

            betap_data[(region,)] = betap
            betag_data[(region,)] = betag
            betas_data[(region,)] = betas
            phip_data[(region,)] = phip
            phi_data[(region,)] = phi
            yc_bench = betap * (phi / max(phip, 1e-12)) * regy_bench

            private_den = max(private_total, 1e-12)
            government_den = max(government_total, 1e-12)
            investment_den = max(investment_total, 1e-12)
            cde_alpha_den = 0.0
            for commodity in self.sets.i:
                private_val = _private_total(region, commodity)
                government_val = _government_total(region, commodity)
                investment_val = _investment_total(region, commodity)
                private_share_data[(region, commodity)] = private_val / private_den if private_total > 0.0 else 0.0
                government_share_data[(region, commodity)] = government_val / government_den if government_total > 0.0 else 0.0
                investment_share_data[(region, commodity)] = investment_val / investment_den if investment_total > 0.0 else 0.0
                eh_val = float(self.params.elasticities.incpar.get((region, commodity), 1.0) or 1.0)
                bh_val = float(self.params.elasticities.subpar.get((region, commodity), 1.0) or 1.0)
                if abs(bh_val) < 1e-12:
                    bh_val = 1.0
                eh_data[(region, commodity)] = eh_val
                bh_data[(region, commodity)] = bh_val
                xcshr_val = private_val / max(yc_bench, 1e-12) if private_val > 0.0 else 0.0
                if xcshr_val > 0.0:
                    cde_alpha_den += xcshr_val / bh_val

            yc_pc = yc_bench / max(pop_data[(region,)], 1e-12) if yc_bench > 0.0 else 0.0
            yc_pc = max(yc_pc, 1e-12)
            for commodity in self.sets.i:
                private_val = _private_total(region, commodity)
                xcshr_val = private_val / max(yc_bench, 1e-12) if private_val > 0.0 else 0.0
                bh_val = bh_data[(region, commodity)]
                eh_val = eh_data[(region, commodity)]
                pa_val = 1.0  # GAMS numerario initialization
                uh_val = 1.0  # GAMS benchmark utility initialization
                if yc_bench > 0.0 and xcshr_val > 0.0 and cde_alpha_den > 0.0:
                    alphaa_hhd_data[(region, commodity)] = ((xcshr_val / bh_val) * (((yc_pc / pa_val) ** bh_val)) * (uh_val ** (-eh_val * bh_val))) / cde_alpha_den
                else:
                    alphaa_hhd_data[(region, commodity)] = 0.0

            if private_total > 0.0:
                prod_term = 1.0
                for commodity in self.sets.i:
                    share = private_share_data[(region, commodity)]
                    if share <= 0.0:
                        continue
                    # Use same quantity as get_xc_init (= get_private_demand total),
                    # not raw vpm which may only hold the domestic component.
                    level = max(self.params.benchmark.get_private_demand(region, commodity)[0], 1e-12)
                    prod_term *= level ** share
                auh_data[(region,)] = 1.0 / max(prod_term, 1e-12)
            else:
                auh_data[(region,)] = 1.0

            aug_data[(region,)] = pop_data[(region,)] / max(government_total, 1e-12)
            aus_data[(region,)] = pop_data[(region,)] / max(savings_total, 1e-12)
            au_data[(region,)] = 1.0

        inv_weight_den = 0.0
        sav_weight_den = 0.0
        net_inv_by_region: Dict[str, float] = {}
        for region in self.sets.r:
            investment_total = sum(self.params.benchmark.vim.get((region, commodity), 0.0) for commodity in self.sets.i)
            vkb = float(self.params.benchmark.vkb.get(region, 0.0))
            vdep = float(self.params.benchmark.vdep.get(region, 0.0))
            depr = (vdep / vkb) if vkb > 0.0 else 0.0
            net_inv = max(investment_total - depr * vkb, 0.0)
            net_inv_by_region[region] = net_inv
            inv_weight_den += net_inv
            sav_weight_den += max(regional_savings_data[(region,)], 0.0)

        for region in self.sets.r:
            invwgt_data[(region,)] = net_inv_by_region[region] / inv_weight_den if inv_weight_den > 0.0 else 0.0
            save_level = max(regional_savings_data[(region,)], 0.0)
            savwgt_data[(region,)] = save_level / sav_weight_den if sav_weight_den > 0.0 else 0.0

        savf_balance_gap = sum(savf_bar_data.values())
        if abs(savf_balance_gap) > 1e-10 and self.sets.r:
            anchor_region = "RestofWorld" if "RestofWorld" in self.sets.r else next(iter(self.sets.r))
            savf_bar_data[(anchor_region,)] = savf_bar_data.get((anchor_region,), 0.0) - savf_balance_gap

        # GAMS calibrates chif from the final foreign-savings benchmark:
        # chif.l(r) = savf.l(r) / regY.l(r). This must happen after the
        # residual-region capital-account rebalance on savf_bar_data.
        for region in self.sets.r:
            regy_bench = max(regy_bench_data.get((region,), 0.0), 1e-8)
            chif_data[(region,)] = savf_bar_data[(region,)] / regy_bench if abs(regy_bench) > 1e-12 else 0.0

        # Override investment_share_data with GAMS alphaa(r,i,"inv") if cal_dump is available.
        # GAMS alphaa uses VDIB-based (basic prices) shares, while the default Python
        # computation uses VDIP (purchaser prices). Using GAMS alphaa ensures the
        # eq_xi/eq_pi benchmark is exactly satisfied when xaa[r,i,"inv"] also comes from
        # the cal_dump, eliminating the ~2.5e-3 initial residual mismatch.
        if self.gams_calibration_dump is not None:
            alphaa_raw = self.gams_calibration_dump.derived_params.get("alphaa", {})
            if alphaa_raw:
                overridden = 0
                for key, val in alphaa_raw.items():
                    if len(key) >= 3 and str(key[-1]).lower() == "inv":
                        r, i = str(key[0]), str(key[1])
                        if (r, i) in investment_share_data:
                            investment_share_data[(r, i)] = float(val)
                            overridden += 1
                if overridden:
                    logger.info(
                        "i_share: overrode %d values from GAMS alphaa(r,i,'inv') "
                        "(basic-price shares → matches xaa[inv] benchmark)",
                        overridden,
                    )

        create_indexed_param("yc_share_reg", ["r"], regional_income_share_data, 0.0)
        create_indexed_param("yg_share_reg", ["r"], regional_government_share_data, 0.0)
        create_indexed_param("yi_share_reg", ["r"], regional_investment_share_data, 0.0)
        create_indexed_param("c_share", ["r", "i"], private_share_data, 0.0)
        create_indexed_param("g_share", ["r", "i"], government_share_data, 0.0)
        create_indexed_param("i_share", ["r", "i"], investment_share_data, 0.0)
        create_indexed_param("axi", ["r"], axi_data, 1.0)
        create_indexed_param("invwgt", ["r"], invwgt_data, 0.0)
        create_indexed_param("savwgt", ["r"], savwgt_data, 0.0)
        create_indexed_param("auh", ["r"], auh_data, 1.0)
        create_indexed_param("aug", ["r"], aug_data, 1.0)
        create_indexed_param("aus", ["r"], aus_data, 1.0, mutable=True)
        create_indexed_param("au", ["r"], au_data, 1.0)
        create_indexed_param("betap", ["r"], betap_data, 0.0, mutable=True)
        create_indexed_param("betag", ["r"], betag_data, 0.0, mutable=True)
        create_indexed_param("betas", ["r"], betas_data, 0.0, mutable=True)
        create_indexed_param("phip", ["r"], phip_data, 1.0)
        create_indexed_param("phi", ["r"], phi_data, 1.0)
        create_indexed_param("chif0", ["r"], chif_data, 0.0)
        create_indexed_param("eh", ["r", "i"], eh_data, 1.0)
        create_indexed_param("bh", ["r", "i"], bh_data, 1.0)
        create_indexed_param("alphaa_hhd", ["r", "i"], alphaa_hhd_data, 0.0)
        create_indexed_param("fdepr", ["r"], fdepr_data, 0.0)
        create_indexed_param("depr", ["r"], depr_data, 0.0)
        create_indexed_param("rorflex", ["r"], rorflex_data, 1.0)
        create_indexed_param("pop", ["r"], pop_data, 1.0)
        create_indexed_param("savf_bar", ["r"], savf_bar_data, 0.0)
        create_indexed_param("aft", ["r", "f"], aft_data, 0.0)
        create_indexed_param("etaf", ["r", "f"], etaf_data, 0.0)
    
    def _add_variables(self, model: "ConcreteModel") -> None:
        """Add all variables for square system.
        
        Initialize with SAM benchmark values (like GAMS cal.gms):
        - Prices = 1.0 (normalized)
        - Quantities = SAM values (millions)
        """
        from pyomo.environ import Var, Reals, NonNegativeReals, Expression, value
        
        # Helper to get SAM value initialization
        def get_vom_init(m, r, a):
            """Get production level from SAM."""
            if self.reference_snapshot:
                ref_xp = self.reference_snapshot.xp.get((r, a))
                if ref_xp is not None and ref_xp > 0.0:
                    return float(ref_xp)

            # GAMS benchmark-consistent production quantity identity:
            # xp = nd + va at purchaser values.
            nd_val = sum(
                float(self.params.benchmark.vdfp.get((r, i, a), 0.0) or 0.0)
                + float(self.params.benchmark.vmfp.get((r, i, a), 0.0) or 0.0)
                for i in self.sets.i
            )
            va_val = 0.0
            for f in self.sets.f:
                evfb_val = float(
                    self.params.benchmark.evfb.get((r, f, a), self.params.benchmark.vfm.get((r, f, a), 0.0))
                    or 0.0
                )
                if evfb_val <= 0.0:
                    continue
                factor_tax = float(self.params.taxes.rtf.get((r, f, a), 0.0) or 0.0)
                va_val += evfb_val * (1.0 + factor_tax)

            xp_val = nd_val + va_val
            if xp_val > 0.0:
                return max(xp_val, 1e-8)

            val = self.params.benchmark.vom.get((r, a), 0.0)
            return max(val, 1e-8)
        
        def get_vfm_init(m, r, f, a):
            """Initialize factor demand from benchmark SAM data."""
            if self.reference_snapshot:
                ref_xf = self.reference_snapshot.xf.get((r, f, a))
                if ref_xf is not None and ref_xf > 0.0:
                    return float(ref_xf)
            # GAMS cal.gms: xf.l = EVFB / pf.l
            vfm_val = max(
                float(
                    self.params.benchmark.evfb.get((r, f, a), self.params.benchmark.vfm.get((r, f, a), 0.0)) or 0.0
                ),
                0.0,
            )
            pf_val = max(get_pf_init(m, r, f, a), 1e-12)
            return max(vfm_val / pf_val, 0.0)

        def get_pf_init(m, r, f, a):
            # First, try to use baseline from reference snapshot (if available)
            if self.reference_snapshot:
                ref_pf = self.reference_snapshot.pf.get((r, f, a))
                if ref_pf is not None and ref_pf > 0.0:
                    return float(ref_pf)
            
            # Fallback to tax-based initialization
            kappa = self.params.taxes.kappaf_activity.get((r, f, a), 0.0)
            return max(1.0 / max(1.0 - kappa, 1e-8), 1e-8)

        def get_pfa_init(m, r, f, a):
            pf_val = get_pf_init(m, r, f, a)
            factor_tax = float(self.params.taxes.rtf.get((r, f, a), 0.0) or 0.0)
            return max(pf_val * (1.0 + factor_tax), 1e-8)

        def get_pfy_init(m, r, f, a):
            pf_val = get_pf_init(m, r, f, a)
            kappa = float(self.params.taxes.kappaf_activity.get((r, f, a), 0.0) or 0.0)
            return max(pf_val * max(1.0 - kappa, 1e-8), 1e-8)
        
        def get_vpm_init(m, r, i):
            """Get total private Armington demand from SAM."""
            val, _, _ = self.params.benchmark.get_private_demand(r, i)
            return max(val, 1e-8)

        def get_xc_init(m, r, i):
            """Household Armington quantity consistent with pa*xc = purchaser-value demand."""
            vpm_val = get_vpm_init(m, r, i)
            pa_hhd = get_pa_benchmark_init(m, r, i, GTAP_HOUSEHOLD_AGENT)
            return max(vpm_val / max(pa_hhd, 1e-12), 0.0)
        
        def get_vgm_init(m, r, i):
            """Get total government Armington demand from SAM."""
            val, _, _ = self.params.benchmark.get_government_demand(r, i)
            return max(val, 1e-8)
        
        def get_vim_init(m, r, i):
            """Get total investment Armington demand from SAM."""
            val, _, _ = self.params.benchmark.get_investment_demand(r, i)
            return max(val, 1e-8)

        def get_xscale(m, r, aa):
            return max(float(m.xscale[r, aa]), 1e-12)

        def get_xaa_raw_init(m, r, i, aa):
            if aa in self.sets.a:
                # GAMS benchmark quantity identity: xa = xd + xm with xd=vdfb/pd,
                # xm=vmfb/pmt and pd=pmt=1 at benchmark.
                val = self.params.benchmark.vdfb.get((r, i, aa), 0.0) + self.params.benchmark.vmfb.get((r, i, aa), 0.0)
            elif aa == GTAP_HOUSEHOLD_AGENT:
                # Keep xaa(hhd) coherent with xc(hhd) quantity before macro refresh.
                val = get_xc_init(m, r, i)
            elif aa == GTAP_GOVERNMENT_AGENT:
                val = self.params.benchmark.vdgb.get((r, i), 0.0) + self.params.benchmark.vmgb.get((r, i), 0.0)
            elif aa == GTAP_INVESTMENT_AGENT:
                val = self.params.benchmark.vdib.get((r, i), 0.0) + self.params.benchmark.vmib.get((r, i), 0.0)
            elif aa == GTAP_MARGIN_AGENT:
                val = self._vst_value(str(r), str(i))
            else:
                val = 0.0
            return max(val, 0.0)

        agent_trade_cache: Dict[tuple[str, str, str], tuple[float, float]] = {}

        def _raw_agent_domestic_import(r, i, aa):
            if aa in self.sets.a:
                raw_domestic = self.params.benchmark.vdfb.get((r, i, aa), 0.0)
                raw_import = self.params.benchmark.vmfb.get((r, i, aa), 0.0)
                if raw_domestic + raw_import <= 0.0:
                    raw_domestic = self.params.benchmark.vdfm.get((r, i, aa), 0.0)
                    raw_import = self.params.benchmark.vifm.get((r, i, aa), 0.0)
            elif aa == GTAP_HOUSEHOLD_AGENT:
                raw_domestic = self.params.benchmark.vdpb.get((r, i), 0.0)
                raw_import = self.params.benchmark.vmpb.get((r, i), 0.0)
                if raw_domestic + raw_import <= 0.0:
                    _, raw_domestic, raw_import = self.params.benchmark.get_private_demand(r, i)
            elif aa == GTAP_GOVERNMENT_AGENT:
                raw_domestic = self.params.benchmark.vdgb.get((r, i), 0.0)
                raw_import = self.params.benchmark.vmgb.get((r, i), 0.0)
                if raw_domestic + raw_import <= 0.0:
                    _, raw_domestic, raw_import = self.params.benchmark.get_government_demand(r, i)
            elif aa == GTAP_INVESTMENT_AGENT:
                raw_domestic = self.params.benchmark.vdib.get((r, i), 0.0)
                raw_import = self.params.benchmark.vmib.get((r, i), 0.0)
                if raw_domestic + raw_import <= 0.0:
                    _, raw_domestic, raw_import = self.params.benchmark.get_investment_demand(r, i)
            elif aa == GTAP_MARGIN_AGENT:
                raw_domestic = self._vst_value(str(r), str(i))
                raw_import = 0.0
            else:
                raw_domestic = 0.0
                raw_import = 0.0
            return raw_domestic, raw_import

        def build_agent_trade_cache():
            for r in self.sets.r:
                for i in self.sets.i:
                    raw_levels: Dict[str, tuple[float, float]] = {}
                    total_raw_import = 0.0
                    for aa in list(self.sets.a) + [
                        GTAP_HOUSEHOLD_AGENT,
                        GTAP_GOVERNMENT_AGENT,
                        GTAP_INVESTMENT_AGENT,
                        GTAP_MARGIN_AGENT,
                    ]:
                        raw_domestic, raw_import = _raw_agent_domestic_import(r, i, aa)
                        domestic = max(raw_domestic, 0.0)
                        imported = max(raw_import, 0.0)
                        raw_levels[aa] = (domestic, imported)
                        total_raw_import += imported

                    target_import_total = sum(
                        float(self.params.benchmark.vmsb.get((rp, i, r), 0.0) or 0.0)
                        for rp in self.sets.r
                    )
                    import_scale = (target_import_total / total_raw_import) if total_raw_import > 0.0 else 1.0

                    for aa, (domestic, imported) in raw_levels.items():
                        agent_trade_cache[(r, i, aa)] = (domestic, imported * import_scale)

        def get_agent_trade_levels(m, r, i, aa):
            return agent_trade_cache.get((r, i, aa), (0.0, 0.0))

        def get_xda_init(m, r, i, aa):
            if self.reference_snapshot:
                ref_xd = self.reference_snapshot.xd.get((r, i, aa))
                if ref_xd is not None and ref_xd > 0.0:
                    return float(ref_xd)
            domestic_value, _ = get_agent_trade_levels(m, r, i, aa)
            pd_bench = 1.0
            xda = domestic_value / max(pd_bench, 1e-12)
            return max(xda, 0.0)

        def get_xma_init(m, r, i, aa):
            if self.reference_snapshot:
                ref_xaa = self.reference_snapshot.xaa.get((r, i, aa))
                ref_xd = self.reference_snapshot.xd.get((r, i, aa))
                if ref_xaa is not None and ref_xd is not None:
                    ref_xm = max(float(ref_xaa) - float(ref_xd), 0.0)
                    if ref_xm > 0.0:
                        return ref_xm
            _, imported_value = get_agent_trade_levels(m, r, i, aa)
            xma = imported_value
            return max(xma, 0.0)

        def get_pdp_init(m, r, i, aa):
            return max(1.0 + get_dintx_init(m, r, i, aa), 1e-8)

        def get_dintx_init(m, r, i, aa):
            def _two_key(raw_map, region, commodity):
                val = raw_map.get((region, commodity), None)
                if val is None:
                    val = raw_map.get((commodity, region), 0.0)
                return float(val or 0.0)

            if aa in self.sets.a:
                numerator = float(self.params.benchmark.vdfp.get((r, i, aa), 0.0) or 0.0) - float(
                    self.params.benchmark.vdfb.get((r, i, aa), 0.0) or 0.0
                )
                denom = max(float(self.params.benchmark.vdfb.get((r, i, aa), 0.0) or 0.0), 0.0)
            elif aa == GTAP_HOUSEHOLD_AGENT:
                numerator = _two_key(self.params.benchmark.vdpp, r, i) - _two_key(self.params.benchmark.vdpb, r, i)
                denom = max(_two_key(self.params.benchmark.vdpb, r, i), 0.0)
            elif aa == GTAP_GOVERNMENT_AGENT:
                numerator = _two_key(self.params.benchmark.vdgp, r, i) - _two_key(self.params.benchmark.vdgb, r, i)
                denom = max(_two_key(self.params.benchmark.vdgb, r, i), 0.0)
            elif aa == GTAP_INVESTMENT_AGENT:
                numerator = _two_key(self.params.benchmark.vdip, r, i) - _two_key(self.params.benchmark.vdib, r, i)
                denom = max(_two_key(self.params.benchmark.vdib, r, i), 0.0)
            elif aa == GTAP_MARGIN_AGENT:
                return 0.0
            else:
                numerator = 0.0
                denom = 0.0

            if denom > 0.0:
                return numerator / denom
            return float(self.params.taxes.dintx0.get((r, i, aa), 0.0) or 0.0)

        def get_mintx_init(m, r, i, aa):
            def _two_key(raw_map, region, commodity):
                val = raw_map.get((region, commodity), None)
                if val is None:
                    val = raw_map.get((commodity, region), 0.0)
                return float(val or 0.0)

            if aa in self.sets.a:
                numerator = float(self.params.benchmark.vmfp.get((r, i, aa), 0.0) or 0.0) - float(
                    self.params.benchmark.vmfb.get((r, i, aa), 0.0) or 0.0
                )
                denom = max(float(self.params.benchmark.vmfb.get((r, i, aa), 0.0) or 0.0), 0.0)
            elif aa == GTAP_HOUSEHOLD_AGENT:
                numerator = _two_key(self.params.benchmark.vmpp, r, i) - _two_key(self.params.benchmark.vmpb, r, i)
                denom = max(_two_key(self.params.benchmark.vmpb, r, i), 0.0)
            elif aa == GTAP_GOVERNMENT_AGENT:
                numerator = _two_key(self.params.benchmark.vmgp, r, i) - _two_key(self.params.benchmark.vmgb, r, i)
                denom = max(_two_key(self.params.benchmark.vmgb, r, i), 0.0)
            elif aa == GTAP_INVESTMENT_AGENT:
                numerator = _two_key(self.params.benchmark.vmip, r, i) - _two_key(self.params.benchmark.vmib, r, i)
                denom = max(_two_key(self.params.benchmark.vmib, r, i), 0.0)
            elif aa == GTAP_MARGIN_AGENT:
                return 0.0
            else:
                numerator = 0.0
                denom = 0.0

            if denom > 0.0:
                return numerator / denom
            return float(self.params.taxes.mintx0.get((r, i, aa), 0.0) or 0.0)

        def get_pa_benchmark_init(m, r, i, aa):
            # GAMS cal.gms benchmark seed keeps pa.l normalized at 1.0.
            return 1.0

        def get_xaa_purchaser_value_init(m, r, i, aa):
            def _two_key(raw_map, region, commodity):
                val = raw_map.get((region, commodity), None)
                if val is None:
                    val = raw_map.get((commodity, region), 0.0)
                return float(val or 0.0)

            if aa in self.sets.a:
                return max(
                    float(self.params.benchmark.vdfp.get((r, i, aa), 0.0) or 0.0)
                    + float(self.params.benchmark.vmfp.get((r, i, aa), 0.0) or 0.0),
                    0.0,
                )
            if aa == GTAP_HOUSEHOLD_AGENT:
                return max(_two_key(self.params.benchmark.vdpp, r, i) + _two_key(self.params.benchmark.vmpp, r, i), 0.0)
            if aa == GTAP_GOVERNMENT_AGENT:
                return max(_two_key(self.params.benchmark.vdgp, r, i) + _two_key(self.params.benchmark.vmgp, r, i), 0.0)
            if aa == GTAP_INVESTMENT_AGENT:
                return max(_two_key(self.params.benchmark.vdip, r, i) + _two_key(self.params.benchmark.vmip, r, i), 0.0)
            if aa == GTAP_MARGIN_AGENT:
                return max(float(self._vst_value(str(r), str(i)) or 0.0), 0.0)
            return 0.0

        def get_xaa_init(m, r, i, aa):
            if self.reference_snapshot:
                ref_xaa = self.reference_snapshot.xaa.get((r, i, aa))
                if ref_xaa is not None and ref_xaa > 0.0:
                    return float(ref_xaa)

            # Benchmark purchaser-value demand level (GAMS xa.l seed).
            return get_xaa_purchaser_value_init(m, r, i, aa)

        def get_make_init(m, r, a, i):
            """Get benchmark output by activity-commodity pair from SAM."""
            outputs = self.sets.activity_commodities.get(a, [])
            if outputs and i not in outputs:
                return 0.0
            # Initialize x consistent with eq_x: x = gx*(xp/xscale)*(p_rai/px)^omega
            gx_val = value(m.gx_param[r, a, i]) if hasattr(m, "gx_param") else self.params.calibrated.gx_param.get((r, a, i), 0.0)
            if gx_val <= 0.0:
                return 0.0
            omega = self._get_omegas(r, a)
            if omega == float("inf"):
                # omega=inf: eq_x is p_rai = px (no quantity constraint), use makb
                val = self.params.benchmark.makb.get((r, a, i), 0.0)
                return max(val, 1e-8) if val > 0 else max(gx_val * get_vom_init(m, r, a), 1e-8)
            # get_vom_init returns physical (unscaled) xp; _calibrate_initial_state will
            # multiply xp by xscale later. eq_x uses xp_model/xscale = xp_phys, so:
            # x_init = gx * xp_phys * (p_rai/px)^omega (no xscale division needed here)
            xp_phys = get_vom_init(m, r, a)
            p_rai_v = get_p_rai_init(m, r, a, i)
            return max(gx_val * xp_phys * (p_rai_v ** omega), 1e-8)

        def get_export_init(r, i):
            _, _, xet, _, _ = self.params.benchmark.get_trade_totals(self.sets, r, i)
            return xet

        def get_import_init(r, i):
            intermediate_imports = sum(self.params.benchmark.vifm.get((r, i, a), 0.0) for a in self.sets.a)
            final_imports = (
                self.params.benchmark.vmpp.get((r, i), 0.0)
                + self.params.benchmark.vmgp.get((r, i), 0.0)
                + self.params.benchmark.vmip.get((r, i), 0.0)
            )
            return intermediate_imports + final_imports

        def get_intermediate_use(r, i):
            return sum(
                self.params.benchmark.vdfm.get((r, i, a), 0.0) + self.params.benchmark.vifm.get((r, i, a), 0.0)
                for a in self.sets.a
            )

        def get_final_use(r, i):
            private_total, _, _ = self.params.benchmark.get_private_demand(r, i)
            government_total, _, _ = self.params.benchmark.get_government_demand(r, i)
            investment_total, _, _ = self.params.benchmark.get_investment_demand(r, i)
            return private_total + government_total + investment_total

        def get_total_use(r, i):
            return (
                get_intermediate_use(r, i)
                + get_final_use(r, i)
                + self._vst_value(str(r), str(i))
            )

        def get_xs_init(m, r, i):
            # xs = sum_a x(r,a,i) at benchmark — must match get_make_init to satisfy eq_xs.
            total = sum(get_make_init(m, r, a, i) for a in self.sets.a)
            if total <= 0.0:
                total = sum(self.params.benchmark.makb.get((r, a, i), 0.0) for a in self.sets.a)
            if total <= 0.0:
                total = self.params.benchmark.vom_i.get((r, i), 0.0)
            if total <= 0.0:
                total, _, _, _, _ = self.params.benchmark.get_trade_totals(self.sets, r, i)
            if total <= 0.0:
                total = max(get_total_use(r, i) - get_import_init(r, i), 0.0) + get_export_init(r, i)
            return max(total, 1e-8)

        def get_xds_init(m, r, i):
            # Keep domestic market clearing benchmark-consistent:
            # pdeq requires xds(r,i) = sum_aa xd(r,i,aa)/xScale(r,aa).
            # When a GAMS calibration dump is available, prefer its xds level
            # (which is guaranteed consistent with the dump's gd/ge shares and
            # all price/quantity identities at the GAMS benchmark point).
            if self.reference_snapshot:
                ref_xds = self.reference_snapshot.xds.get((r, i))
                if ref_xds is not None and ref_xds > 0.0:
                    return float(ref_xds)

            # Seed xds from the same agent-level domestic demands used by xda.
            xds_from_agents = sum(
                get_xda_init(m, r, i, aa) / get_xscale(m, r, aa)
                for aa in m.aa
            )
            if xds_from_agents > 0.0:
                return max(xds_from_agents, 1e-8)

            xs_bench, _, _, _, _ = self.params.benchmark.get_trade_totals(self.sets, r, i)
            export_flow = sum(
                float(self.params.benchmark.vxsb.get((r, i, rp), 0.0) or 0.0)
                for rp in self.sets.r
            )
            return max(xs_bench - export_flow, 1e-8)

        build_agent_trade_cache()

        def get_xd_init(m, r, i):
            total = sum(get_xda_init(m, r, i, aa) / get_xscale(m, r, aa) for aa in model.aa)
            return max(total, 1e-8)

        def get_xmt_init(m, r, i):
            if self.reference_snapshot:
                ref_total = sum(
                    float(self.reference_snapshot.xw.get((rp, i, r), 0.0) or 0.0)
                    for rp in self.sets.r
                )
                if ref_total > 0.0:
                    return max(ref_total, 1e-8)
            total = sum(float(self.params.benchmark.vmsb.get((rp, i, r), 0.0) or 0.0) for rp in self.sets.r)
            return max(total, 1e-8)

        def get_pmt_init(m, r, i):
            esubm = self.params.elasticities.esubm.get((r, i), 5.0)
            expo = 1.0 - esubm
            terms = []
            for rp in self.sets.r:
                amw = float(self.params.shares.normalized.import_source_share.get((r, i, rp), 0.0) or 0.0)
                if amw <= 0.0:
                    continue
                xw_ref = float(self.reference_snapshot.xw.get((rp, i, r), 0.0) or 0.0) if self.reference_snapshot else 0.0
                bilateral_exports = float(self.params.benchmark.vxmd.get((rp, i, r), 0.0) or 0.0)
                bilateral_imports = float(self.params.benchmark.vcif.get((rp, i, r), 0.0) or 0.0)
                if bilateral_exports <= 0.0 and bilateral_imports <= 0.0:
                    continue
                if bilateral_exports > 0.0 and bilateral_imports > 0.0:
                    pmcif = max(bilateral_imports / bilateral_exports, 1e-8)
                elif bilateral_imports > 0.0:
                    pmcif = max(bilateral_imports, 1e-8)
                else:
                    export_tax = float(self.params.taxes.rtxs.get((rp, i, r), 0.0) or 0.0)
                    tmarg = sum(self.params.benchmark.vtwr.get((rp, i, r, margin), 0.0) for margin in self.sets.m)
                    tmarg = tmarg / max(bilateral_exports, 1e-12) if bilateral_exports > 0.0 else 0.0
                    pmcif = max(1.0 + export_tax + tmarg, 1e-8)
                imptx = float(self.params.taxes.imptx.get((rp, i, r), 0.0) or 0.0)
                pm = max((1.0 + imptx) * pmcif, 1e-8)
                terms.append(amw * (pm ** expo))
            if terms:
                rhs = sum(terms)
                if rhs > 0.0:
                    return max(rhs ** (1.0 / expo), 1e-8)
            total_imports = sum(float(self.params.benchmark.viws.get((rp, i, r), 0.0) or 0.0) for rp in self.sets.r)
            return 1.0 if total_imports > 0.0 else 1.0

        def get_xet_init(m, r, i):
            if self.reference_snapshot:
                ref_xet = self.reference_snapshot.xet.get((r, i))
                if ref_xet is not None and ref_xet > 0.0:
                    return max(float(ref_xet), 1e-8)
            total_vxsb = sum(
                float(self.params.benchmark.vxsb.get((r, i, rp), 0.0) or 0.0)
                for rp in self.sets.r
            )
            if total_vxsb > 0.0:
                return max(total_vxsb, 1e-8)
            xs_bench, xd_bench, _, _, _ = self.params.benchmark.get_trade_totals(self.sets, r, i)
            numerator = value(model.ps[r, i]) * xs_bench - value(model.pd[r, i]) * xd_bench
            pet_val = 1.0
            if self.reference_snapshot:
                ref_pet = self.reference_snapshot.pet.get((r, i))
                if ref_pet is not None and ref_pet > 0.0:
                    pet_val = float(ref_pet)
            return max(numerator / pet_val, 1e-8)

        def get_xa_init(m, r, i):
            total = sum(get_xaa_init(m, r, i, aa) / get_xscale(m, r, aa) for aa in model.aa)
            return max(total, 1e-8)

        def get_va_init(m, r, a):
            # GAMS cal.gms: va.l = sum(fp, pfa.l*xf.l) / pva.l, with pva.l≈1 at benchmark.
            total = sum(get_pfa_init(m, r, f, a) * get_vfm_init(m, r, f, a) for f in self.sets.f)
            return max(total, 1e-8)

        def get_nd_init(m, r, a):
            total_intermediate = sum(
                float(self.params.benchmark.vdfp.get((r, i, a), 0.0) or 0.0)
                + float(self.params.benchmark.vmfp.get((r, i, a), 0.0) or 0.0)
                for i in self.sets.i
            )
            if total_intermediate > 0.0:
                return max(total_intermediate, 1e-8)
            return max(get_vom_init(m, r, a) - get_va_init(m, r, a), 1e-8)

        def get_factor_supply_init(m, r, f):
            if str(f) == "NatRes":
                return 0.0
            # GAMS if(1) branch: xft.l = sum_a pfy.l*xf.l / pft.l, with pft.l≈1.
            total = sum(get_pfy_init(m, r, f, a) * get_vfm_init(m, r, f, a) for a in self.sets.a)
            return max(total, 0.0)

        def get_pft_init(m, r, f):
            if str(f) == "NatRes":
                return 1e-8
            supply = get_factor_supply_init(m, r, f)
            if supply <= 0.0:
                return 1e-8
            return 1.0

        def get_kstock_init(m, r):
            raw_vkb = self.params.benchmark.vkb
            vkb_val = raw_vkb.get(r)
            if vkb_val is None:
                vkb_val = raw_vkb.get((r,), 0.0)
            vkb = float(vkb_val or 0.0)
            if vkb > 0.0:
                return max(vkb, 1e-8)
            total = sum(self.params.benchmark.vfm.get((r, "Capital", a), 0.0) for a in self.sets.a)
            return max(total, 1e-8)

        def get_kapend_init(m, r):
            xi_bench = get_benchmark_yi(r)
            vkb = get_kstock_init(m, r)
            vdep = float(self.params.benchmark.vdep.get(r, 0.0))
            depr = (vdep / vkb) if vkb > 0.0 else 0.0
            return max((1.0 - depr) * vkb + xi_bench, 1e-8)

        def get_gdpmp_init(m, r):
            absorption = sum(self.params.benchmark.vpm.get((r, i), 0.0) for i in self.sets.i)
            absorption += sum(self.params.benchmark.vgm.get((r, i), 0.0) for i in self.sets.i)
            absorption += sum(self.params.benchmark.vim.get((r, i), 0.0) for i in self.sets.i)

            exports = sum(
                self.params.benchmark.vxsb.get((r, i, rp), 0.0)
                for i in self.sets.i
                for rp in self.sets.r
            )
            imports = sum(
                float(self.params.benchmark.vcif.get((rp, i, r), 0.0) or 0.0)
                for i in self.sets.i
                for rp in self.sets.r
            )
            return max(absorption + exports - imports, 1e-8)

        def get_xigbl_init(m):
            total = 0.0
            for r in self.sets.r:
                xi_bench = get_benchmark_yi(r)
                vkb = get_kstock_init(m, r)
                vdep = float(self.params.benchmark.vdep.get(r, 0.0))
                depr = (vdep / vkb) if vkb > 0.0 else 0.0
                total += xi_bench - depr * vkb
            return max(total, 1e-8)

        def get_pigbl_init(m):
            numer = 0.0
            denom = 0.0
            for r in self.sets.r:
                xi_bench = get_benchmark_yi(r)
                vkb = get_kstock_init(m, r)
                vdep = float(self.params.benchmark.vdep.get(r, 0.0))
                depr = (vdep / vkb) if vkb > 0.0 else 0.0
                net = xi_bench - depr * vkb
                numer += 1.0 * net
                denom += net
            if denom <= 1e-12:
                return 1.0
            return max(numer / denom, 1e-8)
        
        # Production (4 vars per r,a)
        def get_p_rai_init(m, r, a, i):
            # GAMS cal.gms (lines 19391-19394): prdtx per commodity = makb/maks - 1;
            # p.l = ps/(1+prdtx) = 1/(1+prdtx).  Use commodity-specific prdtx so that
            # gx_param (calibrated with the same formula) is consistent with eq_po at init.
            makb_val = float(self.params.benchmark.makb.get((r, a, i), 0.0) or 0.0)
            maks_val = float(self.params.benchmark.maks.get((r, a, i), 0.0) or 0.0)
            if maks_val > 0 and makb_val > 0:
                prdtx = makb_val / maks_val - 1.0
            else:
                prdtx = float(self.params.taxes.rto.get((r, a), 0.0) or 0.0)
            return max(1.0 / max(1.0 + prdtx, 1e-12), 1e-8)

        def get_pp_rai_init(m, r, a, i):
            # pp.l = (1+prdtx)*p.l = 1.0 at benchmark (ps=1).
            # Compute prdtx consistently with get_p_rai_init.
            makb_val = float(self.params.benchmark.makb.get((r, a, i), 0.0) or 0.0)
            maks_val = float(self.params.benchmark.maks.get((r, a, i), 0.0) or 0.0)
            if maks_val > 0 and makb_val > 0:
                prdtx = makb_val / maks_val - 1.0
            else:
                prdtx = float(self.params.taxes.rto.get((r, a), 0.0) or 0.0)
            return max((1.0 + prdtx) * get_p_rai_init(m, r, a, i), 1e-8)

        model.xp = Var(model.r, model.a, within=NonNegativeReals, initialize=get_vom_init, doc="Production")
        model.x = Var(model.r, model.a, model.i, within=NonNegativeReals, initialize=get_make_init, doc="Output")
        model.px = Var(model.r, model.a, within=NonNegativeReals, initialize=1.0, doc="Unit cost")
        model.p_rai = Var(model.r, model.a, model.i, within=NonNegativeReals, initialize=get_p_rai_init, doc="Pre-tax producer price by activity-commodity")
        model.pp = Var(model.r, model.a, within=NonNegativeReals, initialize=1.0, doc="Producer price (activity aggregate)")
        model.pp_rai = Var(model.r, model.a, model.i, within=NonNegativeReals, initialize=get_pp_rai_init, doc="Producer price by activity-commodity")
        
        # Supply (3 vars per r,i)
        model.xs = Var(model.r, model.i, within=NonNegativeReals, initialize=get_xs_init, doc="Domestic supply")
        model.xds = Var(model.r, model.i, within=NonNegativeReals, initialize=get_xds_init, doc="Supply of domestically produced goods")
        model.ps = Var(model.r, model.i, within=NonNegativeReals, initialize=1.0, doc="Supply price")
        model.pd = Var(model.r, model.i, within=NonNegativeReals, initialize=1.0, doc="Domestic price")
        
        # Armington (xa per r,i, pa per r,i,aa)
        model.xa = Var(model.r, model.i, within=NonNegativeReals, initialize=get_xa_init, doc="Armington demand")
        # GAMS: pa(r,i,aa,t) - Agent-specific Armington price
        def get_pa_init(m, r, i, aa):
            return get_pa_benchmark_init(m, r, i, aa)

        model.pa = Var(model.r, model.i, model.aa, within=NonNegativeReals, initialize=get_pa_init, doc="Armington price by agent")
        # Keep no-demand agent/commodity prices at their benchmark normalization.
        # In GAMS these combinations are filtered out by equation-domain flags.
        for r in self.sets.r:
            for i in self.sets.i:
                for aa in model.aa:
                    domestic_value, imported_value = get_agent_trade_levels(model, r, i, aa)
                    if domestic_value <= 0.0 and imported_value <= 0.0:
                        model.pa[r, i, aa].fix(1.0)
        model.dintx = Var(
            model.r,
            model.i,
            model.aa,
            within=Reals,
            initialize=get_dintx_init,
            doc="Indirect tax on domestic consumption",
        )
        model.mintx = Var(
            model.r,
            model.i,
            model.aa,
            within=Reals,
            initialize=get_mintx_init,
            doc="Indirect tax on import consumption",
        )
        
        # Trade - Domestic/Import split (4 vars per r,i)
        # GAMS has xmt and xds as Variables with defining equations (xmteq, xdseq)
        model.xd = Var(model.r, model.i, within=NonNegativeReals, initialize=get_xd_init, doc="Domestic demand")
        model.xmt = Var(model.r, model.i, within=NonNegativeReals, initialize=get_xmt_init, doc="Import demand")
        model.pmt = Var(model.r, model.i, within=NonNegativeReals, initialize=get_pmt_init, doc="Import price")
        model.xda = Var(model.r, model.i, model.aa, within=NonNegativeReals, initialize=get_xda_init, doc="Domestic Armington demand by agent")
        model.xma = Var(model.r, model.i, model.aa, within=NonNegativeReals, initialize=get_xma_init, doc="Imported Armington demand by agent")
        
        # Price pass-through expressions (NOT variables - derived from aggregate prices)
        # These are calculated values, not decision variables (matching GAMS structure)
        # GAMS: pdp(r,i,aa) = pd(r,i) * (1 + dintx(r,i,aa))
        # GAMS: pmp(r,i,aa) = pmt(r,i) * (1 + mintx(r,i,aa))
        def _get_dintx0(r, i, aa):
            """Get domestic consumption tax rate for agent."""
            return self.params.taxes.dintx0.get((r, i, aa), 0.0)
        
        def _get_mintx0(r, i, aa):
            """Get import consumption tax rate for agent."""
            return self.params.taxes.mintx0.get((r, i, aa), 0.0)
        
        # Armington share parameters (alphad, alpham) - calibrated from benchmark
        # GAMS: alphad(r,i,aa) = (xd/xa)*(pdp/pa)**sigma at benchmark
        # At benchmark with prices=1: alphad = xd/xa, alpham = xm/xa
        def _get_alphad(r, i, aa):
            """Get domestic Armington share for agent."""
            # Try agent-specific share first
            dom_val, imp_val = get_agent_trade_levels(model, r, i, aa)
            total = dom_val + imp_val
            if total > 0:
                return dom_val / total
            # Fallback to aggregate share
            return self.params.shares.p_alphad.get((r, i), 0.5)  # Default 50% domestic
        
        def _get_alpham(r, i, aa):
            """Get import Armington share for agent."""
            dom_val, imp_val = get_agent_trade_levels(model, r, i, aa)
            total = dom_val + imp_val
            if total > 0:
                return imp_val / total
            return self.params.shares.p_alpham.get((r, i), 0.5)  # Default 50% import
        
        # paa is now just an alias to pa[r,i,aa]
        def paa_expr_rule(m, r, i, aa):
            return m.pa[r, i, aa]
        model.paa = Expression(model.r, model.i, model.aa, rule=paa_expr_rule, doc="Agent Armington price (expression alias)")
        
        def pdp_expr_rule(m, r, i, aa):
            # GAMS: pdp = pd * (1 + dintx)
            return (1.0 + m.dintx[r, i, aa]) * m.pd[r, i]
        model.pdp = Expression(model.r, model.i, model.aa, rule=pdp_expr_rule, doc="Agent domestic demand price (expression)")
        
        def pmp_expr_rule(m, r, i, aa):
            # GAMS: pmp = pmt * (1 + mintx)
            return (1.0 + m.mintx[r, i, aa]) * m.pmt[r, i]
        model.pmp = Expression(model.r, model.i, model.aa, rule=pmp_expr_rule, doc="Agent import demand price (expression)")
        
        # Trade - Domestic/Export split (4 vars per r,i)
        model.xet = Var(model.r, model.i, within=NonNegativeReals, initialize=get_xet_init, doc="Export supply")
        def get_pet_init(m, r, i):
            if self.reference_snapshot:
                ref_pet = self.reference_snapshot.pet.get((r, i))
                if ref_pet is not None and ref_pet > 0.0:
                    return float(ref_pet)
            return 1.0

        model.pet = Var(model.r, model.i, within=NonNegativeReals, initialize=get_pet_init, doc="Export price")
        
        # Value added/intermediate bundles
        model.va = Var(model.r, model.a, within=NonNegativeReals, initialize=get_va_init, doc="Value added bundle")
        model.nd = Var(model.r, model.a, within=NonNegativeReals, initialize=get_nd_init, doc="Intermediate bundle")
        model.pva = Var(model.r, model.a, within=NonNegativeReals, initialize=1.0, doc="Value added price")
        model.pnd = Var(model.r, model.a, within=NonNegativeReals, initialize=1.0, doc="Intermediate price")

        # Bilateral trade (2 vars per r,i,rp)  
        def get_pe_init(m, r, i, rp):
            if r == rp:
                return 1.0
            # Keep bilateral trade-price seeds aligned with GAMS cal.gms:
            # baseline pe.l starts at 1.0 on active routes.
            bilateral_exports = self.params.benchmark.vxmd.get((r, i, rp), 0.0)
            mirror_imports = self.params.benchmark.viws.get((rp, i, r), 0.0)
            return 1.0 if (bilateral_exports > 0.0 or mirror_imports > 0.0) else 0.0

        def get_xe_init(m, r, i, rp):
            # eq_xe_xw: xe = xw, so seed xe from same source as xw.
            # GAMS cal dump stores xw (CIF-based), not xe separately.
            if self.reference_snapshot:
                ref_xw = self.reference_snapshot.xw.get((r, i, rp))
                if ref_xw is not None and ref_xw >= 0.0:
                    return float(ref_xw)
            return max(float(self.params.benchmark.vxsb.get((r, i, rp), 0.0) or 0.0), 0.0)

        def get_xw_init(m, r, i, rp):
            # Use GAMS cal dump xw (CIF/VIWS-based quantities) so that eq_xweq is
            # satisfied at initialization: xw = amw * xmt (prices=1 at benchmark).
            # The vxsb-based fallback uses FOB prices which differ from VIWS/amw basis.
            if self.reference_snapshot:
                ref_xw = self.reference_snapshot.xw.get((r, i, rp))
                if ref_xw is not None and ref_xw >= 0.0:
                    return float(ref_xw)
            vxsb = float(self.params.benchmark.vxsb.get((r, i, rp), 0.0) or 0.0)
            if vxsb > 0.0:
                pe_val = max(get_pe_init(m, r, i, rp), 1e-12)
                return vxsb / pe_val
            return 0.0

        model.xe = Var(model.r, model.i, model.rp, within=NonNegativeReals, initialize=get_xe_init, doc="Bilateral exports")
        model.xw = Var(model.r, model.i, model.rp, within=NonNegativeReals, initialize=get_xw_init, doc="Bilateral imports")
        model.pe = Var(
            model.r,
            model.i,
            model.rp,
            within=NonNegativeReals,
            initialize=get_pe_init,
            doc="Bilateral export price by route",
        )

        def get_pwmg_init(m, r, i, rp):
            if self.reference_snapshot:
                ref_pwmg = self.reference_snapshot.pwmg.get((r, i, rp))
                if ref_pwmg is not None and ref_pwmg > 0.0:
                    return float(ref_pwmg)
            return 1.0

        def get_xwmg_init(m, r, i, rp):
            return max(
                sum(self.params.benchmark.vtwr.get((r, i, rp, margin), 0.0) for margin in self.sets.m),
                0.0,
            )

        def get_xmgm_init(m, margin, r, i, rp):
            return max(self.params.benchmark.vtwr.get((r, i, rp, margin), 0.0), 0.0)

        def get_xtmg_init(m, margin):
            total = 0.0
            for r in self.sets.r:
                for i in self.sets.i:
                    for rp in self.sets.r:
                        total += self.params.benchmark.vtwr.get((r, i, rp, margin), 0.0)
            return max(total, 0.0)

        model.pwmg = Var(
            model.r,
            model.i,
            model.rp,
            within=NonNegativeReals,
            initialize=get_pwmg_init,
            doc="Route-specific trade and transport margin price",
        )
        model.xwmg = Var(
            model.r,
            model.i,
            model.rp,
            within=NonNegativeReals,
            initialize=get_xwmg_init,
            doc="Demand for trade and transport services by route",
        )
        model.xmgm = Var(
            model.m,
            model.r,
            model.i,
            model.rp,
            within=NonNegativeReals,
            initialize=get_xmgm_init,
            doc="Demand for TT services by mode and route",
        )
        model.xtmg = Var(
            model.m,
            within=NonNegativeReals,
            initialize=get_xtmg_init,
            doc="Global demand for TT services by mode",
        )
        model.ptmg = Var(
            model.m,
            within=NonNegativeReals,
            initialize=1.0,
            doc="Margin commodity trade price index",
        )
        model.etax = Var(
            model.r,
            model.i,
            within=Reals,
            initialize=0.0,
            doc="Export tax shifter uniform across destinations",
        )
        model.mtax = Var(
            model.r,
            model.i,
            within=Reals,
            initialize=0.0,
            doc="Import tax shifter uniform across sources",
        )
        model.lambdam = Var(
            model.r,
            model.i,
            model.rp,
            within=NonNegativeReals,
            initialize=1.0,
            doc="Second-level Armington preference shifter",
        )

        # GAMS baseline calibration (cal.gms):
        #   etax.fx(r,i,t) = 0 ;
        #   mtax.fx(r,i,t) = 0 ;
        #   lambdam.fx(rp,i,r,t) = 1 ;
        # Keep these exogenous unless a scenario explicitly unfixes them.
        for r in self.sets.r:
            for i in self.sets.i:
                model.etax[r, i].fix(0.0)
                model.mtax[r, i].fix(0.0)
        for rp in self.sets.r:
            for i in self.sets.i:
                for r in self.sets.r:
                    model.lambdam[rp, i, r].fix(1.0)

        def _safe_trade_component_value(component_name, key, default: float) -> float:
            component = getattr(model, component_name, None)
            if component is None:
                return default
            try:
                return float(value(component[key]))
            except Exception:
                return default

        def _trade_lambdam_value(exporter, commodity, importer) -> float:
            return max(_safe_trade_component_value("lambdam", (exporter, commodity, importer), 1.0), 1e-12)

        def _trade_chipm_value(exporter, commodity, importer) -> float:
            return max(_safe_trade_component_value("chipm", (exporter, commodity, importer), 1.0), 1e-12)

        def _trade_import_price_value(exporter, commodity, importer) -> float:
            if exporter == importer:
                return 1.0
            pmcif = get_pmcif_init(model, exporter, commodity, importer)
            imptx = float(self.params.taxes.imptx.get((exporter, commodity, importer), 0.0) or 0.0)
            mtax = _trade_mtax_value(importer, commodity, exporter)
            chipm = _trade_chipm_value(exporter, commodity, importer)
            return max(((1.0 + imptx + mtax) * pmcif) / chipm, 1e-12)

        def _trade_mtax_value(importer, commodity, exporter) -> float:
            # GAMS mtax(r,i) is importer/commodity-specific and uniform across sources.
            return _safe_trade_component_value("mtax", (importer, commodity), 0.0)

        def _trade_etax_value(exporter, commodity, importer) -> float:
            # GAMS etax(r,i) is exporter/commodity-specific and uniform across destinations.
            return _safe_trade_component_value("etax", (exporter, commodity), 0.0)
        
        # Bilateral trade prices (GAMS pmeq, pmcifeq, pefobeq variables)
        def get_pm_init(m, rp, i, r):
            """Initialize bilateral import price (tariff-inclusive)."""
            if rp == r:
                xw_bench = float(self.params.benchmark.vxsb.get((rp, i, r), 0.0) or 0.0)
                if xw_bench <= 0.0:
                    return 1.0
            else:
                bilateral_imports = float(self.params.benchmark.vcif.get((rp, i, r), 0.0) or 0.0)
                bilateral_exports = float(self.params.benchmark.vxmd.get((rp, i, r), 0.0) or 0.0)
                if bilateral_imports <= 0.0 and bilateral_exports <= 0.0:
                    return 1.0
            pmcif = get_pmcif_init(m, rp, i, r)
            imptx = float(self.params.taxes.imptx.get((rp, i, r), 0.0) or 0.0)
            mtax = _trade_mtax_value(r, i, rp)
            chipm = _trade_chipm_value(rp, i, r)
            return max(((1.0 + imptx + mtax) * pmcif) / chipm, 1e-8)
            
        def get_pmcif_init(m, rp, i, r):
            """Initialize CIF import price."""
            if rp == r:
                xw_bench = float(self.params.benchmark.vxsb.get((rp, i, r), 0.0) or 0.0)
                if xw_bench <= 0.0:
                    return 1.0
                bilateral_imports = float(self.params.benchmark.vcif.get((rp, i, r), 0.0) or 0.0)
                if bilateral_imports > 0.0:
                    return max(bilateral_imports / max(xw_bench, 1e-12), 1e-8)
                export_tax = float(self.params.taxes.rtxs.get((rp, i, r), 0.0) or 0.0)
                etax = _trade_etax_value(rp, i, r)
                tmarg = float(m.tmarg[rp, i, r]) if hasattr(m, "tmarg") else 0.0
                return max(1.0 + export_tax + etax + tmarg, 1e-8)

            bilateral_exports = float(self.params.benchmark.vxmd.get((rp, i, r), 0.0) or 0.0)
            bilateral_imports = float(self.params.benchmark.vcif.get((rp, i, r), 0.0) or 0.0)
            if bilateral_exports > 0.0 and bilateral_imports > 0.0:
                return max(bilateral_imports / bilateral_exports, 1e-8)
            if bilateral_imports > 0.0:
                return max(bilateral_imports, 1e-8)
            if bilateral_exports > 0.0:
                export_tax = float(self.params.taxes.rtxs.get((rp, i, r), 0.0) or 0.0)
                etax = _trade_etax_value(rp, i, r)
                tmarg = float(m.tmarg[rp, i, r]) if hasattr(m, "tmarg") else 0.0
                return max(1.0 + export_tax + etax + tmarg, 1e-8)
            return 1.0
            
        def get_pefob_init(m, r, i, rp):
            """Initialize FOB export price."""
            if r == rp:
                xw_bench = float(self.params.benchmark.vxsb.get((r, i, rp), 0.0) or 0.0)
                if xw_bench <= 0.0:
                    return 1.0
            else:
                bilateral_exports = self.params.benchmark.vxmd.get((r, i, rp), 0.0)
                if bilateral_exports <= 0.0:
                    return 1.0
            export_tax = float(self.params.taxes.rtxs.get((r, i, rp), 0.0) or 0.0)
            etax = _trade_etax_value(r, i, rp)
            return max(1.0 + export_tax + etax, 1e-8)
            
        model.pm = Var(
            model.rp,
            model.i,
            model.r,
            within=NonNegativeReals,
            initialize=get_pm_init,
            doc="Bilateral import price (tariff-inclusive)",
        )
        model.pmcif = Var(
            model.rp,
            model.i,
            model.r,
            within=NonNegativeReals,
            initialize=get_pmcif_init,
            doc="CIF import price (FOB + margins)",
        )
        model.pefob = Var(
            model.r,
            model.i,
            model.rp,
            within=NonNegativeReals,
            initialize=get_pefob_init,
            doc="FOB export price",
        )
        
        # Factors (4 vars per r,f)
        model.xft = Var(model.r, model.f, within=NonNegativeReals, initialize=get_factor_supply_init, doc="Factor supply")
        model.pft = Var(model.r, model.f, within=NonNegativeReals, initialize=get_pft_init, doc="Factor price")
        model.xf = Var(model.r, model.f, model.a, within=NonNegativeReals, initialize=get_vfm_init, doc="Factor demand")
        model.pf = Var(model.r, model.f, model.a, within=NonNegativeReals, initialize=get_pf_init, doc="Factor price by activity")
        model.pfa = Var(model.r, model.f, model.a, within=NonNegativeReals, initialize=get_pfa_init, doc="Factor price tax inclusive")
        model.pfy = Var(model.r, model.f, model.a, within=NonNegativeReals, initialize=get_pfy_init, doc="After-tax factor price")
        model.pwfact = Var(within=NonNegativeReals, initialize=1.0, doc="World factor price")
        
        # Income (3 vars per r) - GAMS-style benchmark calibration
        def get_benchmark_yc(r):
            return sum(get_vpm_init(None, r, i) for i in self.sets.i)

        def get_benchmark_yg(r):
            return sum(get_vgm_init(None, r, i) for i in self.sets.i)

        def get_benchmark_yi(r):
            return sum(get_vim_init(None, r, i) for i in self.sets.i)

        def get_xiagg_init(m, r):
            return max(get_benchmark_yi(r), 1e-8)

        def get_pi_benchmark_init(m, r):
            sigmai_raw = float(self.params.elasticities.esubi.get(r, 1.0))
            # At benchmark all pa=1 and axi=1, so pi=1 regardless of sigmai.
            # GAMS cal.gms bumps sigmai=1 → 1.01 (CES not CD)
            if abs(sigmai_raw - 1.0) < 1e-8:
                sigmai_raw = 1.01
            expo = 1.0 - sigmai_raw
            terms = []
            for i in self.sets.i:
                share = float(m.i_share[r, i])
                if share <= 0.0:
                    continue
                pa_inv = get_pa_benchmark_init(m, r, i, GTAP_INVESTMENT_AGENT)
                terms.append(share * (pa_inv ** expo))
            if not terms:
                return 1.0
            return max((sum(terms) ** (1.0 / expo)) / max(float(m.axi[r]), 1e-12), 1e-8)

        def get_xi_init(m, r, i):
            share = float(m.i_share[r, i])
            if share <= 0.0:
                return 0.0
            sigmai_raw = float(self.params.elasticities.esubi.get(r, 1.0))
            # GAMS cal.gms bumps sigmai=1 → 1.01
            if abs(sigmai_raw - 1.0) < 1e-8:
                sigmai_raw = 1.01
            xiagg = get_xiagg_init(m, r)
            pi_bench = get_pi_benchmark_init(m, r)
            pa_inv = get_pa_benchmark_init(m, r, i, GTAP_INVESTMENT_AGENT)
            return max(share * xiagg * (pi_bench / max(pa_inv, 1e-12)) ** sigmai_raw, 0.0)

        # Final demand (3 vars per r,i)
        model.xc = Var(model.r, model.i, within=NonNegativeReals, initialize=get_xc_init, doc="Private consumption")
        model.xg = Var(model.r, model.i, within=NonNegativeReals, initialize=get_vgm_init, doc="Government consumption")
        model.xi = Var(model.r, model.i, within=NonNegativeReals, initialize=get_xi_init, doc="Investment")
        model.xaa = Var(model.r, model.i, model.aa, within=NonNegativeReals, initialize=get_xaa_init, doc="Agent/activity Armington demand")
        
        # Income (3 vars per r) - GAMS-style benchmark calibration

        def get_benchmark_regy(r):
            facty = sum(
                self.params.benchmark.vfm.get((r, f, a), 0.0)
                for f in self.sets.f
                for a in self.sets.a
            )
            vdep = float(self.params.benchmark.vdep.get(r, 0.0))
            facty = max(facty - vdep, 0.0)
            ytax = get_ytax_ind_init(None, r)
            return facty + ytax

        def get_regy_init(m, r):
            """Regional income initialized from benchmark regy."""
            return get_benchmark_regy(r)

        def get_income_share(share_param_name: str, region: str) -> float:
            share_param = getattr(model, share_param_name)
            share = float(share_param[region])
            return max(share, 1e-8)

        def get_rsav_init(m, r):
            base = float(self.params.benchmark.save.get(r, 0.0))
            if base > 0.0:
                return base
            regy_val = get_regy_init(m, r)
            yc_val = get_benchmark_yc(r)
            yg_val = get_benchmark_yg(r)
            return regy_val - yc_val - yg_val

        def get_savf_init(m, r):
            savf_flag = getattr(self.closure, "savf_flag", "capFix") if self.closure else "capFix"
            if savf_flag == "capFix":
                return float(model.savf_bar[r])
            yi_val = get_benchmark_yi(r)
            rsav_val = get_rsav_init(m, r)
            dep_val = float(self.params.benchmark.vdep.get(r, 0.0))
            return yi_val - dep_val - rsav_val

        def get_yi_init(m, r):
            if self.reference_snapshot and self.reference_snapshot.yi.get(r) is not None:
                return float(self.reference_snapshot.yi.get(r))
            # Use direct investment flows from SAM (sum of vdip+vmip per commodity).
            # This aligns yi with the basis used for i_share calibration and xiagg_init,
            # so eq_xiagg (pi*xiagg=yi) starts consistent at benchmark (avoids 5% drift
            # from the dep+rsav+savf identity which may not balance exactly at initialization).
            return get_benchmark_yi(r)

        def get_facty_init(m, r):
            factor_income = sum(
                self.params.benchmark.vfm.get((r, f, a), 0.0)
                for f in self.sets.f
                for a in self.sets.a
            )
            dep = float(self.params.benchmark.vdep.get(r, 0.0))
            return max(factor_income - dep, 0.0)

        def _imptx_rate(key: tuple[str, str, str]) -> float:
            raw = self.params.taxes.imptx.get(key)
            if raw is None:
                return 0.0
            return float(raw)

        def get_ytax_ind_init(m, r):
            return self._compute_ytax_ind_bench(r)

        def get_ytax_tot_init(m, r):
            return get_ytax_ind_init(m, r)

        def get_ytax_stream_init(m, r, gy):
            if gy == "pt":
                total = 0.0
                for a in self.sets.a:
                    outputs = self.sets.activity_commodities.get(a, list(self.sets.i))
                    for i in outputs:
                        total += (
                            float(self.params.benchmark.makb.get((r, a, i), 0.0) or 0.0)
                            - float(self.params.benchmark.maks.get((r, a, i), 0.0) or 0.0)
                        )
                return total

            if gy == "ft":
                total = 0.0
                for (rr, f, a), rtf in self.params.taxes.rtf.items():
                    if rr == r:
                        total += float(rtf) * self.params.benchmark.vfm.get((r, f, a), 0.0)
                return total

            if gy == "fs":
                return 0.0

            if gy in ("fc", "pc", "gc", "ic"):
                if gy == "fc":
                    total = 0.0
                    for (rr, i, a), rtpd in self.params.taxes.rtpd.items():
                        if rr == r:
                            total += float(rtpd) * self.params.benchmark.vdfb.get((r, i, a), 0.0)
                    for (rr, i, a), rtpi in self.params.taxes.rtpi.items():
                        if rr == r:
                            total += float(rtpi) * self.params.benchmark.vmfb.get((r, i, a), 0.0)
                    return total
                if gy == "pc":
                    total = 0.0
                    for i in self.sets.i:
                        total += (
                            float(self.params.benchmark.vdpp.get((r, i), 0.0) or 0.0)
                            - float(self.params.benchmark.vdpb.get((r, i), 0.0) or 0.0)
                        )
                        total += (
                            float(self.params.benchmark.vmpp.get((r, i), 0.0) or 0.0)
                            - float(self.params.benchmark.vmpb.get((r, i), 0.0) or 0.0)
                        )
                    return total
                if gy == "gc":
                    total = 0.0
                    for (rr, i), rate in self.params.taxes.rtgd.items():
                        if rr == r:
                            total += float(rate) * self.params.benchmark.vdgb.get((r, i), 0.0)
                    for (rr, i), rate in self.params.taxes.rtgi.items():
                        if rr == r:
                            total += float(rate) * self.params.benchmark.vmgb.get((r, i), 0.0)
                    return total
                if gy == "ic":
                    total = 0.0
                    for i in self.sets.i:
                        total += (
                            float(self.params.benchmark.vdip.get((r, i), 0.0) or 0.0)
                            - float(self.params.benchmark.vdib.get((r, i), 0.0) or 0.0)
                        )
                        total += (
                            float(self.params.benchmark.vmip.get((r, i), 0.0) or 0.0)
                            - float(self.params.benchmark.vmib.get((r, i), 0.0) or 0.0)
                        )
                    return total
                return 0.0

            if gy == "gc":
                total = 0.0
                for (rr, i), rate in self.params.taxes.rtgd.items():
                    if rr == r:
                        total += float(rate) * self.params.benchmark.vdgb.get((r, i), 0.0)
                for (rr, i), rate in self.params.taxes.rtgi.items():
                    if rr == r:
                        total += float(rate) * self.params.benchmark.vmgb.get((r, i), 0.0)
                return total

            if gy == "et":
                total = 0.0
                for (rr, i, rp), rtxs in self.params.taxes.rtxs.items():
                    if rr == r:
                        # Benchmark initialization follows cal.gms where
                        # etax is fixed to zero at the benchmark.
                        total += float(rtxs) * self.params.benchmark.vxsb.get((r, i, rp), 0.0)
                return total

            if gy == "mt":
                total = 0.0
                for (exporter, i, importer), rate in self.params.taxes.imptx.items():
                    if importer == r:
                        # Benchmark initialization follows cal.gms where
                        # mtax is fixed to zero at the benchmark.
                        total += float(rate) * float(self.params.benchmark.vcif.get((exporter, i, r), 0.0) or 0.0)
                return total

            if gy == "dt":
                total = 0.0
                for f in self.sets.f:
                    for a in self.sets.a:
                        kappa = float(self.params.taxes.kappaf_activity.get((r, f, a), 0.0))
                        if kappa == 0.0:
                            kappa = float(self.params.taxes.kappaf.get((r, f), 0.0))
                        total += (
                            kappa
                            * self.params.benchmark.vfm.get((r, f, a), 0.0)
                            / max(value(model.xscale[r, a]), 1e-12)
                        )
                return total

            return 0.0

        def get_xcshr_init(m, r, i):
            # Use same demand total as c_share = private_demand / private_total
            total = sum(self.params.benchmark.get_private_demand(r, j)[0] for j in self.sets.i)
            if total <= 0.0:
                return 0.0
            return self.params.benchmark.get_private_demand(r, i)[0] / total
        
        model.regy = Var(model.r, within=Reals, initialize=get_regy_init, doc="Regional income")
        model.yc = Var(
            model.r,
            within=Reals,
            initialize=lambda m, r: float(self.reference_snapshot.yc.get(r)) if self.reference_snapshot and self.reference_snapshot.yc.get(r) is not None else get_benchmark_yc(r),
            doc="Private income",
        )
        model.yg = Var(
            model.r,
            within=Reals,
            initialize=lambda m, r: float(self.reference_snapshot.yg.get(r)) if self.reference_snapshot and self.reference_snapshot.yg.get(r) is not None else get_benchmark_yg(r),
            doc="Government income",
        )
        model.yi = Var(
            model.r,
            within=Reals,
            initialize=get_yi_init,
            doc="Investment income",
        )
        model.rsav = Var(model.r, within=Reals, initialize=get_rsav_init, doc="Regional savings")
        model.facty = Var(model.r, within=NonNegativeReals, initialize=get_facty_init, doc="Factor income net of depreciation")
        model.ytax = Var(model.r, model.gy, within=Reals, initialize=get_ytax_stream_init, doc="Government tax revenue by stream")
        model.ytaxTot = Var(model.r, within=Reals, initialize=get_ytax_tot_init, doc="Total government revenue")
        model.ytax_ind = Var(model.r, within=Reals, initialize=get_ytax_ind_init, doc="Indirect tax revenue")
        model.ytaxshr = Var(
            model.r,
            model.gy,
            within=Reals,
            initialize=lambda m, r, gy: float(get_ytax_stream_init(m, r, gy)) / max(float(get_regy_init(m, r)), 1e-12),
            doc="Indirect tax revenues as share of regional income",
        )
        
        # Numeraire (all prices = 1.0 like GAMS)
        model.pnum = Var(within=NonNegativeReals, initialize=1.0, doc="Numeraire")
        model.pabs = Var(model.r, within=NonNegativeReals, initialize=1.0, doc="Aggregate absorption price")
        model.pi = Var(model.r, within=NonNegativeReals, initialize=get_pi_benchmark_init, doc="Investment price deflator")
        model.pfact = Var(model.r, within=NonNegativeReals, initialize=1.0, doc="Regional factor price")
        model.kstock = Var(model.r, within=NonNegativeReals, initialize=get_kstock_init, doc="Capital stock")
        model.kapEnd = Var(model.r, within=NonNegativeReals, initialize=get_kapend_init, doc="End-of-period capital stock")
        model.arent = Var(model.r, within=NonNegativeReals, initialize=0.05, doc="Rate of return after tax")
        model.rorc = Var(model.r, within=Reals, initialize=0.05, doc="Net rate of return to capital")
        model.rore = Var(model.r, within=Reals, initialize=0.05, doc="Expected rate of return")
        model.gdpmp = Var(model.r, within=NonNegativeReals, initialize=get_gdpmp_init, doc="Nominal GDP at market prices")
        model.rgdpmp = Var(model.r, within=NonNegativeReals, initialize=get_gdpmp_init, doc="Real GDP at market prices")
        model.pgdpmp = Var(model.r, within=NonNegativeReals, initialize=1.0, doc="GDP price deflator")
        model.xiagg = Var(model.r, within=NonNegativeReals, initialize=get_xiagg_init, doc="Aggregate investment volume")
        # Utility and savings aggregates (single household representative)
        model.pcons = Var(model.r, within=NonNegativeReals, initialize=1.0, doc="Consumer price index")
        model.xcshr = Var(model.r, model.i, within=NonNegativeReals, initialize=get_xcshr_init, doc="Household budget share")
        model.uh = Var(model.r, within=NonNegativeReals, initialize=1.0, doc="Private utility per capita")
        model.ug = Var(model.r, within=NonNegativeReals, initialize=1.0, doc="Government utility per capita")
        model.us = Var(model.r, within=NonNegativeReals, initialize=1.0, doc="Savings utility per capita")
        model.u = Var(model.r, within=NonNegativeReals, initialize=1.0, doc="Total utility")
        model.psave = Var(model.r, within=NonNegativeReals, initialize=1.0, doc="Price of savings")
        model.savf = Var(model.r, within=Reals, initialize=get_savf_init, doc="Foreign savings")
        model.chif = Var(
            model.r,
            within=Reals,
            initialize=lambda m, r: float(m.chif0[r]),
            doc="Share of nominal foreign savings in regional income",
        )
        model.ev = Var(
            model.r,
            within=NonNegativeReals,
            initialize=lambda m, r: max(float(m.yc[r].value if m.yc[r].value is not None else 1.0), 1e-8),
            doc="Equivalent variation",
        )
        model.cv = Var(
            model.r,
            within=NonNegativeReals,
            initialize=lambda m, r: max(float(m.yc[r].value if m.yc[r].value is not None else 1.0), 1e-8),
            doc="Compensating variation",
        )
        model.xigbl = Var(within=NonNegativeReals, initialize=get_xigbl_init, doc="Global net investment")
        model.pigbl = Var(within=NonNegativeReals, initialize=get_pigbl_init, doc="Price of global investment")
        model.chiSave = Var(within=NonNegativeReals, initialize=1.0, doc="Savings price adjustment")
        model.rorg = Var(within=NonNegativeReals, initialize=1.0, doc="Global rate of return")
        model.walras = Var(within=Reals, initialize=0.0, doc="Walras check")

        self._refresh_macro_initial_state(model)
        
        # Set strict positive lower bounds to prevent division by zero and negative powers.
        # Follow GAMS where possible:
        # - broad "always-positive" variables keep a small absolute floor
        # - utility/absorption variables use a relative floor based on the base level
        #   (GAMS iterloop uses 0.001 * previous solution level)
        # NOTE: Only set bounds on variables with positive initial values; those initialized to 0 must stay at 0.
        MIN_QUANTITY = 1e-8
        GAMS_REL_LOWER_BOUND = 1e-3

        def _set_relative_positive_lower_bound(var) -> None:
            for vardata in var.values():
                init_val = value(vardata)
                if init_val is None or init_val <= 0.0:
                    continue
                vardata.setlb(max(MIN_QUANTITY, GAMS_REL_LOWER_BOUND * float(init_val)))

        price_vars = [
            'px', 'pp', 'p_rai', 'pp_rai', 'ps', 'pd', 'pa', 'pmt', 'pet', 'pva', 'pnd', 'pft', 'pf', 'pfa', 'pfy',
            'pnum', 'pabs', 'pfact', 'pwfact', 'pgdpmp', 'psave', 'pigbl',
        ]
        # GAMS lower-bounds prices aggressively, but not Armington/trade quantities.
        # Keeping positive floors on quantities like xmt/xd/xa can create artificial
        # states where volumes are stuck at 1e-8 while CES prices explode.
        strictly_positive_level_vars = ['xiagg', 'kstock', 'kapEnd', 'xigbl', 'chiSave', 'rorg', 'ev', 'cv']

        for var_name in price_vars + strictly_positive_level_vars:
            if hasattr(model, var_name):
                var = getattr(model, var_name)
                for idx in var:
                    var[idx].setlb(MIN_QUANTITY)

        # Keep positive price variables on a benchmark-relative path instead of
        # letting them collapse all the way to MIN_QUANTITY. This is especially
        # important for CES/CET blocks that use negative exponents (for example
        # pmt in pmteq and pd/pdp in Armington demand equations).
        for var_name in price_vars:
            if hasattr(model, var_name):
                _set_relative_positive_lower_bound(getattr(model, var_name))

        # Match the GTAP/GAMS iterloop policy for utility and top-level absorption prices:
        # uh.lo, ug.lo, us.lo, u.lo, pcons.lo, pi.lo, ptmg.lo = 0.001 * previous/base level
        # Extend the same policy to macro prices/stocks that appear in ratios/powers.
        for var_name in ['uh', 'ug', 'us', 'u', 'pcons', 'pi', 'ptmg', 'psave', 'pigbl', 'kapEnd', 'xigbl', 'chiSave', 'rorg', 'ev', 'cv']:
            if hasattr(model, var_name):
                _set_relative_positive_lower_bound(getattr(model, var_name))

        # Trade-route prices appear with negative exponents in pmteq/xweq and
        # need route-specific positive lower bounds as well.
        for var_name in ['pm', 'pmcif', 'pefob', 'pwmg', 'pe']:
            if hasattr(model, var_name):
                _set_relative_positive_lower_bound(getattr(model, var_name))
        
        # Conditional quantity vars: can be zero if benchmark is zero
        # Only set lb=1e-8 if initial value is positive
        # xf: factor demand can be zero (e.g., Land/NatRes not used in services)
        # xw, xe: bilateral trade can be zero for certain trade pairs
        conditional_quantity_vars = ['xet', 'xda', 'xma', 'xe', 'xc', 'xg', 'xi', 'xaa', 'xds', 'xf', 'xw', 'xwmg', 'xmgm', 'xtmg']
        for var_name in conditional_quantity_vars:
            if hasattr(model, var_name):
                var = getattr(model, var_name)
                for idx in var:
                    init_val = value(var[idx])
                    if init_val is not None and init_val > MIN_QUANTITY:
                        var[idx].setlb(MIN_QUANTITY)
    
    def _add_equations(self, model: "ConcreteModel") -> None:
        """Add all equations for square system."""
        from pyomo.environ import Constraint, exp, log, value

        def _vmsb_value(region, commodity, partner) -> float:
            val = self.params.benchmark.vmsb.get((partner, commodity, region))
            if val is None:
                val = self.params.benchmark.vmsb.get((region, commodity, partner), 0.0)
            if not val or val <= 0.0:
                val = self.params.benchmark.viws.get((partner, commodity, region), 0.0)
            if not val or val <= 0.0:
                val = self.params.benchmark.vcif.get((partner, commodity, region), 0.0)
            return float(val or 0.0)

        def _imptx_rate_importer(importer, commodity, exporter) -> float:
            raw = self.params.taxes.imptx.get((exporter, commodity, importer))
            if raw is None:
                raw = self.params.taxes.imptx.get((importer, commodity, exporter), 0.0)
            return float(raw or 0.0)

        def _safe_component_value(component_name, key, default: float) -> float:
            component = getattr(model, component_name, None)
            if component is None:
                return default
            try:
                return float(value(component[key]))
            except Exception:
                return default

        def _lambdam_value(exporter, commodity, importer) -> float:
            return max(_safe_component_value("lambdam", (exporter, commodity, importer), 1.0), 1e-12)

        def _chipm_value(exporter, commodity, importer) -> float:
            return max(_safe_component_value("chipm", (exporter, commodity, importer), 1.0), 1e-12)

        def _mtax_value(importer, commodity, exporter) -> float:
            # GAMS mtax(r,i) is indexed by importer and commodity only.
            return _safe_component_value("mtax", (importer, commodity), 0.0)

        def _etax_value(exporter, commodity, importer) -> float:
            # GAMS etax(r,i) is indexed by exporter and commodity only.
            return _safe_component_value("etax", (exporter, commodity), 0.0)

        if_sub = bool(getattr(self.closure, "if_sub", True)) if self.closure is not None else True

        def _factor_tax_value(region, factor, activity) -> float:
            return float(self.params.taxes.rtf.get((region, factor, activity), 0.0) or 0.0)

        def _kappaf_value(region, factor, activity) -> float:
            kappa = float(self.params.taxes.kappaf_activity.get((region, factor, activity), 0.0) or 0.0)
            if kappa == 0.0:
                kappa = float(self.params.taxes.kappaf.get((region, factor), 0.0) or 0.0)
            return kappa

        # --------------------------------------------------------------------
        # GAMS-style substitution macros (ifSUB)
        # --------------------------------------------------------------------
        def _m_pp(region, activity, commodity):
            if if_sub:
                return (1.0 + value(model.prdtx_rai[region, activity, commodity])) * model.p_rai[region, activity, commodity]
            return model.pp_rai[region, activity, commodity]

        def _m_xwmg(exporter, commodity, importer):
            if if_sub:
                return model.tmarg[exporter, commodity, importer] * model.xw[exporter, commodity, importer]
            return model.xwmg[exporter, commodity, importer]

        def _m_xmgm(mode, exporter, commodity, importer):
            if if_sub:
                share = model.amgm[mode, exporter, commodity, importer]
                lambdamg = model.lambdamg[mode, exporter, commodity, importer] + 1e-12
                return share * _m_xwmg(exporter, commodity, importer) / lambdamg
            return model.xmgm[mode, exporter, commodity, importer]

        def _m_pwmg(exporter, commodity, importer):
            if if_sub:
                return sum(
                    model.amgm[m, exporter, commodity, importer] * model.ptmg[m]
                    / (model.lambdamg[m, exporter, commodity, importer] + 1e-12)
                    for m in model.m
                )
            return model.pwmg[exporter, commodity, importer]

        def _m_pefob(exporter, commodity, importer):
            if if_sub:
                export_tax = float(self.params.taxes.rtxs.get((exporter, commodity, importer), 0.0) or 0.0)
                etax = _etax_value(exporter, commodity, importer)
                return (1.0 + export_tax + etax) * model.pe[exporter, commodity, importer]
            return model.pefob[exporter, commodity, importer]

        def _m_pmcif(exporter, commodity, importer):
            if if_sub:
                tmarg = model.tmarg[exporter, commodity, importer]
                return _m_pefob(exporter, commodity, importer) + _m_pwmg(exporter, commodity, importer) * tmarg
            return model.pmcif[exporter, commodity, importer]

        def _m_pm(exporter, commodity, importer):
            if if_sub:
                imptx = _imptx_rate_importer(importer, commodity, exporter)
                mtax = _mtax_value(importer, commodity, exporter)
                chipm = _chipm_value(exporter, commodity, importer)
                return ((1.0 + imptx + mtax) * _m_pmcif(exporter, commodity, importer)) / chipm
            return model.pm[exporter, commodity, importer]

        def _m_pfa(region, factor, activity):
            if if_sub:
                return model.pf[region, factor, activity] * (1.0 + _factor_tax_value(region, factor, activity))
            return model.pfa[region, factor, activity]

        def _m_pfy(region, factor, activity):
            if if_sub:
                return model.pf[region, factor, activity] * (1.0 - _kappaf_value(region, factor, activity))
            return model.pfy[region, factor, activity]
        
        # ========================================================================
        # PRODUCTION BLOCK
        # ========================================================================
        
        # Legacy Pyomo-only profit identity. Keep the component for compatibility
        # with existing tooling/tests, but leave it inactive because GAMS uses the
        # explicit xpeq/pxeq make block instead of an extra px == pp equation.
        def prf_y_rule(model, r, a):
            return model.px[r, a] == model.pp[r, a]
        model.prf_y = Constraint(model.r, model.a, rule=prf_y_rule)
        model.prf_y.deactivate()

        # Value-added and intermediate nests (GAMS exact formulation)
        # GAMS: nd(r,a,t) =e= and(r,a,t)*xp(r,a,t)*(px(r,a,t)/pnd(r,a,t))**sigmap(r,a)
        #                      * (axp(r,a,t)*lambdand(r,a,t))**(sigmap(r,a)-1)
        def eq_nd_rule(model, r, a):
            and_val = value(model.and_param[r, a])  # GAMS calibrated parameter
            if and_val <= 0.0:
                return Constraint.Skip
            sigmap = self._get_sigmap(r, a)
            px = model.px[r, a]
            pnd = model.pnd[r, a]
            if value(pnd) <= 0:
                return Constraint.Skip
            ratio = px / pnd
            shift = self._axp_shift(r, a) * self._lambdand(r, a)
            return model.nd[r, a] == and_val * model.xp[r, a] * ratio**sigmap * shift**(sigmap - 1)
        model.eq_nd = Constraint(model.r, model.a, rule=eq_nd_rule)

        # GAMS: va(r,a,t) =e= ava(r,a,t)*xp(r,a,t)*(px(r,a,t)/pva(r,a,t))**sigmap(r,a)
        #                      * (axp(r,a,t)*lambdava(r,a,t))**(sigmap(r,a)-1)
        def eq_va_rule(model, r, a):
            ava_val = value(model.ava_param[r, a])  # GAMS calibrated parameter
            if ava_val <= 0.0:
                return Constraint.Skip
            sigmap = self._get_sigmap(r, a)
            px = model.px[r, a]
            pva = model.pva[r, a]
            if value(pva) <= 0:
                return Constraint.Skip
            ratio = px / pva
            shift = self._axp_shift(r, a) * self._lambdava(r, a)
            return model.va[r, a] == ava_val * model.xp[r, a] * ratio**sigmap * shift**(sigmap - 1)
        model.eq_va = Constraint(model.r, model.a, rule=eq_va_rule)
        
        # ========================================================================
        # PRICE EQUATIONS - CES COST FUNCTIONS (GAMS style)
        # ========================================================================

        # Unit cost definition (GAMS pxeq)
        # px**(1-sigmap) = (axp**(sigmap-1)) * [and*(pnd/lambdand)**(1-sigmap) + ava*(pva/lambdava)**(1-sigmap)]
        def eq_pxeq_rule(model, r, a):
            and_val = value(model.and_param[r, a])
            ava_val = value(model.ava_param[r, a])

            # Some benchmark datasets do not provide shift levels (axp/lambda*).
            # In that case, fall back to normalized VA/ND shares to preserve the
            # CES price identity at the benchmark point. Use 1-nd_share for ava
            # to guarantee and+ava=1 regardless of p_va source.
            if not self.params.shifts.axp and not self.params.shifts.lambdand and not self.params.shifts.lambdava:
                and_val = value(model.nd_share[r, a])
                ava_val = 1.0 - and_val

            if and_val <= 0.0 and ava_val <= 0.0:
                return Constraint.Skip

            sigmap = self._get_sigmap(r, a)
            expo = 1.0 - sigmap
            if abs(expo) < 1e-8:
                nd_term = (model.pnd[r, a] / max(self._lambdand(r, a), 1e-8)) ** and_val if and_val > 0.0 else 1.0
                va_term = (model.pva[r, a] / max(self._lambdava(r, a), 1e-8)) ** ava_val if ava_val > 0.0 else 1.0
                return model.px[r, a] == nd_term * va_term

            shift = self._axp_shift(r, a) ** (sigmap - 1.0)
            lambdand = max(self._lambdand(r, a), 1e-8)
            lambdava = max(self._lambdava(r, a), 1e-8)

            term_nd = and_val * (model.pnd[r, a] / lambdand) ** expo if and_val > 0.0 else 0.0
            term_va = ava_val * (model.pva[r, a] / lambdava) ** expo if ava_val > 0.0 else 0.0
            return model.px[r, a] ** expo == shift * (term_nd + term_va)
        model.eq_pxeq = Constraint(model.r, model.a, rule=eq_pxeq_rule)

        # Price of ND bundle (GAMS pndeq)
        # pnd**(1-sigmand) = sum(i, io(r,i,a)*[pa(r,i)/lambdaio(r,i,a)]**(1-sigmand))
        def eq_pndeq_rule(model, r, a):
            sigmand = self._get_sigmand(r, a)
            expo = 1.0 - sigmand
            if abs(expo) < 1e-8:
                terms = []
                for i in model.i:
                    io_val = (
                        value(model.io_param[r, i, a])
                        if hasattr(model, "io_param")
                        else value(model.p_io[r, i, a])
                    )
                    if not self.params.shifts.lambdaio:
                        io_val = value(model.p_io[r, i, a])
                    if io_val <= 0.0:
                        continue
                    lambdaio = max(value(model.lambdaio[r, i, a]), 1e-8)
                    terms.append((model.pa[r, i, a] / lambdaio) ** io_val)
                if not terms:
                    return Constraint.Skip
                prod = 1.0
                for term in terms:
                    prod *= term
                return model.pnd[r, a] == prod

            terms = []
            for i in model.i:
                io_val = (
                    value(model.io_param[r, i, a])
                    if hasattr(model, "io_param")
                    else value(model.p_io[r, i, a])
                )

                # If lambdaio is unavailable in the benchmark input, use
                # normalized intermediate shares to keep pndeq benchmark-consistent.
                if not self.params.shifts.lambdaio:
                    io_val = value(model.p_io[r, i, a])

                if io_val <= 0.0:
                    continue
                lambdaio = max(value(model.lambdaio[r, i, a]), 1e-8)
                # Use agent-specific Armington price for activity a
                terms.append(io_val * (model.pa[r, i, a] / lambdaio) ** expo)

            if not terms:
                return Constraint.Skip
            return model.pnd[r, a] ** expo == sum(terms)
        model.eq_pndeq = Constraint(model.r, model.a, rule=eq_pndeq_rule)

        # Price of VA bundle (GAMS pvaeq)
        # pva**(1-sigmav) = sum(f, af(r,f,a)*[pfa(r,f,a)/lambdaf(r,f,a)]**(1-sigmav))
        def eq_pvaeq_rule(model, r, a):
            sigmav = self._get_sigmav(r, a)
            expo = 1.0 - sigmav
            if abs(expo) < 1e-8:
                terms = []
                for f in model.f:
                    af_val = (
                        value(model.af_param[r, f, a])
                        if hasattr(model, "af_param")
                        else value(model.af_share[r, f, a])
                    )
                    if af_val <= 0.0:
                        continue
                    lambdaf = max(self._lambdaf(r, f, a), 1e-8)
                    terms.append((_m_pfa(r, f, a) / lambdaf) ** af_val)
                if not terms:
                    return Constraint.Skip
                prod = 1.0
                for term in terms:
                    prod *= term
                return model.pva[r, a] == prod

            terms = []
            for f in model.f:
                af_val = (
                    value(model.af_param[r, f, a])
                    if hasattr(model, "af_param")
                    else value(model.af_share[r, f, a])
                )
                if af_val <= 0.0:
                    continue
                lambdaf = max(self._lambdaf(r, f, a), 1e-8)
                terms.append(af_val * (_m_pfa(r, f, a) / lambdaf) ** expo)

            if not terms:
                return Constraint.Skip
            return model.pva[r, a] ** expo == sum(terms)
        model.eq_pvaeq = Constraint(model.r, model.a, rule=eq_pvaeq_rule)

        # Output allocation (GAMS xeq)
        def eq_x_rule(model, r, a, i):
            outputs = self.sets.activity_commodities.get(a, [])
            if outputs and i not in outputs:
                return Constraint.Skip
            if value(model.xflag[r, a, i]) <= 0.0:
                return Constraint.Skip
            share = value(model.gx_param[r, a, i])
            make_base = self.params.benchmark.makb.get((r, a, i), 0.0)
            if share <= 0.0 and make_base <= 0.0:
                return Constraint.Skip

            omega = self._get_omegas(r, a)
            if omega == float("inf"):
                return model.p_rai[r, a, i] == model.px[r, a]

            return model.x[r, a, i] == share * (model.xp[r, a] / model.xscale[r, a]) * (model.p_rai[r, a, i] / model.px[r, a]) ** omega
        model.eq_x = Constraint(model.r, model.a, model.i, rule=eq_x_rule)

        def eq_po_rule(model, r, a):
            outputs = self.sets.activity_commodities.get(a)
            if not outputs:
                return Constraint.Skip

            active_outputs = [
                i for i in outputs
                if value(model.gx_param[r, a, i]) > 0.0 or self.params.benchmark.makb.get((r, a, i), 0.0) > 0.0
            ]
            if not active_outputs:
                return Constraint.Skip

            omega = self._get_omegas(r, a)
            if omega == float("inf"):
                # GAMS xpeq with omegas=inf uses xp/xscale on the quantity side.
                return (model.xp[r, a] / model.xscale[r, a]) == sum(model.x[r, a, i] for i in active_outputs)

            exponent = 1.0 + omega
            return model.px[r, a] ** exponent == sum(
                value(model.gx_param[r, a, i]) * model.p_rai[r, a, i] ** exponent
                for i in active_outputs
            )
        model.eq_po = Constraint(model.r, model.a, rule=eq_po_rule)

        # Commodity-level output tax mapping (GAMS ppeq structure):
        # pp(r,a,i) = (1 + prdtx(r,a,i)) * p(r,a,i)
        def eq_pp_rai_rule(model, r, a, i):
            outputs = self.sets.activity_commodities.get(a, [])
            if outputs and i not in outputs:
                return Constraint.Skip
            if value(model.xflag[r, a, i]) <= 0.0:
                return Constraint.Skip
            share = value(model.gx_param[r, a, i])
            make_base = self.params.benchmark.makb.get((r, a, i), 0.0)
            if share <= 0.0 and make_base <= 0.0:
                return Constraint.Skip
            return model.pp_rai[r, a, i] == (1.0 + value(model.prdtx_rai[r, a, i])) * model.p_rai[r, a, i]
        model.eq_pp_rai = Constraint(model.r, model.a, model.i, rule=eq_pp_rai_rule)

        # Commodity aggregation by make-route (GAMS peq)
        def eq_peq_rule(model, r, a, i):
            outputs = self.sets.activity_commodities.get(a, [])
            if outputs and i not in outputs:
                return Constraint.Skip
            if value(model.xflag[r, a, i]) <= 0.0:
                return Constraint.Skip

            share = value(model.gx_param[r, a, i])
            make_base = self.params.benchmark.makb.get((r, a, i), 0.0)
            ax_val = value(model.p_ax[r, a, i])
            if share <= 0.0 and make_base <= 0.0 and ax_val <= 0.0:
                return Constraint.Skip

            sigma = self._get_sigmas(r, i)
            if sigma == float("inf"):
                return _m_pp(r, a, i) == model.ps[r, i]

            return model.x[r, a, i] == ax_val * model.xs[r, i] * (model.ps[r, i] / _m_pp(r, a, i)) ** sigma
        model.eq_peq = Constraint(model.r, model.a, model.i, rule=eq_peq_rule)
        
        # ========================================================================
        # SUPPLY BLOCK
        # ========================================================================
        
        # Domestic supply (GAMS pseq)
        def eq_xs_rule(model, r, i):
            producing_activities = self.sets.commodity_activities.get(i, list(model.a))
            active_activities = [
                a for a in producing_activities
                if value(model.gx_param[r, a, i]) > 0.0 or self.params.benchmark.makb.get((r, a, i), 0.0) > 0.0
            ]
            if not active_activities:
                return Constraint.Skip

            sigma = self._get_sigmas(r, i)
            if sigma == float("inf"):
                return model.xs[r, i] == sum(model.x[r, a, i] for a in active_activities)

            exponent = 1.0 - sigma
            if abs(exponent) < 1e-8:
                return Constraint.Skip
            return model.ps[r, i] ** exponent == sum(
                value(model.p_ax[r, a, i]) * _m_pp(r, a, i) ** exponent
                for a in active_activities
            )
        model.eq_xs = Constraint(model.r, model.i, rule=eq_xs_rule)
        
        # Legacy Pyomo simplification. The CET block should determine ps/pd/pet
        # without an extra identity constraint.
        def eq_ps_rule(model, r, i):
            return model.ps[r, i] == model.pd[r, i]
        model.eq_ps = Constraint(model.r, model.i, rule=eq_ps_rule)
        model.eq_ps.deactivate()
        
        # ========================================================================
        # TRADE - CET DOMESTIC/EXPORT ALLOCATION
        # ========================================================================
        
        def eq_xds_rule(model, r, i):
            omega = self.params.elasticities.omegax.get((r, i), float("inf"))
            gd_share = value(model.gd_share[r, i])
            if gd_share <= 0.0:
                return model.xds[r, i] == 0.0
            if omega == float("inf"):
                return model.pd[r, i] == model.ps[r, i]
            return model.xds[r, i] == model.gd_share[r, i] * model.xs[r, i] * (model.pd[r, i] / model.ps[r, i]) ** omega
        model.eq_xds = Constraint(model.r, model.i, rule=eq_xds_rule)

        def eq_xet_rule(model, r, i):
            omega = self.params.elasticities.omegax.get((r, i), float("inf"))
            if value(model.xet_flag[r, i]) <= 0.0:
                return Constraint.Skip
            ge_share = value(model.ge_share[r, i])
            if ge_share <= 0.0:
                return model.xet[r, i] == 0.0
            if omega == float("inf"):
                return model.pet[r, i] == model.ps[r, i]
            return (
                model.xet[r, i]
                == model.ge_share[r, i]
                * model.xs[r, i]
                * (model.pet[r, i] / model.ps[r, i]) ** omega
            )
        model.eq_xet = Constraint(model.r, model.i, rule=eq_xet_rule)

        def eq_xseq_rule(model, r, i):
            omega = self.params.elasticities.omegax.get((r, i), float("inf"))
            gd_share = value(model.gd_share[r, i])
            ge_share = value(model.ge_share[r, i])
            if omega == float("inf"):
                return model.xs[r, i] == model.xds[r, i] + model.xet[r, i]
            exponent = 1.0 + omega
            return (
                model.ps[r, i] ** exponent
                == model.gd_share[r, i] * model.pd[r, i] ** exponent
                + model.ge_share[r, i] * model.pet[r, i] ** exponent
            )
        model.eq_xseq = Constraint(model.r, model.i, rule=eq_xseq_rule)
        
        # Legacy Pyomo simplification. Aggregate export price should be governed
        # by peeq/peteq, not an extra pet == ps identity.
        def eq_pe_rule(model, r, i):
            return model.pet[r, i] == model.ps[r, i]
        model.eq_pe = Constraint(model.r, model.i, rule=eq_pe_rule)
        model.eq_pe.deactivate()

        # Legacy Pyomo-only route price identity. Keep the component for
        # compatibility with tooling, but leave it inactive because GAMS
        # determines route prices through peeq/peteq instead.
        def eq_pe_route_rule(model, r, i, rp):
            if r == rp:
                return Constraint.Skip
            bilateral_exports = self.params.benchmark.vxmd.get((r, i, rp), 0.0)
            mirror_imports = self.params.benchmark.viws.get((rp, i, r), 0.0)
            if bilateral_exports <= 0.0 and mirror_imports <= 0.0:
                return Constraint.Skip
            return model.pe[r, i, rp] == model.pet[r, i]
        model.eq_pe_route = Constraint(model.r, model.i, model.rp, rule=eq_pe_route_rule)
        model.eq_pe_route.deactivate()
        
        # Aggregate exports over bilateral flows. GAMS uses xw directly in the
        # CET block, so keep the same aggregation object here.
        def eq_xet_agg_rule(model, r, i):
            active_partners = [rp for rp in model.rp if rp != r and self.params.benchmark.vxmd.get((r, i, rp), 0.0) > 0.0]
            if not active_partners:
                return model.xet[r, i] == 0.0
            return model.xet[r, i] == sum(model.xw[r, i, rp] for rp in active_partners)
        model.eq_xet_agg = Constraint(model.r, model.i, rule=eq_xet_agg_rule)
        model.eq_xet_agg.deactivate()
        
        # Legacy Pyomo helper. Keep the component for compatibility with
        # snapshots/reporting, but do not include it in the active MCP.
        def eq_xe_xw_rule(model, r, i, rp):
            if r == rp:
                return Constraint.Skip
            bilateral_exports = self.params.benchmark.vxmd.get((r, i, rp), 0.0)
            if bilateral_exports <= 0.0:
                return Constraint.Skip
            return model.xe[r, i, rp] == model.xw[r, i, rp]
        model.eq_xe_xw = Constraint(model.r, model.i, model.rp, rule=eq_xe_xw_rule)
        model.eq_xe_xw.deactivate()
        
        # ========================================================================
        # TRADE - CES ARMINGTON DOMESTIC/IMPORT
        # ========================================================================
        
        # Armington aggregation (Leontief for simplicity)
        def eq_xa_rule(model, r, i):
            return model.xa[r, i] == sum(model.xaa[r, i, aa] / model.xscale[r, aa] for aa in model.aa)
        model.eq_xa = Constraint(model.r, model.i, rule=eq_xa_rule)
        # Aggregate xa(r,i) is a Pyomo helper not present as a standalone
        # equation in GTAP. Keep it for reporting but exclude from MCP.
        model.eq_xa.deactivate()

        # Agent/activity demand for intermediate inputs by activity.
        def eq_xaa_activity_rule(model, r, i, a):
            io_val = (
                value(model.io_param[r, i, a])
                if hasattr(model, "io_param")
                else value(model.p_io[r, i, a])
            )
            if not self.params.shifts.lambdaio:
                io_val = value(model.p_io[r, i, a])

            if io_val <= 0.0:
                return model.xaa[r, i, a] == 0.0

            sigmand = self._get_sigmand(r, a)
            lambdaio = max(value(model.lambdaio[r, i, a]), 1e-8)
            return model.xaa[r, i, a] == (
                io_val
                * model.nd[r, a]
                * (model.pnd[r, a] / model.pa[r, i, a]) ** sigmand
                * (lambdaio ** (sigmand - 1.0))
            )
        model.eq_xaa_activity = Constraint(model.r, model.i, model.a, rule=eq_xaa_activity_rule)

        def eq_xaa_hhd_rule(model, r, i):
            return model.xaa[r, i, GTAP_HOUSEHOLD_AGENT] == model.xc[r, i]
        model.eq_xaa_hhd = Constraint(model.r, model.i, rule=eq_xaa_hhd_rule)

        def eq_xaa_gov_rule(model, r, i):
            return model.xaa[r, i, GTAP_GOVERNMENT_AGENT] == model.xg[r, i]
        model.eq_xaa_gov = Constraint(model.r, model.i, rule=eq_xaa_gov_rule)

        def eq_xaa_inv_rule(model, r, i):
            return model.xaa[r, i, GTAP_INVESTMENT_AGENT] == model.xi[r, i]
        model.eq_xaa_inv = Constraint(model.r, model.i, rule=eq_xaa_inv_rule)

        def eq_xaa_tmg_rule(model, r, i):
            benchmark_margin = self._vst_value(str(r), str(i))
            return model.xaa[r, i, GTAP_MARGIN_AGENT] == model.xscale[r, GTAP_MARGIN_AGENT] * benchmark_margin
        model.eq_xaa_tmg = Constraint(model.r, model.i, rule=eq_xaa_tmg_rule)

        def _raw_agent_domestic_import_eq(r, i, aa):
            if aa in self.sets.a:
                raw_domestic = self.params.benchmark.vdfb.get((r, i, aa), 0.0)
                raw_import = self.params.benchmark.vmfb.get((r, i, aa), 0.0)
                if raw_domestic + raw_import <= 0.0:
                    raw_domestic = self.params.benchmark.vdfm.get((r, i, aa), 0.0)
                    raw_import = self.params.benchmark.vifm.get((r, i, aa), 0.0)
            elif aa == GTAP_HOUSEHOLD_AGENT:
                raw_domestic = self.params.benchmark.vdpb.get((r, i), 0.0)
                raw_import = self.params.benchmark.vmpb.get((r, i), 0.0)
                if raw_domestic + raw_import <= 0.0:
                    _, raw_domestic, raw_import = self.params.benchmark.get_private_demand(r, i)
            elif aa == GTAP_GOVERNMENT_AGENT:
                raw_domestic = self.params.benchmark.vdgb.get((r, i), 0.0)
                raw_import = self.params.benchmark.vmgb.get((r, i), 0.0)
                if raw_domestic + raw_import <= 0.0:
                    _, raw_domestic, raw_import = self.params.benchmark.get_government_demand(r, i)
            elif aa == GTAP_INVESTMENT_AGENT:
                raw_domestic = self.params.benchmark.vdib.get((r, i), 0.0)
                raw_import = self.params.benchmark.vmib.get((r, i), 0.0)
                if raw_domestic + raw_import <= 0.0:
                    _, raw_domestic, raw_import = self.params.benchmark.get_investment_demand(r, i)
            elif aa == GTAP_MARGIN_AGENT:
                raw_domestic = self._vst_value(str(r), str(i))
                raw_import = 0.0
            else:
                raw_domestic = 0.0
                raw_import = 0.0
            return max(raw_domestic, 0.0), max(raw_import, 0.0)

        benchmark_agent_trade_share_cache: Dict[tuple[str, str, str], tuple[float, float]] = {}
        for r in self.sets.r:
            for i in self.sets.i:
                raw_levels: Dict[str, tuple[float, float]] = {}
                total_raw_import = 0.0
                for aa in list(self.sets.a) + [
                    GTAP_HOUSEHOLD_AGENT,
                    GTAP_GOVERNMENT_AGENT,
                    GTAP_INVESTMENT_AGENT,
                    GTAP_MARGIN_AGENT,
                ]:
                    domestic, imported = _raw_agent_domestic_import_eq(r, i, aa)
                    raw_levels[aa] = (domestic, imported)
                    total_raw_import += imported

                target_import_total = sum(
                    float(self.params.benchmark.vmsb.get((rp, i, r), 0.0) or 0.0)
                    for rp in self.sets.r
                )
                import_scale = (target_import_total / total_raw_import) if total_raw_import > 0.0 else 1.0

                for aa, (domestic, imported) in raw_levels.items():
                    total = domestic + imported * import_scale
                    if total <= 0.0:
                        benchmark_agent_trade_share_cache[(r, i, aa)] = (0.0, 0.0)
                    else:
                        benchmark_agent_trade_share_cache[(r, i, aa)] = (
                            domestic / total,
                            (imported * import_scale) / total,
                        )

        def _top_armington_sigma(r, i, aa):
            # GAMS calibrates sigmam(r,i,aa) and defaults it to esubd(i,r)
            # when no agent-specific override is available.
            return float(self.params.elasticities.esubd.get((r, i), 2.0))

        benchmark_agent_armington_param_cache: Dict[tuple[str, str, str], tuple[float, float]] = {}
        for r in self.sets.r:
            for i in self.sets.i:
                for aa in list(self.sets.a) + [
                    GTAP_HOUSEHOLD_AGENT,
                    GTAP_GOVERNMENT_AGENT,
                    GTAP_INVESTMENT_AGENT,
                    GTAP_MARGIN_AGENT,
                ]:
                    xaa_bench = max(float(value(model.xaa[r, i, aa])), 0.0)
                    xda_bench = max(float(value(model.xda[r, i, aa])), 0.0)
                    xma_bench = max(float(value(model.xma[r, i, aa])), 0.0)
                    if xaa_bench <= 0.0 or (xda_bench <= 0.0 and xma_bench <= 0.0):
                        benchmark_agent_armington_param_cache[(r, i, aa)] = (0.0, 0.0)
                        continue

                    sigma_m = _top_armington_sigma(r, i, aa)
                    pdp_bench = float(value(model.pdp[r, i, aa]))
                    pmp_bench = float(value(model.pmp[r, i, aa]))
                    pa_bench = float(value(model.pa[r, i, aa]))
                    if pa_bench <= 0.0:
                        benchmark_agent_armington_param_cache[(r, i, aa)] = (0.0, 0.0)
                        continue

                    alphad = (xda_bench / xaa_bench) * (pdp_bench / pa_bench) ** sigma_m if xda_bench > 0.0 else 0.0
                    alpham = (xma_bench / xaa_bench) * (pmp_bench / pa_bench) ** sigma_m if xma_bench > 0.0 else 0.0
                    benchmark_agent_armington_param_cache[(r, i, aa)] = (alphad, alpham)

        def get_benchmark_agent_armington_shares(r, i, aa):
            return benchmark_agent_armington_param_cache.get((r, i, aa), (0.0, 0.0))

        # NOTE: eq_pdp, eq_pmp, eq_paa removed - these are now Expression, not Var
        # The price pass-through relationships are encoded directly in the Expression definitions

        def eq_dintxeq_rule(model, r, i, aa):
            def _two_key(raw_map, region, commodity):
                val = raw_map.get((region, commodity), None)
                if val is None:
                    val = raw_map.get((commodity, region), 0.0)
                return float(val or 0.0)

            if aa in self.sets.a:
                numerator = float(self.params.benchmark.vdfp.get((r, i, aa), 0.0) or 0.0) - float(
                    self.params.benchmark.vdfb.get((r, i, aa), 0.0) or 0.0
                )
                denom = max(float(self.params.benchmark.vdfb.get((r, i, aa), 0.0) or 0.0), 0.0)
            elif aa == GTAP_HOUSEHOLD_AGENT:
                numerator = _two_key(self.params.benchmark.vdpp, r, i) - _two_key(self.params.benchmark.vdpb, r, i)
                denom = max(_two_key(self.params.benchmark.vdpb, r, i), 0.0)
            elif aa == GTAP_GOVERNMENT_AGENT:
                numerator = _two_key(self.params.benchmark.vdgp, r, i) - _two_key(self.params.benchmark.vdgb, r, i)
                denom = max(_two_key(self.params.benchmark.vdgb, r, i), 0.0)
            elif aa == GTAP_INVESTMENT_AGENT:
                numerator = _two_key(self.params.benchmark.vdip, r, i) - _two_key(self.params.benchmark.vdib, r, i)
                denom = max(_two_key(self.params.benchmark.vdib, r, i), 0.0)
            else:
                numerator = 0.0
                denom = 0.0

            target = (numerator / denom) if denom > 0.0 else float(self.params.taxes.dintx0.get((r, i, aa), 0.0) or 0.0)
            return model.dintx[r, i, aa] == target
        model.eq_dintxeq = Constraint(model.r, model.i, model.aa, rule=eq_dintxeq_rule)

        def eq_mintxeq_rule(model, r, i, aa):
            def _two_key(raw_map, region, commodity):
                val = raw_map.get((region, commodity), None)
                if val is None:
                    val = raw_map.get((commodity, region), 0.0)
                return float(val or 0.0)

            if aa in self.sets.a:
                numerator = float(self.params.benchmark.vmfp.get((r, i, aa), 0.0) or 0.0) - float(
                    self.params.benchmark.vmfb.get((r, i, aa), 0.0) or 0.0
                )
                denom = max(float(self.params.benchmark.vmfb.get((r, i, aa), 0.0) or 0.0), 0.0)
            elif aa == GTAP_HOUSEHOLD_AGENT:
                numerator = _two_key(self.params.benchmark.vmpp, r, i) - _two_key(self.params.benchmark.vmpb, r, i)
                denom = max(_two_key(self.params.benchmark.vmpb, r, i), 0.0)
            elif aa == GTAP_GOVERNMENT_AGENT:
                numerator = _two_key(self.params.benchmark.vmgp, r, i) - _two_key(self.params.benchmark.vmgb, r, i)
                denom = max(_two_key(self.params.benchmark.vmgb, r, i), 0.0)
            elif aa == GTAP_INVESTMENT_AGENT:
                numerator = _two_key(self.params.benchmark.vmip, r, i) - _two_key(self.params.benchmark.vmib, r, i)
                denom = max(_two_key(self.params.benchmark.vmib, r, i), 0.0)
            else:
                numerator = 0.0
                denom = 0.0

            target = (numerator / denom) if denom > 0.0 else float(self.params.taxes.mintx0.get((r, i, aa), 0.0) or 0.0)
            return model.mintx[r, i, aa] == target
        model.eq_mintxeq = Constraint(model.r, model.i, model.aa, rule=eq_mintxeq_rule)

        def eq_xda_rule(model, r, i, aa):
            domestic_share, _ = get_benchmark_agent_armington_shares(r, i, aa)
            if domestic_share <= 0.0:
                return model.xda[r, i, aa] == 0.0
            sigma_m = _top_armington_sigma(r, i, aa)
            if sigma_m == float("inf"):
                return model.pdp[r, i, aa] == model.paa[r, i, aa]
            return model.xda[r, i, aa] == domestic_share * model.xaa[r, i, aa] * (model.paa[r, i, aa] / model.pdp[r, i, aa]) ** sigma_m
        model.eq_xda = Constraint(model.r, model.i, model.aa, rule=eq_xda_rule)

        def eq_xma_rule(model, r, i, aa):
            _, import_share = get_benchmark_agent_armington_shares(r, i, aa)
            if import_share <= 0.0:
                return model.xma[r, i, aa] == 0.0
            sigma_m = _top_armington_sigma(r, i, aa)
            if sigma_m == float("inf"):
                return model.pmp[r, i, aa] == model.paa[r, i, aa]
            return model.xma[r, i, aa] == import_share * model.xaa[r, i, aa] * (model.paa[r, i, aa] / model.pmp[r, i, aa]) ** sigma_m
        model.eq_xma = Constraint(model.r, model.i, model.aa, rule=eq_xma_rule)

        # GAMS xmteq/xdseq: aggregate demands defined as sum over agents
        def eq_xd_agg_rule(model, r, i):
            return model.xd[r, i] == sum(model.xda[r, i, aa] / model.xscale[r, aa] for aa in model.aa)
        model.eq_xd_agg = Constraint(model.r, model.i, rule=eq_xd_agg_rule)

        def eq_xmt_agg_rule(model, r, i):
            return model.xmt[r, i] == sum(model.xma[r, i, aa] / model.xscale[r, aa] for aa in model.aa)
        model.eq_xmt_agg = Constraint(model.r, model.i, rule=eq_xmt_agg_rule)
        
        # Armington price CES aggregator by agent (GAMS paeq)
        # pa(r,i,aa)**(1-sigmam) = alphad*pdp**(1-sigmam) + alpham*pmp**(1-sigmam)
        def eq_paa_rule(model, r, i, aa):
            # Get shares from benchmark data
            dom_share, imp_share = get_benchmark_agent_armington_shares(r, i, aa)
            alphad = dom_share
            alpham = imp_share
            # Skip if no demand from this agent
            if alphad <= 0.0 and alpham <= 0.0:
                return Constraint.Skip
            sigma_m = _top_armington_sigma(r, i, aa)
            expo = 1.0 - sigma_m
            if abs(expo) < 1e-8:  # Cobb-Douglas case
                # pa = pdp^alphad * pmp^alpham
                return model.pa[r, i, aa] == model.pdp[r, i, aa] ** alphad * model.pmp[r, i, aa] ** alpham
            # CES case
            return model.pa[r, i, aa] ** expo == alphad * model.pdp[r, i, aa] ** expo + alpham * model.pmp[r, i, aa] ** expo
        model.eq_paa = Constraint(model.r, model.i, model.aa, rule=eq_paa_rule)
        
        # NOTE: eq_pmt moved to eq_pmteq (CES formulation) in bilateral trade section below

        # Trade margins (GAMS xwmgeq/xmgmeq/pwmgeq/xtmgeq/ptmgeq - simplified static)
        def eq_xwmg_rule(model, r, i, rp):
            if value(model.tmarg[r, i, rp]) <= 0.0:
                return Constraint.Skip
            return model.xwmg[r, i, rp] == model.tmarg[r, i, rp] * model.xw[r, i, rp]
        model.eq_xwmg = Constraint(model.r, model.i, model.rp, rule=eq_xwmg_rule)

        def eq_xmgm_rule(model, m, r, i, rp):
            share = value(model.amgm[m, r, i, rp])
            if share <= 0.0:
                return Constraint.Skip
            return model.xmgm[m, r, i, rp] == share * _m_xwmg(r, i, rp) / (model.lambdamg[m, r, i, rp] + 1e-12)
        model.eq_xmgm = Constraint(model.m, model.r, model.i, model.rp, rule=eq_xmgm_rule)

        def eq_pwmg_rule(model, r, i, rp):
            if value(model.tmarg[r, i, rp]) <= 0.0:
                return Constraint.Skip
            total = sum(
                model.amgm[m, r, i, rp] * model.ptmg[m] / (model.lambdamg[m, r, i, rp] + 1e-12)
                for m in model.m
            )
            return model.pwmg[r, i, rp] == total
        model.eq_pwmg = Constraint(model.r, model.i, model.rp, rule=eq_pwmg_rule)

        def eq_xtmg_rule(model, m):
            return model.xtmg[m] == sum(_m_xmgm(m, r, i, rp) for r in model.r for i in model.i for rp in model.rp)
        model.eq_xtmg = Constraint(model.m, rule=eq_xtmg_rule)

        def eq_ptmg_rule(model, m):
            return model.ptmg[m] == model.pnum
        model.eq_ptmg = Constraint(model.m, rule=eq_ptmg_rule)

        # ========================================================================
        # BILATERAL TRADE - IMPORT SOURCE ALLOCATION (GAMS xweq, pmteq, pmeq)
        # ========================================================================

        import_source_share_cache: Dict[Tuple[str, str, str], float] = {}

        def _get_import_source_share(model, importer, commodity, exporter) -> float:
            key = (importer, commodity, exporter)
            cached = import_source_share_cache.get(key)
            if cached is not None:
                return cached

            share = float(self.params.shares.normalized.import_source_share.get(key, 0.0) or 0.0)

            import_source_share_cache[key] = share
            return share
        
        # Bilateral import demand (GAMS xweq)
        # xw(rp,i,r) = amw(rp,i,r)*xmt(r,i)*(pmt(r,i)/pm(rp,i,r))**sigmaw(r,i)
        def eq_xweq_rule(model, rp, i, r):
            amw = _get_import_source_share(model, r, i, rp)
            if amw <= 0.0:
                return Constraint.Skip
            esubm = self.params.elasticities.esubm.get((r, i), 5.0)
            lambdam = _lambdam_value(rp, i, r)
            return model.xw[rp, i, r] == (
                amw
                * model.xmt[r, i]
                * (model.pmt[r, i] / _m_pm(rp, i, r)) ** esubm
                * (lambdam ** (esubm - 1.0))
            )
        model.eq_xweq = Constraint(model.rp, model.i, model.r, rule=eq_xweq_rule)
        
        # Aggregate import price CES (GAMS pmteq)
        # pmt(r,i)**(1-esubm) = sum(rp, amw(rp,i,r)*pm(rp,i,r)**(1-esubm))
        def eq_pmteq_rule(model, r, i):
            esubm = self.params.elasticities.esubm.get((r, i), 5.0)
            expo = 1.0 - esubm
            if abs(expo) < 1e-8:
                return Constraint.Skip  # Cobb-Douglas handled differently
            active_shares = [
                _get_import_source_share(model, r, i, rp)
                for rp in model.rp
            ]
            if not any(share > 0.0 for share in active_shares):
                return model.pmt[r, i] == 1.0  # Default price if no imports
            terms = []
            for rp in model.rp:
                amw = _get_import_source_share(model, r, i, rp)
                if amw <= 0.0:
                    continue
                lambdam = _lambdam_value(rp, i, r)
                terms.append(amw * (_m_pm(rp, i, r) / lambdam) ** expo)
            if not terms:
                return model.pmt[r, i] == 1.0
            return model.pmt[r, i] ** expo == sum(terms)
        model.eq_pmteq = Constraint(model.r, model.i, rule=eq_pmteq_rule)
        
        # Bilateral import price tariff-inclusive (GAMS pmeq)
        # pm(rp,i,r) = (1 + imptx(rp,i,r) + mtax(r,i))*pmcif(rp,i,r)
        def eq_pmeq_rule(model, rp, i, r):
            if value(model.xw_flag[rp, i, r]) <= 0.0:
                return Constraint.Skip
            imptx = _imptx_rate_importer(r, i, rp)
            mtax = _mtax_value(r, i, rp)
            chipm = _chipm_value(rp, i, r)
            return model.pm[rp, i, r] == ((1.0 + imptx + mtax) * model.pmcif[rp, i, r]) / chipm
        model.eq_pmeq = Constraint(model.rp, model.i, model.r, rule=eq_pmeq_rule)
        
        # CIF import price (GAMS pmcifeq)
        # pmcif(rp,i,r) = pefob(rp,i,r) + pwmg(rp,i,r)*tmarg(rp,i,r)
        def eq_pmcifeq_rule(model, rp, i, r):
            if value(model.xw_flag[rp, i, r]) <= 0.0:
                return Constraint.Skip
            tmarg = value(model.tmarg[rp, i, r])
            return model.pmcif[rp, i, r] == model.pefob[rp, i, r] + model.pwmg[rp, i, r] * tmarg
        model.eq_pmcifeq = Constraint(model.rp, model.i, model.r, rule=eq_pmcifeq_rule)
        
        # FOB export price (GAMS pefobeq)
        # pefob(r,i,rp) = (1 + exptx(r,i,rp) + etax(r,i))*pe(r,i,rp)
        def eq_pefobeq_rule(model, r, i, rp):
            if value(model.xw_flag[r, i, rp]) <= 0.0:
                return Constraint.Skip
            export_tax = float(self.params.taxes.rtxs.get((r, i, rp), 0.0))
            etax = _etax_value(r, i, rp)
            return model.pefob[r, i, rp] == (1.0 + export_tax + etax) * model.pe[r, i, rp]
        model.eq_pefobeq = Constraint(model.r, model.i, model.rp, rule=eq_pefobeq_rule)
        
        # Bilateral export supply CET (GAMS peeq)
        # xw(r,i,rp) = gw(r,i,rp)*xet(r,i)*(pe(r,i,rp)/pet(r,i))**omegaw(r,i)
        def eq_peeq_rule(model, r, i, rp):
            if value(model.xw_flag[r, i, rp]) <= 0.0:
                return Constraint.Skip
            omegaw = self.params.elasticities.omegaw.get((r, i), float("inf"))
            if omegaw == float("inf"):
                return model.pe[r, i, rp] == model.pet[r, i]
            return (
                model.xw[r, i, rp]
                == model.gw_share[r, i, rp] * model.xet[r, i] * (model.pe[r, i, rp] / model.pet[r, i]) ** omegaw
            )
        model.eq_peeq = Constraint(model.r, model.i, model.rp, rule=eq_peeq_rule)
        
        # Aggregate export price CET (GAMS peteq)
        # pet(r,i)**(1+omegaw) = sum(rp, gw(r,i,rp)*pe(r,i,rp)**(1+omegaw))
        def eq_peteq_rule(model, r, i):
            if value(model.xet_flag[r, i]) <= 0.0:
                return Constraint.Skip
            active_routes: list[str] = []
            for rp in model.rp:
                if value(model.xw_flag[r, i, rp]) > 0.0:
                    active_routes.append(rp)
            if not active_routes:
                return Constraint.Skip
            omegaw = self.params.elasticities.omegaw.get((r, i), float("inf"))
            if omegaw == float("inf"):
                return model.xet[r, i] == sum(
                    model.xw[r, i, rp] 
                    for rp in active_routes
                )
            exponent = 1.0 + omegaw
            terms = []
            for rp in active_routes:
                gw = (
                    model.gw_share[r, i, rp]
                    if hasattr(model, "gw_share")
                    else float(self.params.shares.p_gw.get((r, i, rp), 0.0))
                )
                terms.append(gw * model.pe[r, i, rp] ** exponent)
            if not terms:
                return Constraint.Skip
            return model.pet[r, i] ** exponent == sum(terms)
        model.eq_peteq = Constraint(model.r, model.i, rule=eq_peteq_rule)
        
        # ========================================================================
        # DOMESTIC MARKET EQUILIBRIUM (GAMS pdeq)
        # ========================================================================
        
        # Domestic goods market equilibrium
        # xds(r,i) = sum(aa, xd(r,i,aa)/xScale(r,aa))
        def eq_pdeq_rule(model, r, i):
            return model.xds[r, i] == sum(
                model.xda[r, i, aa] / model.xscale[r, aa] 
                for aa in model.aa 
                if get_benchmark_agent_armington_shares(r, i, aa)[0] > 0.0
            )
        model.eq_pdeq = Constraint(model.r, model.i, rule=eq_pdeq_rule)

        # ========================================================================
        # FACTOR BLOCK
        # ========================================================================

        # Factor market clearing (distribute xft using gf share)
        def eq_xft_rule(model, r, f):
            if f not in self.sets.mf:
                return Constraint.Skip
            if value(model.xftflag[r, f]) <= 0.0:
                return Constraint.Skip
            return model.xft[r, f] == sum(model.xf[r, f, a] / model.xscale[r, a] for a in model.a)
        model.eq_xft = Constraint(model.r, model.f, rule=eq_xft_rule)

        # Factor demand (GAMS exact formulation)
        # GAMS: xf(r,f,a,t) =e= af(r,f,a,t)*va(r,a,t)*(pva(r,a,t)/pfa(r,f,a,t))**sigmav(r,a)
        #                       * (lambdaf(r,f,a,t))**(sigmav(r,a)-1)
        def eq_xfeq_rule(model, r, f, a):
            af_val = (
                value(model.af_param[r, f, a])
                if hasattr(model, "af_param")
                else value(model.af_share[r, f, a])
            )
            if af_val <= 0.0:
                return Constraint.Skip
            if value(model.xfflag[r, f, a]) <= 0.0:
                return Constraint.Skip
            ratio = model.pva[r, a] / _m_pfa(r, f, a)
            sigmav = self._get_sigmav(r, a)
            lambdaf = self._lambdaf(r, f, a)
            return model.xf[r, f, a] == af_val * model.va[r, a] * ratio**sigmav * lambdaf ** (sigmav - 1)
        model.eq_xfeq = Constraint(model.r, model.f, model.a, rule=eq_xfeq_rule)

        # Aggregate supply of factors from production shares
        def eq_xfteq_rule(model, r, f):
            if f not in self.sets.mf:
                return Constraint.Skip
            if value(model.xftflag[r, f]) <= 0.0:
                return Constraint.Skip
            benchmark_supply = float(value(model.aft[r, f]))
            if benchmark_supply <= 0:
                return Constraint.Skip
            elasticity = float(value(model.etaf[r, f]))
            return model.xft[r, f] == model.aft[r, f] * (model.pft[r, f] / (model.pabs[r] + 1e-12)) ** elasticity
        model.eq_xfteq = Constraint(model.r, model.f, rule=eq_xfteq_rule)

        # Factor supply / law-of-one-price (GAMS pfeq(r,fp,a)).
        #
        # Three branches matching model.gms:1079 exactly:
        #   (1) fm + omegaf = inf  (perfect mobility): M_PFY(r,f,a) = pft(r,f)
        #   (2) fm + omegaf finite (partial CET):       xf = xscale*gf*xft*(M_PFY/pft)^omegaf
        #   (3) fnm (sector-specific supply curve):     xf = xscale*gf*(M_PFY/pabs)^etaff
        # One equation per active (r,f,a) so that pf(r,f,a) is pinned 1:1
        # against its MCP pair, without resorting to aggressive fixing.
        def _omegaf(region, factor):
            omega = self.params.elasticities.omegaf.get((region, factor))
            if omega is not None:
                return float(omega)
            # GAMS: mobile factors default to inf; sluggish come from -etrae.
            if factor in self.sets.mf:
                return float("inf")
            etrae = self.params.elasticities.etrae.get(factor, float("inf"))
            if etrae == float("inf"):
                return float("inf")
            return -float(etrae)

        def _etaff(region, factor, activity):
            return float(
                self.params.elasticities.etaff.get((region, factor, activity), 0.0)
            )

        def eq_pfeq_rule(model, r, f, a):
            if value(model.xfflag[r, f, a]) <= 0.0:
                return Constraint.Skip
            gf = float(value(model.gf_share[r, f, a]))
            if gf <= 0.0:
                return Constraint.Skip
            pfy = _m_pfy(r, f, a)
            if f in self.sets.mf:
                omegaf = _omegaf(r, f)
                if omegaf == float("inf"):
                    # Perfect mobility: law of one price.
                    return pfy == model.pft[r, f]
                # Partial CET across activities.
                return model.xf[r, f, a] == (
                    model.xscale[r, a] * model.gf_share[r, f, a] * model.xft[r, f]
                    * (pfy / model.pft[r, f]) ** omegaf
                )
            # Sector-specific factor supply curve (fnm).
            etaff = _etaff(r, f, a)
            return model.xf[r, f, a] == (
                model.xscale[r, a] * model.gf_share[r, f, a]
                * (pfy / model.pabs[r]) ** etaff
            )
        model.eq_pfeq = Constraint(model.r, model.f, model.a, rule=eq_pfeq_rule)

        # Factor prices tax inclusive (GAMS pfaeq)
        def eq_pfaeq_rule(model, r, f, a):
            if value(model.xfflag[r, f, a]) <= 0.0:
                return Constraint.Skip
            factor_tax = float(self.params.taxes.rtf.get((r, f, a), 0.0))
            return model.pfa[r, f, a] == model.pf[r, f, a] * (1.0 + factor_tax)
        model.eq_pfaeq = Constraint(model.r, model.f, model.a, rule=eq_pfaeq_rule)

        # Factor prices post-tax/subsidy (GAMS pfyeq)
        def eq_pfyeq_rule(model, r, f, a):
            if value(model.xfflag[r, f, a]) <= 0.0:
                return Constraint.Skip
            kappa = float(self.params.taxes.kappaf_activity.get((r, f, a), 0.0))
            if kappa == 0.0:
                kappa = float(self.params.taxes.kappaf.get((r, f), 0.0))
            return model.pfy[r, f, a] == model.pf[r, f, a] * (1.0 - kappa)
        model.eq_pfyeq = Constraint(model.r, model.f, model.a, rule=eq_pfyeq_rule)

        # Note: eq_pvaeq already defined above with GAMS formulation

        # Regional factor price index aggregates all factor prices by supply share.
        def eq_pfact_rule(model, r):
            total_share = 0.0
            weighted = 0.0
            for f in model.f:
                if str(f) == "NatRes":
                    continue
                if value(model.xftflag[r, f]) <= 0.0:
                    continue
                factor_share = sum(value(model.gf_share[r, f, a]) for a in model.a)
                if factor_share <= 0:
                    continue
                total_share += factor_share
                weighted += factor_share * model.pft[r, f]
            if total_share <= 0:
                return model.pfact[r] == model.pnum
            return model.pfact[r] * total_share == weighted
        model.eq_pfact = Constraint(model.r, rule=eq_pfact_rule)

        # Capital stock equals total capital demand across activities.
        def eq_kstock_rule(model, r):
            # GAMS kstockeq: krat(r)*kstock(r) = sum(cap, xft(r,cap))
            # krat = xft.l / kstock.l at benchmark (cal.gms:413)
            capital = "Capital"
            if capital not in model.f:
                return Constraint.Skip
            if value(model.xftflag[r, capital]) <= 0.0:
                return Constraint.Skip
            xft_bench = float(value(model.aft[r, capital]))
            vkb_val = self.params.benchmark.vkb.get(r)
            if vkb_val is None:
                vkb_val = self.params.benchmark.vkb.get((r,), 0.0)
            kstock_bench = float(vkb_val or 0.0)
            if kstock_bench <= 0.0 or xft_bench <= 0.0:
                return Constraint.Skip
            krat = xft_bench / kstock_bench
            return krat * model.kstock[r] == model.xft[r, capital]
        model.eq_kstock = Constraint(model.r, rule=eq_kstock_rule)
        
        # ========================================================================
        # DEMAND BLOCK
        # ========================================================================
        
        # Private consumption (fixed shares for simplicity)
        # GAMS uses pa(r,i,h,t) where h = household agent 'hhd'
        def eq_xc_rule(model, r, i):
            share = value(model.c_share[r, i])
            if share <= 0.0:
                return model.xc[r, i] == 0.0
            return model.xc[r, i] == share * model.yc[r] / (model.pa[r, i, "hhd"] + 1e-12)
        model.eq_xc = Constraint(model.r, model.i, rule=eq_xc_rule)
        
        # Government consumption
        # GAMS uses pa(r,i,gov,t) where gov = government agent
        def eq_xg_rule(model, r, i):
            share = value(model.g_share[r, i])
            if share <= 0.0:
                return model.xg[r, i] == 0.0
            return model.xg[r, i] == share * model.yg[r] / (model.pa[r, i, "gov"] + 1e-12)
        model.eq_xg = Constraint(model.r, model.i, rule=eq_xg_rule)
        
        # Investment demand
        # GAMS uses pa(r,i,inv,t) where inv = investment agent
        # GAMS cal.gms line 795: sigmai(r)$(sigmai(r) eq 1) = 1.01  [exact CD form avoided]
        def eq_xi_rule(model, r, i):
            alphaa = value(model.i_share[r, i])
            if alphaa <= 0.0:
                return model.xi[r, i] == 0.0
            sigmai_raw = float(self.params.elasticities.esubi.get(r, 1.0))
            if abs(sigmai_raw - 1.0) < 1e-8:
                sigmai_raw = 1.01  # match GAMS cal.gms: sigmai=1 → 1.01
            return model.xi[r, i] == alphaa * model.xiagg[r] * (model.pi[r] / (model.pa[r, i, "inv"] + 1e-12)) ** sigmai_raw
        model.eq_xi = Constraint(model.r, model.i, rule=eq_xi_rule)

        def eq_xiagg_rule(model, r):
            return model.pi[r] * model.xiagg[r] == model.yi[r]
        model.eq_xiagg = Constraint(model.r, rule=eq_xiagg_rule)

        # ========================================================================
        # UTILITY AND SAVINGS BLOCK (GAMS phiPeq/uh/ug/us/ueq/psave)
        # ========================================================================

        # Household budget shares (xcshr) from benchmark shares
        def eq_xcshr_rule(model, r, i):
            return model.xcshr[r, i] == model.c_share[r, i]
        model.eq_xcshr = Constraint(model.r, model.i, rule=eq_xcshr_rule)

        # Consumer expenditure deflator (pcons) using shares
        def eq_pcons_rule(model, r):
            return model.pcons[r] == sum(model.xcshr[r, i] * model.pa[r, i, "hhd"] for i in model.i)
        model.eq_pcons = Constraint(model.r, rule=eq_pcons_rule)

        # Investment expenditure deflator (GAMS pieq, CES form)
        # GAMS cal.gms line 795: sigmai(r)$(sigmai(r) eq 1) = 1.01  [exact CD avoided]
        # CES: (axi*pi)^(1-sigmai) = sum_i[alphaa_i*(pa_i)^(1-sigmai)]
        # With sigmai=1.01, expo=-0.01: well-initialized from benchmark (axi=1, pa=1 → pi=1)
        def eq_pi_rule(model, r):
            sigmai_raw = float(self.params.elasticities.esubi.get(r, 1.0))
            if abs(sigmai_raw - 1.0) < 1e-8:
                sigmai_raw = 1.01  # match GAMS cal.gms: sigmai=1 → 1.01
            expo = 1.0 - sigmai_raw
            terms = [
                value(model.i_share[r, i]) * model.pa[r, i, "inv"] ** expo
                for i in model.i
                if value(model.i_share[r, i]) > 0.0
            ]
            if not terms:
                return model.pi[r] == 1.0
            return (model.axi[r] * model.pi[r]) ** expo == sum(terms)
        model.eq_pi = Constraint(model.r, rule=eq_pi_rule)

        # Private utility per capita (GAMS uheq, CD form)
        def eq_uh_rule(model, r):
            terms = [
                model.xaa[r, i, GTAP_HOUSEHOLD_AGENT] ** value(model.c_share[r, i])
                for i in model.i
                if value(model.c_share[r, i]) > 0.0
            ]
            if not terms:
                return model.uh[r] == 1.0
            prod_expr = 1.0
            for term in terms:
                prod_expr *= term
            return model.uh[r] == model.auh[r] * prod_expr
        model.eq_uh = Constraint(model.r, rule=eq_uh_rule)

        # Government utility per capita (GAMS ugeq with calibrated shifter)
        def eq_ug_rule(model, r):
            return model.ug[r] == model.aug[r] * model.yg[r] / (model.pcons[r] * model.pop[r] + 1e-12)
        model.eq_ug = Constraint(model.r, rule=eq_ug_rule)

        # Savings price (GAMS psaveeq, compStat-style static form)
        def eq_psave_rule(model, r):
            return model.psave[r] == model.chiSave * model.pi[r]
        model.eq_psave = Constraint(model.r, rule=eq_psave_rule)

        def eq_us_rule(model, r):
            return model.us[r] == model.aus[r] * model.rsav[r] / (model.psave[r] * model.pop[r] + 1e-12)
        model.eq_us = Constraint(model.r, rule=eq_us_rule)

        # Total utility (GAMS ueq, static Cobb-Douglas form)
        def eq_u_rule(model, r):
            return model.u[r] == (
                model.au[r]
                * (model.uh[r] ** model.betap[r])
                * (model.ug[r] ** model.betag[r])
                * (model.us[r] ** model.betas[r])
            )
        model.eq_u = Constraint(model.r, rule=eq_u_rule)

        # Savings price adjustment (GAMS chiSaveeq, compStat-style static form)
        def eq_chisave_rule(model):
            numer = sum(model.invwgt[r] * model.pi[r] for r in model.r)
            # In static base-year form with psave0 = pi0 = 1 and
            # psave(r) = chiSave * pi(r), GAMS chiSaveeq reduces to:
            # chiSave^2 * sum_r savwgt(r) * pi(r) = sum_r invwgt(r) * pi(r)
            denom = sum(model.savwgt[r] * model.pi[r] for r in model.r)
            return (model.chiSave ** 2) * denom == numer
        model.eq_chisave = Constraint(rule=eq_chisave_rule)

        def eq_pigbl_rule(model):
            net_inv_value = sum(
                model.pi[r] * (model.xiagg[r] - model.depr[r] * model.kstock[r])
                for r in model.r
            )
            return model.pigbl * model.xigbl == net_inv_value
        model.eq_pigbl = Constraint(rule=eq_pigbl_rule)

        # Global net investment (GAMS xigbleq)
        def eq_xigbl_rule(model):
            return model.xigbl == sum(
                model.xiagg[r] - model.depr[r] * model.kstock[r]
                for r in model.r
            )
        model.eq_xigbl = Constraint(rule=eq_xigbl_rule)

        # Foreign savings (GAMS savfeq)
        def eq_savf_rule(model, r):
            savf_flag = getattr(self.closure, "savf_flag", "capFix") if self.closure else "capFix"
            is_residual = str(r) == "RestofWorld"
            if savf_flag == "capFix":
                if is_residual:
                    return Constraint.Skip
                return model.savf[r] == model.pigbl * model.savf_bar[r]
            if savf_flag == "capSFix":
                if is_residual:
                    return Constraint.Skip
                return model.savf[r] == model.chif[r] * model.regy[r]
            return model.savf[r] == model.chif[r] * model.regy[r]
        model.eq_savf = Constraint(model.r, rule=eq_savf_rule)

        # Share of nominal foreign savings in regional income (GAMS chifeq)
        def eq_chif_rule(model, r):
            if self.closure and getattr(self.closure, "savf_flag", "capFix") == "capSFix":
                return model.chif[r] == model.chif0[r]
            return model.savf[r] == model.chif[r] * model.regy[r]
        model.eq_chif = Constraint(model.r, rule=eq_chif_rule)

        # Capital account balance (GAMS capAccteq)
        def eq_capacct_rule(model):
            return sum(model.savf[r] for r in model.r) == 0.0
        model.eq_capAcct = Constraint(rule=eq_capacct_rule)

        # Rate of return block (GAMS arent/rorc/rore/kapEnd)
        def eq_kapend_rule(model, r):
            return model.kapEnd[r] == (1.0 - model.depr[r]) * model.kstock[r] + model.xiagg[r]
        model.eq_kapEnd = Constraint(model.r, rule=eq_kapend_rule)

        def eq_arent_rule(model, r):
            capital_factors = [f for f in model.f if str(f).lower() in ("capital", "cap", "k", "kap")]
            if not capital_factors:
                return model.arent[r] == 0.0
            cap_return = 0.0
            for f in capital_factors:
                for a in model.a:
                    kappa = float(self.params.taxes.kappaf_activity.get((r, f, a), 0.0))
                    cap_return += (1.0 - kappa) * model.pf[r, f, a] * model.xf[r, f, a] / model.xscale[r, a]
            return model.arent[r] == cap_return / (model.kstock[r] + 1e-12)
        model.eq_arent = Constraint(model.r, rule=eq_arent_rule)

        def eq_rorc_rule(model, r):
            return model.rorc[r] == model.arent[r] / (model.pi[r] + 1e-12) - model.fdepr[r]
        model.eq_rorc = Constraint(model.r, rule=eq_rorc_rule)

        def eq_rore_rule(model, r):
            return model.rore[r] == model.rorc[r] * (model.kstock[r] / (model.kapEnd[r] + 1e-12)) ** model.rorflex[r]
        model.eq_rore = Constraint(model.r, rule=eq_rore_rule)

        # Global rate of return (GAMS rorgeq, weighted by net investment)
        def eq_rorg_rule(model):
            numer = 0.0
            denom = 0.0
            for r in model.r:
                net_invest = model.pi[r] * (model.xiagg[r] - model.depr[r] * model.kstock[r])
                numer += model.rore[r] * net_invest
                denom += net_invest
            return model.rorg == numer / (denom + 1e-12)
        model.eq_rorg = Constraint(rule=eq_rorg_rule)
        
        # ========================================================================
        # INCOME BLOCK
        # ========================================================================

        residual_regions = tuple(r for r in model.r if str(r) == "RestofWorld")
        
        # Factor income net of depreciation (GAMS factYeq)
        def eq_facty_rule(model, r):
            return model.facty[r] == (
                sum(model.pf[r, f, a] * model.xf[r, f, a] / model.xscale[r, a] for f in model.f for a in model.a)
                - model.fdepr[r] * model.pi[r] * model.kstock[r]
            )
        model.eq_facty = Constraint(model.r, rule=eq_facty_rule)

        # Tax revenues by stream (GAMS ytaxeq)
        def eq_ytax_rule(model, r, gy):
            if gy == "pt":
                total = 0.0
                for a in model.a:
                    for i in self.sets.activity_commodities.get(a, list(model.i)):
                        if value(model.xflag[r, a, i]) <= 0.0:
                            continue
                        total += model.prdtx_rai[r, a, i] * model.p_rai[r, a, i] * model.x[r, a, i]
                return model.ytax[r, gy] == total

            if gy == "ft":
                total = 0.0
                for (rr, f, a), rtf in self.params.taxes.rtf.items():
                    if rr != r:
                        continue
                    total += float(rtf) * model.pf[r, f, a] * model.xf[r, f, a] / model.xscale[r, a]
                return model.ytax[r, gy] == total

            if gy == "fs":
                return model.ytax[r, gy] == 0.0

            if gy in ("fc", "pc", "gc", "ic"):
                if gy == "fc":
                    agents = list(model.a)
                elif gy == "pc":
                    agents = [GTAP_HOUSEHOLD_AGENT]
                elif gy == "gc":
                    agents = [GTAP_GOVERNMENT_AGENT]
                else:
                    agents = [GTAP_INVESTMENT_AGENT]

                total = 0.0
                for aa in agents:
                    for i in model.i:
                        dintx = float(self.params.taxes.dintx0.get((r, i, aa), 0.0))
                        mintx = float(self.params.taxes.mintx0.get((r, i, aa), 0.0))
                        scale = model.xscale[r, aa] if aa in model.a else 1.0
                        total += dintx * model.pd[r, i] * model.xda[r, i, aa] / scale
                        total += mintx * model.pmt[r, i] * model.xma[r, i, aa] / scale
                return model.ytax[r, gy] == total

            if gy == "et":
                total = 0.0
                for (rr, i, rp), rtxs in self.params.taxes.rtxs.items():
                    if rr != r:
                        continue
                    etax = _etax_value(r, i, rp)
                    total += (float(rtxs) + etax) * model.pe[r, i, rp] * model.xw[r, i, rp]
                return model.ytax[r, gy] == total

            if gy == "mt":
                total = 0.0
                for (exporter, i, importer), imptx in self.params.taxes.imptx.items():
                    if importer != r:
                        continue
                    mtax = _mtax_value(r, i, exporter)
                    total += (float(imptx) + mtax) * _m_pmcif(exporter, i, r) * model.xw[exporter, i, r]
                return model.ytax[r, gy] == total

            if gy == "dt":
                total = 0.0
                for f in model.f:
                    for a in model.a:
                        kappa = float(self.params.taxes.kappaf_activity.get((r, f, a), 0.0))
                        if kappa == 0.0:
                            kappa = float(self.params.taxes.kappaf.get((r, f), 0.0))
                        if kappa == 0.0:
                            continue
                        total += kappa * model.pf[r, f, a] * model.xf[r, f, a] / model.xscale[r, a]
                return model.ytax[r, gy] == total

            return model.ytax[r, gy] == 0.0

        model.eq_ytax = Constraint(model.r, model.gy, rule=eq_ytax_rule)

        # Total tax revenue (GAMS ytaxToteq)
        def eq_ytax_tot_rule(model, r):
            return model.ytaxTot[r] == sum(model.ytax[r, gy] for gy in model.gy)
        model.eq_ytax_tot = Constraint(model.r, rule=eq_ytax_tot_rule)

        # Indirect tax revenues (GAMS ytaxIndeq)
        def eq_ytax_ind_rule(model, r):
            return model.ytax_ind[r] == model.ytaxTot[r] - model.ytax[r, "dt"]
        model.eq_ytax_ind = Constraint(model.r, rule=eq_ytax_ind_rule)

        # Regional income (GAMS regYeq)
        def eq_regy_rule(model, r):
            return model.regy[r] == model.facty[r] + model.ytax_ind[r]
        model.eq_regy = Constraint(model.r, rule=eq_regy_rule)

        def eq_ytaxshreq_rule(model, r, gy):
            return model.ytaxshr[r, gy] == model.ytax[r, gy] / (model.regy[r] + 1e-12)
        model.eq_ytaxshreq = Constraint(model.r, model.gy, rule=eq_ytaxshreq_rule)
        
        # Private consumption expenditure (GAMS yceq)
        def eq_yc_rule(model, r):
            return model.yc[r] == model.betap[r] * (model.phi[r] / model.phip[r]) * model.regy[r]
        model.eq_yc = Constraint(model.r, rule=eq_yc_rule)
        
        # Government expenditure (GAMS ygeq)
        def eq_yg_rule(model, r):
            return model.yg[r] == model.betag[r] * model.phi[r] * model.regy[r]
        model.eq_yg = Constraint(model.r, rule=eq_yg_rule)

        # Investment income share (region-specific benchmark calibration)
        def eq_yi_rule(model, r):
            if r in residual_regions:
                return Constraint.Skip
            # GAMS yieq: yi = pi*depr*kstock + rsav + savf
            return model.yi[r] == model.pi[r] * model.depr[r] * model.kstock[r] + model.rsav[r] + model.savf[r]
        model.eq_yi = Constraint(model.r, rule=eq_yi_rule)

        # Regional savings (GAMS rsaveq)
        def eq_rsav_rule(model, r):
            return model.rsav[r] == model.betas[r] * model.phi[r] * model.regy[r]
        model.eq_rsav = Constraint(model.r, rule=eq_rsav_rule)

        final_demand_agents = (
            GTAP_HOUSEHOLD_AGENT,
            GTAP_GOVERNMENT_AGENT,
            GTAP_INVESTMENT_AGENT,
        )

        # Base-year levels for compStat Fisher indices.
        base_pa = {
            (r, i, agent): float(value(model.pa[r, i, agent]))
            for r in model.r
            for i in model.i
            for agent in final_demand_agents
        }
        base_xaa = {
            (r, i, agent): float(value(model.xaa[r, i, agent]))
            for r in model.r
            for i in model.i
            for agent in final_demand_agents
        }
        base_pefob = {
            (r, i, rp): float(value(model.pefob[r, i, rp]))
            for r in model.r
            for i in model.i
            for rp in model.rp
        }
        base_pmcif = {
            (rp, i, r): float(value(model.pmcif[rp, i, r]))
            for rp in model.rp
            for i in model.i
            for r in model.r
        }
        base_xw = {
            (r, i, rp): float(value(model.xw[r, i, rp]))
            for r in model.r
            for i in model.i
            for rp in model.rp
        }
        base_pabs = {r: max(float(value(model.pabs[r])), 1e-8) for r in model.r}
        base_rgdpmp = {r: max(float(value(model.rgdpmp[r])), 1e-8) for r in model.r}

        def _mqabs(model, region, *, price_base: bool, quantity_base: bool):
            total = 0.0
            for i in model.i:
                for agent in final_demand_agents:
                    pa = base_pa[(region, i, agent)] if price_base else model.pa[region, i, agent]
                    xa = base_xaa[(region, i, agent)] if quantity_base else model.xaa[region, i, agent]
                    total += pa * xa
            return total

        def _mqtrade(model, region, *, price_base: bool, quantity_base: bool):
            total = 0.0
            for i in model.i:
                for rp in model.rp:
                    pexp = base_pefob[(region, i, rp)] if price_base else _m_pefob(region, i, rp)
                    pimp = base_pmcif[(rp, i, region)] if price_base else _m_pmcif(rp, i, region)
                    xexp = base_xw[(region, i, rp)] if quantity_base else model.xw[region, i, rp]
                    ximp = base_xw[(rp, i, region)] if quantity_base else model.xw[rp, i, region]
                    total += pexp * xexp - pimp * ximp
            return total

        def _mqgdp(model, region, *, price_base: bool, quantity_base: bool):
            return _mqabs(model, region, price_base=price_base, quantity_base=quantity_base) + _mqtrade(
                model,
                region,
                price_base=price_base,
                quantity_base=quantity_base,
            )

        base_mqabs_00 = {
            r: float(_mqabs(model, r, price_base=True, quantity_base=True))
            for r in model.r
        }
        base_mqgdp_00 = {
            r: float(_mqgdp(model, r, price_base=True, quantity_base=True))
            for r in model.r
        }

        def eq_pabs_rule(model, r):
            mqabs_00 = base_mqabs_00[r]
            if mqabs_00 <= 1e-12:
                return Constraint.Skip
            mqabs_t0 = _mqabs(model, r, price_base=False, quantity_base=True)
            mqabs_tt = _mqabs(model, r, price_base=False, quantity_base=False)
            mqabs_0t = _mqabs(model, r, price_base=True, quantity_base=False)
            scale = max((base_pabs[r] ** 2) * (mqabs_00 ** 2), 1e-12)
            return (model.pabs[r] ** 2) * mqabs_00 * mqabs_0t / scale == (base_pabs[r] ** 2) * mqabs_t0 * mqabs_tt / scale
        model.eq_pabs = Constraint(model.r, rule=eq_pabs_rule)

        # Nominal GDP at market prices (GAMS gdpmpeq)
        def eq_gdpmp_rule(model, r):
            return model.gdpmp[r] == _mqgdp(model, r, price_base=False, quantity_base=False)
        model.eq_gdpmp = Constraint(model.r, rule=eq_gdpmp_rule)

        # Real GDP at market prices.
        # Baseline (is_counterfactual=False): GAMS sets rgdpmp.l = gdpmp.l (assignment, no
        # active constraint).  We replicate this with a simple identity so the solver does
        # not see a residual from an approximate Fisher formula at benchmark.
        # Counterfactual (is_counterfactual=True): use the compStat Fisher chain-volume index
        # (GAMS rgdpmpeq, active only when ts(t) condition holds).
        if self.is_counterfactual:
            def eq_rgdpmp_rule(model, r):
                gdp_00 = base_mqgdp_00[r]
                if gdp_00 <= 1e-12:
                    return Constraint.Skip
                mqgdp_0t = _mqgdp(model, r, price_base=True, quantity_base=False)
                mqgdp_t0 = _mqgdp(model, r, price_base=False, quantity_base=True)
                scale = max((base_rgdpmp[r] ** 2) * (gdp_00 ** 2), 1e-12)
                return (model.rgdpmp[r] ** 2) * gdp_00 * mqgdp_t0 / scale == (base_rgdpmp[r] ** 2) * model.gdpmp[r] * mqgdp_0t / scale
        else:
            def eq_rgdpmp_rule(model, r):  # type: ignore[misc]
                # Baseline: rgdpmp = gdpmp (real = nominal when all prices at benchmark).
                return model.rgdpmp[r] == model.gdpmp[r]
        model.eq_rgdpmp = Constraint(model.r, rule=eq_rgdpmp_rule)

        def eq_pgdpmp_rule(model, r):
            return model.pgdpmp[r] * model.rgdpmp[r] == model.gdpmp[r]
        model.eq_pgdpmp = Constraint(model.r, rule=eq_pgdpmp_rule)

        # Welfare measures (GAMS eveq / cveq) for the representative household.
        def eq_ev_rule(model, r):
            terms = []
            for i in model.i:
                alpha = value(model.alphaa_hhd[r, i])
                bh = value(model.bh[r, i])
                eh = value(model.eh[r, i])
                share = value(model.c_share[r, i])
                if alpha <= 0.0 or share <= 0.0:
                    continue
                terms.append(
                    alpha
                    * (model.uh[r] ** (bh * eh))
                    * ((model.pop[r] / model.ev[r]) ** bh)
                )
            if not terms:
                return Constraint.Skip
            return sum(terms) == 1.0
        model.eq_ev = Constraint(model.r, rule=eq_ev_rule)
        model.eq_ev.deactivate()

        def eq_cv_rule(model, r):
            terms = []
            for i in model.i:
                alpha = value(model.alphaa_hhd[r, i])
                bh = value(model.bh[r, i])
                share = value(model.c_share[r, i])
                if alpha <= 0.0 or share <= 0.0:
                    continue
                terms.append(
                    alpha
                    * ((model.pa[r, i, GTAP_HOUSEHOLD_AGENT] * model.pop[r] / model.cv[r]) ** bh)
                )
            if not terms:
                return Constraint.Skip
            return sum(terms) == 1.0
        model.eq_cv = Constraint(model.r, rule=eq_cv_rule)
        model.eq_cv.deactivate()

        # ========================================================================
        # MARKET CLEARING
        # ========================================================================
        
        # Goods market clearing: Supply = Demand
        def mkt_goods_rule(model, r, i):
            absorption = sum(model.xaa[r, i, aa] / model.xscale[r, aa] for aa in model.aa)
            inventory = self._vst_value(str(r), str(i))
            return model.xa[r, i] == absorption + inventory
        model.mkt_goods = Constraint(model.r, model.i, rule=mkt_goods_rule)
        # Duplicate of eq_xa; keep for reporting compatibility but exclude from MCP.
        model.mkt_goods.deactivate()
        
        # ========================================================================
        # NUMERAIRE
        # ========================================================================
        
        def eq_pwfact_rule(model):
            n_regions = len(list(model.r))
            if n_regions == 0:
                return Constraint.Skip
            return model.pwfact == sum(model.pfact[r] for r in model.r) / n_regions
        model.eq_pwfact = Constraint(rule=eq_pwfact_rule)

        def eq_pnum_rule(model):
            return model.pnum == model.pwfact
        model.eq_pnum = Constraint(rule=eq_pnum_rule)

        def eq_walras_rule(model):
            target_regions = residual_regions if residual_regions else tuple(model.r)
            return model.walras == sum(
                model.yi[r] - (model.pi[r] * model.depr[r] * model.kstock[r] + model.rsav[r] + model.savf[r])
                for r in target_regions
            )
        model.eq_walras = Constraint(rule=eq_walras_rule)

        # Mirror GAMS ifSUB=1 behavior: substitute macro identities directly
        # into the active system and remove the standalone defining equations.
        if if_sub:
            for con_name in (
                "eq_pp_rai",
                "eq_xwmg",
                "eq_xmgm",
                "eq_pwmg",
                "eq_pefobeq",
                "eq_pmcifeq",
                "eq_pmeq",
                "eq_pfaeq",
                "eq_pfyeq",
            ):
                if hasattr(model, con_name):
                    getattr(model, con_name).deactivate()

            def _fix_component(vardata, raw_value: float) -> None:
                target = float(raw_value)
                lb = vardata.lb
                ub = vardata.ub
                if lb is not None and target < float(lb):
                    target = float(lb)
                if ub is not None and target > float(ub):
                    target = float(ub)
                vardata.fix(target)

            for r in model.r:
                for a in model.a:
                    for i in model.i:
                        if (r, a, i) not in model.pp_rai:
                            continue
                        _fix_component(model.pp_rai[r, a, i], value(_m_pp(r, a, i)))

            for r in model.r:
                for i in model.i:
                    for rp in model.rp:
                        if (r, i, rp) in model.xwmg:
                            _fix_component(model.xwmg[r, i, rp], value(_m_xwmg(r, i, rp)))
                        if (r, i, rp) in model.pwmg:
                            _fix_component(model.pwmg[r, i, rp], value(_m_pwmg(r, i, rp)))
                        if (r, i, rp) in model.pefob:
                            _fix_component(model.pefob[r, i, rp], value(_m_pefob(r, i, rp)))
                        if (r, i, rp) in model.pmcif:
                            _fix_component(model.pmcif[r, i, rp], value(_m_pmcif(r, i, rp)))
                        if (r, i, rp) in model.pm:
                            _fix_component(model.pm[r, i, rp], value(_m_pm(r, i, rp)))
                        for m in model.m:
                            if (m, r, i, rp) in model.xmgm:
                                _fix_component(model.xmgm[m, r, i, rp], value(_m_xmgm(m, r, i, rp)))

            for r in model.r:
                for f in model.f:
                    for a in model.a:
                        if (r, f, a) in model.pfa:
                            _fix_component(model.pfa[r, f, a], value(_m_pfa(r, f, a)))
                        if (r, f, a) in model.pfy:
                            _fix_component(model.pfy[r, f, a], value(_m_pfy(r, f, a)))
    
    def _add_objective(self, model: "ConcreteModel") -> None:
        """Add dummy objective for NLP."""
        from pyomo.environ import Objective, minimize

        def dummy_obj(model):
            return 1.0

        model.OBJ = Objective(rule=dummy_obj, sense=minimize)

    def _get_sigmap(self, r: str, a: str) -> float:
        return self.params.elasticities.sigmap.get((r, a), 1.0)

    def _get_sigmand(self, r: str, a: str) -> float:
        return self.params.elasticities.sigmand.get((r, a), 1.0)

    def _get_sigmav(self, r: str, a: str) -> float:
        return self.params.elasticities.sigmav.get((r, a), 1.0)

    def _get_omegas(self, r: str, a: str) -> float:
        return self.params.elasticities.omegas.get((r, a), 1.0)

    def _get_sigmas(self, r: str, i: str) -> float:
        return self.params.elasticities.sigmas.get((r, i), 2.0)

    def _axp_shift(self, r: str, a: str) -> float:
        return self.params.shifts.axp.get((r, a), 1.0)

    def _lambdand(self, r: str, a: str) -> float:
        return self.params.shifts.lambdand.get((r, a), 1.0)

    def _lambdava(self, r: str, a: str) -> float:
        return self.params.shifts.lambdava.get((r, a), 1.0)

    def _lambdaf(self, r: str, f: str, a: str) -> float:
        return self.params.shifts.lambdaf.get((r, f, a), 1.0)

    def _factor_price_term(self, model: "ConcreteModel", r: str, f: str, a: str):
        if hasattr(model, "pfa"):
            return model.pfa[r, f, a]
        return model.pf[r, f, a]
