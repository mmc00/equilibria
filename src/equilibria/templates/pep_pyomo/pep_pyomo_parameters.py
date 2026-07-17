"""PEP calibrated parameters for the Pyomo model.

Wraps ``PEPModelState`` blocks (production/income/trade/consumption/les_parameters/gdp)
into one flat, case-preserving lookup so the equation rules read parameters by their
GAMS-style names (e.g. ``P["io","agr"]``, ``P["beta_VA","agr"]``, ``P["KDO","cap","agr"]``).

Benchmark levels end in ``O`` (KDO, YHO, IMO, …). CES/share params keep their block
names (rho_VA, beta_KD, B_XT, sigma_M, …). Derived params that the cyipopt solver
computes in ``_extract_parameters`` (gamma_INV, gamma_GVT, sigma_*, eta, kmob) are
recomputed here from the same benchmark inputs so the Pyomo model is self-contained.
"""
from __future__ import annotations

from typing import Any


class PEPParams:
    """Flat calibrated-parameter accessor over a PEPModelState.

    ``p[name]`` → scalar or dict; ``p[name, i]`` / ``p[name, i, j]`` → indexed value.
    Missing indexed entries return 0.0 (matches the GAMS ``$``-masked default), missing
    names raise KeyError (a real typo, not a benign zero).
    """

    _BLOCKS = ("production", "income", "trade", "consumption",
               "les_parameters", "gdp", "real_variables")

    def __init__(self, state: Any):
        self._flat: dict[str, Any] = {}
        for blk in self._BLOCKS:
            d = getattr(state, blk, None)
            if isinstance(d, dict):
                for k, v in d.items():
                    self._flat.setdefault(k, v)
        self._derive()

    def _derive(self) -> None:
        """Recompute the derived params the cyipopt solver builds in _extract_parameters."""
        self._flat.setdefault("eta", 1.0)
        self._flat.setdefault("kmob", 1.0)
        self._flat.setdefault("sigma_XD", 2.0)
        # Rate params the solver reads as `state.income.get(name+"O", {})` — the stripped
        # name defaults to an empty dict (→ 0.0 per index) when the calibration omits it.
        # Mirrors _extract_parameters (pep_model_solver_ipopt.py:1010-1103).
        for stripped, benchO in (("sh0", "sh0O"), ("sh1", "sh1O"), ("tr0", "tr0O"),
                                 ("tr1", "tr1O"), ("ttdh0", "ttdh0O"), ("ttdh1", "ttdh1O"),
                                 ("ttdf0", "ttdf0O"), ("ttdf1", "ttdf1O"),
                                 ("ttiw", "ttiwO"), ("ttik", "ttikO"), ("ttip", "ttipO"),
                                 ("ttic", "tticO"), ("ttim", "ttimO"), ("ttix", "ttixO")):
            if stripped not in self._flat:
                self._flat[stripped] = dict(self._flat.get(benchO, {}))
        # gamma_INV[i] = PCO[i]*INVO[i] / Σ_i PCO*INVO ; gamma_GVT[i] = PCO[i]*CGO[i] / Σ
        pco = self._flat.get("PCO", {})
        invo = self._flat.get("INVO", {})
        cgo = self._flat.get("CGO", {})
        tot_inv = sum(pco.get(i, 0.0) * invo.get(i, 0.0) for i in invo) or 1.0
        tot_gvt = sum(pco.get(i, 0.0) * cgo.get(i, 0.0) for i in cgo) or 1.0
        self._flat.setdefault(
            "gamma_INV", {i: pco.get(i, 0.0) * invo.get(i, 0.0) / tot_inv for i in invo})
        self._flat.setdefault(
            "gamma_GVT", {i: pco.get(i, 0.0) * cgo.get(i, 0.0) / tot_gvt for i in cgo})
        # world prices: equations use PWM/PWX, calibration stores PWMO/PWXO (held fixed)
        self._flat.setdefault("PWM", dict(self._flat.get("PWMO", {})))
        self._flat.setdefault("PWX", dict(self._flat.get("PWXO", {})))
        # CES elasticities from rho (pep_model_solver_ipopt.py:1074/1078/1082):
        #   sigma_XT = 1/(rho_XT - 1) ; sigma_X = 1/(rho_X - 1) ; sigma_M = 1/(1 + rho_M)
        def _sig(rho_name, f):
            rho = self._flat.get(rho_name, {})
            return {k: f(v) for k, v in rho.items()} if isinstance(rho, dict) else {}
        self._flat.setdefault("sigma_XT", _sig("rho_XT", lambda r: 1.0 / (r - 1) if r != 1 else 0.0))
        self._flat.setdefault("sigma_X", _sig("rho_X", lambda r: 1.0 / (r - 1) if r != 1 else 0.0))
        self._flat.setdefault("sigma_M", _sig("rho_M", lambda r: 1.0 / (1 + r) if r != -1 else 0.0))
        # TIPO[j] = ttip[j]*PPO[j]*XSTO[j] (benchmark production tax; ipopt:1120-1125)
        ttip, ppo, xsto = (self._flat.get(n, {}) for n in ("ttip", "PPO", "XSTO"))
        self._flat.setdefault("TIPO", {j: ttip.get(j, 0.0) * ppo.get(j, 0.0) * xsto.get(j, 0.0)
                                       for j in xsto})

    def __contains__(self, name: str) -> bool:
        return name in self._flat

    def get(self, name: str, *idx, default: float = 0.0) -> Any:
        if name not in self._flat:
            raise KeyError(f"PEP param {name!r} not calibrated")
        v = self._flat[name]
        if not idx:
            return v
        if isinstance(v, dict):
            key = idx[0] if len(idx) == 1 else tuple(idx)
            return v.get(key, default)
        return v

    def __getitem__(self, key):
        if isinstance(key, tuple):
            return self.get(key[0], *key[1:])
        return self.get(key)

    def names(self) -> list[str]:
        return sorted(self._flat)
