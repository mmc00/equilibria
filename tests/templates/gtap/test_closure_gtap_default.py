"""``GTAPClosureConfig.gtap_default``: the RunGTAP default closure as a named starting point.

It must be the closure ``scripts/gtap/run_burfisher.py`` runs against GEMPACK (100 %
on Burfisher) and must leave the field defaults of ``GTAPClosureConfig()`` alone.
"""

from __future__ import annotations

import pydantic
import pytest

from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig

# GTAPClosureConfig().model_dump() on main (2c84f805), before MAR-249
MAIN_DEFAULTS = {
    "name": "gtap_standard",
    "numeraire": "pnum",
    "numeraire_mode": "fixed_benchmark",
    "capital_mobility": "mobile",
    "fixed": (
        "prdtx", "fctts", "fcttx", "exptx", "imptx", "dintx", "mintx", "itxshftGen",
        "kappashft", "kappaf", "kappafG", "etax", "mtax", "axp", "lambdaN",
        "lambdaio", "lambdaf", "lambdaxm", "lambdam", "lambdamg", "lambdaDN", "xft",
        "pop", "tmarg",
    ),
    "endogenous": (
        "psave", "yi", "xi", "xigbl", "pigbl", "rorg", "chiInv", "netInv",
        "gblValNetInv",
    ),
    "label": None,
    "closure_type": "CNS",
    "savf_flag": "capFix",
    "va_subsidy_basis": "gams",
    "unit_sigma": "gams",
    "if_sub": True,
    "calibration_source": "python",
    "calibration_dump": None,
    "apply_flag_fixing": True,
    "close_mcp_gap": False,
    "fix_taxes": True,
    "fix_technology": True,
    "fix_endowments": True,
    "fix_world_prices": False,
    "gams_factor_pairing": False,
    "rmuv": (),
    "imuv": (),
}  # fmt: skip


def _run_burfisher_closure() -> GTAPClosureConfig:
    """The closure of ``scripts/gtap/run_burfisher.py`` (``_solve``), copied."""
    return GTAPClosureConfig(
        name="base",
        closure_type="MCP",
        capital_mobility="sluggish",
        fix_endowments=False,
        fix_taxes=False,
        fix_technology=False,
        if_sub=False,
        savf_flag="capFlex",
        numeraire="pnum",
    )


def _dump(c: GTAPClosureConfig) -> dict:
    d = c.model_dump()
    d.pop("name")
    return d


def test_same_as_run_burfisher() -> None:
    assert _dump(GTAPClosureConfig.gtap_default()) == _dump(_run_burfisher_closure())


def test_changes_override() -> None:
    c = GTAPClosureConfig.gtap_default(savf_flag="capFix")
    assert c.savf_flag == "capFix"
    want = _dump(GTAPClosureConfig.gtap_default())
    want["savf_flag"] = "capFix"
    assert _dump(c) == want


def test_changes_are_validated() -> None:
    with pytest.raises(pydantic.ValidationError):
        GTAPClosureConfig.gtap_default(savf_flag="nope")


def test_unknown_field_raises() -> None:
    with pytest.raises(pydantic.ValidationError):
        GTAPClosureConfig.gtap_default(nonexistent=1)


def test_defaults_untouched() -> None:
    d = GTAPClosureConfig().model_dump()
    assert {
        k: tuple(v) if isinstance(v, list | tuple) else v for k, v in d.items()
    } == (MAIN_DEFAULTS)
