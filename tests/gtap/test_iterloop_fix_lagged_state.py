"""Tests for fix_lagged_state — Phase C.2 of GTAP v62 multi-period MCP refactor.

We use small Pyomo `ConcreteModel` mocks rather than building the full GTAP
model from a dataset. Rationale: `fix_lagged_state` depends only on the
Pyomo Var/Param protocol (indexed iteration + .value + .fix()), not on any
GTAP-specific structure. Mocks are faster, more deterministic, and exercise
exactly the contract the function relies on. The real-model behaviour is
covered by the downstream multi-period integration test (Phase C.3+).
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest
from pyomo.environ import ConcreteModel, Param, Reals, Set, Var

sys.path.insert(0, str(Path('src').resolve()))

from equilibria.templates.gtap.gtap_iterloop import LAGGED_VARS, fix_lagged_state


# --- Helpers -----------------------------------------------------------------


def _build_pair():
    """Build two structurally-identical small ConcreteModels containing every
    name in LAGGED_VARS plus pmuv as a Param."""
    def _one() -> ConcreteModel:
        m = ConcreteModel()
        m.r = Set(initialize=["USA", "EU"])
        m.a = Set(initialize=["agr", "ind"])
        m.i = Set(initialize=["agr", "ind"])
        m.aa = Set(initialize=["agr", "ind", "hh"])
        m.f = Set(initialize=["lab", "cap"])
        m.rp = Set(initialize=["USA", "EU"])
        m.m_set = Set(initialize=["trp"])
        # Indexed Vars
        m.axp = Var(m.r, m.a, within=Reals, initialize=0.0)
        m.lambdand = Var(m.r, m.a, within=Reals, initialize=0.0)
        m.lambdava = Var(m.r, m.a, within=Reals, initialize=0.0)
        m.aioall = Var(m.r, m.i, m.a, within=Reals, initialize=0.0)
        m.lambdaf = Var(m.r, m.f, m.a, within=Reals, initialize=0.0)
        m.pf = Var(m.r, m.f, m.a, within=Reals, initialize=0.0)
        m.xf = Var(m.r, m.f, m.a, within=Reals, initialize=0.0)
        m.pa = Var(m.r, m.i, m.aa, within=Reals, initialize=0.0)
        m.xaa = Var(m.r, m.i, m.aa, within=Reals, initialize=0.0)
        m.pe = Var(m.r, m.i, m.rp, within=Reals, initialize=0.0)
        m.pefob = Var(m.r, m.i, m.rp, within=Reals, initialize=0.0)
        m.pmcif = Var(m.rp, m.i, m.r, within=Reals, initialize=0.0)
        m.pm = Var(m.rp, m.i, m.r, within=Reals, initialize=0.0)
        m.xw = Var(m.r, m.i, m.rp, within=Reals, initialize=0.0)
        m.ptmg = Var(m.m_set, within=Reals, initialize=0.0)
        m.psave = Var(m.r, within=Reals, initialize=0.0)
        m.pi = Var(m.r, within=Reals, initialize=0.0)
        m.uh = Var(m.r, within=Reals, initialize=0.0)
        m.pabs = Var(m.r, within=Reals, initialize=0.0)
        m.pfact = Var(m.r, within=Reals, initialize=0.0)
        m.pwfact = Var(within=Reals, initialize=0.0)
        m.gdpmp = Var(m.r, within=Reals, initialize=0.0)
        m.rgdpmp = Var(m.r, within=Reals, initialize=0.0)
        m.pgdpmp = Var(m.r, within=Reals, initialize=0.0)
        # pmuv as Param (default closure)
        m.pmuv = Param(within=Reals, initialize=1.0, mutable=True)
        return m

    return _one(), _one()


# --- The 24-name contract ----------------------------------------------------


def test_lagged_vars_matches_map_summary():
    """LAGGED_VARS matches the documented 24-name list."""
    expected = (
        "axp", "lambdand", "lambdava", "aioall", "lambdaf",
        "pf", "xf",
        "pa", "xaa", "pe", "pefob", "pmcif", "pm", "xw", "ptmg",
        "psave", "pi",
        "uh",
        "pabs",
        "pfact", "pwfact",
        "gdpmp", "rgdpmp", "pgdpmp",
    )
    assert LAGGED_VARS == expected
    assert len(LAGGED_VARS) == 24


# --- Happy path --------------------------------------------------------------


def test_copy_and_fix_happy_path():
    prev, new = _build_pair()
    prev.axp["USA", "agr"].value = 1.5
    prev.axp["USA", "ind"].value = 2.5
    prev.axp["EU", "agr"].value = 3.5
    prev.axp["EU", "ind"].value = 4.5

    n = fix_lagged_state(new, prev, lagged_var_names=("axp",))

    assert n == 4
    assert new.axp["USA", "agr"].value == 1.5
    assert new.axp["USA", "ind"].value == 2.5
    assert new.axp["EU", "agr"].value == 3.5
    assert new.axp["EU", "ind"].value == 4.5
    for idx in new.axp:
        assert new.axp[idx].fixed is True


# --- Missing-var safety ------------------------------------------------------


def test_missing_var_does_not_raise(caplog):
    prev, new = _build_pair()
    with caplog.at_level(logging.WARNING):
        n = fix_lagged_state(new, prev, lagged_var_names=("nonexistent_var",))
    assert n == 0
    assert any("nonexistent_var" in r.message for r in caplog.records)


# --- None values are skipped -------------------------------------------------


def test_none_values_skipped():
    prev, new = _build_pair()
    for idx in prev.axp:
        prev.axp[idx].value = 7.0
    # Wipe one back to None
    prev.axp["USA", "agr"].value = None

    n = fix_lagged_state(new, prev, lagged_var_names=("axp",))

    assert n == 3  # 4 indices, 1 skipped
    # Skipped index is left at its initial value (0.0) and NOT fixed.
    assert new.axp["USA", "agr"].fixed is False
    assert new.axp["USA", "agr"].value == 0.0
    assert new.axp["USA", "ind"].fixed is True
    assert new.axp["USA", "ind"].value == 7.0


# --- Return count is correct -------------------------------------------------


def test_return_count_matches_total_fixable():
    prev, new = _build_pair()
    # Set every prev value so all indices are fixable
    for name in ("axp", "psave", "pwfact", "ptmg"):
        v = getattr(prev, name)
        if v.is_indexed():
            for idx in v:
                v[idx].value = 1.0
        else:
            v.value = 1.0

    # axp: 2*2=4, psave: 2, pwfact: 1 scalar, ptmg: 1
    expected = 4 + 2 + 1 + 1
    n = fix_lagged_state(
        new, prev,
        lagged_var_names=("axp", "psave", "pwfact", "ptmg"),
    )
    assert n == expected


# --- Param on pmuv is skipped ------------------------------------------------


def test_param_pmuv_is_skipped(caplog):
    prev, new = _build_pair()
    with caplog.at_level(logging.WARNING):
        n = fix_lagged_state(new, prev, lagged_var_names=("pmuv",))
    assert n == 0
    assert any("pmuv" in r.message for r in caplog.records)


# --- Full default LAGGED_VARS list runs end-to-end without errors -----------


def test_full_lagged_vars_runs_without_raising(caplog):
    prev, new = _build_pair()
    # Seed every fixable Var on prev side
    for name in LAGGED_VARS:
        v = getattr(prev, name, None)
        if v is None or not isinstance(v, Var):
            continue
        if v.is_indexed():
            for idx in v:
                v[idx].value = 0.5
        else:
            v.value = 0.5

    with caplog.at_level(logging.WARNING):
        n = fix_lagged_state(new, prev)  # default LAGGED_VARS

    # The count must be > 0 and pwfact (scalar) should be fixed.
    assert n > 0
    assert new.pwfact.fixed is True
