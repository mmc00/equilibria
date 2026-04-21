"""Tests for GTAP template structure handling."""

from __future__ import annotations

from pathlib import Path

import pytest

from equilibria.templates.gtap.gtap_model_equations import GTAPModelEquations
from equilibria.templates.gtap.gtap_parameters import GTAP_MARGIN_AGENT, GTAPParameters
from equilibria.templates.gtap.gtap_sets import GTAPSets


def _set_symbol(name: str, elements: list[str]) -> dict[str, object]:
    """Build a minimal decoded set symbol for mocked GDX payloads."""
    return {
        "name": name,
        "type": 0,
        "type_name": "set",
        "elements": elements,
    }


@pytest.fixture
def multi_output_gdx_payload() -> dict[str, object]:
    """Mocked standard_gtap_7-style GDX metadata."""
    return {
        "filepath": "mock.gdx",
        "symbols": [
            _set_symbol("reg", ["NAmerica", "EU_28"]),
            _set_symbol("comm", ["c_Crops", "c_MeatLstk", "c_Extraction"]),
            _set_symbol("acts", ["a_agricultur", "a_Extraction"]),
            _set_symbol("fp", ["Land", "Capital", "NatRes"]),
            _set_symbol("fm", ["Capital"]),
            _set_symbol("fnm", ["Land", "NatRes"]),
        ],
    }


@pytest.fixture
def one_to_one_gdx_payload() -> dict[str, object]:
    """Mocked non-diagonal but bijective GDX metadata."""
    return {
        "filepath": "mock.gdx",
        "symbols": [
            _set_symbol("reg", ["NAmerica"]),
            _set_symbol("comm", ["c_Food", "c_Extraction"]),
            _set_symbol("acts", ["a_AgroProd", "a_Extraction"]),
            _set_symbol("fp", ["Land", "Capital"]),
            _set_symbol("fm", ["Capital"]),
            _set_symbol("fnm", ["Land"]),
        ],
    }


def test_loads_multi_output_structure_from_make_symbol(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    multi_output_gdx_payload: dict[str, object],
) -> None:
    """GTAP sets should preserve non-diagonal multi-output structure."""
    gdx_path = tmp_path / "standard_gtap_7.gdx"
    gdx_path.write_bytes(b"GDX")

    monkeypatch.setattr(
        "equilibria.templates.gtap.gtap_sets.read_gdx",
        lambda _: multi_output_gdx_payload,
    )

    def mock_read_parameter_values(_: dict[str, object], name: str) -> dict[tuple[str, ...], float]:
        if name == "makb":
            return {
                ("c_Crops", "a_agricultur", "NAmerica"): 10.0,
                ("c_MeatLstk", "a_agricultur", "NAmerica"): 12.0,
                ("c_Extraction", "a_Extraction", "EU_28"): 15.0,
            }
        raise ValueError(name)

    monkeypatch.setattr(
        "equilibria.templates.gtap.gtap_sets.read_parameter_values",
        mock_read_parameter_values,
    )

    sets = GTAPSets()
    sets.load_from_gdx(gdx_path)

    assert sets.r == ["NAmerica", "EU_28"]
    assert sets.i == ["c_Crops", "c_MeatLstk", "c_Extraction"]
    assert sets.a == ["a_agricultur", "a_Extraction"]
    assert sets.structure == "multi_output"
    assert sets.output_pairs == [
        ("a_agricultur", "c_Crops"),
        ("a_agricultur", "c_MeatLstk"),
        ("a_Extraction", "c_Extraction"),
    ]
    assert sets.activity_commodities["a_agricultur"] == ["c_Crops", "c_MeatLstk"]
    assert sets.commodity_activities["c_Extraction"] == ["a_Extraction"]
    assert not sets.is_bijective_output_structure
    assert sets.validate() == (True, [])


def test_loads_bijective_non_diagonal_structure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    one_to_one_gdx_payload: dict[str, object],
) -> None:
    """GTAP sets should build one-to-one mappings when make pairs are bijective."""
    gdx_path = tmp_path / "gtap_one_to_one.gdx"
    gdx_path.write_bytes(b"GDX")

    monkeypatch.setattr(
        "equilibria.templates.gtap.gtap_sets.read_gdx",
        lambda _: one_to_one_gdx_payload,
    )

    def mock_read_parameter_values(_: dict[str, object], name: str) -> dict[tuple[str, ...], float]:
        if name == "makb":
            return {
                ("c_Food", "a_AgroProd", "NAmerica"): 10.0,
                ("c_Extraction", "a_Extraction", "NAmerica"): 15.0,
            }
        raise ValueError(name)

    monkeypatch.setattr(
        "equilibria.templates.gtap.gtap_sets.read_parameter_values",
        mock_read_parameter_values,
    )

    sets = GTAPSets()
    sets.load_from_gdx(gdx_path)

    assert sets.structure == "non_diagonal"
    assert sets.is_bijective_output_structure
    assert sets.a_to_i == {
        "a_AgroProd": "c_Food",
        "a_Extraction": "c_Extraction",
    }
    assert sets.i_to_a == {
        "c_Food": "a_AgroProd",
        "c_Extraction": "a_Extraction",
    }
    assert sets.validate() == (True, [])


def test_parameters_calibrate_make_shares_from_multi_output_benchmark(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    multi_output_gdx_payload: dict[str, object],
) -> None:
    """GTAP parameters should load make values and calibrate output shares."""
    gdx_path = tmp_path / "standard_gtap_7.gdx"
    gdx_path.write_bytes(b"GDX")

    def mock_read_parameter_values(_: dict[str, object], name: str) -> dict[tuple[str, ...], float]:
        if name == "makb":
            return {
                ("c_Crops", "a_agricultur", "NAmerica"): 10.0,
                ("c_MeatLstk", "a_agricultur", "NAmerica"): 12.0,
                ("c_Extraction", "a_Extraction", "EU_28"): 15.0,
            }
        if name == "vom":
            return {
                ("NAmerica", "a_agricultur"): 22.0,
                ("EU_28", "a_Extraction"): 15.0,
            }
        raise ValueError(name)

    monkeypatch.setattr(
        "equilibria.templates.gtap.gtap_sets.read_gdx",
        lambda _: multi_output_gdx_payload,
    )
    monkeypatch.setattr(
        "equilibria.templates.gtap.gtap_sets.read_parameter_values",
        mock_read_parameter_values,
    )
    monkeypatch.setattr(
        "equilibria.templates.gtap.gtap_parameters.read_gdx",
        lambda _: multi_output_gdx_payload,
    )
    monkeypatch.setattr(
        "equilibria.templates.gtap.gtap_parameters.read_parameter_values",
        mock_read_parameter_values,
    )

    params = GTAPParameters()
    params.load_from_gdx(gdx_path)

    assert params.benchmark.makb[("NAmerica", "a_agricultur", "c_Crops")] == 10.0
    assert params.benchmark.makb[("NAmerica", "a_agricultur", "c_MeatLstk")] == 12.0
    assert params.benchmark.vom_i[("NAmerica", "c_Crops")] == 10.0
    assert params.benchmark.vom_i[("NAmerica", "c_MeatLstk")] == 12.0
    assert params.shares.p_gx[("NAmerica", "a_agricultur", "c_Crops")] == pytest.approx(10.0 / 22.0)
    assert params.shares.p_gx[("NAmerica", "a_agricultur", "c_MeatLstk")] == pytest.approx(12.0 / 22.0)


def test_parameters_calibrate_demand_shares_from_benchmark() -> None:
    """Demand and income shares should come from benchmark values, not constants."""
    sets = GTAPSets(
        r=["NAmerica"],
        i=["c_Crops", "c_MeatLstk"],
        a=["a_agricultur"],
        f=["Land", "Capital"],
        mf=["Capital"],
        sf=["Land"],
    )
    params = GTAPParameters(sets=sets)
    params.benchmark.vpm[("NAmerica", "c_Crops")] = 30.0
    params.benchmark.vpm[("NAmerica", "c_MeatLstk")] = 10.0
    params.benchmark.vgm[("NAmerica", "c_Crops")] = 9.0
    params.benchmark.vgm[("NAmerica", "c_MeatLstk")] = 6.0
    params.benchmark.vim[("NAmerica", "c_Crops")] = 5.0
    params.benchmark.vim[("NAmerica", "c_MeatLstk")] = 5.0
    params.benchmark.vfm[("NAmerica", "Land", "a_agricultur")] = 40.0
    params.benchmark.vfm[("NAmerica", "Capital", "a_agricultur")] = 60.0

    params.shares.calibrate(params.benchmark, params.elasticities, sets)

    assert params.shares.p_cons[("NAmerica", "c_Crops")] == pytest.approx(0.75)
    assert params.shares.p_cons[("NAmerica", "c_MeatLstk")] == pytest.approx(0.25)
    assert params.shares.p_gov[("NAmerica", "c_Crops")] == pytest.approx(0.6)
    assert params.shares.p_gov[("NAmerica", "c_MeatLstk")] == pytest.approx(0.4)
    assert params.shares.p_inv[("NAmerica", "c_Crops")] == pytest.approx(0.5)
    assert params.shares.p_inv[("NAmerica", "c_MeatLstk")] == pytest.approx(0.5)
    assert params.shares.p_yc["NAmerica"] == pytest.approx(0.4)
    assert params.shares.p_yg["NAmerica"] == pytest.approx(0.15)
    assert params.shares.p_yi["NAmerica"] == pytest.approx(0.1)


def test_final_demand_split_reconstructs_total_when_total_is_domestic_only() -> None:
    """If vpm/vgm/vim behave like domestic demand, recover total as domestic plus imports."""
    params = GTAPParameters()

    total, domestic, imported = params.benchmark._resolve_final_demand_split(
        total_flow=139.962662,
        domestic_flow=139.962662,
        import_flow=665.824646,
    )

    assert total == pytest.approx(805.787308)
    assert domestic == pytest.approx(139.962662)
    assert imported == pytest.approx(665.824646)


def test_final_demand_split_preserves_consistent_armington_total() -> None:
    """If the reported total already matches domestic plus imports, keep it unchanged."""
    params = GTAPParameters()

    total, domestic, imported = params.benchmark._resolve_final_demand_split(
        total_flow=30.0,
        domestic_flow=24.0,
        import_flow=6.0,
    )

    assert total == pytest.approx(30.0)
    assert domestic == pytest.approx(24.0)
    assert imported == pytest.approx(6.0)


def test_parameters_calibrate_value_added_factor_shares() -> None:
    """Value-added calibration should produce factor shares within activity and VA scale."""
    sets = GTAPSets(
        r=["NAmerica"],
        i=["c_Crops"],
        a=["a_agricultur"],
        f=["Land", "Capital"],
        mf=["Capital"],
        sf=["Land"],
    )
    params = GTAPParameters(sets=sets)
    params.benchmark.vom[("NAmerica", "a_agricultur")] = 100.0
    params.benchmark.vfm[("NAmerica", "Land", "a_agricultur")] = 40.0
    params.benchmark.vfm[("NAmerica", "Capital", "a_agricultur")] = 20.0

    params.shares.calibrate(params.benchmark, params.elasticities, sets)

    assert params.shares.p_af[("NAmerica", "Land", "a_agricultur")] == pytest.approx(2.0 / 3.0)
    assert params.shares.p_af[("NAmerica", "Capital", "a_agricultur")] == pytest.approx(1.0 / 3.0)
    assert params.shares.p_va[("NAmerica", "a_agricultur")] == pytest.approx(0.6)


def test_model_equations_build_price_indices_and_numeraire() -> None:
    """Regional absorption and factor-price indices should feed the numeraire."""
    sets = GTAPSets(
        r=["NAmerica"],
        i=["c_Crops"],
        a=["a_agricultur"],
        f=["Land", "Capital"],
        mf=["Capital"],
        sf=["Land"],
        s=["NAmerica"],
        output_pairs=[("a_agricultur", "c_Crops")],
        activity_commodities={"a_agricultur": ["c_Crops"]},
        commodity_activities={"c_Crops": ["a_agricultur"]},
    )
    params = GTAPParameters(sets=sets)
    params.benchmark.vdfm[("NAmerica", "c_Crops", "a_agricultur")] = 50.0
    params.benchmark.vpm[("NAmerica", "c_Crops")] = 30.0
    params.benchmark.vgm[("NAmerica", "c_Crops")] = 15.0
    params.benchmark.vim[("NAmerica", "c_Crops")] = 5.0
    params.benchmark.vfm[("NAmerica", "Land", "a_agricultur")] = 60.0
    params.benchmark.vfm[("NAmerica", "Capital", "a_agricultur")] = 40.0
    params.benchmark.vom[("NAmerica", "a_agricultur")] = 100.0

    model = GTAPModelEquations(sets=sets, params=params).build_model()

    eq_pabs = model.eq_pabs["NAmerica"]
    eq_pfact = model.eq_pfact["NAmerica"]
    eq_pwfact = model.eq_pwfact
    eq_pnum = model.eq_pnum

    assert "pabs[NAmerica] - (" in str(eq_pabs.body)
    assert "0.5*paa[NAmerica,c_Crops,a_agricultur]" in str(eq_pabs.body)
    assert "0.3*paa[NAmerica,c_Crops,hhd]" in str(eq_pabs.body)
    assert "0.15*paa[NAmerica,c_Crops,gov]" in str(eq_pabs.body)
    assert "0.05*paa[NAmerica,c_Crops,inv]" in str(eq_pabs.body)
    assert "pfact[NAmerica] - (0.6*pft[NAmerica,Land] + 0.4*pft[NAmerica,Capital])" == str(eq_pfact.body)
    assert "pwfact - pfact[NAmerica]" == str(eq_pwfact.body)
    assert str(eq_pnum.body) == "pnum"
    assert eq_pnum.lower == 1.0
    assert eq_pnum.upper == 1.0


def test_parameters_calibrate_trade_and_armington_shares_from_benchmark() -> None:
    """Trade and Armington shares should come from benchmark flows."""
    sets = GTAPSets(
        r=["NAmerica", "EU_28"],
        i=["c_Crops"],
        a=["a_agricultur"],
        f=["Land", "Capital"],
        mf=["Capital"],
        sf=["Land"],
    )
    params = GTAPParameters(sets=sets)
    params.benchmark.vom_i[("NAmerica", "c_Crops")] = 100.0
    params.benchmark.vom_i[("EU_28", "c_Crops")] = 80.0
    params.benchmark.vxmd[("NAmerica", "c_Crops", "EU_28")] = 40.0
    params.benchmark.vxmd[("EU_28", "c_Crops", "NAmerica")] = 20.0
    params.benchmark.viws[("EU_28", "c_Crops", "NAmerica")] = 20.0
    params.benchmark.viws[("NAmerica", "c_Crops", "EU_28")] = 40.0

    params.shares.calibrate(params.benchmark, params.elasticities, sets)

    assert params.shares.p_alphad[("NAmerica", "c_Crops")] == pytest.approx(0.75)
    assert params.shares.p_alpham[("NAmerica", "c_Crops")] == pytest.approx(0.25)
    assert params.shares.p_gd[("NAmerica", "c_Crops")] == pytest.approx(0.6)
    assert params.shares.p_ge[("NAmerica", "c_Crops")] == pytest.approx(0.4)
    assert params.shares.p_alphad[("EU_28", "c_Crops")] == pytest.approx(0.6)
    assert params.shares.p_alpham[("EU_28", "c_Crops")] == pytest.approx(0.4)
    assert params.shares.p_gd[("EU_28", "c_Crops")] == pytest.approx(0.75)
    assert params.shares.p_ge[("EU_28", "c_Crops")] == pytest.approx(0.25)
    assert params.shares.p_gw[("NAmerica", "c_Crops", "EU_28")] == pytest.approx(1.0)
    assert params.shares.p_gw[("EU_28", "c_Crops", "NAmerica")] == pytest.approx(1.0)
    assert params.shares.p_amw[("NAmerica", "c_Crops", "EU_28")] == pytest.approx(1.0)
    assert params.shares.p_amw[("EU_28", "c_Crops", "NAmerica")] == pytest.approx(1.0)


def test_parameters_calibrate_agent_armington_and_io_shares() -> None:
    """Agent-level domestic/import splits and IO shares should come from benchmark data."""
    sets = GTAPSets(
        r=["NAmerica"],
        i=["c_Crops", "c_MeatLstk"],
        a=["a_agricultur"],
        f=["Land", "Capital"],
        mf=["Capital"],
        sf=["Land"],
    )
    params = GTAPParameters(sets=sets)
    params.benchmark.vdfm[("NAmerica", "c_Crops", "a_agricultur")] = 30.0
    params.benchmark.vifm[("NAmerica", "c_Crops", "a_agricultur")] = 10.0
    params.benchmark.vdfm[("NAmerica", "c_MeatLstk", "a_agricultur")] = 15.0
    params.benchmark.vifm[("NAmerica", "c_MeatLstk", "a_agricultur")] = 15.0
    params.benchmark.vdpp[("NAmerica", "c_Crops")] = 24.0
    params.benchmark.vmpp[("NAmerica", "c_Crops")] = 6.0
    params.benchmark.vdgp[("NAmerica", "c_Crops")] = 10.0
    params.benchmark.vdip[("NAmerica", "c_Crops")] = 3.0
    params.benchmark.vmip[("NAmerica", "c_Crops")] = 1.0
    params.benchmark.vpm[("NAmerica", "c_Crops")] = 30.0
    params.benchmark.vgm[("NAmerica", "c_Crops")] = 10.0
    params.benchmark.vim[("NAmerica", "c_Crops")] = 4.0

    params.shares.calibrate(params.benchmark, params.elasticities, sets)

    assert params.shares.p_io[("NAmerica", "c_Crops", "a_agricultur")] == pytest.approx(40.0 / 70.0)
    assert params.shares.p_io[("NAmerica", "c_MeatLstk", "a_agricultur")] == pytest.approx(30.0 / 70.0)
    assert params.shares.p_alphad_agent[("NAmerica", "c_Crops", "a_agricultur")] == pytest.approx(0.75)
    assert params.shares.p_alpham_agent[("NAmerica", "c_Crops", "a_agricultur")] == pytest.approx(0.25)
    assert params.shares.p_alphad_agent[("NAmerica", "c_MeatLstk", "a_agricultur")] == pytest.approx(0.5)
    assert params.shares.p_alpham_agent[("NAmerica", "c_MeatLstk", "a_agricultur")] == pytest.approx(0.5)
    assert params.shares.p_alphad_agent[("NAmerica", "c_Crops", "hhd")] == pytest.approx(0.8)
    assert params.shares.p_alpham_agent[("NAmerica", "c_Crops", "hhd")] == pytest.approx(0.2)
    assert params.shares.p_alphad_agent[("NAmerica", "c_Crops", "gov")] == pytest.approx(1.0)
    assert params.shares.p_alpham_agent[("NAmerica", "c_Crops", "gov")] == pytest.approx(0.0)
    assert params.shares.p_alphad_agent[("NAmerica", "c_Crops", "inv")] == pytest.approx(0.75)
    assert params.shares.p_alpham_agent[("NAmerica", "c_Crops", "inv")] == pytest.approx(0.25)


def test_tax_rates_derive_agent_consumption_wedges_from_benchmark() -> None:
    """Consumption tax wedges should be derivable from purchaser/basic benchmark values."""
    sets = GTAPSets(
        r=["NAmerica"],
        i=["c_Crops"],
        a=["a_agricultur"],
        f=["Land"],
        mf=[],
        sf=["Land"],
    )
    params = GTAPParameters(sets=sets)
    params.benchmark.vdfb[("NAmerica", "c_Crops", "a_agricultur")] = 100.0
    params.benchmark.vdfp[("NAmerica", "c_Crops", "a_agricultur")] = 115.0
    params.benchmark.vmfb[("NAmerica", "c_Crops", "a_agricultur")] = 80.0
    params.benchmark.vmfp[("NAmerica", "c_Crops", "a_agricultur")] = 100.0
    params.benchmark.vdpb[("NAmerica", "c_Crops")] = 50.0
    params.benchmark.vdpp[("NAmerica", "c_Crops")] = 55.0
    params.benchmark.vmpb[("NAmerica", "c_Crops")] = 20.0
    params.benchmark.vmpp[("NAmerica", "c_Crops")] = 25.0
    params.benchmark.vdgb[("NAmerica", "c_Crops")] = 10.0
    params.benchmark.vdgp[("NAmerica", "c_Crops")] = 12.0
    params.benchmark.vmgb[("NAmerica", "c_Crops")] = 8.0
    params.benchmark.vmgp[("NAmerica", "c_Crops")] = 10.0
    params.benchmark.vdib[("NAmerica", "c_Crops")] = 30.0
    params.benchmark.vdip[("NAmerica", "c_Crops")] = 33.0
    params.benchmark.vmib[("NAmerica", "c_Crops")] = 12.0
    params.benchmark.vmip[("NAmerica", "c_Crops")] = 15.0

    params.taxes.derive_agent_consumption_taxes(params.benchmark, sets)

    assert params.taxes.dintx_agent[("NAmerica", "c_Crops", "a_agricultur")] == pytest.approx(0.15)
    assert params.taxes.mintx_agent[("NAmerica", "c_Crops", "a_agricultur")] == pytest.approx(0.25)
    assert params.taxes.dintx_agent[("NAmerica", "c_Crops", "hhd")] == pytest.approx(0.10)
    assert params.taxes.mintx_agent[("NAmerica", "c_Crops", "hhd")] == pytest.approx(0.25)
    assert params.taxes.dintx_agent[("NAmerica", "c_Crops", "gov")] == pytest.approx(0.20)
    assert params.taxes.mintx_agent[("NAmerica", "c_Crops", "gov")] == pytest.approx(0.25)
    assert params.taxes.dintx_agent[("NAmerica", "c_Crops", "inv")] == pytest.approx(0.10)
    assert params.taxes.mintx_agent[("NAmerica", "c_Crops", "inv")] == pytest.approx(0.25)


def test_tax_rates_derive_trade_route_wedges_from_benchmark() -> None:
    """Trade-route wedges should be recoverable from raw GTAP border values."""
    sets = GTAPSets(
        r=["NAmerica", "EU_28"],
        i=["c_Crops"],
        a=["a_agricultur"],
        f=["Land"],
        mf=[],
        sf=["Land"],
    )
    params = GTAPParameters(sets=sets)
    params.benchmark.vxsb[("NAmerica", "c_Crops", "EU_28")] = 100.0
    params.benchmark.vfob[("NAmerica", "c_Crops", "EU_28")] = 105.0
    params.benchmark.vcif[("NAmerica", "c_Crops", "EU_28")] = 110.0
    params.benchmark.vmsb[("NAmerica", "c_Crops", "EU_28")] = 121.0

    params.taxes.derive_trade_route_wedges(params.benchmark, sets)

    assert params.taxes.exptx_route[("NAmerica", "c_Crops", "EU_28")] == pytest.approx(0.05)
    assert params.taxes.tmarg_route[("NAmerica", "c_Crops", "EU_28")] == pytest.approx((110.0 - 105.0) / 100.0)
    assert params.taxes.imptx_route[("NAmerica", "c_Crops", "EU_28")] == pytest.approx(0.10)


def test_model_equations_build_with_multi_output_structure() -> None:
    """Multi-output make structures should now build through explicit output shares."""
    sets = GTAPSets(
        r=["NAmerica"],
        i=["c_Crops", "c_MeatLstk", "c_Extraction"],
        a=["a_agricultur", "a_Extraction"],
        f=["Land", "Capital"],
        mf=["Capital"],
        sf=["Land"],
        s=["NAmerica"],
        output_pairs=[
            ("a_agricultur", "c_Crops"),
            ("a_agricultur", "c_MeatLstk"),
            ("a_Extraction", "c_Extraction"),
        ],
        activity_commodities={
            "a_agricultur": ["c_Crops", "c_MeatLstk"],
            "a_Extraction": ["c_Extraction"],
        },
        commodity_activities={
            "c_Crops": ["a_agricultur"],
            "c_MeatLstk": ["a_agricultur"],
            "c_Extraction": ["a_Extraction"],
        },
    )
    params = GTAPParameters(sets=sets)
    params.benchmark.vom[("NAmerica", "a_agricultur")] = 100.0
    params.benchmark.vom[("NAmerica", "a_Extraction")] = 50.0
    params.benchmark.makb[("NAmerica", "a_agricultur", "c_Crops")] = 25.0
    params.benchmark.makb[("NAmerica", "a_agricultur", "c_MeatLstk")] = 75.0
    params.benchmark.makb[("NAmerica", "a_Extraction", "c_Extraction")] = 50.0
    params.shares.p_gx[("NAmerica", "a_agricultur", "c_Crops")] = 0.25
    params.shares.p_gx[("NAmerica", "a_agricultur", "c_MeatLstk")] = 0.75
    params.shares.p_gx[("NAmerica", "a_Extraction", "c_Extraction")] = 1.0
    params.shares.p_cons[("NAmerica", "c_Crops")] = 0.8
    params.shares.p_cons[("NAmerica", "c_MeatLstk")] = 0.2
    params.shares.p_gov[("NAmerica", "c_Crops")] = 0.6
    params.shares.p_gov[("NAmerica", "c_MeatLstk")] = 0.4
    params.shares.p_inv[("NAmerica", "c_Crops")] = 0.5
    params.shares.p_inv[("NAmerica", "c_MeatLstk")] = 0.5
    params.shares.p_yc["NAmerica"] = 0.55
    params.shares.p_yg["NAmerica"] = 0.25
    params.shares.p_yi["NAmerica"] = 0.20

    model = GTAPModelEquations(sets=sets, params=params).build_model()

    crop_eq = model.eq_x["NAmerica", "a_agricultur", "c_Crops"]
    meat_eq = model.eq_x["NAmerica", "a_agricultur", "c_MeatLstk"]
    unmapped = model.eq_x["NAmerica", "a_agricultur", "c_Extraction"]
    po_eq = model.eq_po["NAmerica", "a_agricultur", "c_Crops"]
    xc_eq = model.eq_xc["NAmerica", "c_Crops"]
    yi_eq = model.eq_yi["NAmerica"]

    assert "0.25*xp[NAmerica,a_agricultur]" in str(crop_eq.body)
    assert "0.75*xp[NAmerica,a_agricultur]" in str(meat_eq.body)
    assert str(unmapped.body) == "x[NAmerica,a_agricultur,c_Extraction]"
    assert unmapped.lower == 0.0
    assert unmapped.upper == 0.0
    assert "po[NAmerica,a_agricultur,c_Crops] - ps[NAmerica,c_Crops]" == str(po_eq.body)
    assert "xc[NAmerica,c_Crops] - 0.8*yc[NAmerica]" == str(xc_eq.body)
    assert "yi[NAmerica] - 0.2*regy[NAmerica]" == str(yi_eq.body)


def test_model_equations_build_with_agent_armington_demands() -> None:
    """Agent-specific Armington demand blocks should build for activities and final demand."""
    sets = GTAPSets(
        r=["NAmerica"],
        i=["c_Crops", "c_MeatLstk"],
        a=["a_agricultur"],
        f=["Land", "Capital"],
        mf=["Capital"],
        sf=["Land"],
        s=["NAmerica"],
        output_pairs=[("a_agricultur", "c_Crops"), ("a_agricultur", "c_MeatLstk")],
        activity_commodities={"a_agricultur": ["c_Crops", "c_MeatLstk"]},
        commodity_activities={"c_Crops": ["a_agricultur"], "c_MeatLstk": ["a_agricultur"]},
    )
    params = GTAPParameters(sets=sets)
    params.benchmark.vom[("NAmerica", "a_agricultur")] = 100.0
    params.benchmark.makb[("NAmerica", "a_agricultur", "c_Crops")] = 50.0
    params.benchmark.makb[("NAmerica", "a_agricultur", "c_MeatLstk")] = 50.0
    params.shares.p_gx[("NAmerica", "a_agricultur", "c_Crops")] = 0.5
    params.shares.p_gx[("NAmerica", "a_agricultur", "c_MeatLstk")] = 0.5
    params.shares.p_io[("NAmerica", "c_Crops", "a_agricultur")] = 0.8
    params.shares.p_io[("NAmerica", "c_MeatLstk", "a_agricultur")] = 0.2
    params.shares.p_alphad_agent[("NAmerica", "c_Crops", "a_agricultur")] = 0.75
    params.shares.p_alpham_agent[("NAmerica", "c_Crops", "a_agricultur")] = 0.25
    params.shares.p_alphad_agent[("NAmerica", "c_Crops", "hhd")] = 0.8
    params.shares.p_alpham_agent[("NAmerica", "c_Crops", "hhd")] = 0.2
    params.shares.p_alphad_agent[("NAmerica", "c_Crops", "gov")] = 1.0
    params.shares.p_alpham_agent[("NAmerica", "c_Crops", "gov")] = 0.0
    params.shares.p_alphad_agent[("NAmerica", "c_Crops", "inv")] = 0.75
    params.shares.p_alpham_agent[("NAmerica", "c_Crops", "inv")] = 0.25
    params.shares.p_cons[("NAmerica", "c_Crops")] = 0.8
    params.shares.p_gov[("NAmerica", "c_Crops")] = 1.0
    params.shares.p_inv[("NAmerica", "c_Crops")] = 1.0
    params.shares.p_yc["NAmerica"] = 0.5
    params.shares.p_yg["NAmerica"] = 0.3
    params.shares.p_yi["NAmerica"] = 0.2
    params.shares.p_af[("NAmerica", "Land", "a_agricultur")] = 0.6
    params.shares.p_af[("NAmerica", "Capital", "a_agricultur")] = 0.4
    params.shares.p_va[("NAmerica", "a_agricultur")] = 0.5

    model = GTAPModelEquations(sets=sets, params=params).build_model()

    assert list(model.aa) == ["a_agricultur", "hhd", "gov", "inv"]
    xaa_prod_eq = model.eq_xaa_prod["NAmerica", "c_Crops", "a_agricultur"]
    prf_y_eq = model.prf_y["NAmerica", "a_agricultur"]
    xaa_hhd_eq = model.eq_xaa_hhd["NAmerica", "c_Crops"]
    pdp_hhd_eq = model.eq_pdp["NAmerica", "c_Crops", "hhd"]
    paa_hhd_eq = model.eq_paa["NAmerica", "c_Crops", "hhd"]
    xda_hhd_eq = model.eq_xda["NAmerica", "c_Crops", "hhd"]
    xma_inv_eq = model.eq_xma["NAmerica", "c_Crops", "inv"]
    xmt_eq = model.eq_xmt["NAmerica", "c_Crops"]
    mkt_eq = model.mkt_goods["NAmerica", "c_Crops"]

    assert "0.8*xp[NAmerica,a_agricultur]" in str(xaa_prod_eq.body)
    assert "px[NAmerica,a_agricultur] - pp[NAmerica,a_agricultur]" == str(prf_y_eq.body)
    assert "xaa[NAmerica,c_Crops,hhd] - xc[NAmerica,c_Crops]" == str(xaa_hhd_eq.body)
    assert "pdp[NAmerica,c_Crops,hhd] - pd[NAmerica,c_Crops]" == str(pdp_hhd_eq.body)
    assert "paa[NAmerica,c_Crops,hhd]**(-1.0)" in str(paa_hhd_eq.body)
    assert "0.8*pdp[NAmerica,c_Crops,hhd]**(-1.0)" in str(paa_hhd_eq.body)
    assert "0.2*pmp[NAmerica,c_Crops,hhd]**(-1.0)" in str(paa_hhd_eq.body)
    assert "xda[NAmerica,c_Crops,hhd]" in str(xda_hhd_eq.body)
    assert "(paa[NAmerica,c_Crops,hhd]/pdp[NAmerica,c_Crops,hhd])**2.0" in str(xda_hhd_eq.body)
    assert "xma[NAmerica,c_Crops,inv]" in str(xma_inv_eq.body)
    assert "(paa[NAmerica,c_Crops,inv]/pmp[NAmerica,c_Crops,inv])**2.0" in str(xma_inv_eq.body)
    assert "xmt[NAmerica,c_Crops] - (xma[NAmerica,c_Crops,a_agricultur] + xma[NAmerica,c_Crops,hhd] + xma[NAmerica,c_Crops,gov] + xma[NAmerica,c_Crops,inv])" == str(xmt_eq.body)
    assert "xaa[NAmerica,c_Crops,a_agricultur]" in str(mkt_eq.body)
    assert "xaa[NAmerica,c_Crops,hhd]" in str(mkt_eq.body)
    assert "xaa[NAmerica,c_Crops,gov]" in str(mkt_eq.body)
    assert "xaa[NAmerica,c_Crops,inv]" in str(mkt_eq.body)


def test_model_equations_build_trade_block_from_benchmark_shares() -> None:
    """Trade equations should use calibrated Armington and bilateral shares."""
    sets = GTAPSets(
        r=["NAmerica", "EU_28"],
        i=["c_Crops"],
        a=["a_agricultur"],
        f=["Land", "Capital"],
        mf=["Capital"],
        sf=["Land"],
        s=["NAmerica", "EU_28"],
        output_pairs=[("a_agricultur", "c_Crops")],
        activity_commodities={"a_agricultur": ["c_Crops"]},
        commodity_activities={"c_Crops": ["a_agricultur"]},
    )
    params = GTAPParameters(sets=sets)
    params.benchmark.vom[("NAmerica", "a_agricultur")] = 100.0
    params.benchmark.vom[("EU_28", "a_agricultur")] = 80.0
    params.benchmark.vom_i[("NAmerica", "c_Crops")] = 100.0
    params.benchmark.vom_i[("EU_28", "c_Crops")] = 80.0
    params.benchmark.makb[("NAmerica", "a_agricultur", "c_Crops")] = 100.0
    params.benchmark.makb[("EU_28", "a_agricultur", "c_Crops")] = 80.0
    params.shares.p_gx[("NAmerica", "a_agricultur", "c_Crops")] = 1.0
    params.shares.p_gx[("EU_28", "a_agricultur", "c_Crops")] = 1.0
    params.shares.p_alphad[("NAmerica", "c_Crops")] = 0.75
    params.shares.p_alpham[("NAmerica", "c_Crops")] = 0.25
    params.shares.p_alphad[("EU_28", "c_Crops")] = 0.50
    params.shares.p_alpham[("EU_28", "c_Crops")] = 0.50
    params.shares.p_gd[("NAmerica", "c_Crops")] = 0.5
    params.shares.p_ge[("NAmerica", "c_Crops")] = 0.5
    params.shares.p_gd[("EU_28", "c_Crops")] = 0.5
    params.shares.p_ge[("EU_28", "c_Crops")] = 0.5
    params.shares.p_gw[("NAmerica", "c_Crops", "EU_28")] = 1.0
    params.shares.p_gw[("EU_28", "c_Crops", "NAmerica")] = 1.0
    params.shares.p_amw[("NAmerica", "c_Crops", "EU_28")] = 1.0
    params.shares.p_amw[("EU_28", "c_Crops", "NAmerica")] = 1.0
    params.shares.p_alphad_agent[("NAmerica", "c_Crops", "a_agricultur")] = 0.75
    params.shares.p_alpham_agent[("NAmerica", "c_Crops", "a_agricultur")] = 0.25
    params.shares.p_alphad_agent[("NAmerica", "c_Crops", "hhd")] = 0.75
    params.shares.p_alpham_agent[("NAmerica", "c_Crops", "hhd")] = 0.25
    params.shares.p_alphad_agent[("NAmerica", "c_Crops", "gov")] = 0.75
    params.shares.p_alpham_agent[("NAmerica", "c_Crops", "gov")] = 0.25
    params.shares.p_alphad_agent[("NAmerica", "c_Crops", "inv")] = 0.75
    params.shares.p_alpham_agent[("NAmerica", "c_Crops", "inv")] = 0.25
    params.shares.p_alphad_agent[("EU_28", "c_Crops", "a_agricultur")] = 0.50
    params.shares.p_alpham_agent[("EU_28", "c_Crops", "a_agricultur")] = 0.50
    params.shares.p_alphad_agent[("EU_28", "c_Crops", "hhd")] = 0.50
    params.shares.p_alpham_agent[("EU_28", "c_Crops", "hhd")] = 0.50
    params.shares.p_alphad_agent[("EU_28", "c_Crops", "gov")] = 0.50
    params.shares.p_alpham_agent[("EU_28", "c_Crops", "gov")] = 0.50
    params.shares.p_alphad_agent[("EU_28", "c_Crops", "inv")] = 0.50
    params.shares.p_alpham_agent[("EU_28", "c_Crops", "inv")] = 0.50
    params.shares.p_io[("NAmerica", "c_Crops", "a_agricultur")] = 1.0
    params.shares.p_io[("EU_28", "c_Crops", "a_agricultur")] = 1.0
    params.elasticities.omegax[("NAmerica", "c_Crops")] = 2.0
    params.elasticities.omegax[("EU_28", "c_Crops")] = 2.0
    params.elasticities.esubm[("NAmerica", "c_Crops")] = 4.0
    params.elasticities.esubm[("EU_28", "c_Crops")] = 4.0
    params.elasticities.omegaw[("NAmerica", "c_Crops")] = 2.0
    params.elasticities.omegaw[("EU_28", "c_Crops")] = 2.0
    params.taxes.exptx_route[("NAmerica", "c_Crops", "EU_28")] = 0.05
    params.taxes.exptx_route[("EU_28", "c_Crops", "NAmerica")] = 0.06
    params.taxes.tmarg_route[("NAmerica", "c_Crops", "EU_28")] = 0.02
    params.taxes.tmarg_route[("EU_28", "c_Crops", "NAmerica")] = 0.03
    params.taxes.imptx_route[("NAmerica", "c_Crops", "EU_28")] = 0.10
    params.taxes.imptx_route[("EU_28", "c_Crops", "NAmerica")] = 0.08

    model = GTAPModelEquations(sets=sets, params=params).build_model()

    eq_xmt = model.eq_xmt["NAmerica", "c_Crops"]
    eq_xw = model.eq_xw["NAmerica", "c_Crops", "EU_28"]
    eq_trade = model.eq_trade_balance["NAmerica", "c_Crops", "EU_28"]
    eq_xe = model.eq_xe["NAmerica", "c_Crops", "EU_28"]
    eq_pa = model.eq_pa["NAmerica", "c_Crops"]
    eq_pmt = model.eq_pmt["NAmerica", "c_Crops"]
    eq_pet = model.eq_pet["NAmerica", "c_Crops"]
    eq_xds_cet = model.eq_xds_cet["NAmerica", "c_Crops"]
    eq_xds_market = model.eq_xds_market["NAmerica", "c_Crops"]
    eq_pefob = model.eq_pefob["NAmerica", "c_Crops", "EU_28"]
    eq_pmcif = model.eq_pmcif["NAmerica", "c_Crops", "EU_28"]
    eq_pm = model.eq_pm["NAmerica", "c_Crops", "EU_28"]
    eq_ps = model.eq_ps["NAmerica", "c_Crops"]

    assert "xmt[NAmerica,c_Crops] - (xma[NAmerica,c_Crops,a_agricultur] + xma[NAmerica,c_Crops,hhd] + xma[NAmerica,c_Crops,gov] + xma[NAmerica,c_Crops,inv])" == str(eq_xmt.body)
    assert "xw[NAmerica,c_Crops,EU_28]" in str(eq_xw.body)
    assert "(pmt[NAmerica,c_Crops]/pm[NAmerica,c_Crops,EU_28])**4.0" in str(eq_xw.body)
    assert "xw[NAmerica,c_Crops,EU_28] - xe[EU_28,c_Crops,NAmerica]" == str(eq_trade.body)
    assert "xe[NAmerica,c_Crops,EU_28]" in str(eq_xe.body)
    assert "(pe[EU_28,c_Crops,NAmerica]/pet[NAmerica,c_Crops])**2.0" in str(eq_xe.body)
    assert "pa[NAmerica,c_Crops]*xa[NAmerica,c_Crops]" == str(eq_pa.body).split(" - ")[0]
    assert "paa[NAmerica,c_Crops,a_agricultur]*xaa[NAmerica,c_Crops,a_agricultur]" in str(eq_pa.body)
    assert "paa[NAmerica,c_Crops,hhd]*xaa[NAmerica,c_Crops,hhd]" in str(eq_pa.body)
    assert "pmt[NAmerica,c_Crops]" in str(eq_pmt.body)
    assert "pm[NAmerica,c_Crops,EU_28]" in str(eq_pmt.body)
    assert "-3.0" in str(eq_pmt.body)
    assert "pet[NAmerica,c_Crops]" in str(eq_pet.body)
    assert "pe[EU_28,c_Crops,NAmerica]" in str(eq_pet.body)
    assert "**3.0" in str(eq_pet.body)
    assert "xds[NAmerica,c_Crops]" in str(eq_xds_cet.body)
    assert "(pd[NAmerica,c_Crops]/ps[NAmerica,c_Crops])**2.0" in str(eq_xds_cet.body)
    assert "xds[NAmerica,c_Crops] - xd[NAmerica,c_Crops]" == str(eq_xds_market.body)
    assert "pefob[NAmerica,c_Crops,EU_28] - 1.05*pe[NAmerica,c_Crops,EU_28]" == str(eq_pefob.body)
    assert "pmcif[NAmerica,c_Crops,EU_28] - (pefob[NAmerica,c_Crops,EU_28] + 0.02*pwmg[NAmerica,c_Crops,EU_28])" == str(eq_pmcif.body)
    assert "pm[NAmerica,c_Crops,EU_28] - 1.1*pmcif[NAmerica,c_Crops,EU_28]" == str(eq_pm.body)
    assert "ps[NAmerica,c_Crops]**3.0" in str(eq_ps.body)
    assert "0.5*pd[NAmerica,c_Crops]**3.0" in str(eq_ps.body)
    assert "0.5*pet[NAmerica,c_Crops]**3.0" in str(eq_ps.body)




def test_model_equations_build_trade_margin_block() -> None:
    """Trade-margin equations should build on margin commodities and the tmg agent."""
    sets = GTAPSets(
        r=["NAmerica", "EU_28"],
        i=["c_Crops", "c_Transport"],
        a=["a_agricultur", "a_transport"],
        f=["Land", "Capital"],
        mf=["Capital"],
        sf=["Land"],
        m=["c_Transport"],
        s=["NAmerica", "EU_28"],
        output_pairs=[("a_agricultur", "c_Crops"), ("a_transport", "c_Transport")],
        activity_commodities={"a_agricultur": ["c_Crops"], "a_transport": ["c_Transport"]},
        commodity_activities={"c_Crops": ["a_agricultur"], "c_Transport": ["a_transport"]},
    )
    params = GTAPParameters(sets=sets)
    params.shares.p_gx[("NAmerica", "a_agricultur", "c_Crops")] = 1.0
    params.shares.p_gx[("EU_28", "a_agricultur", "c_Crops")] = 1.0
    params.shares.p_gx[("NAmerica", "a_transport", "c_Transport")] = 1.0
    params.shares.p_gx[("EU_28", "a_transport", "c_Transport")] = 1.0
    params.shares.p_amw[("NAmerica", "c_Crops", "EU_28")] = 1.0
    params.shares.p_gw[("EU_28", "c_Crops", "NAmerica")] = 1.0
    params.shares.p_amgm[("c_Transport", "NAmerica", "c_Crops", "EU_28")] = 1.0
    params.shares.p_tmg[("NAmerica", "c_Transport")] = 0.6
    params.shares.p_tmg[("EU_28", "c_Transport")] = 0.4
    params.shares.p_io[("NAmerica", "c_Crops", "a_agricultur")] = 1.0
    params.shares.p_io[("EU_28", "c_Crops", "a_agricultur")] = 1.0
    params.shares.p_io[("NAmerica", "c_Transport", "a_transport")] = 1.0
    params.shares.p_io[("EU_28", "c_Transport", "a_transport")] = 1.0
    params.elasticities.sigmam["c_Transport"] = 3.0
    params.taxes.tmarg_route[("NAmerica", "c_Crops", "EU_28")] = 0.05

    model = GTAPModelEquations(sets=sets, params=params).build_model()

    assert GTAP_MARGIN_AGENT in list(model.aa)

    eq_xaa_tmg = model.eq_xaa_tmg["NAmerica", "c_Transport"]
    eq_xaa_tmg_zero = model.eq_xaa_tmg_zero["NAmerica", "c_Crops"]
    eq_xwmg = model.eq_xwmg["NAmerica", "c_Crops", "EU_28"]
    eq_xmgm = model.eq_xmgm["c_Transport", "NAmerica", "c_Crops", "EU_28"]
    eq_pwmg = model.eq_pwmg["NAmerica", "c_Crops", "EU_28"]
    eq_xtmg = model.eq_xtmg["c_Transport"]
    eq_ptmg = model.eq_ptmg["c_Transport"]
    eq_pdp_tmg_zero = model.eq_pdp["NAmerica", "c_Crops", "tmg"]
    eq_pmp_tmg_zero = model.eq_pmp["NAmerica", "c_Crops", "tmg"]
    eq_pdp_tmg_margin = model.eq_pdp["NAmerica", "c_Transport", "tmg"]
    eq_paa_tmg_zero = model.eq_paa["NAmerica", "c_Crops", "tmg"]

    assert "0.6*xtmg[c_Transport]" in str(eq_xaa_tmg.body)
    assert "(ptmg[c_Transport]/paa[NAmerica,c_Transport,tmg])**3.0" in str(eq_xaa_tmg.body)
    assert str(eq_xaa_tmg_zero.body) == "xaa[NAmerica,c_Crops,tmg]"
    assert eq_xaa_tmg_zero.lower == 0.0
    assert eq_xaa_tmg_zero.upper == 0.0
    assert str(eq_pdp_tmg_zero.body) == "pdp[NAmerica,c_Crops,tmg]"
    assert eq_pdp_tmg_zero.lower == 0.0
    assert eq_pdp_tmg_zero.upper == 0.0
    assert str(eq_pmp_tmg_zero.body) == "pmp[NAmerica,c_Crops,tmg]"
    assert eq_pmp_tmg_zero.lower == 0.0
    assert eq_pmp_tmg_zero.upper == 0.0
    assert "pdp[NAmerica,c_Transport,tmg] - pd[NAmerica,c_Transport]" == str(eq_pdp_tmg_margin.body)
    assert str(eq_paa_tmg_zero.body) == "paa[NAmerica,c_Crops,tmg]"
    assert eq_paa_tmg_zero.lower == 0.0
    assert eq_paa_tmg_zero.upper == 0.0
    assert "xwmg[NAmerica,c_Crops,EU_28] - 0.05*xw[NAmerica,c_Crops,EU_28]" == str(eq_xwmg.body)
    assert "xmgm[c_Transport,NAmerica,c_Crops,EU_28] - xwmg[NAmerica,c_Crops,EU_28]" == str(eq_xmgm.body)
    assert "pwmg[NAmerica,c_Crops,EU_28] - ptmg[c_Transport]" == str(eq_pwmg.body)
    assert "xtmg[c_Transport]" in str(eq_xtmg.body)
    assert "xmgm[c_Transport,NAmerica,c_Crops,EU_28]" in str(eq_xtmg.body)
    assert "xmgm[c_Transport,EU_28,c_Crops,NAmerica]" in str(eq_xtmg.body)
    assert "ptmg[c_Transport]" in str(eq_ptmg.body)
    assert "paa[NAmerica,c_Transport,tmg]" in str(eq_ptmg.body)


def test_model_equations_build_with_bijective_non_diagonal_structure() -> None:
    """One-to-one non-diagonal make structures should still build."""
    sets = GTAPSets(
        r=["NAmerica"],
        i=["c_Food", "c_Extraction"],
        a=["a_AgroProd", "a_Extraction"],
        f=["Land", "Capital"],
        mf=["Capital"],
        sf=["Land"],
        s=["NAmerica"],
        i_to_a={
            "c_Food": "a_AgroProd",
            "c_Extraction": "a_Extraction",
        },
        a_to_i={
            "a_AgroProd": "c_Food",
            "a_Extraction": "c_Extraction",
        },
        output_pairs=[
            ("a_AgroProd", "c_Food"),
            ("a_Extraction", "c_Extraction"),
        ],
        activity_commodities={
            "a_AgroProd": ["c_Food"],
            "a_Extraction": ["c_Extraction"],
        },
        commodity_activities={
            "c_Food": ["a_AgroProd"],
            "c_Extraction": ["a_Extraction"],
        },
    )

    model = GTAPModelEquations(sets=sets, params=GTAPParameters()).build_model()

    mapped = model.eq_x["NAmerica", "a_AgroProd", "c_Food"]
    unmapped = model.eq_x["NAmerica", "a_AgroProd", "c_Extraction"]

    assert "xp[NAmerica,a_AgroProd]" in str(mapped.body)
    assert str(unmapped.body) == "x[NAmerica,a_AgroProd,c_Extraction]"
    assert unmapped.lower == 0.0
    assert unmapped.upper == 0.0
