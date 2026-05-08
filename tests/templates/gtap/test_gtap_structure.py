"""Tests for GTAP template structure handling.

These tests target the **current** GTAP Standard 7 / GAMS-parity API:
``GTAPShareParameters`` (``p_gx``/``p_va``/``p_af``/``p_alphad``/``p_alpham``/
``p_gd``/``p_ge``/``p_gw``/``p_amw``/``p_io`` …), ``GTAPTaxRates``
(``rto``/``rtf``/``rtpd``/``rtpi``/``rtxs``/``rtms``/``imptx``/``dintx0``/
``mintx0``/``kappaf`` …) and the calibrated parameters (``gx_param``,
``af_param``, ``io_param``) registered on the Pyomo model.
"""

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

    def mock_read_parameter_values(
        _: dict[str, object], name: str
    ) -> dict[tuple[str, ...], float]:
        # The reader uppercases names; accept either form to be robust.
        if name.upper() == "MAKB":
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

    def mock_read_parameter_values(
        _: dict[str, object], name: str
    ) -> dict[tuple[str, ...], float]:
        if name.upper() == "MAKB":
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
    """Multi-output ``makb`` should drive ``p_gx`` (CET output share)."""
    sets = GTAPSets(
        r=["NAmerica", "EU_28"],
        i=["c_Crops", "c_MeatLstk", "c_Extraction"],
        a=["a_agricultur", "a_Extraction"],
        f=["Land", "Capital", "NatRes"],
        mf=["Capital"],
        sf=["Land", "NatRes"],
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
    params.benchmark.vom[("NAmerica", "a_agricultur")] = 22.0
    params.benchmark.vom[("EU_28", "a_Extraction")] = 15.0
    params.benchmark.makb[("NAmerica", "a_agricultur", "c_Crops")] = 10.0
    params.benchmark.makb[("NAmerica", "a_agricultur", "c_MeatLstk")] = 12.0
    params.benchmark.makb[("EU_28", "a_Extraction", "c_Extraction")] = 15.0

    params.shares.calibrate(params.benchmark, params.elasticities, sets)

    assert params.shares.p_gx[("NAmerica", "a_agricultur", "c_Crops")] == pytest.approx(10.0 / 22.0)
    assert params.shares.p_gx[("NAmerica", "a_agricultur", "c_MeatLstk")] == pytest.approx(12.0 / 22.0)
    assert params.shares.p_gx[("EU_28", "a_Extraction", "c_Extraction")] == pytest.approx(1.0)
    # The bijective inverse (CES production share) sums to 1 per (r, i).
    assert params.shares.p_ax[("NAmerica", "a_agricultur", "c_Crops")] == pytest.approx(1.0)
    assert params.shares.p_ax[("NAmerica", "a_agricultur", "c_MeatLstk")] == pytest.approx(1.0)


def test_parameters_calibrate_armington_shares_from_benchmark() -> None:
    """Top-level Armington shares ``p_alphad``/``p_alpham`` should reflect xd/xa."""
    sets = GTAPSets(
        r=["NAmerica"],
        i=["c_Crops"],
        a=["a_agricultur"],
        f=["Land", "Capital"],
        mf=["Capital"],
        sf=["Land"],
    )
    params = GTAPParameters(sets=sets)
    # Domestic absorption: 30 (private) + 9 (gov) + 6 (inv) + 5 (intermediate) = 50
    params.benchmark.vpm[("NAmerica", "c_Crops")] = 30.0
    params.benchmark.vdpp[("NAmerica", "c_Crops")] = 30.0
    params.benchmark.vgm[("NAmerica", "c_Crops")] = 9.0
    params.benchmark.vdgp[("NAmerica", "c_Crops")] = 9.0
    params.benchmark.vim[("NAmerica", "c_Crops")] = 6.0
    params.benchmark.vdip[("NAmerica", "c_Crops")] = 6.0
    params.benchmark.vdfb[("NAmerica", "c_Crops", "a_agricultur")] = 5.0
    # Imports: 50 total (vifm intermediate)
    params.benchmark.vifm[("NAmerica", "c_Crops", "a_agricultur")] = 50.0

    params.shares.calibrate(params.benchmark, params.elasticities, sets)

    # xa = 50 (domestic) + 50 (imports) = 100
    assert params.shares.p_alphad[("NAmerica", "c_Crops")] == pytest.approx(0.5)
    assert params.shares.p_alpham[("NAmerica", "c_Crops")] == pytest.approx(0.5)


def test_parameters_calibrate_value_added_factor_shares() -> None:
    """Value-added calibration should produce factor shares ``p_af`` and VA scale ``p_va``."""
    sets = GTAPSets(
        r=["NAmerica"],
        i=["c_Crops"],
        a=["a_agricultur"],
        f=["Land", "Capital"],
        mf=["Capital"],
        sf=["Land"],
        output_pairs=[("a_agricultur", "c_Crops")],
        activity_commodities={"a_agricultur": ["c_Crops"]},
        commodity_activities={"c_Crops": ["a_agricultur"]},
    )
    params = GTAPParameters(sets=sets)
    params.benchmark.vom[("NAmerica", "a_agricultur")] = 100.0
    params.benchmark.makb[("NAmerica", "a_agricultur", "c_Crops")] = 100.0
    params.benchmark.vfm[("NAmerica", "Land", "a_agricultur")] = 40.0
    params.benchmark.vfm[("NAmerica", "Capital", "a_agricultur")] = 20.0

    params.shares.calibrate(params.benchmark, params.elasticities, sets)

    # va = 60, total output = 100 → p_va = 0.6
    assert params.shares.p_va[("NAmerica", "a_agricultur")] == pytest.approx(0.6)
    # within VA: Land=40/60, Capital=20/60
    assert params.shares.p_af[("NAmerica", "Land", "a_agricultur")] == pytest.approx(2.0 / 3.0)
    assert params.shares.p_af[("NAmerica", "Capital", "a_agricultur")] == pytest.approx(1.0 / 3.0)
    # ND complement
    assert params.shares.p_nd[("NAmerica", "a_agricultur")] == pytest.approx(0.4)


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


def test_parameters_calibrate_trade_and_armington_shares_from_benchmark() -> None:
    """Bilateral trade shares ``p_gd``/``p_ge``/``p_gw``/``p_amw`` should follow benchmark flows."""
    sets = GTAPSets(
        r=["NAmerica", "EU_28"],
        i=["c_Crops"],
        a=["a_agricultur"],
        f=["Land", "Capital"],
        mf=["Capital"],
        sf=["Land"],
        output_pairs=[("a_agricultur", "c_Crops")],
        activity_commodities={"a_agricultur": ["c_Crops"]},
        commodity_activities={"c_Crops": ["a_agricultur"]},
    )
    params = GTAPParameters(sets=sets)
    # NAmerica produces 100 Crops: 60 domestic, 40 exported to EU_28.
    params.benchmark.vom_i[("NAmerica", "c_Crops")] = 100.0
    params.benchmark.vom_i[("EU_28", "c_Crops")] = 80.0
    params.benchmark.makb[("NAmerica", "a_agricultur", "c_Crops")] = 100.0
    params.benchmark.makb[("EU_28", "a_agricultur", "c_Crops")] = 80.0
    params.benchmark.vxsb[("NAmerica", "c_Crops", "EU_28")] = 40.0
    params.benchmark.vxsb[("EU_28", "c_Crops", "NAmerica")] = 20.0
    # Domestic absorption to keep xa positive on each side.
    params.benchmark.vpm[("NAmerica", "c_Crops")] = 60.0
    params.benchmark.vdpp[("NAmerica", "c_Crops")] = 60.0
    params.benchmark.vpm[("EU_28", "c_Crops")] = 60.0
    params.benchmark.vdpp[("EU_28", "c_Crops")] = 60.0
    # Bilateral imports (mirror exports).
    params.benchmark.vmsb[("NAmerica", "c_Crops", "EU_28")] = 40.0
    params.benchmark.vmsb[("EU_28", "c_Crops", "NAmerica")] = 20.0
    params.benchmark.vcif[("NAmerica", "c_Crops", "EU_28")] = 40.0
    params.benchmark.vcif[("EU_28", "c_Crops", "NAmerica")] = 20.0

    params.shares.calibrate(params.benchmark, params.elasticities, sets)

    # Top-level CET split is xet/xs. NAmerica: 40/100 → exports 0.4, domestic 0.6.
    assert params.shares.p_gd[("NAmerica", "c_Crops")] == pytest.approx(0.6)
    assert params.shares.p_ge[("NAmerica", "c_Crops")] == pytest.approx(0.4)
    # EU_28: 20/80 → 0.25 exports, 0.75 domestic.
    assert params.shares.p_gd[("EU_28", "c_Crops")] == pytest.approx(0.75)
    assert params.shares.p_ge[("EU_28", "c_Crops")] == pytest.approx(0.25)
    # Bilateral export shares sum to 1.0 (single destination on each side).
    assert params.shares.p_gw[("NAmerica", "c_Crops", "EU_28")] == pytest.approx(1.0)
    assert params.shares.p_gw[("EU_28", "c_Crops", "NAmerica")] == pytest.approx(1.0)


def test_parameters_calibrate_intermediate_io_shares() -> None:
    """``p_io`` should split intermediate demand by commodity inside ND."""
    sets = GTAPSets(
        r=["NAmerica"],
        i=["c_Crops", "c_MeatLstk"],
        a=["a_agricultur"],
        f=["Land", "Capital"],
        mf=["Capital"],
        sf=["Land"],
        output_pairs=[("a_agricultur", "c_Crops")],
        activity_commodities={"a_agricultur": ["c_Crops"]},
        commodity_activities={"c_Crops": ["a_agricultur"]},
    )
    params = GTAPParameters(sets=sets)
    params.benchmark.vom[("NAmerica", "a_agricultur")] = 100.0
    params.benchmark.makb[("NAmerica", "a_agricultur", "c_Crops")] = 100.0
    params.benchmark.vdfm[("NAmerica", "c_Crops", "a_agricultur")] = 30.0
    params.benchmark.vifm[("NAmerica", "c_Crops", "a_agricultur")] = 10.0
    params.benchmark.vdfm[("NAmerica", "c_MeatLstk", "a_agricultur")] = 15.0
    params.benchmark.vifm[("NAmerica", "c_MeatLstk", "a_agricultur")] = 15.0

    params.shares.calibrate(params.benchmark, params.elasticities, sets)

    # Total intermediate cost = 70. Crops=40/70, MeatLstk=30/70.
    assert params.shares.p_io[("NAmerica", "c_Crops", "a_agricultur")] == pytest.approx(40.0 / 70.0)
    assert params.shares.p_io[("NAmerica", "c_MeatLstk", "a_agricultur")] == pytest.approx(30.0 / 70.0)


def test_tax_rates_derive_agent_consumption_wedges_from_benchmark() -> None:
    """Agent consumption tax wedges ``dintx0``/``mintx0`` derived from purchaser/basic flows."""
    sets = GTAPSets(
        r=["NAmerica"],
        i=["c_Crops"],
        a=["a_agricultur"],
        f=["Land"],
        mf=[],
        sf=["Land"],
    )
    params = GTAPParameters(sets=sets)
    bench = params.benchmark
    bench.vdfb[("NAmerica", "c_Crops", "a_agricultur")] = 100.0
    bench.vdfp[("NAmerica", "c_Crops", "a_agricultur")] = 115.0
    bench.vmfb[("NAmerica", "c_Crops", "a_agricultur")] = 80.0
    bench.vmfp[("NAmerica", "c_Crops", "a_agricultur")] = 100.0
    bench.vdpb[("NAmerica", "c_Crops")] = 50.0
    bench.vdpp[("NAmerica", "c_Crops")] = 55.0
    bench.vmpb[("NAmerica", "c_Crops")] = 20.0
    bench.vmpp[("NAmerica", "c_Crops")] = 25.0
    bench.vdgb[("NAmerica", "c_Crops")] = 10.0
    bench.vdgp[("NAmerica", "c_Crops")] = 12.0
    bench.vmgb[("NAmerica", "c_Crops")] = 8.0
    bench.vmgp[("NAmerica", "c_Crops")] = 10.0
    bench.vdib[("NAmerica", "c_Crops")] = 30.0
    bench.vdip[("NAmerica", "c_Crops")] = 33.0
    bench.vmib[("NAmerica", "c_Crops")] = 12.0
    bench.vmip[("NAmerica", "c_Crops")] = 15.0

    params.taxes.derive_agent_consumption_taxes(bench, sets)

    # Domestic firms: (115-100)/100 = 0.15. Imports: (100-80)/80 = 0.25.
    assert params.taxes.dintx0[("NAmerica", "c_Crops", "a_agricultur")] == pytest.approx(0.15)
    assert params.taxes.mintx0[("NAmerica", "c_Crops", "a_agricultur")] == pytest.approx(0.25)
    # Households: 5/50 = 0.10, imports 5/20 = 0.25.
    assert params.taxes.dintx0[("NAmerica", "c_Crops", "hhd")] == pytest.approx(0.10)
    assert params.taxes.mintx0[("NAmerica", "c_Crops", "hhd")] == pytest.approx(0.25)
    # Government: 2/10 = 0.20, imports 2/8 = 0.25.
    assert params.taxes.dintx0[("NAmerica", "c_Crops", "gov")] == pytest.approx(0.20)
    assert params.taxes.mintx0[("NAmerica", "c_Crops", "gov")] == pytest.approx(0.25)
    # Investment: 3/30 = 0.10, imports 3/12 = 0.25.
    assert params.taxes.dintx0[("NAmerica", "c_Crops", "inv")] == pytest.approx(0.10)
    assert params.taxes.mintx0[("NAmerica", "c_Crops", "inv")] == pytest.approx(0.25)


def test_tax_rates_derive_trade_route_wedges_from_benchmark() -> None:
    """Trade-route wedges ``rtxs``/``imptx`` derivable from raw GTAP border values."""
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

    params.taxes.derive_from_benchmark(params.benchmark, sets)

    # rtxs = (vfob - vxsb) / vxsb = 5/100.
    assert params.taxes.rtxs[("NAmerica", "c_Crops", "EU_28")] == pytest.approx(0.05)
    # imptx = (vmsb - vcif) / vcif = 11/110.
    assert params.taxes.imptx[("NAmerica", "c_Crops", "EU_28")] == pytest.approx(0.10)


def test_model_equations_build_with_multi_output_structure() -> None:
    """Multi-output make structures should build, expose ``gx_param`` per output."""
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
    params.shares.calibrate(params.benchmark, params.elasticities, sets)
    params.calibrated.calibrate_from_benchmark(
        params.benchmark, params.elasticities, sets, params.taxes
    )

    model = GTAPModelEquations(sets=sets, params=params).build_model()

    # gx_param is registered on the Pyomo model with the calibrated CET shares.
    assert model.gx_param[("NAmerica", "a_agricultur", "c_Crops")] == pytest.approx(0.25)
    assert model.gx_param[("NAmerica", "a_agricultur", "c_MeatLstk")] == pytest.approx(0.75)
    assert model.gx_param[("NAmerica", "a_Extraction", "c_Extraction")] == pytest.approx(1.0)

    # x and eq_x are indexed over (r, a, i). Inactive (a, i) cells are Skip'd
    # rather than emitted as zero constraints.
    assert ("NAmerica", "a_agricultur", "c_Crops") in model.eq_x
    assert ("NAmerica", "a_agricultur", "c_MeatLstk") in model.eq_x
    assert ("NAmerica", "a_Extraction", "c_Extraction") in model.eq_x
    assert ("NAmerica", "a_agricultur", "c_Extraction") not in model.eq_x


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
        commodity_activities={
            "c_Crops": ["a_agricultur"],
            "c_MeatLstk": ["a_agricultur"],
        },
    )
    params = GTAPParameters(sets=sets)
    params.benchmark.vom[("NAmerica", "a_agricultur")] = 100.0
    params.benchmark.makb[("NAmerica", "a_agricultur", "c_Crops")] = 50.0
    params.benchmark.makb[("NAmerica", "a_agricultur", "c_MeatLstk")] = 50.0
    params.benchmark.vfm[("NAmerica", "Land", "a_agricultur")] = 30.0
    params.benchmark.vfm[("NAmerica", "Capital", "a_agricultur")] = 20.0
    params.benchmark.vdfm[("NAmerica", "c_Crops", "a_agricultur")] = 30.0
    params.benchmark.vdfm[("NAmerica", "c_MeatLstk", "a_agricultur")] = 20.0
    # Final demand keeps xa positive across agents.
    for r, i, total in [
        ("NAmerica", "c_Crops", 50.0),
        ("NAmerica", "c_MeatLstk", 30.0),
    ]:
        params.benchmark.vpm[(r, i)] = total
        params.benchmark.vdpp[(r, i)] = total
        params.benchmark.vgm[(r, i)] = 5.0
        params.benchmark.vdgp[(r, i)] = 5.0
        params.benchmark.vim[(r, i)] = 5.0
        params.benchmark.vdip[(r, i)] = 5.0
    params.shares.calibrate(params.benchmark, params.elasticities, sets)

    model = GTAPModelEquations(sets=sets, params=params).build_model()

    # The aggregate-agent set ``aa`` should hold the activity plus the three
    # final-demand agents (hhd, gov, inv).
    assert "a_agricultur" in list(model.aa)
    for fd_agent in ("hhd", "gov", "inv"):
        assert fd_agent in list(model.aa)

    # Per-agent xaa Armington demand is built for both activities and final
    # demand — the indices should exist after build.
    assert ("NAmerica", "c_Crops", "a_agricultur") in model.xaa
    assert ("NAmerica", "c_Crops", "hhd") in model.xaa
    assert ("NAmerica", "c_Crops", "gov") in model.xaa
    assert ("NAmerica", "c_Crops", "inv") in model.xaa


def test_model_equations_build_price_indices_and_numeraire() -> None:
    """Regional price-index and numeraire constraints should be built."""
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
    params.benchmark.vom[("NAmerica", "a_agricultur")] = 100.0
    params.benchmark.makb[("NAmerica", "a_agricultur", "c_Crops")] = 100.0
    params.benchmark.vfm[("NAmerica", "Land", "a_agricultur")] = 60.0
    params.benchmark.vfm[("NAmerica", "Capital", "a_agricultur")] = 40.0
    params.benchmark.vdfm[("NAmerica", "c_Crops", "a_agricultur")] = 50.0
    params.benchmark.vpm[("NAmerica", "c_Crops")] = 30.0
    params.benchmark.vdpp[("NAmerica", "c_Crops")] = 30.0
    params.benchmark.vgm[("NAmerica", "c_Crops")] = 15.0
    params.benchmark.vdgp[("NAmerica", "c_Crops")] = 15.0
    params.benchmark.vim[("NAmerica", "c_Crops")] = 5.0
    params.benchmark.vdip[("NAmerica", "c_Crops")] = 5.0
    params.shares.calibrate(params.benchmark, params.elasticities, sets)

    model = GTAPModelEquations(sets=sets, params=params).build_model()

    # eq_pabs / eq_pfact / eq_pwfact / eq_pnum should exist; eq_pnum links
    # the numeraire to the world factor-price index (pnum == pwfact).
    assert "NAmerica" in model.eq_pabs
    assert "NAmerica" in model.eq_pfact
    assert "pnum" in str(model.eq_pnum.body)
    assert "pwfact" in str(model.eq_pnum.body)
    # Factor-price aggregate references both factors.
    pfact_body = str(model.eq_pfact["NAmerica"].body)
    assert "pf[NAmerica,Land,a_agricultur]" in pfact_body
    assert "pf[NAmerica,Capital,a_agricultur]" in pfact_body


def test_model_equations_build_trade_block_from_benchmark_shares() -> None:
    """Trade equations should build when bilateral and Armington benchmarks are populated."""
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
    for r in sets.r:
        params.benchmark.vom[(r, "a_agricultur")] = 100.0
        params.benchmark.vom_i[(r, "c_Crops")] = 100.0
        params.benchmark.makb[(r, "a_agricultur", "c_Crops")] = 100.0
        params.benchmark.vfm[(r, "Land", "a_agricultur")] = 60.0
        params.benchmark.vfm[(r, "Capital", "a_agricultur")] = 40.0
        params.benchmark.vdfm[(r, "c_Crops", "a_agricultur")] = 30.0
        params.benchmark.vpm[(r, "c_Crops")] = 30.0
        params.benchmark.vdpp[(r, "c_Crops")] = 30.0
    # Bilateral exports/imports.
    params.benchmark.vxsb[("NAmerica", "c_Crops", "EU_28")] = 40.0
    params.benchmark.vxsb[("EU_28", "c_Crops", "NAmerica")] = 40.0
    params.benchmark.vmsb[("NAmerica", "c_Crops", "EU_28")] = 40.0
    params.benchmark.vmsb[("EU_28", "c_Crops", "NAmerica")] = 40.0
    params.benchmark.vcif[("NAmerica", "c_Crops", "EU_28")] = 40.0
    params.benchmark.vcif[("EU_28", "c_Crops", "NAmerica")] = 40.0
    params.benchmark.vfob[("NAmerica", "c_Crops", "EU_28")] = 40.0
    params.benchmark.vfob[("EU_28", "c_Crops", "NAmerica")] = 40.0
    params.shares.calibrate(params.benchmark, params.elasticities, sets)
    params.taxes.derive_from_benchmark(params.benchmark, sets)
    params.calibrated.calibrate_from_benchmark(
        params.benchmark, params.elasticities, sets, params.taxes
    )

    model = GTAPModelEquations(sets=sets, params=params).build_model()

    # Bilateral trade variables/constraints exist for the active routes.
    assert ("NAmerica", "c_Crops", "EU_28") in model.eq_pefobeq
    assert ("NAmerica", "c_Crops", "EU_28") in model.eq_peeq
    assert ("EU_28", "c_Crops", "NAmerica") in model.eq_xweq
    # Top-level CET on the supply side and per-region market clear.
    assert ("NAmerica", "c_Crops") in model.eq_xs
    assert ("NAmerica", "c_Crops") in model.eq_pdeq
    # Cif/fob price linkage and bilateral import-price equation.
    assert ("EU_28", "c_Crops", "NAmerica") in model.eq_pmcifeq
    assert ("EU_28", "c_Crops", "NAmerica") in model.eq_pmeq


def test_model_equations_build_trade_margin_block() -> None:
    """Trade-margin equations should build when a margin commodity is declared."""
    sets = GTAPSets(
        r=["NAmerica", "EU_28"],
        i=["c_Crops", "c_Transport"],
        a=["a_agricultur", "a_transport"],
        f=["Land", "Capital"],
        mf=["Capital"],
        sf=["Land"],
        m=["c_Transport"],
        s=["NAmerica", "EU_28"],
        output_pairs=[
            ("a_agricultur", "c_Crops"),
            ("a_transport", "c_Transport"),
        ],
        activity_commodities={
            "a_agricultur": ["c_Crops"],
            "a_transport": ["c_Transport"],
        },
        commodity_activities={
            "c_Crops": ["a_agricultur"],
            "c_Transport": ["a_transport"],
        },
    )
    params = GTAPParameters(sets=sets)
    for r in sets.r:
        for a, i in [("a_agricultur", "c_Crops"), ("a_transport", "c_Transport")]:
            params.benchmark.vom[(r, a)] = 100.0
            params.benchmark.makb[(r, a, i)] = 100.0
            params.benchmark.vfm[(r, "Land", a)] = 60.0
            params.benchmark.vfm[(r, "Capital", a)] = 40.0
            params.benchmark.vdfm[(r, i, a)] = 30.0
        for i in sets.i:
            params.benchmark.vpm[(r, i)] = 30.0
            params.benchmark.vdpp[(r, i)] = 30.0
    # A bilateral trade route plus margin allocation.
    params.benchmark.vxsb[("NAmerica", "c_Crops", "EU_28")] = 40.0
    params.benchmark.vmsb[("NAmerica", "c_Crops", "EU_28")] = 42.0
    params.benchmark.vcif[("NAmerica", "c_Crops", "EU_28")] = 42.0
    params.benchmark.vfob[("NAmerica", "c_Crops", "EU_28")] = 40.0
    params.shares.calibrate(params.benchmark, params.elasticities, sets)

    model = GTAPModelEquations(sets=sets, params=params).build_model()

    # The margin agent should be present in the aggregate-agent set.
    assert GTAP_MARGIN_AGENT in list(model.aa)
    # Margin variables/constraints should exist for the margin commodity and route.
    assert ("c_Transport",) in [(k,) if not isinstance(k, tuple) else k for k in model.eq_xtmg]
    assert ("c_Transport",) in [(k,) if not isinstance(k, tuple) else k for k in model.eq_ptmg]


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

    params = GTAPParameters(sets=sets)
    for a, i in sets.output_pairs:
        params.benchmark.vom[("NAmerica", a)] = 100.0
        params.benchmark.makb[("NAmerica", a, i)] = 100.0
        params.benchmark.vfm[("NAmerica", "Land", a)] = 60.0
        params.benchmark.vfm[("NAmerica", "Capital", a)] = 40.0
    params.shares.calibrate(params.benchmark, params.elasticities, sets)
    params.calibrated.calibrate_from_benchmark(
        params.benchmark, params.elasticities, sets, params.taxes
    )

    model = GTAPModelEquations(sets=sets, params=params).build_model()

    # Mapped (a, i) pairs receive an eq_x constraint, unmapped do not.
    assert ("NAmerica", "a_AgroProd", "c_Food") in model.eq_x
    assert ("NAmerica", "a_AgroProd", "c_Extraction") not in model.eq_x
    assert ("NAmerica", "a_Extraction", "c_Extraction") in model.eq_x
