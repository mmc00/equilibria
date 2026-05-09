"""Tests for GTAP template structure handling."""

from __future__ import annotations

from pathlib import Path

import pytest

from equilibria.templates.gtap.gtap_parameters import GTAPParameters
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

    assert params.taxes.dintx0[("NAmerica", "c_Crops", "a_agricultur")] == pytest.approx(0.15)
    assert params.taxes.mintx0[("NAmerica", "c_Crops", "a_agricultur")] == pytest.approx(0.25)
    assert params.taxes.dintx0[("NAmerica", "c_Crops", "hhd")] == pytest.approx(0.10)
    assert params.taxes.mintx0[("NAmerica", "c_Crops", "hhd")] == pytest.approx(0.25)
    assert params.taxes.dintx0[("NAmerica", "c_Crops", "gov")] == pytest.approx(0.20)
    assert params.taxes.mintx0[("NAmerica", "c_Crops", "gov")] == pytest.approx(0.25)
    assert params.taxes.dintx0[("NAmerica", "c_Crops", "inv")] == pytest.approx(0.10)
    assert params.taxes.mintx0[("NAmerica", "c_Crops", "inv")] == pytest.approx(0.25)


