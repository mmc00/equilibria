from __future__ import annotations

from pathlib import Path
from typing import Any

from equilibria.templates.simple_open_parity_pipeline import (
    SimpleOpenGAMSReference,
    compare_simple_open_gams_parity,
    load_simple_open_gams_reference,
)


def test_load_simple_open_gams_reference_uses_babel_reader(monkeypatch: Any, tmp_path: Path) -> None:
    gdx_path = tmp_path / "simple_open.gdx"
    gdx_path.write_bytes(b"dummy")

    calls: list[str] = []

    def fake_read_gdx(path: str | Path) -> dict[str, Any]:
        assert Path(path) == gdx_path
        return {"filepath": str(gdx_path), "symbols": [], "elements": [], "domains": []}

    def fake_decode(*, gdx_path: Path, gdx_data: dict[str, Any], symbol_name: str, ordered_names: tuple[str, ...], fill_missing_with_zero: bool) -> dict[str, float]:
        calls.append(symbol_name)
        if symbol_name == "benchmark":
            return {"VA": 1.0, "INT": 0.5}
        if symbol_name == "level":
            return {"VA": 1.0, "INT": 0.5}
        if symbol_name == "residual":
            return {"EQ_VA": 0.0, "EQ_INT": 0.0, "EQ_CET": 0.0}
        if symbol_name == "calib":
            return {"alpha_va": 0.4, "modelstat": 1.0, "solvestat": 1.0, "closure_code": 101.0}
        raise AssertionError(symbol_name)

    monkeypatch.setattr(
        "equilibria.templates.simple_open_parity_pipeline.read_gdx",
        fake_read_gdx,
    )
    monkeypatch.setattr(
        "equilibria.templates.simple_open_parity_pipeline._decode_named_parameter_symbol",
        fake_decode,
    )

    payload = load_simple_open_gams_reference(gdx_path)
    assert payload == SimpleOpenGAMSReference(
        closure_names=("simple_open_default",),
        benchmark={"VA": 1.0, "INT": 0.5},
        level={"VA": 1.0, "INT": 0.5},
        residual={"EQ_VA": 0.0, "EQ_INT": 0.0, "EQ_CET": 0.0},
        calib={"alpha_va": 0.4, "modelstat": 1.0, "solvestat": 1.0, "closure_code": 101.0},
    )
    assert calls == ["benchmark", "level", "residual", "calib"]


def test_compare_simple_open_gams_parity_passes_with_matching_reference(monkeypatch: Any, tmp_path: Path) -> None:
    gdx_path = tmp_path / "simple_open.gdx"
    gdx_path.write_bytes(b"dummy")

    monkeypatch.setattr(
        "equilibria.templates.simple_open_parity_pipeline.load_simple_open_gams_reference",
        lambda path: SimpleOpenGAMSReference(
            closure_names=("simple_open_default",),
            benchmark={
                "VA": 1.0,
                "INT": 0.5,
                "X": 1.0,
                "D": 1.0,
                "E": 1.0,
                "ER": 1.0,
                "PFX": 1.0,
                "CAB": 1.0,
                "FSAV": 1.0,
            },
            level={
                "VA": 1.0,
                "INT": 0.5,
                "X": 1.0,
                "D": 1.0,
                "E": 1.0,
                "ER": 1.0,
                "PFX": 1.0,
                "CAB": 1.0,
                "FSAV": 1.0,
            },
            residual={"EQ_VA": 0.0, "EQ_INT": 0.0, "EQ_CET": 0.0},
                calib={
                    "alpha_va": 0.4,
                    "rho_va": 0.75,
                    "a_int": 0.5,
                    "b_ext": 0.1,
                    "theta_cet": 0.6,
                    "phi_cet": 1.2,
                    "closure_code": 101.0,
                    "modelstat": 1.0,
                    "solvestat": 1.0,
                },
            ),
    )

    comparison = compare_simple_open_gams_parity(
        contract={"closure": {"name": "simple_open_default"}},
        gdx_path=gdx_path,
    )

    assert comparison.passed is True
    assert comparison.active_closure_match is True
    assert comparison.level_mismatches == 0
    assert comparison.parameter_mismatches == 0
    assert comparison.residual_mismatches == 0


def test_compare_simple_open_gams_parity_detects_mismatch(monkeypatch: Any, tmp_path: Path) -> None:
    gdx_path = tmp_path / "simple_open.gdx"
    gdx_path.write_bytes(b"dummy")

    monkeypatch.setattr(
        "equilibria.templates.simple_open_parity_pipeline.load_simple_open_gams_reference",
        lambda path: SimpleOpenGAMSReference(
            closure_names=("wrong_closure",),
            benchmark={
                "VA": 1.0,
                "INT": 0.5,
                "X": 1.0,
                "D": 1.0,
                "E": 1.0,
                "ER": 1.0,
                "PFX": 1.0,
                "CAB": 1.0,
                "FSAV": 1.0,
            },
            level={
                "VA": 1.1,
                "INT": 0.5,
                "X": 1.0,
                "D": 1.0,
                "E": 1.0,
                "ER": 1.0,
                "PFX": 1.0,
                "CAB": 1.0,
                "FSAV": 1.0,
            },
            residual={"EQ_VA": 1e-3, "EQ_INT": 0.0, "EQ_CET": 0.0},
                calib={
                    "alpha_va": 0.4,
                    "rho_va": 0.75,
                    "a_int": 0.5,
                    "b_ext": 0.1,
                    "theta_cet": 0.6,
                    "phi_cet": 1.2,
                    "closure_code": 101.0,
                    "modelstat": 7.0,
                    "solvestat": 1.0,
                },
            ),
    )

    comparison = compare_simple_open_gams_parity(
        contract={"closure": {"name": "simple_open_default"}},
        gdx_path=gdx_path,
        abs_tol=1e-9,
    )

    assert comparison.passed is False
    assert comparison.active_closure_match is False
    assert comparison.level_mismatches == 1
    assert comparison.residual_mismatches == 1
    assert comparison.modelstat == 7.0
