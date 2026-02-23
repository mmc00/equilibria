"""Tests for PEP set include parsing and consistency checks."""

from __future__ import annotations

from pathlib import Path

from equilibria.core import Set
from equilibria.templates.pep_sets import PEPSetManager


def _build_manager() -> PEPSetManager:
    manager = PEPSetManager()
    manager.sets = {
        "J": Set(name="J", elements=("agr", "ser"), description="Sectors"),
        "H": Set(name="H", elements=("hrp", "hur"), description="Households"),
    }
    return manager


def test_validate_consistency_generated_include_format(tmp_path: Path) -> None:
    manager = _build_manager()
    include = tmp_path / "sets_definition.inc"
    include.write_text(
        "\n".join(
            [
                "* Generated",
                "J Sectors /",
                "  agr",
                "  ser",
                "/",
                "",
                "H Households /",
                "  hrp",
                "  hur",
                "/",
            ]
        ),
        encoding="utf-8",
    )

    assert manager.validate_consistency(include)


def test_validate_consistency_set_block_format(tmp_path: Path) -> None:
    manager = _build_manager()
    include = tmp_path / "dynamic_sets.inc"
    include.write_text(
        "\n".join(
            [
                "SET",
                "J All industries dynamic",
                "/",
                " ser",
                " agr",
                "/",
                "",
                "H(AG) Households dynamic",
                "/",
                " hur",
                " hrp",
                "/",
                ";",
            ]
        ),
        encoding="utf-8",
    )

    assert manager.validate_consistency(include)


def test_validate_consistency_detects_membership_mismatch(tmp_path: Path) -> None:
    manager = _build_manager()
    include = tmp_path / "sets_definition.inc"
    include.write_text(
        "\n".join(
            [
                "J Sectors /",
                "  agr",
                "  ind",
                "/",
                "",
                "H Households /",
                "  hrp",
                "  hur",
                "/",
            ]
        ),
        encoding="utf-8",
    )

    assert not manager.validate_consistency(include)
