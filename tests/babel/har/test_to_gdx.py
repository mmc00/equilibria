"""Smoke tests for the HAR → three-GDX wrapper.

Skipped automatically when the GTAP Standard 7 reference dataset is not
available locally.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from equilibria.babel.har import convert_har_to_gdx

GTAP_DIR = Path("/Users/marmol/proyectos2/cge_babel/standard_gtap_7")
SUFFIX = "-9x10"
GDXDUMP = Path(
    "/Library/Frameworks/GAMS.framework/Versions/48/Resources/gdxdump"
)

pytestmark = pytest.mark.skipif(
    not (GTAP_DIR / f"basedata{SUFFIX}.har").is_file(),
    reason="GTAP 9x10 HAR dataset not available locally",
)


def test_convert_har_to_gdx_writes_three_files(tmp_path: Path):
    out = convert_har_to_gdx(
        GTAP_DIR, tmp_path, base_name="9x10py", suffix=SUFFIX
    )
    assert set(out.keys()) == {"sets", "dat", "prm"}
    for path in out.values():
        assert path.is_file()
        assert path.stat().st_size > 0


@pytest.mark.skipif(not GDXDUMP.is_file(), reason="GAMS gdxdump not installed")
def test_convert_har_to_gdx_files_are_gams_readable(tmp_path: Path):
    """Each output GDX must be readable by the official GAMS gdxdump."""
    out = convert_har_to_gdx(
        GTAP_DIR, tmp_path, base_name="9x10py", suffix=SUFFIX
    )
    for path in out.values():
        result = subprocess.run(
            [str(GDXDUMP), str(path), "Symbols"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, (
            f"gdxdump failed for {path.name}: {result.stderr}"
        )
        assert "Problem reading" not in result.stdout


def test_convert_har_to_gdx_sets_have_expected_cardinality(tmp_path: Path):
    """Sets.gdx must contain ACTS=9, COMM=10, REG=10, ENDW=5."""
    out = convert_har_to_gdx(
        GTAP_DIR, tmp_path, base_name="9x10py", suffix=SUFFIX
    )
    if not GDXDUMP.is_file():
        pytest.skip("gdxdump unavailable")

    text = subprocess.run(
        [str(GDXDUMP), str(out["sets"]), "Symbols"],
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout

    expected = {"ACTS": 9, "COMM": 10, "REG": 10, "ENDW": 5, "MARG": 1}
    for name, count in expected.items():
        assert any(
            line.strip().split()[1:] and line.strip().split()[1] == name
            and int(line.strip().split()[4]) == count
            for line in text.splitlines()
            if name in line
        ), f"Set {name} should have {count} records in:\n{text}"


def test_convert_har_to_gdx_dat_has_core_sam_flows(tmp_path: Path):
    """Dat.gdx must contain VDFB(900), EVFB(290), MAKB(100), VCIF(1000)."""
    out = convert_har_to_gdx(
        GTAP_DIR, tmp_path, base_name="9x10py", suffix=SUFFIX
    )
    if not GDXDUMP.is_file():
        pytest.skip("gdxdump unavailable")

    text = subprocess.run(
        [str(GDXDUMP), str(out["dat"]), "Symbols"],
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout

    for name, count in (
        ("VDFB", 900),
        ("EVFB", 290),
        ("MAKB", 100),
        ("VCIF", 1000),
        ("VTWR", 700),
    ):
        line = next((l for l in text.splitlines() if name in l), None)
        assert line is not None, f"{name} missing in Dat.gdx symbols"
        parts = line.split()
        # Layout: "<idx> <Name> <dim> <Type> <Records> [text]"
        assert int(parts[4]) == count, (
            f"{name} expected {count} records, got {parts[4]}"
        )
