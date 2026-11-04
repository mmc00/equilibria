"""Where the GDX reader tests find their specimens (data/gdx_tests, ADR-0010)."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SPECIMENS = ROOT / "data" / "gdx_tests" / "specimens"

# File name -> the folder of the generator that writes it. Explicit, because
# gdx_fixtures/ also writes a multidim_test.gdx that is not the one tests read.
_FOLDER = {
    "test_5d.gdx": "5d_test",
    "test_6d.gdx": "6d_test",
    **{
        f"test_{n}.gdx": "compression_tests"
        for n in (
            "arithmetic",
            "arithmetic_compressed",
            "geometric",
            "geometric_compressed",
            "growth",
            "mixed",
            "small_arithmetic",
            "small_geometric",
            "sparse",
        )
    },
    "sparse_test.gdx": "gdx_fixtures",
    "variables_equations_test.gdx": "gdx_fixtures",
    "set_2d_minimal.gdx": "minimal_set",
    "set_2d_multi_col.gdx": "multi_col_set",
    "multidim_test.gdx": "multidim_test",
    **{
        f"{n}.gdx": "set_tests"
        for n in (
            "set_2d_cartesian",
            "set_2d_empty",
            "set_2d_full",
            "set_2d_sparse",
            "set_2d_with_param",
            "set_2d_with_text",
            "set_3d",
            "set_4d",
        )
    },
    "simple_test.gdx": "simple",
    "bigskip_test.gdx": "no_generator",
    "patterns_test.gdx": "no_generator",
    "skip_test.gdx": "no_generator",
}


def specimen(name: str) -> Path:
    """The path of one GDX specimen; an unknown name is an error, not a skip."""
    return SPECIMENS / _FOLDER[name] / name
