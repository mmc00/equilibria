import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "gtap"))

from cascade_layers import LAYER_SPECS, build_cmd, layer_by_name


GDX = Path("/ref/out.gdx")


def _argv(name, dataset, period, gdx=GDX):
    return build_cmd(layer_by_name(name), dataset, period, gdx)


def test_six_subprocess_layers_in_diagnostic_order():
    names = [l.name for l in LAYER_SPECS]
    assert names == ["mcp_pairing", "nl_compare", "calibration",
                     "validate_reference", "probe_seed", "drift_test"]


def test_mcp_pairing_argv():
    argv = _argv("mcp_pairing", "gtap7_3x3", "shock")
    assert argv[-4:] == ["--dataset", "gtap7_3x3", "--period", "shock"][:0] + \
        ["--dataset", "gtap7_3x3", "--period", "shock", "--apply-closure"][-4:]
    assert "diff_mcp_pairing.py" in argv[1]
    assert "--apply-closure" in argv


def test_nl_compare_uses_phase_and_skip_gams():
    argv = _argv("nl_compare", "gtap7_3x3", "shock")
    assert "--phase" in argv and "shock" in argv and "--skip-gams" in argv
    assert "--period" not in argv  # nl_compare takes --phase, not --period


def test_calibration_pins_benchmark_period_check():
    argv = _argv("calibration", "gtap7_3x3", "shock")
    i = argv.index("--period")
    assert argv[i + 1] == "check"  # benchmark period, not the shock axis
    assert "--gdx" in argv and str(GDX) in argv


def test_probe_seed_argv():
    argv = _argv("probe_seed", "gtap7_3x3", "shock")
    for tok in ("--template", "gtap", "--scenario", "altertax_shock",
                "--seed-gams", "shock", "--gdx-ref", "--residuals"):
        assert tok in argv


def test_drift_has_no_base():
    assert build_cmd(layer_by_name("drift_test"), "9x10", "base", GDX) is None


def test_build_cmd_returns_none_for_unsupported_period():
    # bundle family has no 'check'; nl_compare itself supports it, but the
    # orchestrator only builds for available periods — here verify the per-layer
    # guard: drift has periods (check, shock) so 'base' -> None.
    assert build_cmd(layer_by_name("drift_test"), "gtap7_3x3", "base", GDX) is None
