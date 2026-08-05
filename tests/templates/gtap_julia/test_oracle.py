"""Task 1: the Julia oracle harness runs end-to-end and dumps base/shock CSV."""

from pathlib import Path

import pytest
from scripts.gtap_julia.oracle import run_oracle


def _cell(lines: list[str], key: str) -> float | None:
    for ln in lines:
        if ln.startswith(key):
            return float(ln.rsplit(",", 1)[1])
    return None


@pytest.mark.slow
def test_oracle_sample_runs_and_dumps(tmp_path):
    out = run_oracle(dataset="sample", tariff_power=1.10, out_dir=tmp_path)
    base = Path(out["base"]).read_text().splitlines()
    shock = Path(out["shock"]).read_text().splitlines()
    assert len(base) > 500 and len(shock) > 500

    # a known cell moves under the tariff (land use in crops falls)
    b = _cell(base, "qfe,land,crops,")
    s = _cell(shock, "qfe,land,crops,")
    assert b is not None and s is not None
    assert abs(s / b - 1.0) > 1e-4
