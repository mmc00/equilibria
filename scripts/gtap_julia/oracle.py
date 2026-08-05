"""Run the Julia GTAPv7 model as the reference oracle for the gtap_julia port.

Shells out to `run_julia_oracle.jl` (calibrate -> base -> tariff shock) and
returns the paths of the dumped base/shock CSVs. The Julia model solves the same
economics as a levels NLP with IPOPT and converges to machine precision, so it is
the faithful cell-by-cell reference the port is validated against.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_JL_SCRIPT = _HERE / "run_julia_oracle.jl"
_JULIA = Path.home() / ".juliaup" / "bin" / "julia"
_PKG = Path.home() / "proyectos" / "GlobalTradeAnalysisProjectModelV7.jl"


def run_oracle(
    dataset: str = "sample",
    tariff_power: float = 1.10,
    out_dir: Path | str | None = None,
    timeout: int = 900,
) -> dict[str, Path]:
    """Run the Julia oracle and return {"base": <csv>, "shock": <csv>}.

    Args:
        dataset: "sample" (Julia's bundled 7x6) or a directory with
            gsdfdat.har / gsdfset.har / gsdfpar.har.
        tariff_power: import-tariff power applied to every bilateral tms
            (1.10 = +10%).
        out_dir: where to write the CSVs (default: alongside this script).
        timeout: seconds to allow (Julia precompiles on first run).
    """
    if not _JULIA.exists():
        raise FileNotFoundError(f"julia not found at {_JULIA}")
    out = Path(out_dir) if out_dir is not None else _HERE
    out.mkdir(parents=True, exist_ok=True)
    base_csv = out / f"julia_{_slug(dataset)}_base.csv"
    shock_csv = out / f"julia_{_slug(dataset)}_shock.csv"

    cmd = [
        str(_JULIA),
        f"--project={_PKG}",
        str(_JL_SCRIPT),
        dataset,
        str(tariff_power),
        str(base_csv),
        str(shock_csv),
    ]
    res = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout, cwd=str(_HERE)
    )
    if res.returncode != 0 or ">>> DONE" not in res.stdout:
        raise RuntimeError(
            f"Julia oracle failed (rc={res.returncode}).\n"
            f"stdout tail:\n{res.stdout[-2000:]}\n"
            f"stderr tail:\n{res.stderr[-2000:]}"
        )
    if not base_csv.exists() or not shock_csv.exists():
        raise RuntimeError("Julia oracle finished but CSVs are missing")
    return {"base": base_csv, "shock": shock_csv}


def _slug(dataset: str) -> str:
    return "sample" if dataset == "sample" else Path(dataset).name
