"""Manifest model for strict-gams baseline compatibility."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def file_sha256(path: Path | str) -> str:
    """Compute SHA-256 digest for a file."""
    file_path = Path(path)
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _sum_values(values: Any) -> float:
    if isinstance(values, dict):
        return float(sum(float(v) for v in values.values()))
    return float(values or 0.0)


def compute_state_anchors(state: Any) -> dict[str, float]:
    """Compute stable calibration anchors from `PEPModelState`."""
    exd_total = _sum_values(state.trade.get("EXDO", {}))
    im_total = _sum_values(state.trade.get("IMO", {}))
    return {
        "GDP_BP": float(state.gdp.get("GDP_BPO", 0.0)),
        "EXD_TOTAL": exd_total,
        "IM_TOTAL": im_total,
        "TRADE_BALANCE": exd_total - im_total,
        "CAB": float(state.income.get("CABO", 0.0)),
    }


@dataclass
class BaselineManifest:
    """Sidecar manifest that binds baseline levels to source inputs."""

    schema_version: str
    generated_at: str
    gams_slice: str
    results_gdx: str
    results_gdx_sha256: str
    sam_file: str | None
    sam_sha256: str | None
    val_par_file: str | None
    val_par_sha256: str | None
    set_sizes: dict[str, int]
    state_anchors: dict[str, float]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save_json(self, path: Path | str) -> None:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(self.to_dict(), indent=2))


def load_baseline_manifest(path: Path | str) -> BaselineManifest:
    """Load manifest JSON from disk."""
    data = json.loads(Path(path).read_text())
    return BaselineManifest(
        schema_version=str(data.get("schema_version", "")),
        generated_at=str(data.get("generated_at", "")),
        gams_slice=str(data.get("gams_slice", "")).lower(),
        results_gdx=str(data.get("results_gdx", "")),
        results_gdx_sha256=str(data.get("results_gdx_sha256", "")),
        sam_file=data.get("sam_file"),
        sam_sha256=data.get("sam_sha256"),
        val_par_file=data.get("val_par_file"),
        val_par_sha256=data.get("val_par_sha256"),
        set_sizes={str(k): int(v) for k, v in dict(data.get("set_sizes", {})).items()},
        state_anchors={str(k): float(v) for k, v in dict(data.get("state_anchors", {})).items()},
        metadata=dict(data.get("metadata", {})),
    )


def build_baseline_manifest(
    *,
    state: Any,
    results_gdx: Path | str,
    gams_slice: str,
    sam_file: Path | str | None = None,
    val_par_file: Path | str | None = None,
    metadata: dict[str, Any] | None = None,
) -> BaselineManifest:
    """Build manifest from calibrated state and source files."""
    results_path = Path(results_gdx)
    sam_path = Path(sam_file) if sam_file else None
    val_path = Path(val_par_file) if val_par_file else None

    return BaselineManifest(
        schema_version="pep_baseline_manifest/v1",
        generated_at=datetime.now(timezone.utc).isoformat(),
        gams_slice=str(gams_slice).lower(),
        results_gdx=str(results_path),
        results_gdx_sha256=file_sha256(results_path) if results_path.exists() else "",
        sam_file=str(sam_path) if sam_path else None,
        sam_sha256=file_sha256(sam_path) if sam_path and sam_path.exists() else None,
        val_par_file=str(val_path) if val_path else None,
        val_par_sha256=file_sha256(val_path) if val_path and val_path.exists() else None,
        set_sizes={name: len(elements) for name, elements in state.sets.items()},
        state_anchors=compute_state_anchors(state),
        metadata=metadata or {},
    )
