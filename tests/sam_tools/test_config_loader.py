from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from equilibria.sam_tools.config_loader import load_workflow_config


def test_load_workflow_config_resolves_paths_and_aliases(tmp_path: Path) -> None:
    cfg_file = tmp_path / "workflow.yaml"
    payload = {
        "metadata": {"country": "cri"},
        "input": {
            "path": "data/raw.xlsx",
            "format": "ieem_raw",
            "mapping_path": "data/mapping.xlsx",
            "options": {
                "sheet_name": "MCS2016",
                "extra_path": "data/extra.txt",
            },
        },
        "output": {
            "path": "results/out.gdx",
            "symbol": " sam_out ",
        },
        "report_path": "results/report.json",
        "transforms": [{"op": "aggregate_mapping"}],
    }
    cfg_file.write_text(yaml.safe_dump(payload), encoding="utf-8")

    cfg = load_workflow_config(cfg_file)
    assert cfg.name == "workflow"
    assert cfg.country == "cri"
    assert cfg.input_format == "ieem_raw_excel"
    assert cfg.output_format == "gdx"
    assert cfg.output_symbol == "sam_out"
    assert cfg.input_path == (tmp_path / "data/raw.xlsx").resolve()
    assert cfg.output_path == (tmp_path / "results/out.gdx").resolve()
    assert cfg.report_path == (tmp_path / "results/report.json").resolve()
    assert cfg.input_options["sheet_name"] == "MCS2016"
    assert cfg.input_options["mapping_path"] == (tmp_path / "data/mapping.xlsx").resolve()
    assert cfg.input_options["extra_path"] == (tmp_path / "data/extra.txt").resolve()
    assert cfg.transforms == [{"op": "aggregate_mapping"}]


def test_load_workflow_config_rejects_invalid_options_type(tmp_path: Path) -> None:
    cfg_file = tmp_path / "invalid.yaml"
    payload = {
        "metadata": {"name": "bad"},
        "input": {
            "path": "data/raw.xlsx",
            "format": "excel",
            "options": ["not-a-mapping"],
        },
        "output": {"path": "results/out.xlsx", "format": "excel"},
        "transforms": [],
    }
    cfg_file.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="input.options must be a mapping"):
        load_workflow_config(cfg_file)
