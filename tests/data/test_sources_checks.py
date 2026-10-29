"""check_tree(): each check of ADR-0010 can fail on its own."""

from __future__ import annotations

from pathlib import Path

from equilibria.sources import check_tree, sha256_file
from tests.data._records import result_record, write_record


def _clean_tree(root: Path) -> tuple[Path, Path]:
    """A dataset folder (kind data) with one result folder that was made from it."""
    ds = root / "gtap7" / "gtap7_3x3"
    ds.mkdir(parents=True)
    (ds / "basedata.har").write_bytes(b"data")
    write_record(
        ds,
        {
            "kind": "data",
            "model": "gtap7",
            "dataset": "gtap7_3x3",
            "source": "GTAP 3x3 aggregation",
            "license": "public access",
            "outputs": {"basedata.har": sha256_file(ds / "basedata.har")},
            "status": "verified",
        },
    )
    res = ds / "gams_ifsub0"
    res.mkdir()
    (res / "base.nl").write_bytes(b"nl")
    write_record(
        res,
        result_record(
            generated_from={
                "data": {
                    "path": "gtap7/gtap7_3x3",
                    "sha256": {"basedata.har": sha256_file(ds / "basedata.har")},
                }
            },
            outputs={"base.nl": sha256_file(res / "base.nl")},
        ),
    )
    (root / "README.md").write_text("map")
    return ds, res


def test_clean_tree_has_no_messages(tmp_path: Path) -> None:
    _clean_tree(tmp_path)
    assert check_tree(tmp_path) == []


def test_file_without_source_is_reported(tmp_path: Path) -> None:
    _clean_tree(tmp_path)
    (tmp_path / "loose").mkdir()
    (tmp_path / "loose" / "x.gdx").write_bytes(b"x")
    msgs = check_tree(tmp_path)
    assert len(msgs) == 1 and "loose/x.gdx" in msgs[0]


def test_changed_input_is_reported(tmp_path: Path) -> None:
    ds, _ = _clean_tree(tmp_path)
    (ds / "basedata.har").write_bytes(b"edited")
    msgs = check_tree(tmp_path)
    # The dataset's own output hash and the result's input hash both catch it.
    assert any("gams_ifsub0" in m and "basedata.har" in m for m in msgs)


def test_edited_output_is_reported(tmp_path: Path) -> None:
    _, res = _clean_tree(tmp_path)
    (res / "base.nl").write_bytes(b"edited by hand")
    msgs = check_tree(tmp_path)
    assert len(msgs) == 1 and "base.nl" in msgs[0]


def test_extra_output_file_is_reported(tmp_path: Path) -> None:
    _, res = _clean_tree(tmp_path)
    (res / "shock.nl").write_bytes(b"not recorded")
    msgs = check_tree(tmp_path)
    assert len(msgs) == 1 and "shock.nl" in msgs[0]


def test_pending_is_reported(tmp_path: Path) -> None:
    _, res = _clean_tree(tmp_path)
    text = (
        (res / "source.yaml").read_text().replace("status: verified", "status: pending")
    )
    (res / "source.yaml").write_text(text)
    assert any("pending" in m for m in check_tree(tmp_path))
    assert check_tree(tmp_path, allow_pending=True) == []
