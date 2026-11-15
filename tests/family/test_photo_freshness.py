"""The committed Pyomo photos: present, intact, made with MUMPS, and not stale."""

from __future__ import annotations

import gzip
import shutil

import pytest

from equilibria.blocks.gtap import fingerprint as fp
from equilibria.family import photos

VARIANTS = [pytest.param(v, id=f"{v[0]}-ifsub{v[1]}-{v[2]}") for v in photos.VARIANTS]


@pytest.mark.parametrize("variant", VARIANTS)
def test_every_variant_is_fresh_or_shared_changed(variant):
    ds, ifsub, savf, mode, _ = variant
    state = photos.freshness(ds, ifsub, savf, mode)
    assert state in ("fresh", "shared_changed"), (
        f"{ds} ifsub={ifsub} {savf}: the photo is stale (Pyomo-only code, data or "
        f"closure changed); run {photos.REGEN} --dataset {ds}"
    )


def _copy(tmp_path, ds="gtap7_3x3", ifsub=1, savf="capFix"):
    src = photos.photo_dir(ds, ifsub, savf)
    dst = photos.photo_dir(ds, ifsub, savf, root=tmp_path)
    shutil.copytree(src, dst)
    return dst


def test_missing_photo_is_an_error_not_a_skip(tmp_path):
    with pytest.raises(photos.PhotoError, match="regen_photos.py --dataset gtap7_3x3"):
        photos.unpacked("gtap7_3x3", root=tmp_path, cache_root=tmp_path / "c")


def test_truncated_photo_fails_its_hash(tmp_path):
    dst = _copy(tmp_path)
    with (dst / "final.pkl.gz").open("r+b") as fh:
        fh.truncate(100)
    with pytest.raises(photos.PhotoError, match="final.pkl.gz"):
        photos.unpacked("gtap7_3x3", root=tmp_path, cache_root=tmp_path / "c")


def test_linsolve_mismatch_is_named():
    with pytest.raises(photos.PhotoError, match="linsolve"):
        photos.unpacked("gtap7_3x3", linsolve="direct")


def test_unpacked_has_the_plain_files(tmp_path):
    _copy(tmp_path)
    out = photos.unpacked("gtap7_3x3", root=tmp_path, cache_root=tmp_path / "c")
    assert (out / "final.pkl").is_file() and (out / "001_A_base.pkl").is_file()
    assert not list(out.glob("*.gz"))
    packed = photos.photo_dir("gtap7_3x3", root=tmp_path) / "final.pkl.gz"
    assert (out / "final.pkl").read_bytes() == gzip.decompress(packed.read_bytes())


@pytest.mark.parametrize("variant", VARIANTS)
def test_committed_names_are_lowercase(variant):
    ds, ifsub, savf, mode, _ = variant
    folder = photos.photo_dir(ds, ifsub, savf, mode)
    assert [f.name for f in folder.iterdir() if f.name != f.name.lower()] == []


@pytest.mark.parametrize(
    "plain", ["001_A_base.pkl", "002_A2_check.pkl", "005_S_shock.json", "final.pkl"]
)
def test_stored_name_round_trips(plain):
    stored = photos.stored_name(plain)
    assert stored == stored.lower()
    assert photos.plain_name(stored) == plain


def test_pyomo_only_edit_makes_it_stale(monkeypatch):
    real = fp.code_digest
    pyomo = set(fp.pyomo_only_files())
    monkeypatch.setattr(
        fp,
        "code_digest",
        lambda files: "edited" if set(files) == pyomo else real(files),
    )
    assert photos.freshness("gtap7_3x3", 1, "capFix", "gtap") == "stale"


def test_shared_edit_is_shared_changed(monkeypatch):
    real = fp.code_digest
    shared = set(fp.shared_files())
    monkeypatch.setattr(
        fp,
        "code_digest",
        lambda files: "edited" if set(files) == shared else real(files),
    )
    assert photos.freshness("gtap7_3x3", 1, "capFix", "gtap") == "shared_changed"
