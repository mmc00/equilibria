"""Old-key seed files are pruned once abandoned.

The seed cache lives in the user's home and is shared by every worktree, so a
branch still on the old key keeps writing ``seed-*.json``.  Only old-key files
untouched for 30 days are removed; a recent one may belong to a live branch.
"""

import os
import time

from equilibria.blocks.gtap import seed_cache

_DAY = 86400


def _touch(path, age_days):
    path.write_text("{}")
    t = time.time() - age_days * _DAY
    os.utime(path, (t, t))


def test_prune_removes_only_old_key_files_past_the_age(tmp_path, monkeypatch):
    monkeypatch.setenv("EQUILIBRIA_SEED_CACHE", str(tmp_path))
    monkeypatch.delenv("EQUILIBRIA_SEED_CACHE_DISABLE", raising=False)
    old_stale = tmp_path / "seed-aaaa.json"
    old_recent = tmp_path / "seed-bbbb.json"
    new_stale = tmp_path / f"{seed_cache.KEY_PREFIX}-cccc.json"
    other = tmp_path / "notes.txt"
    _touch(old_stale, 31)
    _touch(old_recent, 5)
    _touch(new_stale, 400)
    _touch(other, 400)

    removed = seed_cache.prune_old_keys()

    assert removed == [old_stale.name]
    assert not old_stale.exists()
    assert old_recent.exists(), "a recent old-key file may belong to a live branch"
    assert new_stale.exists(), "current-key files are never pruned"
    assert other.exists(), "files that are not seeds are never touched"


def test_save_prunes(tmp_path, monkeypatch):
    monkeypatch.setenv("EQUILIBRIA_SEED_CACHE", str(tmp_path))
    monkeypatch.delenv("EQUILIBRIA_SEED_CACHE_DISABLE", raising=False)
    old_stale = tmp_path / "seed-aaaa.json"
    _touch(old_stale, 60)
    seed_cache.save(f"{seed_cache.KEY_PREFIX}-x", {"pf": {("USA",): 1.0}})
    assert not old_stale.exists()


def test_disabled_cache_prunes_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv("EQUILIBRIA_SEED_CACHE", str(tmp_path))
    monkeypatch.setenv("EQUILIBRIA_SEED_CACHE_DISABLE", "1")
    old_stale = tmp_path / "seed-aaaa.json"
    _touch(old_stale, 60)
    assert seed_cache.prune_old_keys() == []
    assert old_stale.exists()


def test_seed_key_uses_the_new_prefix():
    from types import SimpleNamespace

    class _C:
        def model_dump(self):
            return {"if_sub": False}

    p = SimpleNamespace(
        benchmark=SimpleNamespace(evfb={("USA",): 1.0}),
        taxes=None,
        elasticities=None,
        shifts=None,
    )
    key = seed_cache.cache_key("gtap-3x2", _C(), "ROW", p)
    assert key is not None and key.startswith(f"{seed_cache.KEY_PREFIX}-")
