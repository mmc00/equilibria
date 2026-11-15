"""The model print of the saved Pyomo photos: code split into Pyomo-only and shared."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

from equilibria.blocks.gtap import fingerprint as fp

ROOT = fp._package_root()


def _params():
    return SimpleNamespace(benchmark=SimpleNamespace(vom={("USA", "agr"): 1.0}))


def _closure(**extra):
    return SimpleNamespace(model_dump=lambda: {"if_sub": True, **extra})


def _module(path: Path) -> str:
    rel = path.relative_to(ROOT.parent).with_suffix("")
    return ".".join(rel.parts).removesuffix(".__init__")


def test_pyomo_only_and_shared_partition_the_code_files():
    pyomo = set(fp.pyomo_only_files())
    shared = set(fp.shared_files())
    assert pyomo and shared
    assert not pyomo & shared
    family = set((ROOT / "family").rglob("*.py"))
    assert pyomo | shared == set(fp._code_files(ROOT)) | family


def test_code_digest_matches_the_cache_key_digest():
    # the caches' fingerprint() must not change: same hashing over the same files
    assert fp.code_digest(fp._code_files(ROOT)) == fp._code_digest()


def test_model_print_ignores_env_in_equality():
    a = fp.model_print(
        params=_params(), closure=_closure(), residual_region="ROW", linsolve="mumps"
    )
    b = fp.model_print(
        params=_params(), closure=_closure(), residual_region="ROW", linsolve="direct"
    )
    assert a is not None and b is not None
    assert a.env["linsolve"] == "mumps" and b.env["linsolve"] == "direct"
    assert a == b
    assert fp.ModelPrint.from_dict(a.to_dict()) == a


def test_model_print_sees_data_and_closure():
    a = fp.model_print(
        params=_params(), closure=_closure(), residual_region="ROW", linsolve="mumps"
    )
    other_closure = fp.model_print(
        params=_params(),
        closure=_closure(savf="capFlex"),
        residual_region="ROW",
        linsolve="mumps",
    )
    other_region = fp.model_print(
        params=_params(), closure=_closure(), residual_region="USA", linsolve="mumps"
    )
    assert a.closure != other_closure.closure
    assert a.closure != other_region.closure
    assert a.data == other_closure.data


def test_no_family_module_imports_pyomo_only_files():
    pyomo = {_module(p) for p in fp.pyomo_only_files()}
    sources = [
        *(ROOT / "family").rglob("*.py"),
        ROOT / "backends" / "family_backend.py",
    ]
    hits = []
    for src in sources:
        for node in ast.walk(ast.parse(src.read_text())):
            names = []
            if isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module] + [f"{node.module}.{a.name}" for a in node.names]
            elif isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            hits += [(src.name, n) for n in names if n in pyomo]
    assert hits == []
