"""Every model door keeps the door contract (ADR-0012, ``equilibria.core.door``)."""

from __future__ import annotations

import sys
import types

import pytest

from equilibria.core.door import DOORS, DoorSpec, check_door


@pytest.mark.parametrize("spec", DOORS, ids=lambda s: s.module)
def test_every_door_keeps_the_contract(spec: DoorSpec) -> None:
    assert check_door(spec) == []


def test_gtap_is_a_door() -> None:
    assert DoorSpec("equilibria.gtap", "GTAP", "gtap_static", "gtap_default") in DOORS


def test_check_door_catches_a_broken_door(monkeypatch: pytest.MonkeyPatch) -> None:
    """A door missing ``results`` and with a ``ClosureConfig`` that is not a
    ``ModelClosureConfig`` breaks two rules, and the check names both."""
    import equilibria.gtap as good

    fake = types.ModuleType("equilibria_fake_door")
    for name in ("load", "GTAP", "gtap_static", "Exercise"):
        setattr(fake, name, getattr(good, name))

    class NotAClosure:
        @classmethod
        def gtap_default(cls, **changes):
            return cls()

    vars(fake).update(
        ClosureConfig=NotAClosure,
        __all__=["load", "GTAP", "gtap_static", "Exercise", "ClosureConfig"],
    )
    monkeypatch.setitem(sys.modules, "equilibria_fake_door", fake)
    msgs = check_door(
        DoorSpec("equilibria_fake_door", "GTAP", "gtap_static", "gtap_default")
    )
    assert any("results" in m for m in msgs), msgs
    assert any("ModelClosureConfig" in m for m in msgs), msgs


# ── every template has a door ───────────────────────────────────────────────


def _write(root, rel: str, text: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _fake_tree(root):
    """A package with one real template, a Protocol stub, an abstract base that
    raises, and a subclass of ModelTemplate that inherits its recipe."""
    _write(root, "pkg/__init__.py", "")
    _write(
        root,
        "pkg/real.py",
        "class Real:\n"
        "    def build(self, periods, overwrite=()):\n        return None\n"
        "    def default_recipe(self):\n        return make_recipe()\n",
    )
    _write(
        root,
        "pkg/stubs.py",
        "class Template(Protocol):\n    def default_recipe(self): ...\n\n"
        "class Base(ABC):\n"
        "    def default_recipe(self):\n"
        '        raise NotImplementedError("no recipe")\n',
    )
    _write(root, "pkg/sub.py", "class Child(ModelTemplate):\n    pass\n")


def test_find_templates_sees_real_templates_only(tmp_path) -> None:
    from equilibria.core.door import find_templates

    _fake_tree(tmp_path)
    assert find_templates(tmp_path / "pkg", "pkg") == {"pkg.real.Real", "pkg.sub.Child"}


def test_find_templates_in_the_package() -> None:
    from equilibria.core.door import find_templates, package_root

    assert find_templates(package_root()) == {
        "equilibria.templates.gtap.template.GTAP",
        "equilibria.templates.simple_open.template.SimpleOpenEconomy",
    }


def test_every_template_has_a_door_or_is_pending() -> None:
    from equilibria.core.door import coverage_problems

    assert coverage_problems() == []


def test_a_template_without_a_door_fails(tmp_path) -> None:
    from equilibria.core.door import coverage_problems

    _fake_tree(tmp_path)
    msgs = coverage_problems(root=tmp_path / "pkg", package="pkg", doors=(), pending={})
    assert any("pkg.real.Real" in m and "no door" in m for m in msgs), msgs
    assert any("pkg.sub.Child" in m for m in msgs), msgs


def test_a_stale_pending_entry_fails(tmp_path) -> None:
    from equilibria.core.door import coverage_problems

    _fake_tree(tmp_path)
    pending = {"pkg.real.Real": "x", "pkg.sub.Child": "x", "pkg.gone.Gone": "x"}
    msgs = coverage_problems(
        root=tmp_path / "pkg", package="pkg", doors=(), pending=pending
    )
    assert msgs == ["pkg.gone.Gone is in PENDING but is not a template class"], msgs


def test_a_pending_template_that_got_a_door_fails() -> None:
    from equilibria.core.door import DOORS, PENDING, coverage_problems

    pending = {**PENDING, "equilibria.templates.gtap.template.GTAP": "x"}
    msgs = coverage_problems(doors=DOORS, pending=pending)
    assert any("GTAP" in m and "has a door" in m for m in msgs), msgs


def test_every_pending_entry_says_why() -> None:
    from equilibria.core.door import PENDING

    assert all(len(why) > 20 for why in PENDING.values()), PENDING


def test_every_door_has_an_api_page() -> None:
    """A door is the public surface: the API reference documents each one."""
    from pathlib import Path

    from equilibria.core.door import DOORS

    api = Path(__file__).resolve().parents[2] / "docs" / "site" / "api"
    pages = "\n".join(p.read_text(encoding="utf-8") for p in api.glob("*.md"))
    for spec in DOORS:
        assert f".. automodule:: {spec.module}\n" in pages, spec.module
