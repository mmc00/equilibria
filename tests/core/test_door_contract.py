"""Every model door keeps the door contract (ADR-0012, ``equilibria.core.door``)."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from equilibria.core.door import DOORS, DoorSpec, check_door

ROOT = Path(__file__).resolve().parents[2]


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
    # the ways a template can hide (MAR-249 review)
    _write(
        root,
        "pkg/hidden.py",
        "from pkg.real import Real\n"
        "from equilibria.templates.base import ModelTemplate as Base\n"
        "from . import real as r\n"
        "class FromTemplate(Real):\n    pass\n"
        "class ViaModule(r.Real):\n    pass\n"
        "class Aliased(Base):\n    pass\n"
        "class Grandchild(FromTemplate):\n    pass\n"
        "class Async:\n    async def default_recipe(self):\n        return 1\n"
        "class Assigned:\n    default_recipe = staticmethod(lambda: 1)\n"
        "try:\n    class InTry:\n        def default_recipe(self):\n            return 1\n"
        "except ImportError:\n    pass\n"
        "class NotOne(Exception):\n    pass\n",
    )


def test_find_templates_sees_real_templates_only(tmp_path) -> None:
    from equilibria.core.door import find_templates

    _fake_tree(tmp_path)
    assert find_templates(tmp_path / "pkg", "pkg") == {
        "pkg.real.Real",
        "pkg.sub.Child",
        "pkg.hidden.FromTemplate",
        "pkg.hidden.ViaModule",
        "pkg.hidden.Aliased",
        "pkg.hidden.Grandchild",
        "pkg.hidden.Async",
        "pkg.hidden.Assigned",
        "pkg.hidden.InTry",
    }


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
    assert any("pkg.hidden.FromTemplate" in m for m in msgs), msgs


def test_a_stale_pending_entry_fails(tmp_path) -> None:
    from equilibria.core.door import coverage_problems, find_templates

    _fake_tree(tmp_path)
    pending = dict.fromkeys(find_templates(tmp_path / "pkg", "pkg"), "x")
    pending["pkg.gone.Gone"] = "x"
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
    from equilibria.core.door import api_page_problems

    assert api_page_problems(ROOT / "docs" / "site" / "api") == []


def test_a_door_without_an_api_page_fails(tmp_path) -> None:
    from equilibria.core.door import DOORS, api_page_problems

    (tmp_path / "other.md").write_text(".. automodule:: equilibria.core\n")
    msgs = api_page_problems(tmp_path, DOORS)
    assert msgs == [
        f"{DOORS[0].module} has no API page (an `.. automodule:: {DOORS[0].module}` "
        f"in {tmp_path})"
    ], msgs


def test_stub_shapes_are_not_templates(tmp_path) -> None:
    """``pass``, ``return NotImplemented`` and ``@abstractmethod`` recipes are stubs: an
    abstract base written that way must not block commits."""
    from equilibria.core.door import find_templates

    _write(tmp_path, "pkg/__init__.py", "")
    _write(
        tmp_path,
        "pkg/bases.py",
        "class A:\n    def default_recipe(self):\n        '''doc'''\n        pass\n\n"
        "class B:\n    def default_recipe(self):\n        return NotImplemented\n\n"
        "class C(ABC):\n    @abstractmethod\n    def default_recipe(self):\n"
        "        return make()\n\n"
        "class D(ABC):\n    @abc.abstractmethod\n    def default_recipe(self):\n"
        "        return make()\n",
    )
    assert find_templates(tmp_path / "pkg", "pkg") == set()


def test_a_door_that_does_not_import_is_a_problem() -> None:
    from equilibria.core.door import DoorSpec, check_door, coverage_problems

    broken = DoorSpec("equilibria_no_such_door", "X", "x", "y")
    msgs = coverage_problems(doors=(broken,))
    assert any("equilibria_no_such_door: cannot import" in m for m in msgs), msgs
    assert check_door(broken) == [
        "equilibria_no_such_door: cannot import (No module named 'equilibria_no_such_door')"
    ]


def test_a_myst_automodule_fence_counts_as_a_page(tmp_path) -> None:
    from equilibria.core.door import DOORS, api_page_problems

    m = DOORS[0].module
    (tmp_path / "p.md").write_text(f"```{{automodule}} {m}\n```\n", encoding="utf-8")
    assert api_page_problems(tmp_path, DOORS) == []
