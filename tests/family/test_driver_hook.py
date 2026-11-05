"""EQUILIBRIA_GTAP_FAMILY_EVAL: the newton_tr driver's NLP is wrapped only when asked."""

import pytest

pytest.importorskip("jax")
pytest.importorskip("pyomo.contrib.pynumero.interfaces.pyomo_nlp")

from tests.family._data import load_params  # noqa: E402
from tests.family._oracle import comp_stat_closure  # noqa: E402

ENV = "EQUILIBRIA_GTAP_FAMILY_EVAL"


def test_off_by_default(monkeypatch):
    from equilibria.family.driver_hook import maybe_wrap

    monkeypatch.delenv(ENV, raising=False)
    nlp = object()
    assert maybe_wrap(nlp, None, comp_stat_closure(), "newton_tr") is nlp


def test_wraps_newton_tr_only(monkeypatch):
    from equilibria.family.driver_hook import maybe_wrap

    monkeypatch.setenv(ENV, "1")
    nlp = object()
    assert maybe_wrap(nlp, None, comp_stat_closure(), "krylov") is nlp


def test_ifsub1_raises(monkeypatch):
    from equilibria.family.driver_hook import maybe_wrap

    monkeypatch.setenv(ENV, "1")
    closure = comp_stat_closure().model_copy(update={"if_sub": True})
    with pytest.raises(NotImplementedError, match="ifSUB"):
        maybe_wrap(object(), None, closure, "newton_tr")


def test_wraps_and_reuses_the_recording(monkeypatch):
    from equilibria.family import driver_hook
    from equilibria.family.hybrid import HybridNLP
    from tests.family.test_hybrid_nlp import _frozen_nlp

    monkeypatch.setenv(ENV, "1")
    monkeypatch.setattr(driver_hook, "_CACHE", {})
    p = load_params("gtap7_3x3")
    closure = comp_stat_closure()
    _, nlp_c = _frozen_nlp("check")
    _, nlp_s = _frozen_nlp("shock")
    h_c = driver_hook.maybe_wrap(nlp_c, p, closure, "newton_tr")
    h_s = driver_hook.maybe_wrap(nlp_s, p, closure, "newton_tr")
    assert isinstance(h_c, HybridNLP) and isinstance(h_s, HybridNLP)
    assert (h_c.period, h_s.period) == ("check", "shock")
    assert len(driver_hook._CACHE) == 1
