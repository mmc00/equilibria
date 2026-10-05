"""Fixtures compartidas de tests/templates/gtap."""

import pytest
from tests.templates.gtap._nus333 import closure, nus333_params


@pytest.fixture(scope="module")
def nus333_mp_model():
    """Modelo de 3 periodos de nus333, sin resolver (instrumentos fijos)."""
    from equilibria.templates.gtap.gtap_block_model import build_block_model

    p = nus333_params()
    gc = closure("capFix")
    m, _ = build_block_model(p, p.sets, gc, "ROW")
    return m
