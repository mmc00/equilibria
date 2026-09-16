from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "gtap" / "check_parity_gates_stamp.py"


def _load_module() -> Any:
    spec = importlib.util.spec_from_file_location("check_parity_gates_stamp", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_input_trees_cubre_los_bloques_gtap() -> None:
    """Editar blocks/gtap debe invalidar el stamp: es el oraculo contra GEMPACK.

    test_gtap7_gempack_parity mide el camino de bloques (build_block_model con
    base_calibrated=True). Si blocks/gtap no esta en INPUT_TREES, cambiar una
    ecuacion ahi no invalida el stamp y el hook block_push_without_gates deja
    pasar el push sin re-correr los gates.
    """
    module = _load_module()

    assert "src/equilibria/blocks/gtap" in module.INPUT_TREES, (
        "blocks/gtap no esta vigilado: un cambio de ecuacion en bloques no "
        "invalidaria el stamp y el hook dejaria pasar el push sin gates"
    )


def test_input_trees_cubre_ambos_caminos() -> None:
    """Los 5 gates cubren monolito y bloques; INPUT_TREES debe vigilar los dos."""
    module = _load_module()

    for tree in (
        "src/equilibria/templates/gtap",
        "src/equilibria/blocks/gtap",
        "src/equilibria/templates/gtap_logvalue",
        "src/equilibria/blocks/gtap_logvalue",
    ):
        assert tree in module.INPUT_TREES, f"{tree} no esta vigilado por el stamp"
