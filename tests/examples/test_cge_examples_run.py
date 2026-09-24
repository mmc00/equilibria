"""Los ejemplos publicados tienen que EJECUTARSE (issue #15).

example_01 y example_06 estuvieron rotos desde junio sin que nada lo
notara: un usuario los reporto antes que el repo. Nada en la suite los
ejecutaba, asi que un ValueError en la primera linea de la guia de
entrada convivia con el CI en verde.

Este gate corre cada ejemplo en un subproceso y exige exit 0.

NO lleva `@pytest.mark.slow` a proposito: el CI corre
`pytest -m "not gams and not slow"` (.github/workflows/tests.yml:157) y
no hay ningun job que recupere los `slow`, asi que marcarlo lo dejaria
fuera del CI -- justo el agujero que este gate existe para tapar.
Medido: 0.24-0.46s por ejemplo, ~2.4s los ocho, presupuesto de unit test.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples" / "cge"
EXAMPLES = sorted(EXAMPLES_DIR.glob("example_*.py"))


def test_examples_are_discovered() -> None:
    """Sin esto el parametrize vacio pasaria por vacuidad (cf. issue #23).

    Va como test y no como assert de modulo a proposito: un assert en la
    importacion aborta la COLECCION del directorio entero en vez de dar
    un fallo legible.
    """
    assert EXAMPLES, f"no se encontraron ejemplos en {EXAMPLES_DIR}"


@pytest.mark.parametrize("example", EXAMPLES, ids=lambda p: p.stem)
def test_example_runs(example: Path) -> None:
    proc = subprocess.run(
        [sys.executable, str(example)],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=example.parent,
    )
    assert proc.returncode == 0, (
        f"{example.name} fallo con exit {proc.returncode}\n"
        f"--- stdout (cola) ---\n{proc.stdout[-2000:]}\n"
        f"--- stderr (cola) ---\n{proc.stderr[-2000:]}"
    )
