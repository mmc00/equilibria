"""Los ejemplos publicados tienen que EJECUTARSE (issue #15).

example_01 y example_06 estuvieron rotos desde junio sin que nada lo
notara: un usuario los reporto antes que el repo. Nada en la suite los
ejecutaba, asi que un ValueError en la primera linea de la guia de
entrada convivia con el CI en verde.

Este gate corre cada ejemplo en un subproceso y exige exit 0. Es lento
comparado con un unit test, pero es la unica forma de que "el ejemplo
anda" sea una afirmacion medida y no una suposicion.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples" / "cge"
EXAMPLES = sorted(EXAMPLES_DIR.glob("example_*.py"))

# Si el glob no encuentra nada, el test pasaria por vacuidad -- exactamente
# el modo de fallo del issue #23. Que sea rojo.
assert EXAMPLES, f"no se encontraron ejemplos en {EXAMPLES_DIR}"


@pytest.mark.parametrize("example", EXAMPLES, ids=lambda p: p.stem)
def test_example_runs(example: Path) -> None:
    proc = subprocess.run(
        [sys.executable, str(example)],
        capture_output=True,
        text=True,
        timeout=600,
        cwd=example.parent,
    )
    assert proc.returncode == 0, (
        f"{example.name} fallo con exit {proc.returncode}\n"
        f"--- stdout (cola) ---\n{proc.stdout[-2000:]}\n"
        f"--- stderr (cola) ---\n{proc.stderr[-2000:]}"
    )
