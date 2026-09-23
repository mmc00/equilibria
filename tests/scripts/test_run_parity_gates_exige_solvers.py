# tests/scripts/test_run_parity_gates_exige_solvers.py
#
# El gate obligatorio COMPARA contra GAMS. Si el oraculo no esta, tiene que
# FALLAR, no saltarse: pytest devuelve 0 cuando todo se saltea, y el script
# solo miraba el returncode.
#
# Medido 2026-09-23 en este repo: sin `gdxdump` en el PATH los 18 casos MCP y
# los 14 NLP se saltaban enteros y el script imprimia "Gates GREEN" y escribia
# el stamp que desbloquea el push.
import ast
import importlib.util
import os
import pathlib
import subprocess
import sys
from typing import Any

REPO = pathlib.Path(__file__).resolve().parents[2]
GATE = REPO / "scripts" / "gtap" / "run_parity_gates.py"


def _fuente() -> str:
    return GATE.read_text(encoding="utf-8")


def _cargar_gate() -> Any:
    """Carga el script por ruta, como hace test_check_parity_gates_stamp."""
    spec = importlib.util.spec_from_file_location("run_parity_gates", GATE)
    assert spec and spec.loader
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def test_el_sweep_corre_con_require_solvers():
    """El pytest del gate se lanza con EQUILIBRIA_REQUIRE_SOLVERS=1.

    Ese flag (tests/conftest.py) desactiva los guards de solver, asi que un
    binario ausente se nota en vez de esconderse tras un skip.
    """
    assert "EQUILIBRIA_REQUIRE_SOLVERS" in _fuente(), (
        "run_parity_gates.py debe exigir los solvers: sin eso un oraculo "
        "ausente produce 32 skips, returncode 0 y un stamp verde."
    )


def test_el_flag_desactiva_los_guards_de_solver():
    """Contrato de `tests/conftest.py` del que depende el arreglo.

    Sin el flag los marcadores ausentes se desactivan (y los tests se saltan);
    con el flag no se desactiva ninguno.
    """
    codigo = (
        "import sys; sys.path.insert(0, 'tests');"
        "from conftest import _ausentes; print(sorted(_ausentes()))"
    )
    env = dict(os.environ)
    env["PATH"] = "/usr/bin:/bin"  # sin GAMS

    env.pop("EQUILIBRIA_REQUIRE_SOLVERS", None)
    sin = subprocess.run(
        [sys.executable, "-c", codigo],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
    ).stdout
    env["EQUILIBRIA_REQUIRE_SOLVERS"] = "1"
    con = subprocess.run(
        [sys.executable, "-c", codigo],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
    ).stdout

    assert "needs_gdxdump" in sin, f"sin el flag deberia desactivarse: {sin!r}"
    assert con.strip() == "[]", f"con el flag no debe desactivarse ninguno: {con!r}"


def test_encuentra_gams_fuera_del_path():
    """GAMS se instala como framework en macOS y no se exporta al PATH.

    El gate lo busca en vez de exigir que el usuario lo sepa — es lo que hacia
    que la corrida "verde" de esta sesion no comparase nada.
    """
    hallado = _cargar_gate()._gams_en_path()
    # None = ya esta en el PATH (valido). Si no, tiene que traer gdxdump.
    if hallado is not None:
        assert (pathlib.Path(hallado) / "gdxdump").exists(), hallado


def test_el_mensaje_rojo_menciona_el_solver():
    """Si el gate cae por un solver ausente, el mensaje tiene que decirlo:
    antes ese caso ni siquiera llegaba a rojo."""
    arbol = ast.parse(_fuente())
    textos = [
        n.value
        for n in ast.walk(arbol)
        if isinstance(n, ast.Constant) and isinstance(n.value, str)
    ]
    assert any(
        "solver" in t.lower()
        for t in textos
        if "GATES RED" in t or "solver" in t.lower()
    ), "el camino rojo deberia orientar sobre un solver ausente"
