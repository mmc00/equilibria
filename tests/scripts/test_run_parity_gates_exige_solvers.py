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


def _cargar_diff_core() -> Any:
    """Carga `_diff_core` por ruta: vive en scripts/, no es paquete importable."""
    ruta = REPO / "scripts" / "gtap" / "_diff_core.py"
    spec = importlib.util.spec_from_file_location("_diff_core", ruta)
    assert spec and spec.loader
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def _cargar_gate() -> Any:
    """Carga el script por ruta, como hace test_check_parity_gates_stamp."""
    spec = importlib.util.spec_from_file_location("run_parity_gates", GATE)
    assert spec and spec.loader
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def test_el_sweep_se_lanza_exigiendo_los_solvers(monkeypatch, tmp_path):
    """COMPORTAMIENTO, no texto: se intercepta subprocess.run y se mira el env
    con el que se lanza pytest.

    La primera version de este test hacia
    `assert "EQUILIBRIA_REQUIRE_SOLVERS" in _fuente()`, que pasaba igual con el
    flag puesto a "0" (verificado). Comprobar que un literal aparece en el
    fichero no prueba nada sobre lo que el script hace.
    """
    gate = _cargar_gate()
    capturado: dict[str, Any] = {}

    class _Res:
        returncode = 0
        stdout = "abc123"

    def _fake_run(argv, **kw):
        if "pytest" in argv:
            capturado["env"] = kw.get("env")
            capturado["argv"] = argv
        return _Res()

    monkeypatch.setattr(gate.subprocess, "run", _fake_run)
    monkeypatch.setattr(gate, "dirty_watched", lambda _: [])
    monkeypatch.setattr(gate, "REGEN_CMDS", [])
    monkeypatch.setattr(gate, "stamp_path", lambda *_: tmp_path / "stamp")
    monkeypatch.setattr(gate, "input_hash", lambda *_: "deadbeef")
    monkeypatch.setattr(gate.sys, "argv", ["run_parity_gates.py"])

    gate.main()

    env = capturado.get("env")
    assert env is not None, "el sweep no se lanzo"
    assert env.get("EQUILIBRIA_REQUIRE_SOLVERS") == "1", (
        "el sweep debe exigir los solvers: con el guard activo un oraculo "
        f"ausente se salta y el gate estampa verde. env={env.get('EQUILIBRIA_REQUIRE_SOLVERS')!r}"
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


def test_el_orden_de_versiones_no_es_alfabetico(tmp_path, monkeypatch):
    """`Current` primero, y el resto por numero REAL de version.

    El codigo original ordenaba strings con `reverse=True` y el comentario
    decia "version mas alta primero". Era falso: "Current" gana siempre (C >
    digitos en ASCII) y entre numeros "9" > "53" > "48" > "10". Acertaba de
    casualidad porque en esta maquina Current->53.
    """
    gate = _cargar_gate()
    raiz = tmp_path / "GAMS.framework" / "Versions"
    for v in ("9", "10", "48", "53", "Current"):
        d = raiz / v / "Resources"
        d.mkdir(parents=True)
        (d / "gdxdump").touch()

    monkeypatch.setattr(gate.shutil, "which", lambda _: None, raising=False)
    monkeypatch.setattr(
        gate, "_PATRONES_GAMS", [str(raiz / "*" / "Resources")], raising=False
    )
    elegido = gate._gams_en_path()
    assert elegido is not None
    assert pathlib.Path(elegido).parent.name == "Current", elegido

    # Sin `Current`, gana la version numerica mas alta (53), no "9" ni "48".
    import shutil as _sh

    _sh.rmtree(raiz / "Current")
    elegido = gate._gams_en_path()
    assert pathlib.Path(elegido).parent.name == "53", elegido


def test_el_oraculo_no_esta_clavado_a_una_version(monkeypatch):
    """`_diff_core.GDXDUMP` es el binario con el que se LEE el oraculo de GAMS.

    Era una ruta absoluta a GAMS 48: funcionaba en una sola maquina, y ademas
    fijaba el oraculo a una version mientras el gate usaba otra (Current->53).
    Inyectar el PATH en el sweep no lo arreglaba, porque esta ruta no mira el
    PATH.
    """
    _diff_core = _cargar_diff_core()

    assert "/Versions/48/" not in _diff_core.GDXDUMP, (
        f"el oraculo vuelve a estar clavado a una version: {_diff_core.GDXDUMP}"
    )

    # El escape hatch explicito manda sobre todo lo demas.
    monkeypatch.setenv("EQUILIBRIA_GDXDUMP", "/ruta/elegida/gdxdump")
    assert _diff_core._resolver_gdxdump() == "/ruta/elegida/gdxdump"


def test_el_oraculo_y_el_gate_usan_la_misma_instalacion():
    """Si divergen, el gate mide con una version y el oraculo lee con otra."""
    _diff_core = _cargar_diff_core()

    gate = _cargar_gate()._gams_en_path()
    if gate is None:
        return  # gdxdump ya en el PATH: ambos lo usan
    assert pathlib.Path(_diff_core.GDXDUMP).parent == pathlib.Path(gate), (
        f"oraculo={_diff_core.GDXDUMP} vs gate={gate}"
    )
