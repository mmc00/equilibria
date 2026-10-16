"""export_sl4_levels.py: niveles antes (LEVB) y despues (LEVA) de cualquier .sl4 (#127).

sltoht solo corre en Windows, asi que en Mac se prueba con ``--from-text`` sobre
``sl4levels_sample_sic.txt``, una muestra de texto ``sltoht -SHL -SIC`` hecha a mano
y commiteada antes del parser (oraculo independiente).

Convencion del .har de salida (la lee #131):
  * LEVA (post-sim): header = id numerico del sl4dump (``"0052"``), ``long_name`` el
    del sl4dump (``"qo # ... #"``) -> identico a los fixtures sl4levels_* de siempre.
  * LEVB (pre-sim): header = ``"B"`` + los 3 ultimos digitos del id (``"B052"``),
    ``long_name`` = ``"LEVB "`` + el del sl4dump.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from equilibria.babel.har.reader import read_har

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "gtap" / "export_sl4_levels.py"
FIXTURES = ROOT / "tests" / "fixtures" / "gtap7_gempack"
SAMPLE = FIXTURES / "sl4levels_sample_sic.txt"
SL4DUMP_3X3 = FIXTURES / "sl4dump_gtap7_3x3_tm10.har"


def _load_module() -> Any:
    spec = importlib.util.spec_from_file_location("export_sl4_levels", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _export_sample(tmp_path: Path) -> dict[str, Any]:
    out = tmp_path / "sl4levels_sample.har"
    rc = _load_module().main(
        [str(SAMPLE), "--from-text", "--sl4dump", str(SL4DUMP_3X3), "--out", str(out)]
    )
    assert rc == 0
    return read_har(str(out))


def test_from_text_exporta_niveles_antes_y_despues(tmp_path: Path) -> None:
    har = _export_sample(tmp_path)

    # qo [ACTS, REG]: valores tal cual estan escritos en la muestra (no recalculados).
    leva, levb = har["0052"], har["B052"]
    assert leva.long_name.startswith("qo # output of activity")
    assert levb.long_name.startswith("LEVB qo # output of activity")
    assert levb.set_names == ["ACTS", "REG"]
    assert levb.set_elements == [["Food", "Mnfcs", "Svces"], ["USA", "EU_28", "ROW"]]
    # componente 1 = (Food, USA); componente 9 = (Svces, ROW); orden Fortran.
    assert levb.array[0, 0] == pytest.approx(1579235.5)
    assert leva.array[0, 0] == pytest.approx(1551671.0)
    assert levb.array[2, 2] == pytest.approx(51690457.0)
    assert leva.array[2, 2] == pytest.approx(51609160.0)
    # componente 2 = (Mnfcs, USA): verifica el orden Fortran, no C.
    assert levb.array[1, 0] == pytest.approx(5737490.6)


def test_variable_sin_nivel_conserva_el_centinela_en_ambos_periodos(
    tmp_path: Path,
) -> None:
    har = _export_sample(tmp_path)

    np.testing.assert_array_equal(har["0027"].array, np.full(3, 1e10, dtype=np.float32))
    np.testing.assert_array_equal(har["B027"].array, np.full(3, 1e10, dtype=np.float32))


def test_variable_con_cambio_pct_distinto_al_sl4dump_se_descarta(
    tmp_path: Path,
) -> None:
    har = _export_sample(tmp_path)

    # globalcgds (id 0049) trae -1.9 en el texto vs -1.3623582 en el sl4dump.
    assert "0049" not in har
    assert "B049" not in har
    assert sorted(har) == ["0027", "0052", "B027", "B052"]


def _sic_text_from(sl4dump: Path, post_levels: Path | None) -> str:
    """Texto SIC sintetico: %-cambio del sl4dump; LEVA del fixture sl4levels (o el
    %-cambio si no hay fixture); LEVB = LEVA - 1. Solo para probar el armado del .har."""
    dump = read_har(str(sl4dump))
    post = read_har(str(post_levels)) if post_levels else dump
    lines: list[str] = []
    for hid, ha in dump.items():
        if hid not in post or "#" not in (ha.long_name or ""):
            continue  # cabeceras XX** de metadatos: no son variables
        lines.append(f"! Variable {ha.long_name.strip()}")
        pct = np.asarray(ha.array, dtype=np.float64).ravel(order="F")
        lev = np.asarray(post[hid].array, dtype=np.float64).ravel(order="F")
        for i, (p, q) in enumerate(zip(pct, lev, strict=True), 1):
            lines.append(
                f"  {float(p)!r} {float(q) - 1.0!r} {float(q)!r} 1.0  !%1= {i:6d}"
            )
    return "\n".join(lines) + "\n"


LEVA_FIXTURES = sorted(FIXTURES.glob("sl4levels_*_tm10.har"))


@pytest.mark.parametrize("leva_fixture", LEVA_FIXTURES, ids=lambda p: p.stem)
def test_los_fixtures_sl4levels_existentes_se_reproducen_igual(
    leva_fixture: Path, tmp_path: Path
) -> None:
    sl4dump = leva_fixture.with_name(
        leva_fixture.name.replace("sl4levels_", "sl4dump_")
    )
    text = tmp_path / "levels.txt"
    text.write_text(_sic_text_from(sl4dump, leva_fixture))
    out = tmp_path / leva_fixture.name

    rc = _load_module().main(
        [str(text), "--from-text", "--sl4dump", str(sl4dump), "--out", str(out)]
    )

    assert rc == 0
    expected, got = read_har(str(leva_fixture)), read_har(str(out))
    leva = {k: v for k, v in got.items() if not k.startswith("B")}
    assert sorted(leva) == sorted(expected)
    for hid, ha in expected.items():
        g = leva[hid]
        assert (g.long_name, g.coeff_name, g.set_names, g.set_elements) == (
            ha.long_name,
            ha.coeff_name,
            ha.set_names,
            ha.set_elements,
        ), hid
        assert g.array.dtype == ha.array.dtype, hid
        np.testing.assert_array_equal(g.array, ha.array, err_msg=hid)


def test_cubre_la_solucion_20x41_capfix(tmp_path: Path) -> None:
    sl4dump = FIXTURES / "sl4dump_gtap7_20x41_tm10_s8-16-32_capfix.har"
    text = tmp_path / "levels.txt"
    text.write_text(_sic_text_from(sl4dump, None))

    rc = _load_module().main(
        [
            str(text),
            "--from-text",
            "--sl4dump",
            str(sl4dump),
            "--out",
            str(tmp_path / "o.har"),
        ]
    )

    assert rc == 0
    got = read_har(str(tmp_path / "o.har"))
    leva = sorted(k for k in got if not k.startswith("B"))
    levb = sorted(k for k in got if k.startswith("B"))
    assert len(leva) == 252  # las 257 cabeceras menos las 5 XX** de metadatos
    assert levb == ["B" + k[1:] for k in leva]
    # qfd [COMM, ACTS, REG] = 20x20x41
    assert got["B002"].array.shape == (20, 20, 41)
