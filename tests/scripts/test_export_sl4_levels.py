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


def test_variable_sin_nivel_no_se_guarda_como_dato_y_queda_declarada_en_nolv(
    tmp_path: Path,
) -> None:
    har = _export_sample(tmp_path)

    # psave (id 0027) trae el centinela 1e10 en ambas columnas de nivel: no hay
    # LEVA/LEVB que confundir con datos; su nombre queda en la cabecera NOLV.
    assert "0027" not in har
    assert "B027" not in har
    assert list(har["NOLV"].array) == ["psave"]


def test_variable_con_cambio_pct_distinto_al_sl4dump_se_descarta(
    tmp_path: Path,
) -> None:
    har = _export_sample(tmp_path)

    # globalcgds (id 0049) trae -1.9 en el texto vs -1.3623582 en el sl4dump.
    assert "0049" not in har
    assert "B049" not in har
    assert sorted(har) == ["0052", "B052", "NOLV"]


def _sic_text_from(sl4dump: Path, post_levels: Path | None) -> str:
    """Texto SIC sintetico CONSISTENTE con el %-cambio del sl4dump, para que la
    validacion de niveles pase por la razon correcta:
      * con fixture sl4levels: post = LEVA real; pre = post / (1 + %/100);
      * sin fixture: pre = 1 (indice de benchmark); post = 1 + %/100;
      * change = post - pre. Variables sin nivel (LEVA = 1e10): centinela en ambas.
    Prueba el armado del .har, no es un texto real de sltoht."""
    dump = read_har(str(sl4dump))
    post_har = read_har(str(post_levels)) if post_levels else None
    lines: list[str] = []
    for hid, ha in dump.items():
        if "#" not in (ha.long_name or ""):
            continue  # cabeceras XX** de metadatos: no son variables
        if post_har is not None and hid not in post_har:
            continue
        lines.append(f"! Variable {ha.long_name.strip()}")
        pct = np.asarray(ha.array, dtype=np.float64).ravel(order="F")
        if post_har is None:
            pre = np.ones_like(pct)
            post = 1.0 + pct / 100.0
        else:
            post = np.asarray(post_har[hid].array, dtype=np.float64).ravel(order="F")
            pre = post / (1.0 + pct / 100.0)
            no_level = post == np.float32(1e10)
            pre[no_level] = post[no_level] = 1e10
        for i, (p, a, b) in enumerate(zip(pct, pre, post, strict=True), 1):
            chg = float(p) if b == 1e10 else float(b - a)
            lines.append(
                f"  {float(p)!r} {float(a)!r} {float(b)!r} {chg!r}  !%1= {i:6d}"
            )
    return "\n".join(lines) + "\n"


LEVA_FIXTURES = sorted(FIXTURES.glob("sl4levels_*_tm10.har"))


@pytest.mark.parametrize("leva_fixture", LEVA_FIXTURES, ids=lambda p: p.stem)
def test_los_fixtures_sl4levels_existentes_se_reproducen_igual(
    leva_fixture: Path, tmp_path: Path
) -> None:
    mod = _load_module()
    sl4dump = leva_fixture.with_name(
        leva_fixture.name.replace("sl4levels_", "sl4dump_")
    )
    text = tmp_path / "levels.txt"
    text.write_text(_sic_text_from(sl4dump, leva_fixture))
    out = tmp_path / leva_fixture.name

    rc = mod.main(
        [str(text), "--from-text", "--sl4dump", str(sl4dump), "--out", str(out)]
    )

    assert rc == 0
    expected, got = read_har(str(leva_fixture)), read_har(str(out))
    sentinel = np.float32(1e10)
    with_level = {k: v for k, v in expected.items() if not np.all(v.array == sentinel)}
    no_level = [
        v.long_name.split("#", 1)[0].strip()
        for v in expected.values()
        if np.all(v.array == sentinel)
    ]
    assert sorted(got) == sorted(
        [*with_level, *(mod.levb_header(k) for k in with_level), "NOLV"]
    )
    assert list(got["NOLV"].array) == no_level
    for hid, ha in with_level.items():
        g = got[hid]
        assert (g.long_name, g.coeff_name, g.set_names, g.set_elements) == (
            ha.long_name,
            ha.coeff_name,
            ha.set_names,
            ha.set_elements,
        ), hid
        assert g.array.dtype == ha.array.dtype, hid
        np.testing.assert_array_equal(g.array, ha.array, err_msg=hid)
        assert got[mod.levb_header(hid)].long_name == mod.levb_long_name(ha.long_name)


def test_cubre_la_solucion_20x41_capfix(tmp_path: Path) -> None:
    mod = _load_module()
    sl4dump = FIXTURES / "sl4dump_gtap7_20x41_tm10_s8-16-32_capfix.har"
    text = tmp_path / "levels.txt"
    text.write_text(_sic_text_from(sl4dump, None))

    rc = mod.main(
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
    leva = sorted(k for k in got if k[0] == "0")
    assert len(leva) == 252  # las 257 cabeceras menos las 5 XX** de metadatos
    assert sorted(k for k in got if k[0] == "B") == sorted(
        mod.levb_header(k) for k in leva
    )
    assert list(got["NOLV"].array) == []
    # qfd [COMM, ACTS, REG] = 20x20x41: pre = 1 (indice), post = 1 + %/100
    dump = read_har(str(sl4dump))
    qfd_pre = got[mod.levb_header("0002")]
    assert qfd_pre.array.shape == (20, 20, 41)
    assert qfd_pre.long_name == mod.levb_long_name(dump["0002"].long_name)
    np.testing.assert_array_equal(qfd_pre.array, 1.0)
    np.testing.assert_allclose(
        got["0002"].array, 1.0 + dump["0002"].array / 100.0, rtol=1e-6
    )


def _swap_level_columns(text: str) -> str:
    """Intercambia las columnas pre-sim y post-sim en cada fila de componente."""
    out = []
    for line in text.splitlines():
        if "!%1=" in line:
            data, tag = line.split("!", 1)
            pct, pre, post, chg = data.split()
            line = f"  {pct} {post} {pre} {chg}   !{tag}"
        out.append(line)
    return "\n".join(out) + "\n"


def test_niveles_pre_y_post_intercambiados_se_rechazan_nombrando_variable_y_celda(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    text = tmp_path / "swapped.txt"
    text.write_text(_swap_level_columns(SAMPLE.read_text()))
    out = tmp_path / "o.har"

    rc = _load_module().main(
        [str(text), "--from-text", "--sl4dump", str(SL4DUMP_3X3), "--out", str(out)]
    )

    assert rc != 0
    assert not out.exists()
    err = capsys.readouterr().err
    # primera celda de qo: componente 1 = (Food, USA)
    assert "qo" in err
    assert "(Food, USA)" in err


def test_variable_de_cambio_ordinario_valida_post_menos_pre(tmp_path: Path) -> None:
    # del_tbal (id 0186) es una variable de CAMBIO ordinario: su columna de solucion
    # (= sl4dump) es post - pre en US$ millones, no un %. En GTAP no trae niveles en
    # el SL4; aqui se le inventan (a mano, sintetico) para ejercer la rama de cambio.
    text = tmp_path / "change.txt"
    text.write_text(
        "! Variable del_tbal # change in trade balance X - M, $ US million #\n"
        "  -110822.77  1000000.0   889177.23  -110822.77  !%1=     1\n"
        "   163810.19  -50000.0    113810.19   163810.19  !%1=     2\n"
        "   -52987.58       0.0    -52987.58   -52987.58  !%1=     3\n"
    )
    out = tmp_path / "o.har"

    rc = _load_module().main(
        [str(text), "--from-text", "--sl4dump", str(SL4DUMP_3X3), "--out", str(out)]
    )

    assert rc == 0
    har = read_har(str(out))
    np.testing.assert_allclose(har["0186"].array, [889177.23, 113810.19, -52987.58])
    np.testing.assert_allclose(har["B186"].array, [1000000.0, -50000.0, 0.0])


def test_variable_de_cambio_ordinario_con_niveles_incoherentes_se_rechaza(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    text = tmp_path / "change_bad.txt"
    text.write_text(
        "! Variable del_tbal # change in trade balance X - M, $ US million #\n"
        "  -110822.77  1000000.0   889177.23  -110822.77  !%1=     1\n"
        "   163810.19  -50000.0    113810.19   163810.19  !%1=     2\n"
        "   -52987.58       0.0    -52000.00   -52987.58  !%1=     3\n"
    )

    rc = _load_module().main(
        [
            str(text),
            "--from-text",
            "--sl4dump",
            str(SL4DUMP_3X3),
            "--out",
            str(tmp_path / "o.har"),
        ]
    )

    assert rc != 0
    err = capsys.readouterr().err
    assert "del_tbal" in err
    assert "(ROW)" in err
