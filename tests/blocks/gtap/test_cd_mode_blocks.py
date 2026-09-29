"""Utilidad Cobb-Douglas en el modelo de BLOQUES (el modelo vivo).

El PR #89 la implemento solo en el monolito. GAMS la selecciona con
``%utility% eq CD`` (model.gms:761-795); aca, como en el monolito, la decide el
dato: SUBPAR=0 en el .prm. Bajo CD:

  eq_zcons   zcons = alphaa                       model.gms:765
  eq_phip    phip  = sum_i xcshr   (sin eh)       model.gms:781
  eq_uh      uh    = auh * prod_i xa^alphaa       model.gms:794
  eq_ev/cv   no existen (CDE-only)                model.gms:1322/1328
  alphaa     xcshr normalizado a sumar 1          cal.gms:762

Sin solver: se construye el modelo de un periodo y se evaluan los residuos
moviendo las variables que la forma CD NO debe contener (bh, eh, pa, yc).
"""

import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[3]
DATASET = ROOT / "datasets" / "gtap7_3x3"


def _paths():
    needed = ["basedata.har", "sets.har", "default.prm", "baserate.har"]
    missing = [n for n in needed if not (DATASET / n).exists()]
    if missing:
        pytest.skip(f"gtap7_3x3: faltan {missing}")
    return {
        "basedata_path": DATASET / "basedata.har",
        "sets_path": DATASET / "sets.har",
        "default_path": DATASET / "default.prm",
        "baserate_path": DATASET / "baserate.har",
    }


def _params(tmp_path=None, cobb_douglas=False, solo_region=None):
    from equilibria.templates.gtap import GTAPParameters

    paths = _paths()
    if cobb_douglas:
        from equilibria.babel.har import read_har, write_har

        har = read_har(paths["default_path"])
        if "SUBP" not in har:
            pytest.skip("default.prm sin header SUBP")
        if solo_region is None:
            har["SUBP"].array[...] = 0.0
        else:
            col = har["SUBP"].set_elements[1].index(solo_region)
            har["SUBP"].array[:, col] = 0.0
        assert tmp_path is not None
        destino = tmp_path / "cobbdouglas.prm"
        write_har(destino, har)
        paths["default_path"] = destino
    p = GTAPParameters()
    p.load_from_har(**paths)
    return p


def _model(p):
    from equilibria.templates.gtap.gtap_block_model import build_block_single_period
    from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig

    gc = GTAPClosureConfig(
        name="base",
        closure_type="MCP",
        capital_mobility="sluggish",
        fix_endowments=False,
        fix_taxes=False,
        fix_technology=False,
        if_sub=False,
        numeraire="pnum",
    )
    return build_block_single_period(p, p.sets, gc, list(p.sets.r)[-1])


def _resid(con):
    from pyomo.environ import value

    return float(value(con.body)) - float(value(con.upper))


def test_cd_regions_sale_del_dato(tmp_path):
    from equilibria.blocks.gtap import _derived_params as dp

    cde = _params()
    assert dp.cd_regions(cde, cde.sets) == frozenset()
    cd = _params(tmp_path, cobb_douglas=True)
    assert dp.cd_regions(cd, cd.sets) == frozenset(cd.sets.r)


def test_alphaa_cd_es_xcshr_normalizado(tmp_path):
    """cal.gms:762: alphaa = xcshr / sum xcshr, y zcons arranca en alphaa."""
    from equilibria.blocks.gtap import _derived_params as dp

    p = _params(tmp_path, cobb_douglas=True)
    calib = dp.demand_income_params(p, p.sets, residual_region=list(p.sets.r)[-1])
    for r in p.sets.r:
        alphas = {i: calib["alphaa_hhd"][(r, i)] for i in p.sets.i}
        assert sum(alphas.values()) == pytest.approx(1.0, abs=1e-12)
        for i, a in alphas.items():
            assert calib["zcons_init"][(r, i)] == pytest.approx(a, abs=1e-15)


def test_eq_zcons_cd_no_depende_de_precio_ni_utilidad(tmp_path):
    m = _model(_params(tmp_path, cobb_douglas=True))
    from pyomo.environ import value

    for (r, i), con in m.eq_zcons.items():
        a = float(value(m.alphaa_hhd[r, i]))
        if a <= 0.0:
            continue
        m.zcons[r, i].set_value(a)
        m.uh[r].set_value(3.7)
        m.pa[r, i, "hhd"].set_value(2.9)
        assert abs(_resid(con)) < 1e-12, f"eq_zcons[{r},{i}] no es zcons = alphaa"


def test_eq_phip_cd_no_usa_eh(tmp_path):
    p = _params(tmp_path, cobb_douglas=True)
    m = _model(p)
    from pyomo.environ import value

    assert any(abs(float(value(v)) - 1.0) > 1e-6 for v in m.eh.values()), (
        "el test necesita algun eh != 1 para distinguir las formas"
    )
    for r, con in m.eq_phip.items():
        s = sum(
            float(value(m.xcshr[r, i]))
            for i in m.i
            if float(value(m.c_share[r, i])) > 0.0
        )
        m.phip[r].set_value(s)
        assert abs(_resid(con)) < 1e-12, f"eq_phip[{r}] no es phip = sum xcshr"


def test_eq_uh_cd_es_la_cobb_douglas(tmp_path):
    m = _model(_params(tmp_path, cobb_douglas=True))
    from pyomo.environ import value

    for r, con in m.eq_uh.items():
        prod = 1.0
        for i in m.i:
            a = float(value(m.alphaa_hhd[r, i]))
            if a > 0.0 and float(value(m.c_share[r, i])) > 0.0:
                prod *= float(value(m.xaa[r, i, "hhd"])) ** a
        m.uh[r].set_value(float(value(m.auh[r])) * prod)
        m.yc[r].set_value(123.0)  # no debe importar bajo CD
        assert abs(_resid(con)) < 1e-9, f"eq_uh[{r}] no es uh = auh*prod xa^alphaa"


def test_ev_cv_se_saltan_y_se_fijan_bajo_cd(tmp_path):
    p = _params(tmp_path, cobb_douglas=True)
    m = _model(p)
    for nombre in ("eq_ev", "eq_cv"):
        con = dict(getattr(m, nombre).items()) if hasattr(m, nombre) else {}
        for r in p.sets.r:
            assert (r,) not in con and r not in con, (
                f"{nombre}[{r}] sigue bajo Cobb-Douglas (CDE-only, model.gms:1322/1328)"
            )
    for nombre in ("ev", "cv"):
        for r in p.sets.r:
            assert getattr(m, nombre)[r].fixed, (
                f"{nombre}[{r}] libre sin su ecuacion: el MCP pierde la cuadratura"
            )


def test_cde_no_cambia(tmp_path):
    """Con el default.prm (SUBPAR>0) eq_zcons sigue siendo la CDE: depende de uh."""
    m = _model(_params())
    from pyomo.environ import value

    (r, i), con = next(
        (k, c) for k, c in m.eq_zcons.items() if float(value(m.alphaa_hhd[k])) > 0.0
    )
    antes = _resid(con)
    m.uh[r].set_value(float(value(m.uh[r])) * 1.5)
    assert abs(_resid(con) - antes) > 1e-9, "bajo CDE eq_zcons debe depender de uh"
    assert not m.ev[r].fixed
    assert not hasattr(m, "auh"), "auh es CD-only: el modelo CDE no debe registrarlo"


def test_auh_normaliza_la_utilidad_en_el_benchmark(tmp_path):
    """cal.gms:768: auh = uh.l / prod xa.l^alphaa, con uh.l = 1 (cal.gms:243)."""
    m = _model(_params(tmp_path, cobb_douglas=True))
    from pyomo.environ import value

    for r in m.r:
        prod = 1.0
        for i in m.i:
            a = float(value(m.alphaa_hhd[r, i]))
            if a > 0.0 and float(value(m.c_share[r, i])) > 0.0:
                prod *= float(value(m.xaa[r, i, "hhd"])) ** a
        assert float(value(m.auh[r])) * prod == pytest.approx(1.0, abs=1e-9), r


def test_mezcla_cd_cde_entre_regiones_avisa(tmp_path):
    """%utility% es global en GAMS: una region CD junto a otras CDE no tiene
    equivalente. Se construye igual, pero AVISA una sola vez."""
    import warnings

    from equilibria.blocks.gtap import _derived_params as dp

    p = _params(tmp_path, cobb_douglas=True, solo_region="USA")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        assert dp.cd_regions(p, p.sets) == frozenset({"USA"})
    assert sum("Cobb-Douglas" in str(x.message) for x in w) == 1, [
        str(x.message) for x in w
    ]


def test_subpar_none_en_region_cd_no_rompe_la_calibracion(tmp_path):
    from equilibria.blocks.gtap import _derived_params as dp

    p = _params(tmp_path, cobb_douglas=True)
    r = list(p.sets.r)[0]
    i = list(p.sets.i)[0]
    p.elasticities.subpar[(r, i)] = None
    calib = dp.demand_income_params(p, p.sets, residual_region=list(p.sets.r)[-1])
    assert calib["bh"][(r, i)] == 1.0
