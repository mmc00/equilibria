"""Oraculos de ``equilibria.compare``, fijados ANTES de escribir el modulo.

Nada de aqui sale de ``equilibria.compare``: los valores esperados se calcularon a
mano (ejemplos 1-3) o se leyeron del GDX de GAMS con ``gdxdump`` (la herramienta de
GAMS, no el lector de equilibria) y se pegaron como literales (ejemplo 4). Si el
modulo y este archivo discrepan, el que esta mal es el modulo hasta demostrar lo
contrario.

Convenciones del mapeo que estos ejemplos fijan (spec 2026-10-15, Q8/Q9/Q16/Q18):

- ``d = |A - B| / max(|B|, abs_floor)`` con B la referencia (``reference="b"``).
- El orden comun de las dimensiones es el del lado A; B se reordena por ``dims``.
- El renombre de elementos va POR CONJUNTO: el mismo nombre en dos conjuntos
  distintos no se confunde.
- Solo se comparan los periodos (ya traducidos) presentes en los dos lados.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Ejemplo 1 — dos conjuntos con elementos de igual nombre no cruzan celdas.
#
# Variable x(COMM, ACT). A guarda (COMM, ACT) con prefijos c_/a_; B guarda
# (ACT, COMM) sin prefijos, asi que en B "Food" existe en los DOS conjuntos.
# El renombre de A es por conjunto: c_Food -> Food en COMM, a_Food -> Food en ACT.
#
#   celda comun (COMM, ACT)   A      B (clave de B = (ACT, COMM))   d
#   (Food, Food)              10.0   B[(Food, Food)] = 10.0         0
#   (Food, Mfg)                2.0   B[(Mfg, Food)]  = 2.002        0.002/2.002
#   (Mfg,  Food)               3.0   B[(Food, Mfg)]  = 3.0          0
#   (Mfg,  Mfg)               20.0   B[(Mfg, Mfg)]   = 20.0         0
#
# Si el alineador cruzara (Food, Mfg) con (Mfg, Food) —p. ej. ordenando etiquetas o
# sin reordenar B— emparejaria 2.0 con 3.0 y daria d = 1/3 en vez de 0.000999.
# ---------------------------------------------------------------------------
EX1_A = {
    "x": {
        ("c_Food", "a_Food", "shock"): 10.0,
        ("c_Food", "a_Mfg", "shock"): 2.0,
        ("c_Mfg", "a_Food", "shock"): 3.0,
        ("c_Mfg", "a_Mfg", "shock"): 20.0,
    }
}
EX1_B = {
    "x": {
        ("Food", "Food", "shock"): 10.0,
        ("Mfg", "Food", "shock"): 2.002,
        ("Food", "Mfg", "shock"): 3.0,
        ("Mfg", "Mfg", "shock"): 20.0,
    }
}
EX1_MAP = {
    "sets": {
        "COMM": {"a": {"c_Food": "Food", "c_Mfg": "Mfg"}},
        "ACT": {"a": {"a_Food": "Food", "a_Mfg": "Mfg"}},
    },
    "variables": [
        {
            "name": "x",
            "a": {"var": "x", "dims": ["COMM", "ACT"]},
            "b": {"var": "x", "dims": ["ACT", "COMM"]},
            "tol": {"rel": 1e-3, "abs_floor": 1e-6},
        }
    ],
    "thresholds": [1e-3],
}
# d por celda comun (periodo, (COMM, ACT)), calculado a mano.
EX1_D = {
    ("shock", ("Food", "Food")): 0.0,
    ("shock", ("Food", "Mfg")): 0.000999000999000999,  # 0.002 / 2.002
    ("shock", ("Mfg", "Food")): 0.0,
    ("shock", ("Mfg", "Mfg")): 0.0,
}

# ---------------------------------------------------------------------------
# Ejemplo 2 — piso absoluto, umbrales, mediana y maximo.
#
# y(REG), tol rel 0.01, abs_floor 1e-3, umbrales [0.001, 0.01].
#
#   celda   A        B       d (ref B)                       d (ref A)
#   USA     1.0      1.0     0                               0
#   EU      0.0005   0.0     0.0005/max(0, 1e-3)    = 0.5    0.0005/max(0.0005,1e-3) = 0.5
#   ROW     100.5    100.0   0.5/100                = 0.005  0.5/100.5 = 0.004975124378109453
#
# Ref B: dentro de 0.001 -> 1/3 (33.333..%); dentro de 0.01 -> 2/3 (66.666..%);
# pasan su tol (0.01) 2/3; mediana de [0, 0.005, 0.5] = 0.005; maximo 0.5 en EU.
# ---------------------------------------------------------------------------
EX2_A = {"y": {("USA", "t1"): 1.0, ("EU", "t1"): 0.0005, ("ROW", "t1"): 100.5}}
EX2_B = {"y": {("USA", "t1"): 1.0, ("EU", "t1"): 0.0, ("ROW", "t1"): 100.0}}
EX2_MAP = {
    "variables": [
        {
            "name": "y",
            "a": {"var": "y", "dims": ["REG"]},
            "b": {"var": "y", "dims": ["REG"]},
            "tol": {"rel": 0.01, "abs_floor": 1e-3},
        }
    ],
    "thresholds": [0.001, 0.01],
}
EX2_D_REF_B = {("t1", ("USA",)): 0.0, ("t1", ("EU",)): 0.5, ("t1", ("ROW",)): 0.005}
EX2_D_REF_A = {
    ("t1", ("USA",)): 0.0,
    ("t1", ("EU",)): 0.5,
    ("t1", ("ROW",)): 0.004975124378109453,
}
EX2_PCT_WITHIN_REF_B = {0.001: 100.0 / 3, 0.01: 200.0 / 3}
EX2_MEDIAN_REF_B = 0.005
EX2_MAX_REF_B = 0.5
EX2_MAX_CELL_REF_B = ("t1", ("EU",))
EX2_PASS_PCT_REF_B = 200.0 / 3

# ---------------------------------------------------------------------------
# Ejemplo 3 — periodos con nombres distintos y celdas sin pareja.
#
# A trae base/check/shock; B trae pre/post y su mapeo dice pre->check, post->shock.
# "base" solo existe en A: NO se compara y queda declarado como periodo solo en A.
#
#   periodo  celda  A    B
#   check    USA    1.0  1.0     pareja, d=0
#   check    ROW    2.0  —       sin pareja en B
#   shock    USA    5.0  5.0     pareja, d=0
#   shock    EU     —    6.0     sin pareja en A
#   base     USA    9.0  —       periodo solo en A: ni pareja ni "sin pareja"
# ---------------------------------------------------------------------------
EX3_A = {
    "z": {
        ("USA", "base"): 9.0,
        ("USA", "check"): 1.0,
        ("ROW", "check"): 2.0,
        ("USA", "shock"): 5.0,
    }
}
EX3_B = {"z": {("USA", "pre"): 1.0, ("USA", "post"): 5.0, ("EU", "post"): 6.0}}
EX3_MAP = {
    "periods": {"b": {"pre": "check", "post": "shock"}},
    "variables": [
        {
            "name": "z",
            "a": {"var": "z", "dims": ["REG"]},
            "b": {"var": "z", "dims": ["REG"]},
            "tol": {"rel": 1e-3, "abs_floor": 1e-6},
        }
    ],
    "exclude": [{"var": "ppriv", "why": "indice Divisia: sin forma en niveles"}],
    "thresholds": [1e-3],
}
EX3_MATCHED = 2
EX3_MISSING_IN_B = [("z", "check", ("ROW",))]  # esta en A, falta en B
EX3_MISSING_IN_A = [("z", "shock", ("EU",))]  # esta en B, falta en A
EX3_PERIODS_ONLY_IN_A = ["base"]
EX3_PERIODS_ONLY_IN_B: list[str] = []

# ---------------------------------------------------------------------------
# Ejemplo 4 — GTAP 3x3, GDX de GAMS (fixture existente).
#
# Literales copiados de:
#   gdxdump tests/fixtures/gtap7/gtap7_3x3/out_gtap_shock_ifsub0.gdx Format=csv Symb=xp
#   gdxdump ... Symb=pft
# (GDXDUMP 53.5.0; 15 cifras significativas, comparar con rel <= 1e-13).
# xp(r, a, t) y pft(r, fp, t): el periodo es la ultima dimension.
# ---------------------------------------------------------------------------
GTAP3X3_GDX = (
    ROOT / "tests" / "fixtures" / "gtap7" / "gtap7_3x3" / "out_gtap_shock_ifsub0.gdx"
)
GDXDUMP_XP = {
    ("USA", "a_Food", "base"): 1.6135371431333,
    ("USA", "a_Food", "check"): 1.56662414651903,
    ("USA", "a_Food", "shock"): 1.54573522356493,
    ("USA", "a_Mnfcs", "base"): 0.5737490772721,
    ("USA", "a_Mnfcs", "check"): 0.573866814794562,
    ("USA", "a_Mnfcs", "shock"): 0.582064529643403,
    ("USA", "a_Svces", "base"): 2.475223454688,
    ("USA", "a_Svces", "check"): 2.47760857784621,
    ("USA", "a_Svces", "shock"): 2.47342234679351,
    ("EU_28", "a_Food", "base"): 2.1325352791219,
    ("EU_28", "a_Food", "check"): 1.9544308367572,
    ("EU_28", "a_Food", "shock"): 1.95870597938654,
    ("EU_28", "a_Mnfcs", "base"): 0.85371807321571,
    ("EU_28", "a_Mnfcs", "check"): 0.855925819747612,
    ("EU_28", "a_Mnfcs", "shock"): 0.841323912703117,
    ("EU_28", "a_Svces", "base"): 2.301447530466,
    ("EU_28", "a_Svces", "check"): 2.31167198822405,
    ("EU_28", "a_Svces", "shock"): 2.31928187711286,
    ("ROW", "a_Food", "base"): 0.95483028868,
    ("ROW", "a_Food", "check"): 0.938423012967756,
    ("ROW", "a_Food", "shock"): 0.942230517475872,
    ("ROW", "a_Mnfcs", "base"): 3.0127091453229,
    ("ROW", "a_Mnfcs", "check"): 3.01695600287339,
    ("ROW", "a_Mnfcs", "shock"): 3.03290056273532,
    ("ROW", "a_Svces", "base"): 0.516904569537,
    ("ROW", "a_Svces", "check"): 0.518049669014743,
    ("ROW", "a_Svces", "shock"): 0.516879257755416,
}
GDXDUMP_PFT_SOME = {
    ("USA", "Land", "base"): 1.0,
    ("USA", "Land", "check"): 0.945923072465676,
    ("EU_28", "SkLab", "shock"): 0.96701677183637,
    ("ROW", "Capital", "shock"): 1.00257706548742,
}

# ---------------------------------------------------------------------------
# Ejemplo 5 — GTAP 3x3 de punta a punta: GDX de GAMS (A) contra un dict al estilo
# de equilibria (B: actividades sin prefijo "a_", solo check y shock).
#
# B = los literales de gdxdump de xp en check/shock, con UNA celda perturbada:
#   B[(USA, Food, shock)] = 1.54573522356493 * 1.002
#   d = |A - B| / |B| = 0.002 / 1.002 = 0.001996007984031936
# Esperado (tol rel 1e-3): 18 parejas (3 reg x 3 act x 2 periodos); 17 dentro de
# 1e-3 (94.444..%), 18 dentro de 1e-2 (100%); maximo en (shock, (USA, Food));
# "base" es periodo solo en A; pft queda excluida con su motivo.
# ---------------------------------------------------------------------------
EX5_B = {
    "xp": {
        (r, a.removeprefix("a_"), t): (
            v * 1.002 if (r, a, t) == ("USA", "a_Food", "shock") else v
        )
        for (r, a, t), v in GDXDUMP_XP.items()
        if t != "base"
    }
}
EX5_MAP = {
    "sets": {"ACT": {"a": {"a_Food": "Food", "a_Mnfcs": "Mnfcs", "a_Svces": "Svces"}}},
    "variables": [
        {
            "name": "output",
            "a": {"var": "xp", "dims": ["REG", "ACT"]},
            "b": {"var": "xp", "dims": ["REG", "ACT"]},
            "tol": {"rel": 1e-3, "abs_floor": 1e-6},
        }
    ],
    "exclude": [{"var": "pft", "why": "ejemplo: excluida a proposito"}],
    "thresholds": [1e-3, 1e-2],
}
EX5_MATCHED = 18
EX5_PCT_WITHIN = {1e-3: 1700.0 / 18, 1e-2: 100.0}
EX5_MAX = 0.001996007984031936
EX5_MAX_CELL = ("shock", ("USA", "Food"))
EX5_PERIODS_ONLY_IN_A = ["base"]

# ---------------------------------------------------------------------------
# Ejemplo 6 — celdas ausentes de un GDX = nivel 0 (decision del usuario, review #126).
#
# GAMS no escribe un registro cuyos atributos estan todos en su default, o sea con
# nivel 0. Regla: en una fuente GDX, una celda que el OTRO lado tiene y el GDX no,
# vale 0 SI la variable del GDX tiene algun registro en ese periodo. Si la variable
# no tiene ningun registro en ese periodo, la celda sigue "sin pareja". Una fuente
# dict/Pyomo nunca rellena: lo que le falta queda "sin pareja".
#
# A = fuente tipo GDX (absent_is_zero), B = dict. w(REG), tol rel 1e-3, piso 1e-6.
# v solo existe para que "base" sea un periodo comun.
#
#   periodo  celda  A (GDX)     B      resultado
#   check    USA    2.0         2.0    pareja, d = 0
#   check    ROW    — (=0)      0.0    pareja, A=0, d = 0/max(0, 1e-6) = 0
#   shock    USA    4.0         —      sin pareja en B (B es dict: no rellena)
#   shock    ROW    — (=0)      0.5    pareja, A=0, d = 0.5/0.5 = 1.0 (no pasa)
#   base     USA    —           1.0    w no tiene registros en "base" en A: sin pareja en A
# ---------------------------------------------------------------------------
EX6_A_LEVELS = {
    "w": {("check", ("USA",)): 2.0, ("shock", ("USA",)): 4.0},
    "v": {("base", ("USA",)): 1.0},
}
EX6_A_NDIM = {"w": 1, "v": 1}
EX6_B = {
    "w": {
        ("USA", "check"): 2.0,
        ("ROW", "check"): 0.0,
        ("ROW", "shock"): 0.5,
        ("USA", "base"): 1.0,
    },
    "v": {("USA", "base"): 1.0},
}
EX6_MAP = {
    "variables": [
        {
            "name": "w",
            "a": {"var": "w", "dims": ["REG"]},
            "b": {"var": "w", "dims": ["REG"]},
            "tol": {"rel": 1e-3, "abs_floor": 1e-6},
        }
    ],
    "thresholds": [1e-3],
}
# (periodo, indices) -> (A, B, d)
EX6_PAIRS = {
    ("check", ("USA",)): (2.0, 2.0, 0.0),
    ("check", ("ROW",)): (0.0, 0.0, 0.0),
    ("shock", ("ROW",)): (0.0, 0.5, 1.0),
}
EX6_MISSING_IN_B = [("w", "shock", ("USA",))]
EX6_MISSING_IN_A = [("w", "base", ("USA",))]

# ---------------------------------------------------------------------------
# Ejemplo 7 — la misma regla sobre el GDX real de GTAP 3x3.
#
# gdxdump ... Symb=pft da 36 registros = 3 regiones x 4 factores moviles x 3
# periodos: NatRes NO esta en el GDX (GAMS no lo escribe: nivel 0).
# B = dict con dos celdas de pft:
#   (USA, Land, base)   = 1.0   GDX 1.0 (gdxdump)  -> d = 0
#   (USA, NatRes, shock) = 0.25  GDX ausente = 0    -> d = 0.25/0.25 = 1.0
# B solo trae base y shock, asi que "check" es periodo solo en A y no se compara
# (12 registros). De los 24 registros de base/shock, 1 se empareja (USA, Land,
# base): los otros 23 faltan en B (dict), sin pareja en B. Ninguna falta en A.
# ---------------------------------------------------------------------------
EX7_B = {"pft": {("USA", "Land", "base"): 1.0, ("USA", "NatRes", "shock"): 0.25}}
EX7_MAP = {
    "variables": [
        {
            "name": "factor_price",
            "a": {"var": "pft", "dims": ["REG", "FP"]},
            "b": {"var": "pft", "dims": ["REG", "FP"]},
            "tol": {"rel": 1e-3, "abs_floor": 1e-6},
        }
    ],
    "thresholds": [1e-3],
}
EX7_PAIRS = {
    ("base", ("USA", "Land")): (1.0, 1.0, 0.0),
    ("shock", ("USA", "NatRes")): (0.0, 0.25, 1.0),
}
EX7_MISSING_IN_A = 0
EX7_MISSING_IN_B = 23
EX7_PERIODS_ONLY_IN_A = ["check"]

# ---------------------------------------------------------------------------
# Ejemplo 8 — Report.to_csv() (spec 3.6): una fila por celda.
#
# Columnas: variable, period, index, a, b, d, status. ``index`` une los elementos
# con "." (como GAMS: USA.Food). status: pass | fail | missing_in_a | missing_in_b;
# una celda sin pareja deja a, b y d vacios. Primero las parejas, luego las sin pareja.
# Valores: los del ejemplo 2 (ref B) y del ejemplo 3, a mano (ver arriba).
# ---------------------------------------------------------------------------
CSV_HEADER = ["variable", "period", "index", "a", "b", "d", "status"]
EX2_CSV_ROWS = [
    ("y", "t1", "USA", 1.0, 1.0, 0.0, "pass"),
    ("y", "t1", "EU", 0.0005, 0.0, 0.5, "fail"),
    ("y", "t1", "ROW", 100.5, 100.0, 0.005, "pass"),
]
EX3_CSV_ROWS = [
    ("z", "check", "USA", 1.0, 1.0, 0.0, "pass"),
    ("z", "shock", "USA", 5.0, 5.0, 0.0, "pass"),
    ("z", "check", "ROW", None, None, None, "missing_in_b"),
    ("z", "shock", "EU", None, None, None, "missing_in_a"),
]
EX1_CSV_INDEX = ["Food.Food", "Food.Mfg", "Mfg.Food", "Mfg.Mfg"]

# ---------------------------------------------------------------------------
# Ejemplo 9 — exclusiones por celda (decision del usuario, opcion B, review #126).
#
# GAMS omite del GDX tanto los ceros como las celdas que su modelo no define (p. ej.
# pft(r, NatRes): GAMS define pft solo para factores moviles). La regla "ausente =
# 0" no distingue: la segunda clase se declara en el mapeo, celda a celda, con su
# motivo. Una celda excluida no se compara, no cuenta como pasa ni como sin pareja,
# y se reporta con el ``why`` y el conteo de su exclusion.
#
# q(REG, FP), A y B dicts, tol rel 1e-3, piso 1e-6. Dos exclusiones, en este orden
# (una celda que caen en las dos se cuenta en la PRIMERA):
#   E1 where {FP: [NatRes]}                       why "NatRes no existe en A"
#   E2 where {REG: [ROW], period: [shock]}        why "ROW en shock: a proposito"
#
#   periodo celda          A      B       sin exclusion          con exclusion
#   check   USA.Lab        1.0    1.0     pareja d=0 pasa        pareja, pasa
#   check   USA.NatRes     0.0    9.0     pareja d=1 falla       E1
#   check   ROW.NatRes     5.0    5.0     pareja d=0 pasa        E1 (aunque pasaria)
#   shock   USA.Lab        2.0    2.002   d=0.002/2.002 pasa     pareja, pasa
#   shock   ROW.Lab        3.0    99.0    d=96/99 falla          E2
#   shock   ROW.NatRes     7.0    —       sin pareja en B        E1 (E2 tambien; gana E1)
#   shock   USA.NatRes     —      1.0     sin pareja en A        E1
#
# Comparadas: 2 celdas, las 2 pasan (pass_pct = 100). Sin pareja: ninguna.
# Excluidas: E1 = 4 celdas, E2 = 1 celda; total 5. 2 + 5 = 7 = celdas de A u B.
# ---------------------------------------------------------------------------
EX9_A = {
    "q": {
        ("USA", "Lab", "check"): 1.0,
        ("USA", "NatRes", "check"): 0.0,
        ("ROW", "NatRes", "check"): 5.0,
        ("USA", "Lab", "shock"): 2.0,
        ("ROW", "Lab", "shock"): 3.0,
        ("ROW", "NatRes", "shock"): 7.0,
    }
}
EX9_B = {
    "q": {
        ("USA", "Lab", "check"): 1.0,
        ("USA", "NatRes", "check"): 9.0,
        ("ROW", "NatRes", "check"): 5.0,
        ("USA", "Lab", "shock"): 2.002,
        ("ROW", "Lab", "shock"): 99.0,
        ("USA", "NatRes", "shock"): 1.0,
    }
}
EX9_WHY_1 = "NatRes no existe en A"
EX9_WHY_2 = "ROW en shock: a proposito"
EX9_MAP = {
    "variables": [
        {
            "name": "q",
            "a": {"var": "q", "dims": ["REG", "FP"]},
            "b": {"var": "q", "dims": ["REG", "FP"]},
            "tol": {"rel": 1e-3, "abs_floor": 1e-6},
        }
    ],
    "exclude": [
        {"var": "q", "where": {"FP": ["NatRes"]}, "why": EX9_WHY_1},
        {"var": "q", "where": {"REG": ["ROW"], "period": ["shock"]}, "why": EX9_WHY_2},
    ],
    "thresholds": [1e-3],
}
EX9_MATCHED = 2
EX9_PASS_PCT = 100.0
EX9_EXCLUSION_COUNTS = [(EX9_WHY_1, 4), (EX9_WHY_2, 1)]
EX9_EXCLUDED_TOTAL = 5
# filas de to_csv con status "excluded" (a/b vacios si la celda falta en ese lado;
# d siempre vacio: no se mide). Se comparan como conjunto.
EX9_CSV_EXCLUDED = {
    ("q", "check", "USA.NatRes", 0.0, 9.0, None, "excluded"),
    ("q", "check", "ROW.NatRes", 5.0, 5.0, None, "excluded"),
    ("q", "shock", "ROW.Lab", 3.0, 99.0, None, "excluded"),
    ("q", "shock", "ROW.NatRes", 7.0, None, None, "excluded"),
    ("q", "shock", "USA.NatRes", None, 1.0, None, "excluded"),
}

# ---------------------------------------------------------------------------
# Ejemplo 10 — exclusion por celda en el GTAP 3x3 real (GAMS vs equilibria resuelto).
#
# Conjuntos del 3x3 (gdxdump): REG = {USA, EU_28, ROW} (3), FP = {Land, SkLab,
# UnSkLab, Capital, NatRes} (5), ACT = {Food, Mnfcs, Svces} (3); periodos del GDX
# comparados: base, check, shock (3).
# Exclusion: factor_price where {FP: [NatRes]} -> 3 reg x 1 factor x 3 periodos = 9.
# factor_price comparadas: 3 x 5 x 3 = 45 celdas en total, menos 9 = 36, todas pasan.
# output: 3 reg x 3 act x 3 periodos = 27 comparadas, todas pasan.
# ---------------------------------------------------------------------------
EX10_OUTPUT_CELLS = 27
EX10_FACTOR_PRICE_CELLS = 36
EX10_NATRES_EXCLUDED = 9

# ---------------------------------------------------------------------------
# Ejemplo 11 — la CLI ``equilibria compare`` (issue #130), sobre los ejemplos 3 y 5.
#
# --fail-below PCT compara PCT con el % de celdas que PASAN sobre TODAS las celdas
# que el mapeo pide comparar: parejas + sin pareja (una celda que falta nunca pasa,
# spec Q5). Las excluidas por el mapeo no cuentan (no se miden).
#
# Ejemplo 5 (GDX de GAMS contra el dict B escrito como JSON): 18 parejas, 17 pasan
# (tol 1e-3; la perturbada tiene d = 0.001996...), 0 sin pareja -> 17/18 = 94.444..%.
#   --fail-below 94 -> sale 0;  --fail-below 95 -> sale 1.
# Ejemplo 3 (dos JSON): 2 parejas que pasan (d = 0) + 1 sin pareja en A + 1 sin
# pareja en B -> 2/4 = 50%. Sobre las parejas solas seria 100%: si la CLI ignorara
# las sin pareja, --fail-below 60 saldria 0.
#   --fail-below 50 -> sale 0;  --fail-below 60 -> sale 1.
# Sin --fail-below la CLI sale 0 aunque haya celdas que fallan.
# ---------------------------------------------------------------------------
EX11_EX5_PASS_PCT_ALL = 1700.0 / 18
EX11_EX5_PASS_PCT_TEXT = "94.44"  # como lo imprime la CLI (2 decimales)
EX11_EX3_PASS_PCT_ALL = 50.0
EX11_FAIL_EXIT = 1
EX11_ERROR_EXIT = 2  # error de uso o de datos (mapeo, fuente, periodo no declarado)
EX11_EX3_PASS_PCT_ALL_TEXT = (
    "50.00"  # la fila TOTAL muestra el mismo % que --fail-below
)

# Ejemplo 12 — from_har: el .har de NIVELES de GEMPACK (issue #131).
#
# HAR escrito en el test con equilibria.babel.har.writer (formato, no numeros: los
# valores esperados son los literales de abajo). Variable qo(ACTS, REG), ACTS =
# {Food, Mnfcs}, REG = {USA}:
#   LEVA "0001" (post) = [10.0, 20.0]   LEVB "B001" (pre) = [8.0, 16.0]
# NOLV = ["psave"]: psave no tiene nivel.
# Periodos: pre (LEVB) y post (LEVA), en ese orden.
#   qo[pre,  (Food, USA)] = 8.0    qo[pre,  (Mnfcs, USA)] = 16.0
#   qo[post, (Food, USA)] = 10.0   qo[post, (Mnfcs, USA)] = 20.0
# Si el mapeo pide psave: no se compara; queda en Report.excluded con el motivo
# EX12_NO_LEVEL_WHY. El mismo motivo para un .har VIEJO (sin B***/NOLV) cuya variable
# trae 1e10 en todas sus celdas. 1e10 en solo ALGUNAS celdas: error.
# Un .har con las cabeceras XXCR..XXTT (cambios % de sltoht) o un .sl4: error con la
# instruccion de conversion (export_sl4_levels.py).
# ---------------------------------------------------------------------------
EX12_QO = {
    ("pre", ("Food", "USA")): 8.0,
    ("pre", ("Mnfcs", "USA")): 16.0,
    ("post", ("Food", "USA")): 10.0,
    ("post", ("Mnfcs", "USA")): 20.0,
}
EX12_PERIODS = ["pre", "post"]
EX12_LEVA_ONLY_PERIODS = ["post"]
EX12_NO_LEVEL_WHY = "sin nivel en GEMPACK"
EX12_CONVERSION_HINT = "export_sl4_levels.py"

# ---------------------------------------------------------------------------
# Ejemplo 13 — divide_by: deshacer las escalas de una fuente (decision 2026-10-18).
#
# A (al estilo GAMS/equilibria) guarda xp(r, a, t) escalado por inScale * xscale[r,a]:
#   xp[USA, a_Food,  shock] = 2.0    xscale[USA, a_Food]  = 1.0
#   xp[USA, a_Mnfcs, shock] = 0.5    xscale[USA, a_Mnfcs] = 0.1
#   divide_by: [{param: xscale, dims: [r, a]}, {value: 1e-6, why: "inScale"}]
#   nivel comparado: 2.0 / (1.0 * 1e-6) = 2.0e6 ;  0.5 / (0.1 * 1e-6) = 5.0e6
# B (al estilo GEMPACK) qo(a, r) en millones, periodo post:
#   qo[Food, USA] = 2.002e6   qo[Mnfcs, USA] = 5.0e6
# d (ref B): |2.0e6 - 2.002e6| / 2.002e6 = 0.002/2.002 = 9.99000999e-4 -> pasa (tol 1e-3)
#            |5.0e6 - 5.0e6| / 5.0e6 = 0
# Sin divide_by, Food daria d = |2 - 2.002e6|/2.002e6 ~ 0.999999: falla.
# Sin el parametro xscale en A: error (nunca se asume 1).
# ---------------------------------------------------------------------------
EX13_A = {"xp": {("USA", "a_Food", "shock"): 2.0, ("USA", "a_Mnfcs", "shock"): 0.5}}
EX13_A_PARAMS = {"xscale": {("USA", "a_Food"): 1.0, ("USA", "a_Mnfcs"): 0.1}}
EX13_B = {"qo": {("Food", "USA", "post"): 2.002e6, ("Mnfcs", "USA", "post"): 5.0e6}}
EX13_INSCALE_WHY = "inScale: el modelo no lo guarda como parametro"
EX13_MAP = {
    "periods": {"a": {"shock": "post"}, "b": {"post": "post"}},
    "sets": {"ACT": {"a": {"a_Food": "Food", "a_Mnfcs": "Mnfcs"}, "b": {}}},
    "variables": [
        {
            "name": "output",
            "a": {
                "var": "xp",
                "dims": ["REG", "ACT"],
                "divide_by": [
                    {"param": "xscale", "dims": ["REG", "ACT"]},
                    {"value": 1e-6, "why": EX13_INSCALE_WHY},
                ],
            },
            "b": {"var": "qo", "dims": ["ACT", "REG"]},
            "tol": {"rel": 1e-3, "abs_floor": 1e-6},
        }
    ],
    "thresholds": [1e-3],
}
EX13_A_UNSCALED = {("post", ("USA", "Food")): 2.0e6, ("post", ("USA", "Mnfcs")): 5.0e6}
EX13_D_FOOD = 0.002 / 2.002
EX13_PASS_PCT = 100.0

# ---------------------------------------------------------------------------
# Ejemplo 14 — GEMPACK <-> equilibria, GTAP 3x3 en NIVELES (issue #131).
#
# A = equilibria resuelto (bloques, PATH, el cierre de test_gtap7_gempack_parity:
# _solve_shock("gtap7_3x3", 0), capFlex); B = sl4levels_gtap7_3x3_tm10.har (LEVA-only,
# solo post). Periodos: A shock -> post. Cantidades de A deshechas a mano:
#   output:   xp(r,a)  / (xscale[r,a] * 1e-6)  <->  qo(a,r)   (xscale de
#             blocks.gtap._derived_params.xscale_data, NO de equilibria.compare)
#   dom_sales: xd(r,c) / 1e-6                   <->  qds(c,r)
# Calculado el 2026-10-18 con tests/compare/oracles/gempack3x3_levels.py (no importa
# equilibria.compare): lee el HAR con babel.har.reader y el modelo con pyomo.value.
#   output:    9 celdas, max d = 2.51e-5, las 9 <= 1e-3
#   dom_sales: 9 celdas, 7 <= 1e-3; Svces EU_28 d = 0.010465, Svces ROW d = 0.004060
#   global: 18 celdas; <= 1e-3: 16; <= 5e-3: 17; <= 1e-2: 17;
#           maximo 0.010465 en dom_sales post (EU_28, Svces)
# Hallazgo (no del comparador): xd se aparta de qds hasta ~1 % en Svces.
# ---------------------------------------------------------------------------
EX14_HAR = (
    ROOT / "tests" / "fixtures" / "gtap7_gempack" / "sl4levels_gtap7_3x3_tm10.har"
)
EX14_CELLS = {"output": 9, "dom_sales": 9}
EX14_WITHIN = {1e-3: 16, 5e-3: 17, 1e-2: 17}
EX14_TOTAL = 18
EX14_OUTPUT_MAX = 2.51e-5
EX14_MAX = 0.010465
EX14_MAX_CELL = ("dom_sales", ("post", ("EU_28", "Svces")))

# ---------------------------------------------------------------------------
# Ejemplo 15 — niveles leidos DIRECTO del .sl4, sin sltoht (issue #148).
#
# Dos oraculos, ninguno sale del lector nuevo:
# 1) Nivel PRE del nus333. Los fixtures de sltoht del nus333 (tm10) dan el post
#    (sl4levels_nus333_tm10.har, LEVA) y el % (sl4dump_nus333_tm10.har); pre =
#    post / (1 + %/100). El pre depende solo de los datos, que son los mismos en
#    todo Burfisher, asi que es el pre de TBL45A. Las 44 variables con nivel del
#    nus333 son de cambio %, ninguna de cambio ordinario.
#    Variables sin nivel: las que el fixture trae con el centinela 1e10. El .sl4
#    trae ademas 12 variables que el fixture no tiene (otra version del modelo).
# 2) Cambio: 100*(post/pre - 1) de cada celda = el volcado de sltoht del mismo .sl4
#    (burfisher_sl4/sl4dump_<EXP>.har). Una celda con pre = 0 no tiene %: GEMPACK
#    escribe uno igual (qes/qfe, flujo cero), pero en niveles es 0 -> 0.
# Precision: LEVB/LEVA del .sl4 son REAL*4 (~6e-8 relativo).
# Medido el 2026-10-19 con un lector de exploracion (scratch, no commiteado): pre
# rel max 4.8e-8 (qo); cambio max 1.04e-5 pp (TBL45A) y 3.05e-5 pp (ME9C).
# ---------------------------------------------------------------------------
EX15_DIR = ROOT / "tests" / "fixtures" / "gtap7_gempack" / "burfisher_sl4"
# GTAPV7 (gtapv7.for, mismo nus333) OMITE 11 variables: VARS/VCNA/VCAR/LVOK/PLEV
# cubren las 252 de la solucion, VCSP/VCSN las 263 del modelo. En TBL45A y ME9C
# las dos listas coinciden y no distinguen cual se usa.
EX15_EXERCISES = ("TBL45A", "ME9C", "GTAPV7")
EX15_SL4 = {e: EX15_DIR / f"{e}.sl4" for e in EX15_EXERCISES}
EX15_DUMP = {e: EX15_DIR / f"sl4dump_{e}.har" for e in EX15_EXERCISES}
EX15_NUS333_POST = (
    ROOT / "tests" / "fixtures" / "gtap7_gempack" / "sl4levels_nus333_tm10.har"
)
EX15_NUS333_PCT = (
    ROOT / "tests" / "fixtures" / "gtap7_gempack" / "sl4dump_nus333_tm10.har"
)
EX15_N_WITH_LEVEL = 44
EX15_PRE_RTOL = 1e-6
EX15_PCT_ATOL = 1e-4  # puntos porcentuales
# qo pre (ACTS x REG), del oraculo 1, para leer a ojo:
EX15_QO_PRE = {
    ("AGR", "USA"): 324849.25648047,
    ("MFG", "USA"): 6587561.07403989,
    ("SER", "USA"): 17701492.85617938,
    ("AGR", "ROW"): 2898823.33469239,
    ("MFG", "ROW"): 31702725.67132671,
    ("SER", "ROW"): 46936497.20200284,
}
EX15_NO_LEVEL_WHY = "sin nivel en GEMPACK"
