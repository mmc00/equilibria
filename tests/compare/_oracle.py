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
# Los otros 35 registros del GDX (36 - (USA, Land, base)) faltan en B (dict): sin
# pareja en B. Ninguna celda falta en A.
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
EX7_MISSING_IN_B = 35

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
