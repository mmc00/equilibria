"""Bipartite matching of PEP equations ↔ variables for the MCP pairing.

For each EQn: its domain (from declaration) + the set of ENDOGENOUS variables that
appear in its body, restricted to those whose declared domain CONFORMS to the eq domain.
Then a maximum bipartite matching (Hopcroft-Karp) assigns each eq a unique variable,
preferring the LHS variable. Guarantees a square, dimension-consistent pairing — exactly
what GAMS MCP needs. Prints the pairing + any unmatched (structural singularity).
"""
import re
from pathlib import Path
from collections import defaultdict

SRC = Path("/Users/marmol/Downloads/pep-1-1/GAMS_Code/PEP-1-1_v2_1.gms")
text = SRC.read_text()
lines = text.splitlines()

# --- variable declarations: name → arity (number of index sets) ---
# PEP declares vars in POSITIVE VARIABLE / VARIABLE blocks: " VAR(i,j)  desc"
var_arity = {}
FIXED = {"e", "LS", "KS", "PWM", "PWX", "CMIN", "VSTK", "G", "CAB",
         "sh1", "tr1", "ttdf1", "ttdh1", "ttiw", "ttix"}  # numeraire + closure-fixed
in_var = False
for ln in lines:
    if re.match(r"^(POSITIVE |FREE )?VARIABLE", ln.upper()):
        in_var = True; continue
    if in_var and (ln.strip() == ";" or ln.strip().endswith(";") and not re.match(r"^ [A-Z]", ln)):
        in_var = False
    m = re.match(r"^ ([A-Za-z_]+)(\(([^)]*)\))?\s", ln)
    if m and in_var:
        name = m.group(1)
        idx = m.group(3)
        var_arity[name] = len([x for x in idx.split(",") if x]) if idx else 0

# --- equation domains ---
eq_dom = {}
for ln in lines:
    m = re.match(r"^ (EQ\d+|WALRAS)(\(([^)]*)\))?\s{2,}\S", ln)
    if m:
        idx = m.group(3)
        eq_dom.setdefault(m.group(1), len([x for x in idx.split(",") if x]) if idx else 0)

# --- equation bodies: which vars appear ---
body = defaultdict(set)
lhs = {}
cur = None
for ln in lines:
    m = re.match(r"^ (EQ\d+|WALRAS)(\([^)]*\))?(\$[^.]*)?\.\.", ln)
    if m:
        cur = m.group(1)
        rest = ln.split("..", 1)[1]
        lm = re.match(r"\s*([A-Za-z_]+)", rest)
        if lm:
            lhs[cur] = lm.group(1)
    if cur:
        for tok in re.findall(r"[A-Za-z_]+", ln):
            if tok in var_arity:
                body[cur].add(tok)
        if ";" in ln:
            cur = None

# --- build adjacency: eq → candidate vars (arity conforms, not fixed) ---
adj = {}
for eq, dom in eq_dom.items():
    cands = [v for v in body[eq] if var_arity.get(v) == dom and v not in FIXED]
    # WALRAS ⊥ LEON specifically
    if eq == "WALRAS":
        cands = ["LEON"]
    adj[eq] = cands

# --- Hopcroft-Karp-ish greedy + augmenting, prefer LHS ---
match_v = {}
def try_assign(eq, seen):
    # prefer LHS var first
    cands = adj[eq]
    if lhs.get(eq) in cands:
        cands = [lhs[eq]] + [c for c in cands if c != lhs[eq]]
    for v in cands:
        if v in seen: continue
        seen.add(v)
        if v not in match_v or try_assign(match_v[v], seen):
            match_v[v] = eq
            return True
    return False

for eq in sorted(eq_dom, key=lambda e: (e != "WALRAS", int(e[2:]) if e[2:].isdigit() else 0)):
    try_assign(eq, set())

pairs = {eq: v for v, eq in match_v.items()}
unmatched = [eq for eq in eq_dom if eq not in pairs]
print("MATCHED:", len(pairs), " UNMATCHED:", len(unmatched))
order = sorted(pairs, key=lambda e: (e != "WALRAS", int(e[2:]) if e[2:].isdigit() else 0))
print(" ".join(f"{eq}.{pairs[eq]}" for eq in order))
if unmatched:
    print("UNMATCHED eqs:", unmatched)
    for eq in unmatched:
        print(f"  {eq} (dom {eq_dom[eq]}) candidates:", adj[eq])
