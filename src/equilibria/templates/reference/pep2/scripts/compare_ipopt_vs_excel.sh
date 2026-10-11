#!/usr/bin/env bash
set -euo pipefail

# El GAMS instalado, no una version fija: la v48 solo existia en el Mac del
# autor y ademas ya no pasa el servidor de licencias. Se puede forzar con
# GAMS_BIN=/ruta/a/gams.
_gams_dir() {
  for d in /Library/Frameworks/GAMS.framework/Versions/Current/Resources \
           /Library/Frameworks/GAMS.framework/Versions/*/Resources \
           /opt/gams/* /usr/local/gams/*; do
    [ -x "$d/gams" ] && { echo "$d"; return; }
  done
  command -v gams >/dev/null 2>&1 && dirname "$(command -v gams)"
}
GAMS_DIR="$(_gams_dir)"
GAMS_BIN="${GAMS_BIN:-$GAMS_DIR/gams}"
GDXDIFF="$GAMS_DIR/gdxdiff"
GDXDUMP="$GAMS_DIR/gdxdump"
WORKDIR="$(cd "$(dirname "$0")" && pwd)"

cd "$WORKDIR"

echo "[1/4] Running baseline model: PEP-1-1_v2_1_ipopt.gms"
"$GAMS_BIN" PEP-1-1_v2_1_ipopt.gms lo=0
cp Results.gdx Results_ipopt.gdx
cp Parameters.gdx Parameters_ipopt.gdx

echo "[2/4] Running Excel-loading model: PEP-1-1_v2_1_ipopt_excel.gms"
"$GAMS_BIN" PEP-1-1_v2_1_ipopt_excel.gms lo=0
cp Results.gdx Results_ipopt_excel.gdx
cp Parameters.gdx Parameters_ipopt_excel.gdx

echo "[3/4] Comparing Results.gdx"
"$GDXDIFF" \
  Results_ipopt.gdx Results_ipopt_excel.gdx \
  > gdxdiff_results_ipopt_vs_excel.txt || true

echo "[4/4] Comparing Parameters.gdx"
"$GDXDIFF" \
  Parameters_ipopt.gdx Parameters_ipopt_excel.gdx \
  > gdxdiff_params_ipopt_vs_excel.txt || true

# diffile.gdx is overwritten by the latest gdxdiff call; regenerate for results diff detail
"$GDXDIFF" \
  Results_ipopt.gdx Results_ipopt_excel.gdx \
  > /dev/null || true
"$GDXDUMP" diffile.gdx symb=valSH format=csv > dif_valSH.csv || true
"$GDXDUMP" diffile.gdx symb=valTR format=csv > dif_valTR.csv || true
"$GDXDUMP" diffile.gdx symb=valYHTR format=csv > dif_valYHTR.csv || true

echo "Done. Files generated:"
echo "  - gdxdiff_results_ipopt_vs_excel.txt"
echo "  - gdxdiff_params_ipopt_vs_excel.txt"
echo "  - dif_valSH.csv"
echo "  - dif_valTR.csv"
echo "  - dif_valYHTR.csv"
