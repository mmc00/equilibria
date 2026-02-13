#!/usr/bin/env bash
set -euo pipefail

GAMS_BIN="${GAMS_BIN:-/Library/Frameworks/GAMS.framework/Versions/48/Resources/gams}"
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
/Library/Frameworks/GAMS.framework/Versions/48/Resources/gdxdiff \
  Results_ipopt.gdx Results_ipopt_excel.gdx \
  > gdxdiff_results_ipopt_vs_excel.txt || true

echo "[4/4] Comparing Parameters.gdx"
/Library/Frameworks/GAMS.framework/Versions/48/Resources/gdxdiff \
  Parameters_ipopt.gdx Parameters_ipopt_excel.gdx \
  > gdxdiff_params_ipopt_vs_excel.txt || true

# diffile.gdx is overwritten by the latest gdxdiff call; regenerate for results diff detail
/Library/Frameworks/GAMS.framework/Versions/48/Resources/gdxdiff \
  Results_ipopt.gdx Results_ipopt_excel.gdx \
  > /dev/null || true
/Library/Frameworks/GAMS.framework/Versions/48/Resources/gdxdump diffile.gdx symb=valSH format=csv > dif_valSH.csv || true
/Library/Frameworks/GAMS.framework/Versions/48/Resources/gdxdump diffile.gdx symb=valTR format=csv > dif_valTR.csv || true
/Library/Frameworks/GAMS.framework/Versions/48/Resources/gdxdump diffile.gdx symb=valYHTR format=csv > dif_valYHTR.csv || true

echo "Done. Files generated:"
echo "  - gdxdiff_results_ipopt_vs_excel.txt"
echo "  - gdxdiff_params_ipopt_vs_excel.txt"
echo "  - dif_valSH.csv"
echo "  - dif_valTR.csv"
echo "  - dif_valYHTR.csv"
