#!/usr/bin/env bash
# Build the Ghana (GHA) data and run MANAGE-WB for the base year plus one
# solved year on macOS, without the GUI and without GDXXRW.
#
#   ./run_gha_one_year.sh
#
# Needs: GAMS 53 (GAMS_SYSDIR below), LibreOffice (soffice, to recalculate
# the bridge workbook after it is written). Each GAMS run gets reslim=600 and
# is timed with /usr/bin/time -l (wall time, maximum resident set size).
#
# Steps
#   1. gtapSamV6.gms  ("Prepare GTAP SAM"): dat/gtapData9.gdx ->
#      reg/GHA/BaseBridge_GHA_GTAP_9.xlsx and reg/GHA/SAM_GHA_GTAP.xlsx
#   2. mac/fix_bridge.py: blank the template placeholder Eps (see the file)
#   3. run.gms, benchmark check (benchmarkCheck=on): the model is generated at
#      the calibrated values with iterlim=0 and run.gms stops on purpose
#      ("Test solve with zero iterations"); the log shows the residual
#   4. run.gms, balanced growth check (BGCheck=on): one solved year
#      -> res/GHA/GHA_BaU_BGCheck.gdx
#   5. run.gms, baseline with shock file none.inc: one solved year
#      -> res/GHA/GHA_BaU.gdx; CONVERT writes the instance to mac/run/neos/
#   6. mac/make_neos_model.py -> mac/neos/gha2015_path.gms (PATH, for NEOS)
#
# The bridge xlsx -> sav/BaseBridge_GHA_GTAP_9.gdx conversion happens inside
# run.gms (inc/opt.inc). The base year comes from the bridge file (set
# BaseYear); the last year is base year + 1.
#
# Solver: MCP solved by NLPEC with IPOPT as subsolver (mac/opt/nlpec.opt).
# PATH and CONOPT4 are not usable here: the local GAMS license only allows
# them for demo-size models.
#
# Optional environment:
#   BRIDGE=<file.xlsx>  skip steps 1-2 and run from this bridge file (a recorded
#                       reg/GHA/BaseBridge_GHA_GTAP_9.xlsx); the shock files are
#                       copied to reg/GHA/sim as gtapSamV6.gms would
#   OUT=<dir>           copy the baseline results there: GHA_BaU.gdx, the
#                       baseline .lst/.log, the CONVERT start and solution
#                       points, the NEOS file and its dictionary
set -euo pipefail

BRIDGE=${BRIDGE:+$(cd "$(dirname "$BRIDGE")" && pwd)/$(basename "$BRIDGE")}
OUT=${OUT:+$(mkdir -p "$OUT" && cd "$OUT" && pwd)}

GAMS_SYSDIR=${GAMS_SYSDIR:-/Library/Frameworks/GAMS.framework/Versions/53/Resources}
export PATH="$GAMS_SYSDIR:$PATH"
PY="$GAMS_SYSDIR/GMSPython/bin/python3"

cd "$(dirname "$0")"
ROOT=$(pwd)
RUN="$ROOT/mac/run"
LOGS="$RUN/logs"

command -v gams >/dev/null || { echo "gams not found in $GAMS_SYSDIR"; exit 1; }
command -v soffice >/dev/null || [ -x /Applications/LibreOffice.app/Contents/MacOS/soffice ] \
  || { echo "LibreOffice (soffice) not found"; exit 1; }

# --- clean generated files from earlier runs
rm -rf "$RUN" reg/GHA res/GHA sav/BaseBridge_GHA_GTAP_9.gdx fromGuiToRun.gms
mkdir -p "$LOGS" reg sav res gui mac/neos

# --- option directory: the model's opt/ plus the macOS NLPEC options
mkdir -p "$RUN/opt"
cp opt/* "$RUN/opt/"
cp mac/opt/* "$RUN/opt/"

# run_gams <name> <file.gms> [allowed abort text]
run_gams() {
  local name=$1 gms=$2 allowed=${3:-}
  local rc=0
  echo "=== $name: gams $gms"
  /usr/bin/time -l gams "$gms" reslim=600 optdir="$RUN/opt" lo=3 \
      o="$LOGS/$name.lst" > "$LOGS/$name.log" 2> "$LOGS/$name.time" || rc=$?
  local wall rss
  wall=$(awk '/ real /{print $1}' "$LOGS/$name.time")
  rss=$(awk '/maximum resident set size/{printf "%.0f", $1/1048576}' "$LOGS/$name.time")
  echo "    rc=$rc wall=${wall}s maxRSS=${rss}MB log=$LOGS/$name.log"
  if [ $rc -ne 0 ]; then
    if [ -n "$allowed" ] && grep -q "$allowed" "$LOGS/$name.log"; then
      echo "    stopped as designed: $allowed"
    else
      tail -20 "$LOGS/$name.log"
      exit $rc
    fi
  fi
}

baseline_settings() {
  # $1 = last year, then extra sed expressions
  local last=$1; shift
  sed -e "s/@LASTYEAR@/$last/" -e "s/@SOLVETYPE@/MCP/" -e "s/@MCPSOLVER@/NLPEC/" "$@" \
      mac/gui_baseline_one_year.gms
  cat mac/gui_meta_items.gms
}

if [ -z "$BRIDGE" ]; then
  # --- 1. GTAP SAM and bridge file
  cp mac/gui_prepare_gtap_sam.gms fromGuiToRun.gms
  run_gams gtapSamV6 gtapSamV6.gms

  # --- 2. template placeholder
  "$PY" mac/fix_bridge.py reg/GHA/BaseBridge_GHA_GTAP_9.xlsx
else
  echo "=== bridge from $BRIDGE (steps 1-2 skipped)"
  mkdir -p reg/GHA/sim
  cp "$BRIDGE" reg/GHA/BaseBridge_GHA_GTAP_9.xlsx
  cp inc/shockFiles/*.inc reg/GHA/sim/
fi

BASEYEAR=$("$PY" - <<'EOF'
import openpyxl
ws = openpyxl.load_workbook("reg/GHA/BaseBridge_GHA_GTAP_9.xlsx", data_only=True)["sets"]
col = [c for c in range(1, ws.max_column + 1) if ws.cell(5, c).value == "BaseYear"][0]
print(int(ws.cell(6, col).value))
EOF
)
LASTYEAR=$((BASEYEAR + 1))
echo "=== base year $BASEYEAR, solved year $LASTYEAR"

# --- 3. benchmark check
baseline_settings $LASTYEAR > fromGuiToRun.gms
run_gams check run.gms "Test solve with zero iterations"
grep -E "Constraint violation|Initial deviation" "$LOGS/check.log" | sed 's/^/    /' || true

# --- 4. balanced growth check
baseline_settings $LASTYEAR -e 's/^\$setglobal benchmarkCheck on/$setglobal benchmarkCheck off/' \
                            -e 's/^\$setglobal BGCheck off/$setglobal BGCheck on/' > fromGuiToRun.gms
run_gams bgcheck run.gms
mv res/GHA/GHA_BaU.gdx res/GHA/GHA_BaU_BGCheck.gdx

# --- 5. baseline; convertMCP=on also writes the 2015 instance with CONVERT
#        (start point and solution, mac/run/neos/) without changing the results
mkdir -p "$RUN/neos"
{ baseline_settings $LASTYEAR -e 's/^\$setglobal benchmarkCheck on/$setglobal benchmarkCheck off/'
  echo '$setglobal convertMCP on'; } > fromGuiToRun.gms
run_gams baseline run.gms

# --- 6. NEOS file: the scalar model with PATH (submit with mac/neos_submit.py)
"$PY" mac/make_neos_model.py "$RUN/neos/gha2015_scalar.gms" mac/neos/gha2015_path.gms
cp "$RUN/neos/gha2015_dict.txt" mac/neos/gha2015_dict.txt

# --- report
for name in bgcheck baseline; do
  echo "=== $name solves"
  grep -E "SOLVE cge|SOLVER STATUS|MODEL STATUS" "$LOGS/$name.lst" | grep -A2 "Solution Report" | sed 's/^/    /' || true
  grep -E "solved with" "$LOGS/$name.log" | sed 's/^/    /' || true
done
ls -l res/GHA/*.gdx sav/BaseBridge_GHA_GTAP_9.gdx reg/GHA/*.xlsx

if [ -n "$OUT" ]; then
  cp res/GHA/GHA_BaU.gdx "$LOGS/baseline.lst" "$LOGS/baseline.log" \
     "$RUN/neos/gha2015_scalar.gms" "$RUN/neos/gha2015_solution.gms" \
     mac/neos/gha2015_path.gms mac/neos/gha2015_dict.txt "$OUT/"
  echo "=== results copied to $OUT"
fi

# run.gms rewrites this tracked file (same content, LF line ends)
git checkout -- inc/model_def_copy.inc 2>/dev/null || true
