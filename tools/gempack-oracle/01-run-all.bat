@echo off
REM ===================================================================
REM  01-run-all.bat - corre los 45 .EXP de NUS333 y vuelca TODO.
REM
REM  Por cada experimento produce en out\<NOMBRE>\ :
REM     <NOMBRE>.sl4   solucion (cambios %) -- lo que imprime el libro
REM     <NOMBRE>.upd   DATOS ACTUALIZADOS = NIVELES post-shock  <-- lo clave
REM     <NOMBRE>.log   log completo del solver
REM     <NOMBRE>.cmf   el command file usado (para auditar)
REM     SUMMARY.har / DECOMP.har / GTAPVol.har  si el modelo los emite
REM
REM  Y ademas, UNA sola vez, los NIVELES BASE en out\_base\.
REM
REM  No se detiene ante un experimento que falle: lo anota y sigue.
REM ===================================================================
setlocal enabledelayedexpansion

if not defined NUS333_DIR set NUS333_DIR=%~dp0nus333
if not defined GTAP_MODEL_DIR (
  echo ERROR: GTAP_MODEL_DIR no definida. Corre 00-check-env.bat primero.
  exit /b 1
)
set OUT=%~dp0out
REM  Los .EXP de NUS333 declaran el cierre de GTAPUv7 (usa atall/avaall/tfe/tfd/
REM  tgd/tid/... que el GTAPV7 condensado NO tiene). Por eso el modelo por defecto
REM  es GTAPUV7; override con GTAP_MODEL_NAME si hace falta otro.
if not defined GTAP_MODEL_NAME set GTAP_MODEL_NAME=GTAPUV7
set MODEL=%GTAP_MODEL_DIR%\%GTAP_MODEL_NAME%

if not exist "%OUT%" mkdir "%OUT%"
set REPORT=%OUT%\_report.txt
echo Corrida iniciada %DATE% %TIME% > "%REPORT%"

REM ---------- compilar el modelo si hace falta ----------
if not exist "%MODEL%.EXE" (
  echo Compilando %GTAP_MODEL_NAME%.TAB con TABLO...
  pushd "%GTAP_MODEL_DIR%"
  tablo -pgs %GTAP_MODEL_NAME%.TAB > "%OUT%\_tablo.log" 2>&1
  if errorlevel 1 (
    echo ERROR: TABLO fallo. Ver %OUT%\_tablo.log
    popd & exit /b 1
  )
  popd
  echo   modelo compilado.
)

REM ---------- 1) NIVELES BASE (sin shock) ----------
REM  Un .CMF sin shock alguno: el .UPD resultante son los niveles del
REM  benchmark tal como los ve GEMPACK. Es el punto de comparacion
REM  contra el que equilibria calibra.
echo.
echo === [0/45] niveles BASE (sin shock) ===
set BASEDIR=%OUT%\_base
if not exist "%BASEDIR%" mkdir "%BASEDIR%"
set CMF=%BASEDIR%\base.cmf
(
  echo auxiliary files = "%GTAP_MODEL_DIR%\%GTAP_MODEL_NAME%";
  echo file GTAPSETS = "%NUS333_DIR%\sets.har";
  echo file GTAPDATA = "%NUS333_DIR%\basedata.har";
  echo file GTAPPARM = "%NUS333_DIR%\default.prm";
  echo file GTAPSUM  = "%BASEDIR%\SUMMARY.har";
  echo file WELVIEW  = "%BASEDIR%\DECOMP.har";
  echo file GTAPVOL  = "%BASEDIR%\GTAPVol.har";
  echo updated file GTAPDATA = "%BASEDIR%\base.upd";
  echo solution file = "%BASEDIR%\base";
  echo Method = Johansen;
  echo Steps = 1;
  echo automatic accuracy = no;
  echo CPU = yes;
  echo NDS = yes;
  echo Exogenous pop psaveslack pfactwld profitslack incomeslack endwslack
  echo    cgdslack tradslack ams atm atf ats atd aosec aoreg avasec avareg
  echo    aintsec aintreg aintall afcom afsec afreg afecom afesec afereg
  echo    aoall afall afeall au dppriv dpgov dpsave to tinc tpreg tm tms
  echo    tx txs qe qesf atall avaall tfe tfd tfm tgd tgm tpdall tpmall tid tim;
  echo Rest endogenous;
  REM  GEMPACK RECHAZA un .cmf sin ningun shock ("E-No shock statements"), asi que
  REM  el base lleva un shock de magnitud CERO: con Johansen 1 paso la solucion es
  REM  identicamente nula y el .UPD son los niveles del benchmark, que es el objetivo.
  echo Shock tm = uniform 0;
  echo verbal description = NUS333 base, shock nulo, para extraer niveles;
) > "%CMF%"
"%MODEL%.EXE" -cmf "%CMF%" > "%BASEDIR%\base.log" 2>&1
if errorlevel 1 (echo   [FALLO] base & echo BASE FALLO >> "%REPORT%") else (echo   [ok] base)

REM ---------- 2) los 45 experimentos ----------
set /a N=0, OK=0, BAD=0
for %%E in ("%NUS333_DIR%\*.EXP") do (
  set /a N+=1
  set NAME=%%~nE
  set D=%OUT%\!NAME!
  if not exist "!D!" mkdir "!D!"
  echo.
  echo === [!N!/45] !NAME! ===

  REM  GTAPV7.EXP declara el cierre del modelo CONDENSADO; el resto es GTAPUv7.
  set MDL=%GTAP_MODEL_NAME%
  if /I "!NAME!"=="GTAPV7" set MDL=GTAPV7
  set CMF=!D!\!NAME!.cmf
  (
    echo auxiliary files = "%GTAP_MODEL_DIR%\!MDL!";
    echo file GTAPSETS = "%NUS333_DIR%\sets.har";
    echo file GTAPDATA = "%NUS333_DIR%\basedata.har";
    echo file GTAPSUM  = "!D!\SUMMARY.har";
    echo file WELVIEW  = "!D!\DECOMP.har";
    echo file GTAPVOL  = "!D!\GTAPVol.har";
    echo updated file GTAPDATA = "!D!\!NAME!.upd";
    echo solution file = "!D!\!NAME!";
    echo CPU = yes;
    echo NDS = yes;
  ) > "!CMF!"
  REM  El .EXP trae: GTAPPARM, Method, Steps, closure y Shock.
  REM  Se anexa verbatim para no reinterpretarlo. Las lineas !@ son
  REM  comentarios de RunGTAP y GEMPACK las ignora.
  type "%%E" >> "!CMF!"

  REM  pushd: el .EXP referencia su .prm por nombre relativo (ver arriba).
  pushd "%NUS333_DIR%"
  "%GTAP_MODEL_DIR%\!MDL!.EXE" -cmf "!CMF!" > "!D!\!NAME!.log" 2>&1
  popd
  if errorlevel 1 (
    set /a BAD+=1
    echo   [FALLO] !NAME!  -- ver !D!\!NAME!.log
    echo FALLO !NAME! >> "%REPORT%"
  ) else (
    set /a OK+=1
    echo   [ok] !NAME!
    echo OK !NAME! >> "%REPORT%"
    REM  volcar .sl4 y .upd a texto plano, asi se leen sin GEMPACK
    if exist "!D!\!NAME!.sl4" sltoht "!D!\!NAME!.sl4" "!D!\!NAME!.sl4.txt" >nul 2>&1
    if exist "!D!\!NAME!.upd" sltoht "!D!\!NAME!.upd" "!D!\!NAME!.upd.txt" >nul 2>&1
  )
)

if exist "%OUT%\_base\base.upd" sltoht "%OUT%\_base\base.upd" "%OUT%\_base\base.upd.txt" >nul 2>&1
if exist "%OUT%\_base\base.sl4" sltoht "%OUT%\_base\base.sl4" "%OUT%\_base\base.sl4.txt" >nul 2>&1

echo.
echo ========================================================
echo  Terminado: !OK! ok, !BAD! fallidos, de !N!
echo  Resultados en: %OUT%
echo  Resumen:       %REPORT%
echo.
echo  Siguiente: 02-pack.bat  (comprime out\ para traerlo)
echo ========================================================
echo Fin %DATE% %TIME% >> "%REPORT%"
endlocal
