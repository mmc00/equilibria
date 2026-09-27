@echo off
REM ===================================================================
REM  Corre los .EXP que la GUI de RunGTAP resolvia y el ejecutable no.
REM
REM  Genera los *-DIR.EXP (traduciendo `rate% N from file X.shk` al
REM  valor directo) y los corre como 01-run-all.bat corre los demas.
REM
REM  Uso:   04-exp-directos.bat
REM ===================================================================
setlocal enabledelayedexpansion

set HERE=%~dp0
set DATA=%HERE%nus333
set OUT=%HERE%out
REM  Mismo criterio que 01-run-all.bat: estos .EXP declaran el cierre de
REM  GTAPUv7 (tfe/tfd/tgd/tpdall/... no existen en el GTAPV7 condensado), y el
REM  ejecutable NO esta en el PATH, hay que llamarlo por ruta completa.
if not defined GTAP_MODEL_DIR set GTAP_MODEL_DIR=C:unGTAP375
if not defined GTAP_MODEL_NAME set GTAP_MODEL_NAME=GTAPUV7

if not exist "%DATA%\ME8.EXP" (
  echo ERROR: no encuentro %DATA%\ME8.EXP
  exit /b 1
)

echo == 1. generando los *-DIR.EXP ==
python "%HERE%mk_exp_directos.py" "%DATA%"
if errorlevel 1 (
  echo ERROR: el generador fallo. No se corre nada.
  exit /b 1
)

echo.
echo == 2. corriendo ==
set OK=0
set FAIL=0

for %%E in (TBL813-DIR ME8-DIR) do (
  set EXP=%DATA%\%%E.EXP
  if not exist "!EXP!" (
    echo   %%E: no se genero, se saltea.
  ) else (
    set D=%OUT%\%%E
    if not exist "!D!" mkdir "!D!"
    set CMF=%DATA%\%%E.cmf
    echo   --- %%E ---

    REM El .cmf se arma como en 01-run-all.bat: el .EXP trae shock+cierre, y
    REM aca se le agregan los archivos y el auxiliar. Se copia el .EXP SIN sus
    REM lineas de Method/Steps solo si hace falta forzarlas; aca se dejan como
    REM vienen, que es lo que el libro usa.
    REM  GTAPPARM NO se declara aca: el .EXP ya lo trae y declararlo dos veces
    REM  da "E-Original file name specified twice". GTAPSUM/WELVIEW/GTAPVOL SI,
    REM  o GEMPACK corta con "E-One data file not named".
    > "!CMF!" echo aux files = %GTAP_MODEL_DIR%\%GTAP_MODEL_NAME%;
    >> "!CMF!" echo file GTAPSETS = sets.har;
    >> "!CMF!" echo file GTAPDATA = basedata.har;
    >> "!CMF!" echo file GTAPSUM  = "!D!\SUMMARY.har";
    >> "!CMF!" echo file WELVIEW  = "!D!\DECOMP.har";
    >> "!CMF!" echo file GTAPVOL  = "!D!\GTAPVol.har";
    >> "!CMF!" echo Updated file GTAPDATA = "!D!\%%E.upd";
    >> "!CMF!" echo Solution file = "!D!\%%E";
    type "!EXP!" >> "!CMF!"

    pushd "%DATA%"
    "%GTAP_MODEL_DIR%\%GTAP_MODEL_NAME%.EXE" -cmf "%%E.cmf" > "!D!\%%E.log" 2>&1
    if errorlevel 1 (
      echo     FALLO ^(ver !D!\%%E.log^)
      REM La variante A (`= file X.shk;`) no se pudo verificar sin GEMPACK.
      REM Si es eso lo que corta, el .EXP trae la variante B comentada:
      REM comentar la linea `Shock ... = file ...;` y descomentar la de abajo.
      echo     Si el error es de sintaxis en el Shock, abrir %%E.EXP:
      echo     trae una "variante B" comentada con el valor directo.
      set /a FAIL+=1
    ) else (
      echo     ok -^> !D!\%%E.sl4
      if exist "!D!\%%E.sl4" sltoht "!D!\%%E.sl4" "!D!\%%E.sl4.txt" ^>nul 2^>^&1
      set /a OK+=1
    )
    popd
  )
)

echo.
echo == resumen:  !OK! ok,  !FAIL! fallaron ==
if !FAIL! GTR 0 exit /b 1
exit /b 0
