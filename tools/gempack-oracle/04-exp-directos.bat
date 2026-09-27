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
    set CMF=%DATA%\%%E.cmf
    echo   --- %%E ---

    REM El .cmf se arma como en 01-run-all.bat: el .EXP trae shock+cierre, y
    REM aca se le agregan los archivos y el auxiliar. Se copia el .EXP SIN sus
    REM lineas de Method/Steps solo si hace falta forzarlas; aca se dejan como
    REM vienen, que es lo que el libro usa.
    > "!CMF!" echo aux files = gtapv7;
    >> "!CMF!" echo file GTAPSETS = sets.har;
    >> "!CMF!" echo file GTAPDATA = basedata.har;
    >> "!CMF!" echo file GTAPPARM = default.prm;
    >> "!CMF!" echo Updated file GTAPDATA = %%E.upd;
    >> "!CMF!" echo Solution file = %%E;
    type "!EXP!" >> "!CMF!"

    pushd "%DATA%"
    gtapv7 -cmf "%%E.cmf" > "%%E.runlog" 2>&1
    if errorlevel 1 (
      echo     FALLO ^(ver %%E.runlog^)
      REM La variante A (`= file X.shk;`) no se pudo verificar sin GEMPACK.
      REM Si es eso lo que corta, el .EXP trae la variante B comentada:
      REM comentar la linea `Shock ... = file ...;` y descomentar la de abajo.
      echo     Si el error es de sintaxis en el Shock, abrir %%E.EXP:
      echo     trae una "variante B" comentada con el valor directo.
      set /a FAIL+=1
    ) else (
      echo     ok -^> %%E.sl4
      set /a OK+=1
    )
    popd
  )
)

echo.
echo == resumen:  !OK! ok,  !FAIL! fallaron ==
if !FAIL! GTR 0 exit /b 1
exit /b 0
