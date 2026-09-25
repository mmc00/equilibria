@echo off
REM ===================================================================
REM  03-gragg.bat - re-corre UN experimento con Gragg 2-4-6 en vez de
REM  Johansen 1 paso, para medir el error de linealizacion.
REM
REM  POR QUE: TBL45A con Johansen reproduce 4 de las 6 celdas del libro
REM  al cuarto decimal (ppa[SER,USA] -5,0720 vs -5,07 del libro;
REM  qpa[SER,USA] 9,6636 vs 9,66). Pero MFG no:
REM
REM                    GEMPACK/Johansen   libro    equilibria
REM     ppa[MFG,USA]        -0,1972       -0,79      -0,9871
REM     qpa[MFG,USA]         5,4572        4,44       4,2154
REM
REM  La HIPOTESIS a medir es que el libro corrio con un metodo
REM  multi-paso y la diferencia en MFG es error de linealizacion de
REM  Johansen. NO esta medido todavia: este .bat es el que lo mide.
REM  Si Gragg NO mueve MFG hacia el libro, la hipotesis queda refutada
REM  y la causa es otra.
REM
REM  Uso:   03-gragg.bat            (por defecto TBL45A)
REM         03-gragg.bat TBL45B     (u otro experimento)
REM
REM  Sale a out-gragg\<NOMBRE>\ para NO pisar la corrida Johansen de
REM  out\, que es la linea base de la comparacion.
REM ===================================================================
setlocal enabledelayedexpansion

if not defined NUS333_DIR set NUS333_DIR=%~dp0nus333
if not defined GTAP_MODEL_DIR (
  echo ERROR: GTAP_MODEL_DIR no definida. Corre 00-check-env.bat primero.
  exit /b 1
)
if not defined GTAP_MODEL_NAME set GTAP_MODEL_NAME=GTAPUV7

set NAME=%~1
if "%NAME%"=="" set NAME=TBL45A

set EXP=%NUS333_DIR%\%NAME%.EXP
if not exist "%EXP%" (
  echo ERROR: no existe %EXP%
  exit /b 1
)

set OUT=%~dp0out-gragg
set D=%OUT%\%NAME%
if not exist "%D%" mkdir "%D%"

REM  GTAPV7.EXP declara el cierre del modelo CONDENSADO; el resto es GTAPUv7.
set MDL=%GTAP_MODEL_NAME%
if /I "%NAME%"=="GTAPV7" set MDL=GTAPV7

echo.
echo === %NAME% con Gragg 2-4-6 ===
echo     salida: %D%

set CMF=%D%\%NAME%.cmf
(
  echo auxiliary files = "%GTAP_MODEL_DIR%\%MDL%";
  echo file GTAPSETS = "%NUS333_DIR%\sets.har";
  echo file GTAPDATA = "%NUS333_DIR%\basedata.har";
  echo file GTAPSUM  = "%D%\SUMMARY.har";
  echo file WELVIEW  = "%D%\DECOMP.har";
  echo file GTAPVOL  = "%D%\GTAPVol.har";
  echo updated file GTAPDATA = "%D%\%NAME%.upd";
  echo solution file = "%D%\%NAME%";
  echo CPU = yes;
  echo NDS = yes;
  echo Method = Gragg;
  echo Steps = 2 4 6;
) > "%CMF%"

REM  El .EXP trae GTAPPARM, closure, Shock -- y tambien SU PROPIO
REM  Method/Steps (Johansen 1). Se anexa TODO menos esas dos lineas:
REM  findstr /V /B /I las descarta por prefijo, asi el .EXP original
REM  queda intacto y lo unico que cambia entre las dos corridas es el
REM  metodo. Cualquier otra diferencia invalidaria la comparacion.
findstr /V /B /I /C:"Method =" /C:"Method=" /C:"Steps =" /C:"Steps=" "%EXP%" >> "%CMF%"

REM  Verificacion: el .cmf final debe tener UNA sola linea Method.
REM  Si el .EXP la escribiera distinto (p.ej. con espacios al inicio),
REM  findstr no la filtraria y GEMPACK tomaria la ultima -- Johansen --
REM  haciendo que esta corrida sea un duplicado silencioso de la base.
set /a NMETH=0
for /f %%C in ('findstr /I /C:"Method" "%CMF%" ^| find /c /v ""') do set NMETH=%%C
if not "!NMETH!"=="1" (
  echo   ERROR: el .cmf quedo con !NMETH! lineas Method, se esperaba 1.
  echo   Revisa %CMF% -- probablemente %NAME%.EXP escribe Method con
  echo   un formato que findstr no filtro. NO se corre: el resultado
  echo   no seria atribuible a Gragg.
  exit /b 1
)

pushd "%NUS333_DIR%"
"%GTAP_MODEL_DIR%\%MDL%.EXE" -cmf "%CMF%" > "%D%\%NAME%.log" 2>&1
set RC=!errorlevel!
popd

if !RC! neq 0 (
  echo   [FALLO] %NAME%  -- ver %D%\%NAME%.log
  exit /b 1
)

echo   [ok] %NAME%
if exist "%D%\%NAME%.sl4" sltoht "%D%\%NAME%.sl4" "%D%\%NAME%.sl4.txt" >nul 2>&1
if exist "%D%\%NAME%.upd" sltoht "%D%\%NAME%.upd" "%D%\%NAME%.upd.txt" >nul 2>&1

echo.
echo ========================================================
echo  Listo. Para ver si Gragg movio MFG hacia el libro:
echo.
echo     python cmp_gragg.py %NAME%
echo.
echo  Compara Johansen (out\) contra Gragg (out-gragg\) celda
echo  por celda, junto a los valores del libro.
echo ========================================================
endlocal
