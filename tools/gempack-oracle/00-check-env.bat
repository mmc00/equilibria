@echo off
REM ===================================================================
REM  00-check-env.bat - verifica el entorno ANTES de correr nada.
REM  No modifica nada. Solo reporta.
REM ===================================================================
setlocal enabledelayedexpansion
set FAIL=0
echo ========================================================
echo  Chequeo de entorno GEMPACK / NUS333
echo ========================================================
echo.

echo --- Ejecutables GEMPACK en el PATH ---
for %%X in (gemsim.exe tablo.exe sltoht.exe seehar.exe modhar.exe) do (
  where %%X >nul 2>&1
  if errorlevel 1 (
    echo   [FALTA] %%X
    if /I "%%X"=="gemsim.exe" set FAIL=1
    if /I "%%X"=="sltoht.exe" set FAIL=1
  ) else (
    for /f "delims=" %%P in ('where %%X') do echo   [ok]    %%X  ^-^> %%P
  )
)
echo.

echo --- Modelo GTAPv7 (.TAB / .EXE) ---
if defined GTAP_MODEL_DIR (
  echo   GTAP_MODEL_DIR = %GTAP_MODEL_DIR%
  if exist "%GTAP_MODEL_DIR%\GTAPV7.TAB" (echo   [ok]    GTAPV7.TAB) else (echo   [FALTA] GTAPV7.TAB en esa carpeta)
  if exist "%GTAP_MODEL_DIR%\GTAPV7.EXE" (echo   [ok]    GTAPV7.EXE ^(ya compilado^)) else (echo   [info]  GTAPV7.EXE ausente: hay que compilar con TABLO)
) else (
  echo   [FALTA] GTAP_MODEL_DIR no esta definida.
  echo           Es la carpeta de RunGTAP que contiene GTAPV7.TAB
  echo           Tipico: C:\RunGTAP\GTAPV7  o  C:\Program Files ^(x86^)\RunGTAP\...
  set FAIL=1
)
echo.

echo --- Datos NUS333 ---
if not defined NUS333_DIR set NUS333_DIR=%~dp0nus333
echo   NUS333_DIR = %NUS333_DIR%
for %%F in (basedata.har sets.har default.prm baserate.har) do (
  if exist "%NUS333_DIR%\%%F" (echo   [ok]    %%F) else (echo   [FALTA] %%F & set FAIL=1)
)
set /a NEXP=0
for %%F in ("%NUS333_DIR%\*.EXP") do set /a NEXP+=1
echo   Experimentos .EXP encontrados: !NEXP!   ^(esperados: 45^)
if !NEXP! LSS 45 echo   [aviso] faltan .EXP
echo.

echo --- Espacio en disco ---
for /f "tokens=3" %%S in ('dir /-c "%~dp0" ^| findstr /C:"bytes free"') do echo   libres: %%S bytes
echo   ^(la corrida completa necesita ~2 GB^)
echo.

echo ========================================================
if !FAIL!==0 (
  echo  LISTO. Podes correr 01-run-all.bat
) else (
  echo  FALTAN COSAS. Ver [FALTA] arriba. NO corras 01-run-all.bat todavia.
  echo.
  echo  Si el problema es GTAP_MODEL_DIR, buscalo asi:
  echo      dir /s /b C:\GTAPV7.TAB
  echo  y despues:
  echo      set GTAP_MODEL_DIR=C:\ruta\que\aparezca
)
echo ========================================================
endlocal
