@echo off
REM  02-pack.bat - comprime out\ en un zip para traer a la Mac.
setlocal
set OUT=%~dp0out
set ZIP=%~dp0nus333-gempack-oracle.zip
if not exist "%OUT%" (echo ERROR: no existe %OUT%. Corre 01-run-all.bat primero. & exit /b 1)
if exist "%ZIP%" del "%ZIP%"
echo Comprimiendo %OUT% ...
powershell -NoProfile -Command "Compress-Archive -Path '%OUT%\*' -DestinationPath '%ZIP%' -CompressionLevel Optimal"
if errorlevel 1 (echo ERROR al comprimir & exit /b 1)
for %%A in ("%ZIP%") do echo   %%~zA bytes  ^-^>  %ZIP%
echo.
echo Traelo a la Mac y corre ahi:
echo    python tools/gempack-oracle/read_oracle.py ^<carpeta-descomprimida^>
endlocal
