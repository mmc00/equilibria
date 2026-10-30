@echo off
setlocal enableDelayedExpansion
set BaseName=10x10

set ifError=0
har2gdx basedata-%BaseName%.har %BaseName%Dat.gdx
echo.ErrorLevel = !errorlevel!
if %errorlevel% NEQ 0 (
   goto convertError
)

set ifError=0
har2gdx default-%BaseName%.prm %BaseName%Prm.gdx
echo.ErrorLevel = !errorlevel!
if %errorlevel% NEQ 0 (
   goto convertError
)

set ifError=0
har2gdx sets-%BaseName%.har %BaseName%Sets.gdx
echo.ErrorLevel = !errorlevel!
if %errorlevel% NEQ 0 (
   goto convertError
)

goto endCMD

:convertError
echo.Conversion failed
set ifError=1
goto endCMD

:endCmd
if %ifError%==0 (
   echo.Successful conclusion...
   exit /b 0
) else (
   exit /b 1
)
