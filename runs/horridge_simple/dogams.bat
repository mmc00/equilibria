echo off
SETLOCAL
REM  edit the next line to reflect YOUR GAMS folder
SET PATH=C:\Program Files\GAMS23.3;%PATH% 
del *.log
del results*.*
echo on

gams.exe simpleNLP  Logoption=2
ren results.gdx ResultsNLP.gdx
gams.exe simpleMCP  Logoption=2
ren results.gdx ResultsMCP.gdx
gams.exe mpsgevh  Logoption=2
ren results.gdx ResultsMPSGE.gdx

REM  next translates the Res*.gdx to HAR equivalent
for %%f in (res*.gdx) do gdx2har %%f  >nul:

dir results*.*
