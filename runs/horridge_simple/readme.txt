This archive contains material to compare implementations of the same
SIMPLE model in GAMS, GEMPACK and MPSGE.

The GEMPACK version is SIMPLE.TAB, with data SIMDATA.HAR, and CMF
FIXCAP.CMF. You could run the model from the command line, or using
WinGEM, or by running GEMPACK.BAT.

The GAMS versions are: 
    MPSGEVH.GMS   MPSGE version
  SIMPLEMCP.GMS   Straight GAMS solving with PATH/MCP
  SIMPLENLP.GMS   Straight GAMS solving with CONOPT/NLP
They all use the same datafile INPUT.GDX

You could run the models from the GAMS IDE or by running DOGAMS.BAT.
[Note: you'll need to edit line 4 of DOGAMS.BAT to reflect the
location of your own GAMS folder.]

The SIZETEST folder contains material to compare these implementations
using datasets of sizes 50 to 500 sectors. That folder has its own README.TXT
file.
