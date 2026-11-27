"""Turn the CONVERT scalar model of the 2015 GHA MCP into a NEOS-ready file.

    python mac/make_neos_model.py mac/run/neos/gha2015_scalar.gms mac/neos/gha2015_path.gms

The CONVERT output (see run.gms, convertMCP) is a self-contained scalar GAMS
model (x1..xN, e1..eN, start levels of the first solve attempt). This script

  * sets PATH as the MCP solver (NEOS cp:PATH:GAMS), with --localsolver=<s>
    to run it here instead (with <s>.opt from the option directory);
  * raises the resource limit to 600 s;
  * after the solve, displays the model status and all levels in a
    one-dimensional parameter lev(j), j = 1..N (lev('7') is x7), with 8
    decimals (the display maximum), so the listing NEOS returns carries the solution. NEOS also
    returns the GDX of all symbols when wantgdx=yes.

No $call, execute or embedded code: it runs under NEOS's execmode=3.
"""
import re
import sys


def main(src, dst):
    text = open(src).read()
    nvar = max(int(m) for m in re.findall(r"\bx(\d+)\b", text))
    if "Solve m using MCP;" not in text.split("* Solve m using MCP;", 1)[1]:
        raise SystemExit("no solve statement found")
    text = text.replace("m.reslim = 300;", "m.reslim = 600;")
    head, tail = text.rsplit("\nSolve m using MCP;", 1)
    pre = """
* --- NEOS (cp:PATH:GAMS) solves with PATH. To run it locally with another
*     MCP solver and its option file: gams <this file> --localsolver=nlpec optdir=<dir>
option mcp = path;
$if set localsolver option mcp = %localsolver%; m.optfile = 1;
m.solprint = 2;
"""
    post = ["",
            "* --- solution: levels of x1..x%d as lev('1')..lev('%d')" % (nvar, nvar),
            "set j 'variable index of the scalar model' / 1*%d /;" % nvar,
            "parameter lev(j) 'level of variable xj', stat(*) 'solve statistics';",
            "stat('solvestat') = m.solvestat; stat('modelstat') = m.modelstat;",
            "stat('numinfes') = m.numinfes; stat('suminfes') = m.suminfes;",
            "stat('iterusd') = m.iterusd; stat('resusd') = m.resusd;"]
    post += ["lev('%d') = x%d.l;" % (i, i) for i in range(1, nvar + 1)]
    post += ["option stat:8:0:1, lev:8:0:1;", "display stat, lev;", ""]
    open(dst, "w").write(head + "\n" + pre + "Solve m using MCP;" + tail + "\n".join(post))
    print("wrote %s (%d variables)" % (dst, nvar))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
