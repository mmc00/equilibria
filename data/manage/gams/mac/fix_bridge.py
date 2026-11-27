"""Remove the placeholder Eps in xfmSupDistrInst from a GTAP_9 bridge file.

dat/BridgeFile_GTAP_9_template.xlsx carries one entry in the block
hohPar!B15 (read as xfmSupDistrInst): h-hhold.f-tech_aspros.a-PDR = 'eps'
(the GTAP_POW template has the block empty). gtapSamV6.gms computes
xfmSupDistr but kills it before writing, so the placeholder survives into
BaseBridge_*.xlsx. Read as Eps it switches off the SAM based factor income
shares for f-tech_aspros and inc/cal/inical.inc stops with 'division by eps'.
This script blanks the placeholder (only when it is that single entry) and
recalculates the workbook.

Usage: python fix_bridge.py reg/GHA/BaseBridge_GHA_GTAP_9.xlsx
"""
import os
import sys

import openpyxl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gdxxrw  # noqa: E402


def main(path):
    wb = openpyxl.load_workbook(path)
    ws = wb["hohPar"]
    cells = [(r, c) for r in range(16, ws.max_row + 1) for c in range(4, ws.max_column + 1)
             if ws.cell(r, c).value not in (None, "")]
    if cells == [(16, 4)] and str(ws.cell(16, 4).value).strip().lower() == "eps" \
            and ws.cell(16, 2).value == "h-hhold" and ws.cell(16, 3).value == "f-tech_aspros":
        ws.cell(16, 4).value = None
        wb.save(path)
        gdxxrw.recalc(path)
        print("fix_bridge.py: blanked placeholder Eps in hohPar!D16 of " + path)
    else:
        print("fix_bridge.py: no placeholder found in " + path + ", file left as is")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
