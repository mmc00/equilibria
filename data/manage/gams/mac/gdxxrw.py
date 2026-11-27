"""Minimal GDXXRW replacement for macOS/Linux (GDXXRW needs Windows and Excel).

Covers what MANAGE-WB uses:

  read  (xlsx -> gdx):  par=, set=, dset= with rng=, rdim=, cdim=, dim=,
                        index=Sheet!Cell, dSetText=Y, MaxDupeErrors=, @file
  write (gdx -> xlsx):  par= with rng=, rdim=, cdim=, @file

Semantics follow the GDXXRW documentation:
  * a range given as top-left cell only is extended right and down until
    skipEmpty+1 (default 2) consecutive empty label rows/columns;
  * empty label cells become the empty label '' (no colMerge);
  * zeros are not stored; strings Eps/Inf/-Inf/NA/Undf map to special values;
  * dset reads a one-dimensional list and drops duplicates;
  * sets with rdim=0 or cdim=0 are read 'dense' (all tuples, text from the
    data cell), otherwise 'YN';
  * dimension defaults: none given -> rdim=1,cdim=1 (read) or rdim=dim-1,
    cdim=1 (write); only one given -> the other is 0 (read) or dim-given
    (write).
  * writing fills the rectangle (row labels, column labels, data) and leaves
    the rest of the sheet untouched (no merge/clear). Afterwards the workbook
    is recalculated with LibreOffice (when found) so formula cells keep values.

Usage mirrors gdxxrw:  python gdxxrw.py i=in.xlsx o=out.gdx index=Layout!B4
"""

import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile

import openpyxl
from openpyxl.utils import column_index_from_string, get_column_letter

import gams.transfer as gt
import pandas as pd

SPECIAL_IN = {
    "eps": gt.SpecialValues.EPS,
    "inf": float("inf"),
    "+inf": float("inf"),
    "-inf": float("-inf"),
    "na": gt.SpecialValues.NA,
    "undf": gt.SpecialValues.UNDEF,
}


def log(msg):
    print("gdxxrw.py: " + msg, flush=True)


# --------------------------------------------------------------------------
# command line
# --------------------------------------------------------------------------
def tokenize(argv):
    tokens = []
    for a in argv:
        if a.startswith("@"):
            with open(a[1:]) as fh:
                for line in fh:
                    line = line.strip()
                    if line and not line.startswith("*"):
                        tokens.extend(shlex.split(line, posix=True))
        else:
            tokens.extend(shlex.split(a, posix=True) if " " in a else [a])
    out = []
    for t in tokens:
        if "=" in t:
            k, v = t.split("=", 1)
            out.append((k.strip().lower(), v.strip().strip("'\"")))
        else:
            out.append((t.strip().lower(), None))
    return out


def parse(argv):
    opts = {"skipempty": 1, "dsettext": False, "maxdupeerrors": 0}
    syms = []
    toks = tokenize(argv)
    for k, v in toks:
        if k in ("i", "input"):
            opts["input"] = v
        elif k in ("o", "output"):
            opts["output"] = v
        elif k == "index":
            opts["index"] = v
        elif k == "dsettext":
            opts["dsettext"] = str(v).lower() in ("y", "yes", "1", "true")
        elif k in ("se", "skipempty"):
            opts["skipempty"] = int(v)
        elif k == "maxdupeerrors":
            opts["maxdupeerrors"] = int(v)
        elif k in ("par", "set", "dset"):
            syms.append({"type": k, "name": v})
        elif k in ("rng", "rdim", "cdim", "dim", "values") and syms:
            syms[-1][k] = v
        elif os.path.splitext(k)[1].lower() in (".xlsx", ".xlsm", ".gdx") and "input" not in opts:
            opts["input"] = k
        # other options (trace, log, ...) are ignored
    return opts, syms


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def split_range(rng, wb):
    if "!" in rng:
        sheet, cells = rng.split("!", 1)
    else:
        sheet, cells = wb.sheetnames[0], rng
    sheet = sheet.strip("'")
    names = {n.lower(): n for n in wb.sheetnames}
    real = names.get(sheet.lower())
    cells = cells.replace("..", ":")
    m = re.match(r"^\$?([A-Za-z]+)\$?(\d+)(?::\$?([A-Za-z]+)\$?(\d+))?$", cells or "A1")
    if not m:
        raise ValueError("cannot parse range " + rng)
    c0, r0 = column_index_from_string(m.group(1)), int(m.group(2))
    end = None
    if m.group(3):
        end = (int(m.group(4)), column_index_from_string(m.group(3)))
    return sheet, real, r0, c0, end


def label(v):
    if v is None:
        return ""
    if isinstance(v, bool):
        return str(v).upper()
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v).strip()


def empty(v):
    return v is None or (isinstance(v, str) and v.strip() == "")


def value(v, where, warn):
    if empty(v):
        return None
    if isinstance(v, bool):
        return 1.0 if v else None
    if isinstance(v, (int, float)):
        return float(v) if v != 0 else None
    s = str(v).strip().lower()
    if s in SPECIAL_IN:
        return SPECIAL_IN[s]
    try:
        f = float(s)
        return f if f != 0 else None
    except ValueError:
        warn.append(where + ": text " + repr(v) + " ignored")
        return None


class Sheet:
    def __init__(self, ws):
        self.ws = ws
        self.cells = {}
        for row in ws.iter_rows():
            for c in row:
                if c.value is not None:
                    self.cells[(c.row, c.column)] = c.value
        self.max_row = max([r for r, _ in self.cells] or [0])
        self.max_col = max([c for _, c in self.cells] or [0])

    def get(self, r, c):
        return self.cells.get((r, c))


def extent(is_empty, start, limit, se):
    """Last index of an open block starting at start."""
    last = start - 1
    gap = 0
    i = start
    while i <= limit:
        if is_empty(i):
            gap += 1
            if gap > se:
                break
        else:
            gap = 0
            last = i
        i += 1
    return last


# --------------------------------------------------------------------------
# reading
# --------------------------------------------------------------------------
def read_symbol(wb, sheets, sym, opts, warn):
    typ = sym["type"]
    rdim = sym.get("rdim")
    cdim = sym.get("cdim")
    dim = sym.get("dim")
    rdim = None if rdim in (None, "") else int(float(rdim))
    cdim = None if cdim in (None, "") else int(float(cdim))
    dim = None if dim in (None, "") else int(float(dim))
    if typ == "dset":
        if rdim is None and cdim is None:
            rdim, cdim = 1, 0
    if rdim is None and cdim is None:
        if dim is None:
            rdim, cdim = 1, 1
        else:
            cdim, rdim = 1, dim - 1
    elif rdim is None:
        rdim = (dim - cdim) if dim is not None else 0
    elif cdim is None:
        cdim = (dim - rdim) if dim is not None else 0

    sname, real, r0, c0, end = split_range(sym.get("rng") or (sym["name"] + "!A1"), wb)
    if real is None:
        raise ValueError("sheet %s not found for %s" % (sname, sym["name"]))
    sh = sheets[real]
    se = opts["skipempty"]

    # column labels / data columns
    if cdim > 0:
        cfirst = c0 + rdim
        if end:
            clast = end[1]
        else:
            clast = extent(lambda c: all(empty(sh.get(r0 + k, c)) for k in range(cdim)), cfirst, sh.max_col, se)
        cols = [(c, tuple(label(sh.get(r0 + k, c)) for k in range(cdim))) for c in range(cfirst, clast + 1)
                if not all(empty(sh.get(r0 + k, c)) for k in range(cdim))]
    else:
        cols = [(c0 + rdim, ())]
    # row labels / data rows
    rfirst = r0 + cdim
    if rdim > 0:
        if end:
            rlast = end[0]
        else:
            rlast = extent(lambda r: all(empty(sh.get(r, c0 + k)) for k in range(rdim)), rfirst, sh.max_row, se)
        rows = [(r, tuple(label(sh.get(r, c0 + k)) for k in range(rdim))) for r in range(rfirst, rlast + 1)
                if not all(empty(sh.get(r, c0 + k)) for k in range(rdim))]
    else:
        rows = [(rfirst, ())]

    recs = []
    if typ == "dset":
        seen = set()
        if rdim == 1:
            items = [(r, key[0]) for r, key in rows]
            txt = lambda r: sh.get(r, c0 + 1)
        else:
            items = [(c, key[0]) for c, key in cols]
            txt = lambda c: sh.get(r0 + 1, c)
        for pos, lab in items:
            if lab.lower() in seen:
                continue
            seen.add(lab.lower())
            t = txt(pos) if opts["dsettext"] else None
            recs.append([lab, "" if empty(t) or not isinstance(t, str) else t.strip()])
        return recs, 1
    if typ == "set":
        mode = (sym.get("values") or "auto").lower()
        if mode == "auto":
            mode = "dense" if (rdim == 0 or cdim == 0) else "yn"
        for r, rkey in rows:
            for c, ckey in cols:
                v = sh.get(r, c)
                if mode in ("dense", "string", "all", "nodata"):
                    t = "" if (empty(v) or mode == "nodata") else label(v)
                    recs.append(list(rkey + ckey) + [t])
                elif mode == "sparse":
                    if not empty(v):
                        recs.append(list(rkey + ckey) + [label(v)])
                else:  # yn
                    if not empty(v) and label(v).lower() not in ("0", "n", "no"):
                        recs.append(list(rkey + ckey) + [""])
        return recs, rdim + cdim
    # parameter
    for r, rkey in rows:
        for c, ckey in cols:
            v = value(sh.get(r, c), "%s!%s%d" % (real, get_column_letter(c), r), warn)
            if v is not None:
                recs.append(list(rkey + ckey) + [v])
    return recs, rdim + cdim


def index_symbols(wb, idx):
    sname, real, r0, c0, end = split_range(idx, wb)
    sh = Sheet(wb[real])
    heads = {}
    c = c0 + 3
    while not empty(sh.get(r0, c)):
        heads[c] = str(sh.get(r0, c)).strip().lower()
        c += 1
    syms = []
    r = r0 + 1
    while not (empty(sh.get(r, c0)) and empty(sh.get(r, c0 + 1))):
        typ = str(sh.get(r, c0) or "").strip().lower()
        name = str(sh.get(r, c0 + 1) or "").strip()
        rng = sh.get(r, c0 + 2)
        if typ in ("par", "set", "dset") and name:
            if empty(rng):
                raise ValueError("index row %d: range of %s has no value (formula not calculated?)" % (r, name))
            s = {"type": typ, "name": name, "rng": str(rng).strip()}
            for cc, h in heads.items():
                v = sh.get(r, cc)
                if not empty(v):
                    s[h] = v
            syms.append(s)
        r += 1
    return syms


def do_read(opts, syms):
    wb = openpyxl.load_workbook(opts["input"], data_only=True)
    if "index" in opts:
        syms = index_symbols(wb, opts["index"]) + syms
    sheets = {n: Sheet(wb[n]) for n in wb.sheetnames}
    m = gt.Container()
    warn = []
    dupes = 0
    for s in syms:
        recs, dim = read_symbol(wb, sheets, s, opts, warn)
        df = pd.DataFrame(recs, columns=list(range(dim)) + ["v"]) if recs else None
        if df is not None and dim > 0:
            key = df[list(range(dim))].apply(lambda col: col.str.lower())
            dup = key.duplicated(keep="first")
            if dup.any():
                dupes += int(dup.sum())
                log("%d duplicate records in %s dropped" % (int(dup.sum()), s["name"]))
                df = df[~dup]
        name = s["name"]
        if name in m:
            log("symbol %s given twice, second ignored" % name)
            continue
        if s["type"] == "par":
            m.addParameter(name, ["*"] * dim, records=df)
        else:
            m.addSet(name, ["*"] * dim, records=df)
        log("read %-4s %-22s %-22s dim=%d records=%d" % (s["type"], name, s.get("rng"), dim, 0 if df is None else len(df)))
    for w in warn[:20]:
        log("warning " + w)
    if len(warn) > 20:
        log("... %d more warnings" % (len(warn) - 20))
    if dupes > opts["maxdupeerrors"]:
        log("%d duplicates exceed MaxDupeErrors=%d" % (dupes, opts["maxdupeerrors"]))
        return 1
    out = opts.get("output") or os.path.splitext(opts["input"])[0] + ".gdx"
    m.write(out, eps_to_zero=False)
    log("wrote " + out)
    return 0


# --------------------------------------------------------------------------
# writing
# --------------------------------------------------------------------------
def fmt_value(v):
    if gt.SpecialValues.isEps(v):
        return "Eps"
    if gt.SpecialValues.isNA(v):
        return "NA"
    if gt.SpecialValues.isUndef(v):
        return "Undf"
    if v == float("inf"):
        return "Inf"
    if v == float("-inf"):
        return "-Inf"
    return float(v)


def recalc(path):
    soffice = shutil.which("soffice") or "/Applications/LibreOffice.app/Contents/MacOS/soffice"
    if not os.path.exists(soffice):
        log("LibreOffice not found: formulas in %s keep no cached values" % path)
        return
    tmp = tempfile.mkdtemp()
    prof = "file://" + tempfile.mkdtemp()
    subprocess.run([soffice, "-env:UserInstallation=" + prof, "--headless", "--calc",
                    "--convert-to", "xlsx:Calc MS Excel 2007 XML", "--outdir", tmp, path],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    shutil.move(os.path.join(tmp, os.path.basename(path)), path)
    log("recalculated " + path + " with LibreOffice")


def do_write(opts, syms):
    m = gt.Container(opts["input"])
    out = opts["output"]
    wb = openpyxl.load_workbook(out) if os.path.exists(out) else openpyxl.Workbook()
    for s in syms:
        sym = m[s["name"]]
        dim = sym.dimension
        rdim = s.get("rdim")
        cdim = s.get("cdim")
        rdim = None if rdim in (None, "") else int(rdim)
        cdim = None if cdim in (None, "") else int(cdim)
        if rdim is None and cdim is None:
            cdim = 1 if dim > 0 else 0
            rdim = dim - cdim
        elif rdim is None:
            rdim = dim - cdim
        elif cdim is None:
            cdim = dim - rdim
        # GDXXRW: no range means the first cell of the first sheet
        sname, real, r0, c0, _ = split_range(s.get("rng") or "A1", wb)
        ws = wb[real] if real else wb.create_sheet(sname)
        df = sym.records
        if df is None or len(df) == 0:
            log("write %s: no records" % s["name"])
            continue
        labs = [df.iloc[:, k].astype(str).tolist() for k in range(dim)]
        vals = df.iloc[:, dim].tolist() if s["type"] == "par" else ["Y"] * len(df)
        rkeys, ckeys, data = [], [], {}
        rpos, cpos = {}, {}
        for i in range(len(df)):
            rk = tuple(labs[k][i] for k in range(rdim))
            ck = tuple(labs[k][i] for k in range(rdim, dim))
            if rk not in rpos:
                rpos[rk] = len(rkeys)
                rkeys.append(rk)
            if ck not in cpos:
                cpos[ck] = len(ckeys)
                ckeys.append(ck)
            data[(rpos[rk], cpos[ck])] = vals[i]
        for j, ck in enumerate(ckeys):
            for k in range(cdim):
                ws.cell(r0 + k, c0 + rdim + j).value = ck[k]
        for i, rk in enumerate(rkeys):
            for k in range(rdim):
                ws.cell(r0 + cdim + i, c0 + k).value = rk[k]
            for j in range(len(ckeys)):
                v = data.get((i, j))
                ws.cell(r0 + cdim + i, c0 + rdim + j).value = None if v is None else (fmt_value(v) if s["type"] == "par" else v)
        log("wrote %-16s to %s!%s%d rdim=%d cdim=%d (%d x %d)" % (s["name"], ws.title, get_column_letter(c0), r0, rdim, cdim, len(rkeys), len(ckeys)))
    wb.save(out)
    recalc(out)
    return 0


def main(argv):
    opts, syms = parse(argv)
    if "input" not in opts:
        log("no input file")
        return 1
    ext = os.path.splitext(opts["input"])[1].lower()
    if ext == ".gdx":
        return do_write(opts, syms)
    return do_read(opts, syms)


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except Exception as e:  # report like gdxxrw: non-zero return code
        log("error: %s" % e)
        raise
