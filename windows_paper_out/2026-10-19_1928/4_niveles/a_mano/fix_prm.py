"""Rewrite a capfix .prm keeping each 1CFULL header's on-disk width from the original .prm."""
import struct, sys
from equilibria.babel.har import write_har, read_har
from equilibria.babel.har import writer, wire

orig, src, dst = sys.argv[1:4]

def widths(path):
    b = open(path, "rb").read(); p = 0; out = {}; name = None
    while p < len(b):
        n = struct.unpack_from("<i", b, p)[0]; rec = b[p+4:p+4+n]; p += 8 + n
        if n == 4: name = rec.decode().strip(); continue
        if rec[4:10] == b"1CFULL" and name:
            out[name] = struct.unpack_from("<2i", rec, 84)[1]; name = None
        else:
            name = None if rec[4:10] in (b"2IFULL", b"2RFULL", b"RESPSE", b"REFULL", b"RLFULL") or name is None else name
    return out

W = widths(orig)
print("anchos originales:", {k: v for k, v in W.items() if v != 12})
_orig_rec = wire.write_set_element_record
_cur = {}
def _w1(out, name, ha):
    elements = [str(e).rstrip() for e in ha.array.tolist()]
    width = W.get(name, max(wire.SET_NAME_WIDTH, max((len(e) for e in elements), default=0)))
    writer._write_name_record(out, name)
    tail = wire.INT.pack(2) + wire.INT.pack(len(elements)) + wire.INT.pack(width)
    writer._write_meta_record(out, wire.TOKEN_1CFULL, ha.long_name, tail)
    wire.write_record(out, wire.write_set_element_record(elements, n_total=len(elements), flag=1, width=width))
writer._write_1cfull = _w1
write_har(dst, read_har(src))
print("escrito", dst, "anchos:", {k: v for k, v in widths(dst).items() if v != 12})
