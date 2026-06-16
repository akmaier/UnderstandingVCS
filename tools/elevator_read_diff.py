#!/usr/bin/env python3
"""Find the first divergent TIA/RIOT *read* between jutari and xitari bus traces.

Both CSVs: global_idx,frame,kind,scanline,scanline_cycle,color_clock,addr,value
(addr hex, value decimal). jutari frames 0-based + addr masked to 0x1FFF;
xitari frames 1-based + full addr. We normalize the read register, group reads
per frame, auto-detect the frame offset, then report the first divergence.
"""
import csv, sys
from collections import defaultdict

def reg_of(addr):
    """Return a normalized register id for a TIA-read or RIOT-IO read, else None."""
    a = addr & 0x1FFF
    if a & 0x1000:          # cartridge / ROM fetch
        return None
    if (a & 0x80) == 0:     # TIA select (A7=0)
        r = a & 0x0F
        return ("TIA", r) if r <= 0x0D else None
    if a & 0x200:           # RIOT I/O (A7=1, A9=1)
        return ("RIOT", 0x280 | (a & 0x1F))
    return None             # RIOT RAM ($80-$FF)

def load(path):
    frames = defaultdict(list)   # frame -> [(reg, value), ...]
    with open(path) as f:
        rd = csv.reader(f)
        next(rd)
        for row in rd:
            if len(row) < 8 or row[2] != "peek":
                continue
            fr = int(row[1]); addr = int(row[6], 16); val = int(row[7])
            r = reg_of(addr)
            if r is not None:
                frames[fr].append((r, val))
    return frames

def load_flat(path):
    """Flat ordered list of (frame, reg, value) reads, plus per-frame counts."""
    out = []
    with open(path) as f:
        rd = csv.reader(f); next(rd)
        for row in rd:
            if len(row) < 8 or row[2] != "peek":
                continue
            fr = int(row[1]); addr = int(row[6], 16); val = int(row[7])
            r = reg_of(addr)
            if r is not None:
                out.append((fr, r, val))
    return out

if "--flat" in sys.argv:
    jf = load_flat("/tmp/ju_bus.csv")
    xf = load_flat("/tmp/xi_bus.csv")
    print(f"jutari {len(jf)} reads (frames {jf[0][0]}..{jf[-1][0]}), xitari {len(xf)} reads (frames {xf[0][0]}..{xf[-1][0]})")
    # longest common prefix of (reg,value), ignoring frame numbers
    n = min(len(jf), len(xf)); i = 0
    while i < n and jf[i][1:] == xf[i][1:]:
        i += 1
    print(f"common prefix = {i} reads")
    if i < n:
        lo = max(0, i-4)
        print(f"\nfirst divergence at global read #{i}:")
        for j in range(lo, min(i+5, n)):
            (jfr,(jk,jr),jv) = jf[j]; (xfr,(xk,xr),xv) = xf[j]
            mark = "  <<<" if j == i else ""
            print(f"  [{j}] ju: f{jfr} {jk}${jr:02x}={jv:#04x}   xi: f{xfr} {xk}${xr:02x}={xv:#04x}{mark}")
    sys.exit(0)

ju = load("/tmp/ju_bus.csv")
xi = load("/tmp/xi_bus.csv")
print(f"jutari frames {min(ju)}..{max(ju)}  xitari frames {min(xi)}..{max(xi)}")

# Auto-detect frame offset: jutari[f] should match xitari[f+off].
def matches(off):
    m = 0
    for f in ju:
        if f + off in xi and ju[f] == xi[f + off]:
            m += 1
    return m
best = max(range(-2, 4), key=matches)
print(f"best frame offset = {best}  ({matches(best)} frames fully matching)")

# Walk frames in order; report first divergent read.
for f in sorted(ju):
    g = f + best
    if g not in xi:
        continue
    a, b = ju[f], xi[g]
    if a == b:
        continue
    print(f"\n=== first divergent frame: jutari {f} / xitari {g} ===")
    print(f"jutari {len(a)} reads, xitari {len(b)} reads")
    for i in range(min(len(a), len(b))):
        if a[i] != b[i]:
            kind, reg = a[i][0]
            print(f"read #{i}: reg={kind}${reg:02x}  jutari_val={a[i][1]:#04x}  xitari_val={b[i][1]:#04x}")
            # context: surrounding reads
            print("  ju context:", [(f'{k}${r:02x}={v:#x}') for (k,r),v in a[max(0,i-2):i+3]])
            print("  xi context:", [(f'{k}${r:02x}={v:#x}') for (k,r),v in b[max(0,i-2):i+3]])
            break
    else:
        print("  (lists share a common prefix; lengths differ)")
    sys.exit(0)
print("no divergence found in TIA/RIOT reads across aligned frames")
