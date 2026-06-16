#!/usr/bin/env python3
"""Frame-aligned per-instruction diff: jutari frame k vs xitari frame k+1.
Finds the FIRST instruction whose (PC,A,X,Y,P,SP) state diverges — the root."""
import re

def parse_ju(path):
    frames = {}; cur = None
    for line in open(path):
        line = line.rstrip()
        m = re.match(r"^F (\d+)$", line)
        if m:
            cur = int(m.group(1)); frames[cur] = []
        elif line.startswith("I ") and cur is not None:
            t = line.split()[1:]          # pc op A X Y P SP
            frames[cur].append((t[0], t[2], t[3], t[4], t[5], t[6]))
    return frames

def parse_xi(path):
    frames = {}; cur = None
    for line in open(path):
        line = line.rstrip()
        m = re.match(r"^XIFRAME (\d+)$", line)
        if m:
            cur = int(m.group(1)); frames[cur] = []
        elif line.startswith("I ") and cur is not None:
            t = line.split()[1:]          # pc A X Y P SP
            frames[cur].append((t[0], t[1], t[2], t[3], t[4], t[5]))
    return frames

ju = parse_ju("/tmp/ju_instr.txt")
xi = parse_xi("/tmp/xi_instr.txt")
print(f"jutari frames {sorted(ju)[:3]}..{sorted(ju)[-1]}; xitari frames {sorted(xi)[:3]}..{sorted(xi)[-1]}")

for k in sorted(ju):
    g = k + 1                       # jutari 0-based, xitari 1-based
    if g not in xi:
        continue
    a, b = ju[k], xi[g]
    n = min(len(a), len(b)); i = 0
    while i < n and a[i] == b[i]:
        i += 1
    if i == n and len(a) == len(b):
        continue                    # frame fully matches
    print(f"\n=== first divergent frame: jutari {k} / xitari {g}  (ju {len(a)} instr, xi {len(b)} instr) ===")
    lo = max(0, i - 4)
    for j in range(lo, min(i + 4, n)):
        mark = "  <<< DIVERGE" if j == i else ""
        print(f"  [{j}] ju PC={a[j][0]:>4} A={a[j][1]:>2} X={a[j][2]:>2} Y={a[j][3]:>2} P={a[j][4]:>2} SP={a[j][5]:>2}   "
              f"xi PC={b[j][0]:>4} A={b[j][1]:>2} X={b[j][2]:>2} Y={b[j][3]:>2} P={b[j][4]:>2} SP={b[j][5]:>2}{mark}")
    if i >= n:
        print(f"  (common prefix exhausted at {i}; lengths differ: ju={len(a)} xi={len(b)})")
    break
else:
    print("no per-instruction divergence found in frames 0..21")
