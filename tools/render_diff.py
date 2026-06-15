#!/usr/bin/env python3
"""Per-scanline xitari-vs-jutari render diff harness.

The reliable replacement for ad-hoc inline render probes (which kept mis-mapping
screen rows to TIA scanlines by hand). For one SCREEN ROW of one frame it shows,
side by side:

  - both emulators' rendered pixel rows (ASCII + the diverging columns),
  - jutari's FULL render state at that scanline (decoded registers, object
    x-positions, per-object pixel-set membership for the diverging columns),
  - xitari's bus-trace pokes on that scanline (what was written, and at which
    color clock), so the responsible register write + its activation timing is
    visible.

The row->scanline mapping is done inside jutari from the env's real `y_start_row`
(target scanline = y_start_row + row) and reused for the xitari bus-trace, so the
mapping is correct once and for all.

Usage:
  jaxtari/.venv/bin/python tools/render_diff.py --rom tools/rom_sweep/roms/tutankham.bin --frame 0 --row 103
  (optional: --actions <file>  --cols A,B  to widen the printed column window)
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
TRACE_DUMP = REPO / "tools" / "trace_dump"
RENDER_JL = REPO / "tools" / "render_diff.jl"
DEF_ACTIONS = REPO / "tools" / "breakout_video" / "output" / "breakout_random_actions.txt"
W = 160

REGNAME = {
    0x00: "VSYNC", 0x01: "VBLANK", 0x02: "WSYNC", 0x04: "NUSIZ0", 0x05: "NUSIZ1",
    0x06: "COLUP0", 0x07: "COLUP1", 0x08: "COLUPF", 0x09: "COLUBK", 0x0A: "CTRLPF",
    0x0B: "REFP0", 0x0C: "REFP1", 0x0D: "PF0", 0x0E: "PF1", 0x0F: "PF2",
    0x10: "RESP0", 0x11: "RESP1", 0x12: "RESM0", 0x13: "RESM1", 0x14: "RESBL",
    0x1B: "GRP0", 0x1C: "GRP1", 0x1D: "ENAM0", 0x1E: "ENAM1", 0x1F: "ENABL",
    0x20: "HMP0", 0x21: "HMP1", 0x22: "HMM0", 0x23: "HMM1", 0x24: "HMBL",
    0x25: "VDELP0", 0x26: "VDELP1", 0x27: "VDELBL", 0x28: "RESMP0", 0x29: "RESMP1",
    0x2A: "HMOVE", 0x2B: "HMCLR", 0x2C: "CXCLR",
}


def xitari_screen_row(rom, acts, frame, row):
    r = subprocess.run([str(TRACE_DUMP), "--rom", str(rom), "--actions", str(acts),
                        "--max-frames", str(frame + 1), "--screen"],
                       capture_output=True, text=True, check=True, timeout=600)
    screens = []
    for line in r.stdout.strip().splitlines():
        o = json.loads(line)
        if o.get("boot_end"):
            continue
        screens.append(np.frombuffer(bytes.fromhex(o["screen"]), dtype=np.uint8)
                       .reshape(o["h"], o["w"]))
    return screens[frame][row]


def xitari_pokes(rom, acts, frame, scanline):
    busf = Path(f"/tmp/_rd_bus_{os.getpid()}.csv")
    subprocess.run([str(TRACE_DUMP), "--rom", str(rom), "--actions", str(acts),
                    "--max-frames", str(frame + 2), "--bus-trace", str(busf),
                    "--bus-trace-frames", f"{frame + 1},{frame + 1}"],
                   capture_output=True, text=True, check=True, timeout=600)
    pokes = []
    for ln in busf.read_text().splitlines()[1:]:
        f = ln.split(",")
        # global_idx,frame,kind,scanline,scanline_cycle,color_clock,addr,value
        if f[2] != "poke" or int(f[3]) != scanline:
            continue
        addr = int(f[6], 16)
        if addr > 0x2C:
            continue
        pokes.append((int(f[5]), addr, int(f[7])))  # (color_clock, addr, value)
    busf.unlink(missing_ok=True)
    return pokes


def jutari_state(rom, acts, frame, row):
    r = subprocess.run(["julia", "--project=" + str(REPO / "jutari"), str(RENDER_JL),
                        "--rom", str(rom), "--actions", str(acts),
                        "--frame", str(frame), "--row", str(row)],
                       capture_output=True, text=True, check=True, timeout=900)
    return json.loads(r.stdout.strip().splitlines()[-1])


def asciirow(vals):
    return "".join("#" if v else "." for v in vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rom", type=Path, required=True)
    ap.add_argument("--actions", type=Path, default=DEF_ACTIONS)
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--row", type=int, required=True)
    ap.add_argument("--cols", type=str, default=None, help="A,B printed window")
    a = ap.parse_args()

    ju = jutari_state(a.rom, a.actions, a.frame, a.row)
    sl = ju["target_scanline"]
    xi_row = xitari_screen_row(a.rom, a.actions, a.frame, a.row)
    ju_row = np.array(ju["screen_row"], dtype=np.uint8)
    diff = np.where(xi_row != ju_row)[0]

    print(f"\n=== {a.rom.stem} frame {a.frame} screen-row {a.row}  "
          f"(y_start={ju['y_start']} -> TIA scanline {sl}) ===")
    print(f"diff: {len(diff)} px" + (f"  cols {diff.min()}-{diff.max()}" if len(diff) else "  (none on this row)"))
    print("col : " + "".join(str(c % 10) for c in range(W)))
    print("xi  : " + asciirow(xi_row))
    print("ju  : " + asciirow(ju_row))
    print("dif : " + "".join("^" if c in set(diff.tolist()) else " " for c in range(W)))

    if ju.get("probe") is None and "regs" not in ju:
        print("\n!! jutari render probe did NOT fire for this scanline "
              "(scanline never rendered in this frame — VBLANK/partial?).")
        return

    regs = bytes.fromhex(ju["regs"])
    def rd(name, addr):
        return f"{name}={regs[addr]:#04x}({regs[addr]})"
    print("\njutari render state @ scanline", sl, ":")
    print("  " + "  ".join(rd(REGNAME[x], x) for x in (0x0A, 0x0D, 0x0E, 0x0F)))
    print("  " + "  ".join(rd(REGNAME[x], x) for x in (0x09, 0x08, 0x06, 0x07)))
    print("  " + "  ".join(rd(REGNAME[x], x) for x in (0x04, 0x05, 0x1B, 0x1C)))
    print("  " + "  ".join(rd(REGNAME[x], x) for x in (0x0B, 0x0C, 0x1F, 0x25)))
    print(f"  obj-x: p0={ju['p0_x']} p1={ju['p1_x']} m0={ju['m0_x']} m1={ju['m1_x']} bl={ju['bl_x']}")

    sets = {k: set(ju[k]) for k in ("pf", "p0", "p1", "m0", "m1", "bl")}
    def who(x):
        t = [k for k in ("pf", "p0", "p1", "m0", "m1", "bl") if x in sets[k]]
        return ",".join(t) if t else "bg"
    if a.cols:
        c0, c1 = (int(v) for v in a.cols.split(","))
        cols = range(c0, c1 + 1)
    else:
        cols = diff if len(diff) else []
    if len(cols):
        print("\n  per-column (diverging):")
        for c in cols:
            print(f"    col {int(c):3d}: xi={int(xi_row[c]):3d}  ju={int(ju_row[c]):3d}  ju-obj={who(int(c))}")

    pk = xitari_pokes(a.rom, a.actions, a.frame, sl)
    print(f"\nxitari pokes on TIA scanline {sl} ({len(pk)}):")
    for cc, addr, val in pk:
        print(f"    cc={cc:3d}  {REGNAME.get(addr, hex(addr)):7s} = {val:#04x}({val})")
    if not pk:
        print("    (none — registers held from an earlier scanline/boot)")


if __name__ == "__main__":
    main()
