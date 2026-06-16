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


def xitari_full_frame(rom, acts, frame):
    """Full (H,W) palette-index frame from xitari for `frame` (0-based)."""
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
    return screens[frame]


def xitari_screen_row(rom, acts, frame, row):
    return xitari_full_frame(rom, acts, frame)[row]


def jutari_full_frame(rom, acts, frame):
    """Full (210,W) palette-index frame from jutari for `frame` (0-based), via
    jutari_screen_dump.jl — same boot/settings as the sweep, so row-finding here
    reproduces the sweep's divergence exactly."""
    outf = Path(f"/tmp/_rd_juscr_{os.getpid()}.bin")
    subprocess.run(["julia", "--project=" + str(REPO / "jutari"),
                    str(REPO / "tools" / "jutari_screen_dump.jl"),
                    "--rom", str(rom), "--actions", str(acts),
                    "--out", str(outf), "--max-frames", str(frame + 1)],
                   capture_output=True, text=True, check=True, timeout=900)
    buf = np.frombuffer(outf.read_bytes(), dtype=np.uint8).reshape(-1, 210, W)
    outf.unlink(missing_ok=True)
    return buf[frame]


def auto_worst_row(rom, acts, frame):
    """Screen row with the most palette diffs in `frame` (and the diff count)."""
    xi = xitari_full_frame(rom, acts, frame)
    ju = jutari_full_frame(rom, acts, frame)
    h = min(xi.shape[0], ju.shape[0])
    per_row = (xi[:h] != ju[:h]).sum(axis=1)
    return int(per_row.argmax()), int(per_row.max())


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


def xitari_pf_activations(rom, acts, frame, scanline):
    """xitari's TRUE frame-relative PF poke activations (XI_POKE_DUMP, env-gated
    dump in xitari TIA::poke's PF-delay branch). Unlike --bus-trace cc, this is
    the real rendered beam position (clock - myClockWhenFrameStarted), so PF
    timing lines up with jutari's pending activations."""
    env = dict(os.environ, XI_POKE_DUMP="1")
    r = subprocess.run([str(TRACE_DUMP), "--rom", str(rom), "--actions", str(acts),
                        "--max-frames", str(frame + 2)],
                       capture_output=True, text=True, timeout=600, env=env)
    out = []
    seen = set()
    for ln in r.stderr.splitlines():
        if not ln.startswith("XIPOKE "):
            continue
        kv = dict(p.split("=") for p in ln.split()[1:])
        if int(kv["sl"]) != scanline:
            continue
        key = (kv["addr"], kv["x"], kv["val"])
        if key in seen:
            continue
        seen.add(key)
        out.append((int(kv["addr"], 16), int(kv["x"]), int(kv["delay"]),
                    int(kv["act"]), int(kv["val"])))
    return out


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
    ap.add_argument("--row", type=int, default=None,
                    help="screen row; omit with --auto to pick the worst-diff row")
    ap.add_argument("--auto", action="store_true",
                    help="auto-select the worst-diff screen row in this frame")
    ap.add_argument("--cols", type=str, default=None, help="A,B printed window")
    a = ap.parse_args()

    if a.row is None or a.auto:
        row, cnt = auto_worst_row(a.rom, a.actions, a.frame)
        print(f"[auto] worst-diff row in frame {a.frame}: row {row} ({cnt} px)")
        a.row = row

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
    # Color attribution: match a palette value against jutari's COLU registers
    # to name which TIA object drew that pixel. xitari only gives the composited
    # framebuffer, so this is how we attribute an xitari pixel to an object class
    # (COLUP0/COLUP1 are shared by player+missile, COLUPF by playfield+ball).
    colu = {regs[0x09]: "bg", regs[0x08]: "pf|bl",
            regs[0x06]: "p0|m0", regs[0x07]: "p1|m1"}
    def colorobj(v):
        return colu.get(int(v), f"?{int(v):#04x}")
    if len(cols):
        print("\n  per-column (diverging)  [xi-obj by color | ju-obj by pixel-set]:")
        for c in cols:
            print(f"    col {int(c):3d}: xi={int(xi_row[c]):3d}({colorobj(xi_row[c]):5s})  "
                  f"ju={int(ju_row[c]):3d}({colorobj(ju_row[c]):5s})  ju-set={who(int(c))}")
        # Summary: which object class accounts for the divergence (by xitari color).
        from collections import Counter
        xi_classes = Counter(colorobj(xi_row[c]) for c in cols)
        ju_classes = Counter(colorobj(ju_row[c]) for c in cols)
        print(f"\n  divergence by color — xitari drew: {dict(xi_classes)}")
        print(f"                        jutari drew: {dict(ju_classes)}")

    pk = xitari_pokes(a.rom, a.actions, a.frame, sl)
    # xitari PF-register poke delay (TIA.cxx:1992-1997): activates at clock+delay,
    # delay = {4,5,2,3}[(x/3)&3] where x is the in-scanline color clock.
    _PFD = (4, 5, 2, 3)
    def xi_activate(cc, addr):
        return cc + (_PFD[(cc // 3) & 3] if addr in (0x0D, 0x0E, 0x0F) else 0)
    print(f"\nxitari pokes on TIA scanline {sl} ({len(pk)})   [PF activate = cc+delay]:")
    for cc, addr, val in pk:
        extra = f"  -> activates cc={xi_activate(cc, addr)}" if addr in (0x0D, 0x0E, 0x0F) else ""
        print(f"    cc={cc:3d}  {REGNAME.get(addr, hex(addr)):7s} = {val:#04x}({val}){extra}")
    if not pk:
        print("    (none — registers held from an earlier scanline/boot)")

    pend = ju.get("pending", [])
    if pend:
        print(f"\njutari pending-write activations on scanline {sl} ({len(pend)}):")
        for act, reg, val in pend:
            print(f"    activates cc={act:3d}  {REGNAME.get(reg, hex(reg)):7s} = {val:#04x}({val})")

    # xitari's TRUE frame-relative PF activations (XI_POKE_DUMP; needs the
    # env-gated dump in xitari TIA::poke). The reliable PF-timing reference —
    # the --bus-trace cc above is CPU-cycle-derived and offset by the startFrame
    # carry, so compare jutari's pending against THESE, not the bus-trace cc.
    xipf = xitari_pf_activations(a.rom, a.actions, a.frame, sl)
    if xipf:
        print(f"\nxitari TRUE PF activations on scanline {sl} (frame-relative; XI_POKE_DUMP):")
        for addr, x, delay, act, val in xipf:
            print(f"    x={x:3d} delay={delay} -> activate cc={act:3d}  {REGNAME.get(addr, hex(addr)):4s} = {val:#04x}({val})")


if __name__ == "__main__":
    main()
