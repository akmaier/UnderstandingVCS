#!/usr/bin/env python3
"""cycle_trace_inspect.py — query the cycle traces emitted by
`cpu_tia_cycle_trace.jl`.

Phase 1 of P3I_G_THREADING_PLAN.md. The trace CSVs are big (~40 MB
for 25 pong frames) — eyeballing them is impractical. This tool
exposes a small query language for the kinds of questions the open
bugs raise:

  * "what TIA writes happen on scanline N of frame M, in beam order?"
  * "what's the color_clock of the first HMOVE write before scanline
     34 of frame N? was it inside the HMOVE-blank trigger window?"
  * "what TIA reads happen on scanline N of frame M?"
  * "diff two traces — find the first event whose (kind, addr, value,
     scanline, scanline_cycle, color_clock) differs."

Subcommands:

  scanline <csv> <frame> <scanline>
      Dump every event (peek/poke/tick/wsync_release) on that
      scanline, in beam-order.

  poke-trace <csv> <frame> <addr-hex>
      Dump every poke to <addr> in <frame>, with scanline +
      color_clock so you can correlate writes with the rendered
      scanline.

  hmove <csv> <frame> <scanline-target>
      Find the HMOVE poke (`$2A`) that fires just before
      <scanline-target>. Print its scanline_cycle so we can check
      against `_HMOVE_BLANK_ENABLE_CYCLES` (true for 0..20 and 75).

  diff <csv-a> <csv-b>
      Walk both traces in parallel and report the first event that
      differs in any field other than `global_idx`. Useful for
      before/after comparison when Phase 2 lands.

  summary <csv>
      Print high-level stats: total events, per-frame event counts,
      kind histogram, peek/poke address histogram.

Usage:
  python3 tools/cycle_trace_inspect.py scanline \\
      tools/fixtures/cycle_traces/pong_jutari_25.csv 1 34

The CSV format (one row per bus event):
  global_idx,frame,kind,scanline,scanline_cycle,color_clock,addr,value
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path
from typing import Iterator, NamedTuple


# TIA register names — for human-readable scanline dumps.
# Index = addr & 0x3F. Names from xitari/emucore/TIA.cxx.
_TIA_WRITE_NAMES = {
    0x00: "VSYNC", 0x01: "VBLANK", 0x02: "WSYNC", 0x03: "RSYNC",
    0x04: "NUSIZ0", 0x05: "NUSIZ1", 0x06: "COLUP0", 0x07: "COLUP1",
    0x08: "COLUPF", 0x09: "COLUBK", 0x0A: "CTRLPF", 0x0B: "REFP0",
    0x0C: "REFP1", 0x0D: "PF0", 0x0E: "PF1", 0x0F: "PF2",
    0x10: "RESP0", 0x11: "RESP1", 0x12: "RESM0", 0x13: "RESM1",
    0x14: "RESBL", 0x15: "AUDC0", 0x16: "AUDC1", 0x17: "AUDF0",
    0x18: "AUDF1", 0x19: "AUDV0", 0x1A: "AUDV1", 0x1B: "GRP0",
    0x1C: "GRP1", 0x1D: "ENAM0", 0x1E: "ENAM1", 0x1F: "ENABL",
    0x20: "HMP0", 0x21: "HMP1", 0x22: "HMM0", 0x23: "HMM1",
    0x24: "HMBL", 0x25: "VDELP0", 0x26: "VDELP1", 0x27: "VDELBL",
    0x28: "RESMP0", 0x29: "RESMP1", 0x2A: "HMOVE", 0x2B: "HMCLR",
    0x2C: "CXCLR",
}
_TIA_READ_NAMES = {
    0x00: "CXM0P", 0x01: "CXM1P", 0x02: "CXP0FB", 0x03: "CXP1FB",
    0x04: "CXM0FB", 0x05: "CXM1FB", 0x06: "CXBLPF", 0x07: "CXPPMM",
    0x08: "INPT0", 0x09: "INPT1", 0x0A: "INPT2", 0x0B: "INPT3",
    0x0C: "INPT4", 0x0D: "INPT5",
}


class Event(NamedTuple):
    idx: int
    frame: int
    kind: str           # peek, poke, tick, wsync_release, frame_boundary
    scanline: int
    scanline_cycle: int
    color_clock: int
    addr: int           # parsed from hex
    value: int


def _iter_events(path: Path) -> Iterator[Event]:
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield Event(
                idx=int(row["global_idx"]),
                frame=int(row["frame"]),
                kind=row["kind"],
                scanline=int(row["scanline"]),
                scanline_cycle=int(row["scanline_cycle"]),
                color_clock=int(row["color_clock"]),
                addr=int(row["addr"], 16),
                value=int(row["value"]),
            )


def _label(ev: Event) -> str:
    """Human-readable register name for TIA peek/poke events.

    addr is the full 13-bit mirrored bus address; the TIA decodes
    only A0..A5 for writes and A0..A3 for reads. So we mask down to
    the relevant nibble per direction.
    """
    masked13 = ev.addr & 0x1FFF
    in_tia_region = (masked13 < 0x80)
    if not in_tia_region:
        return ""
    if ev.kind == "poke":
        reg = ev.addr & 0x3F
        return _TIA_WRITE_NAMES.get(reg, f"TIA-W-${reg:02x}")
    if ev.kind == "peek":
        reg = ev.addr & 0x0F
        return _TIA_READ_NAMES.get(reg, f"TIA-R-${reg:01x}")
    return ""


# ---------------- Subcommand handlers --------------------------------------- #

def cmd_scanline(args: argparse.Namespace) -> int:
    """Dump every event on (frame, scanline) in beam order."""
    target_frame = args.frame
    target_scan = args.scanline
    n = 0
    for ev in _iter_events(args.csv):
        if ev.frame != target_frame:
            continue
        if ev.scanline != target_scan:
            continue
        label = _label(ev)
        suffix = f"  ({label})" if label else ""
        print(f"  idx={ev.idx:<8} sc={ev.scanline_cycle:>2}  "
              f"cc={ev.color_clock:>3}  {ev.kind:<13}  "
              f"addr=${ev.addr:04x}  val=${ev.value:02x}{suffix}")
        n += 1
    print(f"-- {n} events on frame {target_frame}, scanline {target_scan} --",
          file=sys.stderr)
    return 0


def cmd_poke_trace(args: argparse.Namespace) -> int:
    """Dump every poke to <addr> within <frame>."""
    target_frame = args.frame
    target_addr = int(args.addr, 16)
    n = 0
    for ev in _iter_events(args.csv):
        if ev.frame != target_frame:
            continue
        if ev.kind != "poke":
            continue
        if (ev.addr & 0xFFFF) != (target_addr & 0xFFFF):
            continue
        label = _label(ev)
        suffix = f"  ({label})" if label else ""
        print(f"  idx={ev.idx:<8} scanline={ev.scanline:<3} "
              f"sc={ev.scanline_cycle:>2}  cc={ev.color_clock:>3}  "
              f"addr=${ev.addr:04x}  val=${ev.value:02x}{suffix}")
        n += 1
    print(f"-- {n} pokes to ${target_addr:04x} in frame {target_frame} --",
          file=sys.stderr)
    return 0


def cmd_hmove(args: argparse.Namespace) -> int:
    """Find the HMOVE poke just before <scanline-target> in <frame>."""
    target_frame = args.frame
    target_scan = args.scanline
    HMOVE_REG = 0x2A
    # Iterate, remember the most recent HMOVE before target_scan.
    last_hmove = None
    n_total = 0
    for ev in _iter_events(args.csv):
        if ev.frame != target_frame:
            continue
        if ev.kind == "poke" and (ev.addr & 0x3F) == HMOVE_REG:
            n_total += 1
            if ev.scanline < target_scan or (
                ev.scanline == target_scan and ev.scanline_cycle <= 21
            ):
                last_hmove = ev
            else:
                break
    if last_hmove is None:
        print(f"  no HMOVE poke before frame {target_frame} scanline "
              f"{target_scan}", file=sys.stderr)
        return 0
    sc = last_hmove.scanline_cycle
    # _HMOVE_BLANK_ENABLE_CYCLES[sc] is True for sc in 0..20 and sc == 75.
    in_blank_window = sc < 21 or sc == 75
    print(f"  last HMOVE before frame {target_frame} scanline {target_scan}:")
    print(f"    scanline={last_hmove.scanline}  scanline_cycle={sc}  "
          f"color_clock={last_hmove.color_clock}")
    print(f"    -> HMOVE blank window enabled? "
          f"{'YES' if in_blank_window else 'NO'} (sc={sc} in [0..20] or =75)")
    print(f"  ({n_total} total HMOVE pokes in this frame)")
    return 0


def cmd_diff(args: argparse.Namespace) -> int:
    """Walk both CSVs in parallel; report first event that differs."""
    a = _iter_events(args.csv_a)
    b = _iter_events(args.csv_b)
    n = 0
    for ev_a, ev_b in zip(a, b):
        n += 1
        # Compare all fields except global_idx.
        if (ev_a.frame, ev_a.kind, ev_a.scanline, ev_a.scanline_cycle,
            ev_a.color_clock, ev_a.addr, ev_a.value) != \
           (ev_b.frame, ev_b.kind, ev_b.scanline, ev_b.scanline_cycle,
            ev_b.color_clock, ev_b.addr, ev_b.value):
            print(f"  first divergence at event #{n}:")
            print(f"    A: {ev_a}")
            print(f"    B: {ev_b}")
            return 0
    print(f"  identical for {n} events (both traces fully consumed)",
          file=sys.stderr)
    return 0


def cmd_summary(args: argparse.Namespace) -> int:
    """Print high-level stats."""
    total = 0
    by_kind: Counter[str] = Counter()
    by_frame: Counter[int] = Counter()
    poke_addrs: Counter[int] = Counter()
    peek_addrs: Counter[int] = Counter()
    for ev in _iter_events(args.csv):
        total += 1
        by_kind[ev.kind] += 1
        by_frame[ev.frame] += 1
        if ev.kind == "poke" and (ev.addr & 0x1FFF) < 0x80:
            poke_addrs[ev.addr & 0x3F] += 1
        elif ev.kind == "peek" and (ev.addr & 0x1FFF) < 0x80:
            peek_addrs[ev.addr & 0x0F] += 1
    print(f"  total events:   {total:>9}")
    print(f"  frames:         {len(by_frame):>9}")
    print(f"  by kind:")
    for k, v in by_kind.most_common():
        print(f"    {k:<16} {v:>9}")
    print(f"\n  top 12 TIA pokes:")
    for reg, count in poke_addrs.most_common(12):
        name = _TIA_WRITE_NAMES.get(reg, f"${reg:02x}")
        print(f"    {name:<8} ({reg:#04x}) {count:>7}")
    print(f"\n  top 8 TIA peeks:")
    for reg, count in peek_addrs.most_common(8):
        name = _TIA_READ_NAMES.get(reg, f"${reg:02x}")
        print(f"    {name:<8} ({reg:#04x}) {count:>7}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    p_scan = sub.add_parser("scanline", help="dump events on (frame, scanline)")
    p_scan.add_argument("csv", type=Path)
    p_scan.add_argument("frame", type=int)
    p_scan.add_argument("scanline", type=int)
    p_scan.set_defaults(func=cmd_scanline)

    p_poke = sub.add_parser("poke-trace", help="dump every poke to addr in frame")
    p_poke.add_argument("csv", type=Path)
    p_poke.add_argument("frame", type=int)
    p_poke.add_argument("addr", help="hex address, e.g. 2a or 002a")
    p_poke.set_defaults(func=cmd_poke_trace)

    p_hmove = sub.add_parser("hmove", help="find HMOVE before scanline; check blank window")
    p_hmove.add_argument("csv", type=Path)
    p_hmove.add_argument("frame", type=int)
    p_hmove.add_argument("scanline", type=int)
    p_hmove.set_defaults(func=cmd_hmove)

    p_diff = sub.add_parser("diff", help="diff two CSVs event-by-event")
    p_diff.add_argument("csv_a", type=Path)
    p_diff.add_argument("csv_b", type=Path)
    p_diff.set_defaults(func=cmd_diff)

    p_sum = sub.add_parser("summary", help="high-level stats")
    p_sum.add_argument("csv", type=Path)
    p_sum.set_defaults(func=cmd_summary)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
