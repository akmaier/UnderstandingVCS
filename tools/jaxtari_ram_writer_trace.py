#!/usr/bin/env python3
"""PXC1-x diagnostic — log every RAM write inside one or more Pong frames.

The 10-byte conformance gap on `pong_noop_10` (jaxtari vs xitari) is
already known to be data-path-only (CPU registers match at every
frame boundary — see STATUS.md PXC1-x round 2+). The remaining task
is to find which TIA / RIOT read returns a different value at one
specific cycle, propagating through Pong's frame logic to the bytes
that diverge.

This tool gives one half of that picture: it instruments jaxtari's
Bus to record `(PC, addr, value)` for every CPU-driven RAM write
inside a chosen frame range. With the writes-by-PC view in hand,
the bytes that diverge in PXC1's RAM-diff report can be walked
backwards to the specific Pong opcode + the values it computed
from.

Because the bus poke doesn't know the CPU state directly, the
harness single-steps the CPU and threads PC through each step so
every write is attributed to the *instruction that wrote it* (the
PC of the LDA/STA whose memory cycle triggered the write).

The xitari side of the comparison needs a matching emit from
`xitari/emucore/m6502` — either via the `friend class CpuDebug`
mechanism already used by `trace_dump --cpu`, or via a small
xitari-side instrumentation pass. Both are external to this tool.

Usage
-----

    python tools/jaxtari_ram_writer_trace.py \\
        --rom xitari/roms/pong.bin \\
        --frames 1

    # Limit to specific RAM cells (the 10-byte divergence set):
    python tools/jaxtari_ram_writer_trace.py \\
        --rom xitari/roms/pong.bin \\
        --cells 01,04,0f,25,26,27,28,33,3b,3c
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# `tools/` lives next to `jaxtari/`. Add the repo's jaxtari package
# (one level up from this file) to sys.path.
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

import jax.numpy as jnp
import jaxtari.bus.system as _bus_module
from jaxtari.console import (
    Console,
    initial_console,
    console_reset,
    console_step,
    run_until_frame,
)
from jaxtari.cpu.m6502 import step as cpu_step


def _load_rom(path: Path) -> np.ndarray:
    return np.fromfile(path, dtype=np.uint8)


# Small 6502 disassembler — covers all the opcodes the diagnostic tool
# might attribute a write to (every STA/STX/STY/INC/DEC/shift/RMW mode +
# PHA/PHP), plus the loads and branches so `--reads` rows also get
# readable labels. Each entry: (mnemonic, addressing-mode string,
# length). Unimplemented opcodes render as "??? $XX" with length 1.
_DISASM_TABLE: dict[int, tuple[str, str, int]] = {
    # --- Stores
    0x85: ("STA", "zp", 2),     0x95: ("STA", "zp,X", 2),
    0x8D: ("STA", "abs", 3),    0x9D: ("STA", "abs,X", 3),
    0x99: ("STA", "abs,Y", 3),  0x81: ("STA", "(zp,X)", 2),
    0x91: ("STA", "(zp),Y", 2),
    0x86: ("STX", "zp", 2),     0x96: ("STX", "zp,Y", 2),
    0x8E: ("STX", "abs", 3),
    0x84: ("STY", "zp", 2),     0x94: ("STY", "zp,X", 2),
    0x8C: ("STY", "abs", 3),
    # --- INC / DEC memory
    0xE6: ("INC", "zp", 2),     0xF6: ("INC", "zp,X", 2),
    0xEE: ("INC", "abs", 3),    0xFE: ("INC", "abs,X", 3),
    0xC6: ("DEC", "zp", 2),     0xD6: ("DEC", "zp,X", 2),
    0xCE: ("DEC", "abs", 3),    0xDE: ("DEC", "abs,X", 3),
    # --- Shifts (RMW memory modes)
    0x06: ("ASL", "zp", 2),     0x16: ("ASL", "zp,X", 2),
    0x0E: ("ASL", "abs", 3),    0x1E: ("ASL", "abs,X", 3),
    0x46: ("LSR", "zp", 2),     0x56: ("LSR", "zp,X", 2),
    0x4E: ("LSR", "abs", 3),    0x5E: ("LSR", "abs,X", 3),
    0x26: ("ROL", "zp", 2),     0x36: ("ROL", "zp,X", 2),
    0x2E: ("ROL", "abs", 3),    0x3E: ("ROL", "abs,X", 3),
    0x66: ("ROR", "zp", 2),     0x76: ("ROR", "zp,X", 2),
    0x6E: ("ROR", "abs", 3),    0x7E: ("ROR", "abs,X", 3),
    # --- Stack push / pull
    0x48: ("PHA", "impl", 1),   0x08: ("PHP", "impl", 1),
    0x68: ("PLA", "impl", 1),   0x28: ("PLP", "impl", 1),
    # --- Loads (useful when `--reads` shows a TIA peek attributed to LDA)
    0xA9: ("LDA", "imm", 2),    0xA5: ("LDA", "zp", 2),
    0xB5: ("LDA", "zp,X", 2),   0xAD: ("LDA", "abs", 3),
    0xBD: ("LDA", "abs,X", 3),  0xB9: ("LDA", "abs,Y", 3),
    0xA1: ("LDA", "(zp,X)", 2), 0xB1: ("LDA", "(zp),Y", 2),
    0xA2: ("LDX", "imm", 2),    0xA6: ("LDX", "zp", 2),
    0xB6: ("LDX", "zp,Y", 2),   0xAE: ("LDX", "abs", 3),
    0xBE: ("LDX", "abs,Y", 3),
    0xA0: ("LDY", "imm", 2),    0xA4: ("LDY", "zp", 2),
    0xB4: ("LDY", "zp,X", 2),   0xAC: ("LDY", "abs", 3),
    0xBC: ("LDY", "abs,X", 3),
    # --- Transfers
    0xAA: ("TAX", "impl", 1),   0xA8: ("TAY", "impl", 1),
    0x8A: ("TXA", "impl", 1),   0x98: ("TYA", "impl", 1),
    0xBA: ("TSX", "impl", 1),   0x9A: ("TXS", "impl", 1),
    # --- Branches (rel)
    0x10: ("BPL", "rel", 2),    0x30: ("BMI", "rel", 2),
    0x50: ("BVC", "rel", 2),    0x70: ("BVS", "rel", 2),
    0x90: ("BCC", "rel", 2),    0xB0: ("BCS", "rel", 2),
    0xD0: ("BNE", "rel", 2),    0xF0: ("BEQ", "rel", 2),
    # --- Control flow
    0x4C: ("JMP", "abs", 3),    0x6C: ("JMP", "(abs)", 3),
    0x20: ("JSR", "abs", 3),    0x60: ("RTS", "impl", 1),
    0x40: ("RTI", "impl", 1),
    # --- Status flags + NOP/BRK
    0x18: ("CLC", "impl", 1),   0x38: ("SEC", "impl", 1),
    0x58: ("CLI", "impl", 1),   0x78: ("SEI", "impl", 1),
    0xB8: ("CLV", "impl", 1),   0xD8: ("CLD", "impl", 1),
    0xF8: ("SED", "impl", 1),
    0xEA: ("NOP", "impl", 1),   0x00: ("BRK", "impl", 1),
    # --- INX/INY/DEX/DEY
    0xE8: ("INX", "impl", 1),   0xC8: ("INY", "impl", 1),
    0xCA: ("DEX", "impl", 1),   0x88: ("DEY", "impl", 1),
}


def _disasm(rom: np.ndarray, pc: int) -> str:
    """Return a short disassembly string for the opcode at CPU address `pc`.

    The 6507 mirrors a 4K cart at $F000-$FFFF; we just take `pc & 0x0FFF`
    as the ROM offset. Returns "OPCODE mode  (raw bytes)".
    """
    rom_off = pc & 0x0FFF
    if rom_off >= len(rom):
        return f"<rom too small for PC ${pc:04X}>"
    op = int(rom[rom_off])
    entry = _DISASM_TABLE.get(op)
    if entry is None:
        return f"??? ${op:02X}  (unhandled in disasm table)"
    mnem, mode, length = entry
    operand_bytes = rom[rom_off + 1 : rom_off + length]
    operand_hex = " ".join(f"${b:02X}" for b in operand_bytes)
    if length == 1:
        return f"{mnem}                 (raw {op:02X})"
    if length == 2:
        operand = int(operand_bytes[0]) if len(operand_bytes) else 0
        if mode == "imm":
            return f"{mnem} #${operand:02X}           (raw {op:02X} {operand:02X})"
        elif mode == "rel":
            # Signed byte → target = pc + 2 + signed(operand)
            signed = operand - 0x100 if operand >= 0x80 else operand
            target = (pc + 2 + signed) & 0xFFFF
            return (f"{mnem} ${target:04X}         "
                    f"(raw {op:02X} {operand:02X}, rel{signed:+d})")
        elif mode == "zp":
            return f"{mnem} ${operand:02X}            (raw {op:02X} {operand:02X})"
        elif mode == "zp,X":
            return f"{mnem} ${operand:02X},X          (raw {op:02X} {operand:02X})"
        elif mode == "zp,Y":
            return f"{mnem} ${operand:02X},Y          (raw {op:02X} {operand:02X})"
        elif mode == "(zp,X)":
            return f"{mnem} (${operand:02X},X)        (raw {op:02X} {operand:02X})"
        elif mode == "(zp),Y":
            return f"{mnem} (${operand:02X}),Y        (raw {op:02X} {operand:02X})"
    elif length == 3 and len(operand_bytes) == 2:
        addr = int(operand_bytes[0]) | (int(operand_bytes[1]) << 8)
        suffix = ""
        if mode == "abs,X":
            suffix = ",X"
        elif mode == "abs,Y":
            suffix = ",Y"
        elif mode == "(abs)":
            suffix = ""
            return (f"{mnem} (${addr:04X})        "
                    f"(raw {op:02X} {operand_bytes[0]:02X} {operand_bytes[1]:02X})")
        return (f"{mnem} ${addr:04X}{suffix}        "
                f"(raw {op:02X} {operand_bytes[0]:02X} {operand_bytes[1]:02X})")
    return f"{mnem} {mode}  (raw {operand_hex})"


def _switches_apply(console: Console, *, reset_pressed: bool) -> Console:
    """Mirror env.console_switches without importing the env layer."""
    from jaxtari.riot.system import set_swchb_input
    # SWCHB bit 0 is RESET (active-low): pressed → bit 0 = 0, released → 1.
    cur = console.bus.riot.swchb_in
    new = (int(cur) & ~0x01) if reset_pressed else (int(cur) | 0x01)
    new_riot = set_swchb_input(console.bus.riot, new & 0xFF)
    new_bus = console.bus._replace(riot=new_riot)
    return Console(cpu=console.cpu, bus=new_bus)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rom", type=Path, required=True,
                    help="path to the Pong ROM (or any ROM)")
    ap.add_argument("--frames", type=int, default=1,
                    help="number of user frames to trace after boot (default 1)")
    ap.add_argument("--boot-noop", type=int, default=60,
                    help="ALE-equivalent boot NOOP frames (default 60)")
    ap.add_argument("--boot-reset", type=int, default=4,
                    help="ALE-equivalent boot RESET-switch frames (default 4)")
    ap.add_argument("--cells", default=None,
                    help="comma-separated hex RAM addresses to filter; "
                         "shows all writes if omitted")
    ap.add_argument("--max-rows", type=int, default=80,
                    help="cap output rows (default 80)")
    ap.add_argument("--reads", action="store_true",
                    help="also log TIA / RIOT reads (data-path source-of-"
                         "truth for the divergent registers); without it "
                         "only RAM writes are shown")
    args = ap.parse_args()

    target_cells: set[int] | None = None
    if args.cells is not None:
        target_cells = {int(c, 16) for c in args.cells.split(",")}

    rom = _load_rom(args.rom)
    rom_jnp = jnp.asarray(rom, dtype=jnp.uint8)
    console = console_reset(initial_console(rom_jnp))

    # Boot burn — drive 60 NOOP frames + 4 RESET-switch frames.
    for _ in range(args.boot_noop):
        console = run_until_frame(console)
    if args.boot_reset > 0:
        console = _switches_apply(console, reset_pressed=True)
        for _ in range(args.boot_reset):
            console = run_until_frame(console)
        console = _switches_apply(console, reset_pressed=False)

    # Capture writes AND (optionally) reads. Each row is
    # ("W"|"R", pc, a, x, y, p, bus_state_before, addr, value).
    # `bus_state_before` is the data_bus_state at event time — for
    # reads this is what feeds the floating-bus OR; for writes it's
    # what was on the bus just before the store. Single-step the CPU
    # so the PC + register snapshot at each event is bit-accurate.
    rows: list[tuple[str, int, int, int, int, int, int, int, int]] = []
    original_bus_poke = _bus_module._bus_poke
    original_bus_peek = _bus_module._bus_peek

    def make_traced_poke(pc_before: int, a: int, x: int, y: int, p: int):
        def traced(bus, addr, value):
            addr_masked = addr & 0x1FFF
            if (addr_masked & 0x80) and not (addr_masked & 0x1000) \
                    and not (addr_masked & 0x200):
                ram_idx = addr_masked & 0x7F
                if target_cells is None or ram_idx in target_cells:
                    rows.append(("W", pc_before, a, x, y, p,
                                 int(bus.data_bus_state),
                                 ram_idx, value & 0xFF))
            return original_bus_poke(bus, addr, value)
        return traced

    def make_traced_peek(pc_before: int, a: int, x: int, y: int, p: int):
        def traced(bus, addr):
            addr_masked = addr & 0x1FFF
            bus_state_before = int(bus.data_bus_state)
            value, new_bus = original_bus_peek(bus, addr)
            if args.reads:
                # Only log TIA reads (A12=0, A7=0) and RIOT I/O reads
                # (A12=0, A7=1, A9=1) — those are the data-path of
                # interest. RAM reads are too numerous.
                is_tia      = (not (addr_masked & 0x1000)) and (not (addr_masked & 0x80))
                is_riot_io  = ((not (addr_masked & 0x1000)) and
                               (addr_masked & 0x80) and (addr_masked & 0x200))
                if is_tia or is_riot_io:
                    rows.append(("R", pc_before, a, x, y, p,
                                 bus_state_before,
                                 addr_masked, value & 0xFF))
            return value, new_bus
        return traced

    # Walk one frame's worth of instructions at a time (a generous cap
    # of 100k matches console._FRAME_INSTRUCTION_LIMIT).
    INSTR_LIMIT = 100_000
    for _ in range(args.frames):
        start_frame = int(console.bus.tia.frame)
        for _ in range(INSTR_LIMIT):
            pc_before = int(console.cpu.PC)
            a_before  = int(console.cpu.A)
            x_before  = int(console.cpu.X)
            y_before  = int(console.cpu.Y)
            p_before  = int(console.cpu.P)
            _bus_module._bus_poke = make_traced_poke(
                pc_before, a_before, x_before, y_before, p_before)
            _bus_module._bus_peek = make_traced_peek(
                pc_before, a_before, x_before, y_before, p_before)
            try:
                new_cpu, new_bus = cpu_step(console.cpu, console.bus)
            finally:
                _bus_module._bus_poke = original_bus_poke
                _bus_module._bus_peek = original_bus_peek
            console = Console(cpu=new_cpu, bus=new_bus)
            if int(console.bus.tia.frame) != start_frame:
                break

    rows = rows[: args.max_rows]
    if not rows:
        print("(no events matched — try widening --cells / --reads / --frames)")
        return 0

    kind_w = sum(1 for r in rows if r[0] == "W")
    kind_r = sum(1 for r in rows if r[0] == "R")
    print(f"# {len(rows)} events captured during {args.frames} frame(s) "
          f"({kind_w} writes, {kind_r} TIA/RIOT reads)")
    if target_cells is not None:
        print(f"# write filter: cells = {sorted(target_cells)}")
    print()
    print(f"{'op':>2s}  {'PC':>8s}  A    X    Y    P   bus  {'addr':>5s}  {'val':>4s}  opcode")
    print("-" * 100)
    for kind, pc, a, x, y, p, bus_state, addr, value in rows:
        if kind == "W":
            disasm = _disasm(rom, pc)
        else:
            disasm = "(TIA/RIOT read)"
        print(f"  {kind}   ${pc:04X}  ${a:02X}  ${x:02X}  ${y:02X}  ${p:02X}  "
              f"${bus_state:02X}  ${addr:04X}  ${value:02X}  {disasm}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
