"""record_state — per-step full-state trajectory recorder for P2 (SPEC §E0-2).

Replays a ROM under a fixed input-trace and dumps, **for every frame**, the
complete observable VCS state: RIOT RAM, 6502 CPU registers, the executed
opcode, the TIA register file (+ derived object positions / collisions /
inputs), the RIOT timer & I/O ports, and the rendered framebuffer. The stacked
result is the "activations" tensor Phase C (E5) trains SAEs / probes on and the
state input Phase A (E3) lesions and correlates.

Why this exists, and what it reuses
------------------------------------
The emulator is **exact under a fixed action sequence** (Paper-1), so "a
trajectory" is fully identified by *(game, boot convention, action trace)*.
`record_state` is the heavyweight sibling of `replay.trajectory`: where the
latter stacks only RAM (+ optionally screen) for the unit test, this one
threads the jaxtari `Console` after each frame and reads **all** of its
NamedTuple state. It deliberately does NOT touch the emulator core (SCRUM §7):
it only *observes* `env.console` — the same read-only access
`StellaEnvironment.console` documents for "XAI work that needs to inspect
register / RAM state". ROM loading + xitari-parity boot + the action-trace
format all come straight from the E0-1 harness (`loader`, `replay`).

Conformance horizon
-------------------
Paper-1 bit-exactness vs xitari holds only within a bounded window after the
standard boot — ≈ **30 frames of RAM**, **60 frames of screen** on a fixed
action stream (experiment_design.md §3, caveat ii; the comparison videos are
60 fps, so frame = seconds × 60). `record()` enforces this by default: it
records at most `SCREEN_HORIZON_FRAMES` frames (the screen window, the wider of
the two) and flags `ram_horizon_frames` so a consumer that needs bit-exact RAM
knows to slice `[:RAM_HORIZON_FRAMES]`. Pass `enforce_horizon=False` to record
further on purpose (behavioral probes that don't need bit-exactness); the
artifact records the flag either way so nothing reads past the validated window
unknowingly.

Trajectory tensor layout
------------------------
`record()` returns a `Trajectory` dataclass; `save()` writes it as a single
`.npz` (arrays) + a self-describing `.json` sidecar (SPEC §R record + the
layout manifest). For `n` recorded frames:

  ram          (n, 128)   uint8   RIOT internal RAM, post-frame
  cpu          (n, 7)     int64   [A, X, Y, SP, PC, P, cycles] post-frame
  opcode       (n,)       uint8   byte at the post-frame PC (next instruction)
  tia_registers(n, 64)    uint8   the TIA CPU-write register file
  tia_derived  (n, 10)    int32   [p0_x,p1_x,m0_x,m1_x,bl_x,scanline,
                                    scanline_cycle,frame,color_clock,wsync]
  tia_collisions(n, 8)    uint8   latched CX* collision registers $30-$37
  tia_inpt     (n, 6)     uint8   INPT0..INPT5
  riot         (n, 11)    int64   [my_timer,interval_shift,cycles_when_set,
                                    read_after_int,cycles_when_int_reset,
                                    swcha_in,swchb_in,swacnt,swbcnt,
                                    swcha_out,swchb_out]
  cart         (n, 2)     int64   [kind, current_bank]
  framebuffer  (n, H, 160)uint8   cropped visible screen (env.get_screen())
  actions      (n,)       int64   the action applied to reach each frame
  frames       (n,)       int64   1-based frame index (post that many steps)

Column-name lists for the 2-D fields are exported as module constants
(`CPU_COLS`, `TIA_DERIVED_COLS`, `RIOT_COLS`, `CART_COLS`) and copied into the
JSON sidecar's `layout`, so a consumer never has to hard-code indices.

The "executed-opcode" convention: a frame is many CPU instructions; the single
representative opcode we store is the byte at the PC **after** the frame
completes — i.e. the next instruction the CPU will execute. Read together with
the per-frame CPU/PC arrays this gives the instruction-stream snapshot at each
frame boundary without instrumenting the emulator core.

Command-line use
----------------
    PY=/path/to/jaxtari/.venv/bin/python
    XAI_PRIMARY_REPO=/path/to/primary/checkout \\
    PYTHONPATH=/path/to/worktree/tools \\
    $PY -m xai_study.common.record_state --game space_invaders \\
        --frames 60 --out tools/xai_study/common/out/traj_space_invaders

`--actions` takes a path to an actions file (one int per line, `#` comments) or
is omitted to default to an all-NOOP trace of `--frames` length. See
`replay.load_actions` for the trace format (shared with the Paper-1 sweeps).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# E0-1 harness (loader/replay) + results writer. Works both as
# `xai_study.common.record_state` (tools/ on PYTHONPATH) and as the dotted
# package; the relative import keeps it importable as a package module.
from . import loader, replay, results

#: Re-export the horizon constants so consumers can `from record_state import …`.
RAM_HORIZON_FRAMES: int = replay.RAM_HORIZON_FRAMES        # 30
SCREEN_HORIZON_FRAMES: int = replay.SCREEN_HORIZON_FRAMES  # 60

# Column orders for the 2-D state fields (mirrored into the JSON layout).
CPU_COLS: List[str] = ["A", "X", "Y", "SP", "PC", "P", "cycles"]
TIA_DERIVED_COLS: List[str] = [
    "p0_x", "p1_x", "m0_x", "m1_x", "bl_x",
    "scanline", "scanline_cycle", "frame", "color_clock", "wsync_pending",
]
RIOT_COLS: List[str] = [
    "my_timer", "interval_shift", "cycles_when_set", "read_after_int",
    "cycles_when_int_reset", "swcha_in", "swchb_in", "swacnt", "swbcnt",
    "swcha_out", "swchb_out",
]
CART_COLS: List[str] = ["kind", "current_bank"]


@dataclass
class Trajectory:
    """A recorded per-step full-state trajectory (one row per frame).

    All array fields are stacked along axis 0 = frame index. `meta` carries the
    horizon flags and the run parameters; `layout` names the columns of the 2-D
    fields. `n` is the number of recorded frames.
    """
    game: str
    rom_path: str
    actions: np.ndarray            # (n,) int64
    frames: np.ndarray             # (n,) int64, 1-based
    ram: np.ndarray                # (n, 128) uint8
    cpu: np.ndarray                # (n, 7) int64
    opcode: np.ndarray             # (n,) uint8
    tia_registers: np.ndarray      # (n, 64) uint8
    tia_derived: np.ndarray        # (n, 10) int32
    tia_collisions: np.ndarray     # (n, 8) uint8
    tia_inpt: np.ndarray           # (n, 6) uint8
    riot: np.ndarray               # (n, 11) int64
    cart: np.ndarray               # (n, 2) int64
    framebuffer: np.ndarray        # (n, H, 160) uint8
    meta: Dict

    @property
    def n(self) -> int:
        return int(self.frames.shape[0])

    @property
    def layout(self) -> Dict[str, List[str]]:
        return {
            "cpu": CPU_COLS,
            "tia_derived": TIA_DERIVED_COLS,
            "riot": RIOT_COLS,
            "cart": CART_COLS,
        }

    def arrays(self) -> Dict[str, np.ndarray]:
        """The array payload, keyed for the `.npz` (and SPEC §R sibling)."""
        return {
            "actions": self.actions,
            "frames": self.frames,
            "ram": self.ram,
            "cpu": self.cpu,
            "opcode": self.opcode,
            "tia_registers": self.tia_registers,
            "tia_derived": self.tia_derived,
            "tia_collisions": self.tia_collisions,
            "tia_inpt": self.tia_inpt,
            "riot": self.riot,
            "cart": self.cart,
            "framebuffer": self.framebuffer,
        }


# --------------------------------------------------------------------------- #
# Per-frame state extraction (read-only; never mutates the console)
# --------------------------------------------------------------------------- #

def _u8(a) -> np.ndarray:
    return np.asarray(a, dtype=np.uint8)


def _cpu_row(cpu) -> np.ndarray:
    return np.array(
        [int(cpu.A), int(cpu.X), int(cpu.Y), int(cpu.SP),
         int(cpu.PC), int(cpu.P), int(cpu.cycles)],
        dtype=np.int64)


def _tia_derived_row(tia) -> np.ndarray:
    return np.array(
        [int(tia.p0_x), int(tia.p1_x), int(tia.m0_x), int(tia.m1_x),
         int(tia.bl_x), int(tia.scanline), int(tia.scanline_cycle),
         int(tia.frame), int(tia.color_clock), int(bool(tia.wsync_pending))],
        dtype=np.int32)


def _riot_row(riot) -> np.ndarray:
    return np.array(
        [int(riot.my_timer), int(riot.interval_shift),
         int(riot.cycles_when_set), int(bool(riot.read_after_int)),
         int(riot.cycles_when_int_reset), int(riot.swcha_in),
         int(riot.swchb_in), int(riot.swacnt), int(riot.swbcnt),
         int(riot.swcha_out), int(riot.swchb_out)],
        dtype=np.int64)


def _cart_row(cart) -> np.ndarray:
    bank = getattr(cart, "current_bank", 0)
    # E0's bank is a tuple of slot slices; collapse to -1 so the column stays
    # a scalar int (the kind tag already disambiguates the mapper).
    try:
        bank = int(bank)
    except (TypeError, ValueError):
        bank = -1
    return np.array([int(getattr(cart, "kind", -1)), bank], dtype=np.int64)


def _opcode_at_pc(bus, pc: int) -> int:
    """Byte at `pc` via a side-effect-discarding peek.

    Executing code lives in cart ROM space ($F000+ on the 6507), where `peek`
    is side-effect-free; we still discard the returned bus so a stray RIOT-read
    latch can never leak back into the recorded console.
    """
    from jaxtari.bus import peek
    val, _ = peek(bus, int(pc) & 0xFFFF)
    return int(val) & 0xFF


def _snapshot(console) -> Dict[str, np.ndarray]:
    """Read every observable field off a jaxtari `Console` (no mutation)."""
    cpu, bus = console.cpu, console.bus
    tia, riot = bus.tia, bus.riot
    return {
        "ram": _u8(bus.ram),
        "cpu": _cpu_row(cpu),
        "opcode": np.uint8(_opcode_at_pc(bus, int(cpu.PC))),
        "tia_registers": _u8(tia.registers),
        "tia_derived": _tia_derived_row(tia),
        "tia_collisions": _u8(tia.collisions),
        "tia_inpt": _u8(tia.inpt),
        "riot": _riot_row(riot),
        "cart": _cart_row(bus.cart),
    }


# --------------------------------------------------------------------------- #
# Recorder
# --------------------------------------------------------------------------- #

#: xitari-parity boot (the conformance reference). 60 NOOP + 4 RESET + the
#: double-boot construction probe — the boot the Paper-1 sweeps validate against
#: xitari, so the conformance horizon counts from frame 1 of this boot.
BOOT_XITARI_PARITY: Dict = {
    "boot_noop_steps": 60,
    "boot_reset_steps": 4,
    "construction_probe": True,
}


def record(game: str,
           actions: replay.ActionsArg,
           *,
           rom_path: Optional[os.PathLike] = None,
           n: Optional[int] = None,
           enforce_horizon: bool = True,
           collect_framebuffer: bool = True,
           seed: Optional[int] = None,
           boot: Optional[Dict] = None) -> Trajectory:
    r"""Replay `game` under `actions` and record the full per-frame state.

    Parameters
    ----------
    game
        Game name resolved by `loader.resolve_rom` (or pass `rom_path=`).
    actions
        Action trace — a list of ints or a path to an actions file (the
        Paper-1 format; see `replay.load_actions`).
    n
        Cap on frames to record (default: the whole trace). With
        `enforce_horizon`, additionally clamped to `SCREEN_HORIZON_FRAMES`.
    enforce_horizon
        Keep recording inside the Paper-1 conformance window (default True).
    collect_framebuffer
        Record the cropped visible screen per frame (default True). Set False
        for a lighter RAM/register-only trajectory.
    seed
        Forwarded to `loader.load_game` for the boot RNG (None ⇒ deterministic
        boot; the standard xitari-parity boot is deterministic regardless).
    boot
        Boot kwargs for `loader.load_game` (e.g. `boot_noop_steps`,
        `boot_reset_steps`, `construction_probe`). Default ``None`` ⇒
        `BOOT_XITARI_PARITY` (60 NOOP + 4 RESET + construction probe) so the
        recorded trajectory is the Paper-1 conformance reference. The
        construction probe is the slow part of the boot and only matters for a
        handful of games with never-re-inited free-running counters (e.g.
        surround's \$7d); for fast local iteration on a game that doesn't need
        it, pass e.g. ``boot={"boot_noop_steps": 60, "boot_reset_steps": 4,
        "construction_probe": False}`` — the recorded `meta["boot"]` notes when
        a non-reference boot was used so a downstream conformance check knows.

    Returns a `Trajectory`. With the default boot, frame 1 is the first
    user-action frame and the conformance horizon counts from there.
    """
    acts = replay.load_actions(actions)
    total = len(acts) if n is None else min(int(n), len(acts))
    clamped = False
    if enforce_horizon and total > SCREEN_HORIZON_FRAMES:
        total = SCREEN_HORIZON_FRAMES
        clamped = True

    boot_kw = dict(BOOT_XITARI_PARITY if boot is None else boot)
    is_reference_boot = (boot is None or boot_kw == BOOT_XITARI_PARITY)
    env, _ = loader.load_game(game, rom_path, seed=seed, **boot_kw)
    rom_file = str(loader.resolve_rom(game, rom_path))

    rows: List[Dict[str, np.ndarray]] = []
    screens: List[np.ndarray] = []
    applied: List[int] = []
    for i in range(total):
        a = int(acts[i])
        env.step(a)
        applied.append(a)
        rows.append(_snapshot(env.console))
        if collect_framebuffer:
            screens.append(_u8(env.get_screen()))

    def stack(key: str, dtype) -> np.ndarray:
        if not rows:
            return np.empty((0,), dtype=dtype)
        return np.stack([r[key] for r in rows]).astype(dtype, copy=False)

    if collect_framebuffer and screens:
        fb = np.stack(screens).astype(np.uint8, copy=False)
    elif collect_framebuffer:
        fb = np.empty((0, 0, 160), np.uint8)
    else:
        fb = np.empty((0, 0, 160), np.uint8)

    meta = {
        "game": game,
        "rom_path": rom_file,
        "n_frames": total,
        "requested_n": None if n is None else int(n),
        "trace_length": len(acts),
        "enforce_horizon": bool(enforce_horizon),
        "horizon_clamped": bool(clamped),
        "ram_horizon_frames": RAM_HORIZON_FRAMES,
        "screen_horizon_frames": SCREEN_HORIZON_FRAMES,
        "collect_framebuffer": bool(collect_framebuffer),
        "seed": seed,
        "boot": ("xitari-parity (60 NOOP + 4 RESET + construction probe)"
                 if is_reference_boot else f"non-reference: {boot_kw}"),
        "boot_kwargs": boot_kw,
        "reference_boot": bool(is_reference_boot),
        "opcode_convention": "byte at the post-frame PC (next instruction)",
        "screen_shape": list(fb.shape[1:]) if fb.size else [],
    }
    return Trajectory(
        game=game,
        rom_path=rom_file,
        actions=np.asarray(applied, dtype=np.int64),
        frames=np.arange(1, total + 1, dtype=np.int64),
        ram=stack("ram", np.uint8),
        cpu=stack("cpu", np.int64),
        opcode=stack("opcode", np.uint8),
        tia_registers=stack("tia_registers", np.uint8),
        tia_derived=stack("tia_derived", np.int32),
        tia_collisions=stack("tia_collisions", np.uint8),
        tia_inpt=stack("tia_inpt", np.uint8),
        riot=stack("riot", np.int64),
        cart=stack("cart", np.int64),
        framebuffer=fb,
        meta=meta,
    )


def save(traj: Trajectory, out_dir: os.PathLike, *, exp: str = "traj") -> Path:
    """Write `traj` as a SPEC §R record (JSON) + sibling `.npz` of all arrays.

    Layout: `<out_dir>/<exp>_<game>.json` + `<exp>_<game>.npz`. The JSON is the
    self-describing §R record (`metric_name="trajectory_recorded"`,
    `value=n_frames`) with the trajectory `meta` + column `layout` under
    `extra`, and `arrays` pointing at the `.npz`. `read()` round-trips it.
    """
    rec = results.ResultRecord(
        phase="common",
        method="state_trajectory_recorder",
        game=traj.game,
        target_output="full_state",
        metric_name="trajectory_recorded",
        value=traj.n,
        state=f"f1-{traj.n}",
        n=traj.n,
        seed=traj.meta.get("seed"),
        where="local",
        oracle_ref=None,
        extra={"meta": traj.meta, "layout": traj.layout},
    )
    return results.write_record(rec, out_dir, exp=exp, arrays=traj.arrays())


def read(json_path: os.PathLike) -> Trajectory:
    """Reconstruct a `Trajectory` from a `save()`d JSON+npz pair."""
    data = results.read_record(json_path, load_arrays=True)
    arr = data.get("_arrays", {})
    extra = data.get("extra", {}) or {}
    meta = extra.get("meta", {})
    return Trajectory(
        game=data["game"],
        rom_path=meta.get("rom_path", ""),
        actions=arr.get("actions", np.empty((0,), np.int64)),
        frames=arr.get("frames", np.empty((0,), np.int64)),
        ram=arr.get("ram", np.empty((0, 128), np.uint8)),
        cpu=arr.get("cpu", np.empty((0, 7), np.int64)),
        opcode=arr.get("opcode", np.empty((0,), np.uint8)),
        tia_registers=arr.get("tia_registers", np.empty((0, 64), np.uint8)),
        tia_derived=arr.get("tia_derived", np.empty((0, 10), np.int32)),
        tia_collisions=arr.get("tia_collisions", np.empty((0, 8), np.uint8)),
        tia_inpt=arr.get("tia_inpt", np.empty((0, 6), np.uint8)),
        riot=arr.get("riot", np.empty((0, 11), np.int64)),
        cart=arr.get("cart", np.empty((0, 2), np.int64)),
        framebuffer=arr.get("framebuffer", np.empty((0, 0, 160), np.uint8)),
        meta=meta,
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _default_out_dir() -> Path:
    return Path(__file__).resolve().parent / "out"


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Record a per-step full-state VCS trajectory for a ROM.")
    p.add_argument("--game", default="pong",
                   help="game name (loader-resolved); default pong")
    p.add_argument("--rom-path", default=None,
                   help="explicit ROM file (overrides --game resolution)")
    p.add_argument("--actions", default=None,
                   help="actions-file path (one int/line); default all-NOOP")
    p.add_argument("--frames", type=int, default=SCREEN_HORIZON_FRAMES,
                   help=f"frames to record (default {SCREEN_HORIZON_FRAMES} = "
                        "screen horizon)")
    p.add_argument("--out", default=None,
                   help="output stem dir (default tools/xai_study/common/out)")
    p.add_argument("--exp", default="traj", help="filename prefix (default traj)")
    p.add_argument("--no-framebuffer", action="store_true",
                   help="skip recording the screen (lighter trajectory)")
    p.add_argument("--no-horizon", action="store_true",
                   help="record past the conformance horizon (not bit-exact)")
    p.add_argument("--fast-boot", action="store_true",
                   help="skip the slow construction probe (60 NOOP + 4 RESET "
                        "only) — much faster on jaxtari; NOT the conformance "
                        "reference for probe-sensitive games (e.g. surround)")
    p.add_argument("--seed", type=int, default=None, help="boot RNG seed")
    args = p.parse_args(argv)

    acts: replay.ActionsArg
    if args.actions:
        acts = args.actions
    else:
        acts = [0] * int(args.frames)  # all-NOOP trace of the requested length

    boot = None
    if args.fast_boot:
        boot = {"boot_noop_steps": 60, "boot_reset_steps": 4,
                "construction_probe": False}

    traj = record(
        args.game, acts,
        rom_path=args.rom_path,
        n=int(args.frames),
        enforce_horizon=not args.no_horizon,
        collect_framebuffer=not args.no_framebuffer,
        seed=args.seed,
        boot=boot,
    )
    out_dir = Path(args.out) if args.out else _default_out_dir()
    path = save(traj, out_dir, exp=args.exp)
    print(f"recorded {traj.n} frames for '{traj.game}' "
          f"(horizon_clamped={traj.meta['horizon_clamped']})")
    print(f"  ram={traj.ram.shape} cpu={traj.cpu.shape} "
          f"tia_registers={traj.tia_registers.shape} "
          f"framebuffer={traj.framebuffer.shape}")
    print(f"  wrote {path}")
    print(f"        {path.with_suffix('.npz')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
