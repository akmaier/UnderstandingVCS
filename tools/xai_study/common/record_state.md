# `record_state` — per-step full-state trajectory recorder (P2 / SPEC §E0-2)

Replays a ROM under a fixed input-trace and dumps, **for every frame**, the
complete observable VCS state. The stacked result is the **"activations" tensor**
Phase C (E5) trains SAEs / probes on and the **state input** Phase A (E3)
lesions and correlates. Built on the E0-1 harness (`loader` + `replay`) and the
SPEC §R results writer (`results`); it only *observes* the jaxtari `Console`
(`StellaEnvironment.console`) and never modifies the emulator core (SCRUM §7).

## What is recorded (per frame)

| field | shape | dtype | meaning |
|---|---|---|---|
| `ram`            | (n, 128)   | uint8 | RIOT internal RAM, post-frame |
| `cpu`            | (n, 7)     | int64 | 6502 registers — see `CPU_COLS` |
| `opcode`         | (n,)       | uint8 | byte at the **post-frame PC** (next instruction) |
| `tia_registers`  | (n, 64)    | uint8 | the TIA CPU-write register file ($00–$3F) |
| `tia_derived`    | (n, 10)    | int32 | derived TIA state — see `TIA_DERIVED_COLS` |
| `tia_collisions` | (n, 8)     | uint8 | latched CX* collision registers $30–$37 |
| `tia_inpt`       | (n, 6)     | uint8 | INPT0..INPT5 |
| `riot`           | (n, 11)    | int64 | RIOT timer + I/O ports — see `RIOT_COLS` |
| `cart`           | (n, 2)     | int64 | `[kind, current_bank]` (bank −1 for slice-mapped E0) |
| `framebuffer`    | (n, H, 160)| uint8 | cropped visible screen (`env.get_screen()`), H≈210 NTSC |
| `actions`        | (n,)       | int64 | action applied to reach each frame |
| `frames`         | (n,)       | int64 | 1-based frame index (post that many steps) |

`n` = number of recorded frames. Axis 0 is always the frame index, so any field
is sliced as `traj.<field>[:k]` for the first `k` frames.

### Column orders for the 2-D fields

These are exported as module constants **and** copied into the JSON sidecar's
`layout`, so a consumer never hard-codes an index:

- `CPU_COLS = [A, X, Y, SP, PC, P, cycles]`
- `TIA_DERIVED_COLS = [p0_x, p1_x, m0_x, m1_x, bl_x, scanline, scanline_cycle, frame, color_clock, wsync_pending]`
- `RIOT_COLS = [my_timer, interval_shift, cycles_when_set, read_after_int, cycles_when_int_reset, swcha_in, swchb_in, swacnt, swbcnt, swcha_out, swchb_out]`
- `CART_COLS = [kind, current_bank]`

### "Executed-opcode" convention

A single frame is many CPU instructions. The one representative opcode stored
per frame is the byte at the PC **after** the frame completes — i.e. the next
instruction the CPU will execute. Read together with the per-frame CPU/PC arrays
this gives the instruction-stream snapshot at each frame boundary **without**
instrumenting the emulator core. The byte is read via a side-effect-discarding
`peek`; executing code lives in cart-ROM space where `peek` has no side effects.

## Conformance horizon (the hard bound)

Paper-1 bit-exactness vs xitari holds only within a bounded window after the
standard boot — **≈ 30 frames of RAM, 60 frames of screen** on a fixed action
stream (`experiment_design.md` §3, caveat ii; the comparison videos are 60 fps,
so frame = seconds × 60). Constants live in `replay` and are re-exported here:
`RAM_HORIZON_FRAMES = 30`, `SCREEN_HORIZON_FRAMES = 60`.

`record(..., enforce_horizon=True)` (the default) records at most
`SCREEN_HORIZON_FRAMES` frames (the wider window) and sets
`meta["horizon_clamped"]` if it had to truncate. **A consumer that needs
bit-exact RAM must slice `[:RAM_HORIZON_FRAMES]`** — `meta["ram_horizon_frames"]`
records the bound. Pass `enforce_horizon=False` to record further on purpose
(behavioral probes that don't need bit-exactness); the flag is always recorded
so nothing reads past the validated window unknowingly.

## Boot convention

The default boot is the **xitari-parity reference**: 60 NOOP + 4 RESET frames +
the double-boot construction probe (`BOOT_XITARI_PARITY`), the boot the Paper-1
sweeps validate against xitari, so frame 1 is the first user-action frame and the
horizon counts from there. `meta["reference_boot"]` is `True` for this boot.

The construction probe is the **slow** part of the boot and only matters for the
handful of games with a never-re-inited free-running counter (e.g. surround's
`$7d`). For fast local iteration on a game that doesn't need it, pass the
`--fast-boot` CLI flag (or `boot={"boot_noop_steps":60,"boot_reset_steps":4,
"construction_probe":False}`); `meta["reference_boot"]` then records `False` so a
downstream conformance check knows a non-reference boot was used. Space Invaders,
Pong, and Breakout are **not** probe-sensitive, so `--fast-boot` is bit-exact for
them within the horizon.

## Input-trace format

Reused verbatim from `replay.load_actions` (the Paper-1 sweep format):

- an in-memory list of ints — one ALE action per frame; or
- a path to an **actions file**: one integer per line, blank lines and `#`
  comments ignored.

Action ints are the standard ALE/jaxtari `Action` enum (0 = NOOP, etc.). The
CLI defaults to an all-NOOP trace of `--frames` length when `--actions` is
omitted.

## Output artifact

`save(traj, out_dir, exp="traj")` writes a **SPEC §R record pair**:

- `<out_dir>/<exp>_<game>_f1-<n>.json` — the self-describing §R record
  (`method="state_trajectory_recorder"`, `metric_name="trajectory_recorded"`,
  `value=n`), with the full `meta` (boot, horizon flags, run params) and the
  column `layout` under `extra`, and `arrays` naming the sibling `.npz`.
- `<out_dir>/<exp>_<game>_f1-<n>.npz` — all arrays above, compressed.

`read(json_path)` reconstructs the `Trajectory` (round-trips `save`).

> The `out/` artifacts are **generated**, not source: this item's `file_scope`
> is `record_state.*` only, so the recorder code is committed and the artifact is
> reproduced on demand by the command below (matching the ROM-not-redistributed
> discipline — ROMs are gitignored, read in place).

## Running it

```bash
PY=/path/to/jaxtari/.venv/bin/python
XAI_PRIMARY_REPO=/path/to/primary/checkout \
PYTHONPATH=/path/to/worktree/tools \
$PY -m xai_study.common.record_state \
    --game space_invaders --frames 30 --fast-boot \
    --out tools/xai_study/common/out
```

Flags: `--game`, `--rom-path`, `--actions <file>`, `--frames N`, `--out DIR`,
`--exp PREFIX`, `--no-framebuffer` (lighter RAM/register-only trace),
`--no-horizon` (record past the bit-exact window), `--fast-boot`, `--seed`.

> **ENV note (jaxtari ~205× slower):** the boot dominates wall-clock — the
> construction-probe reference boot is ≈ 188 emulated frames and can take tens of
> minutes on the M1; `--fast-boot` (124 boot frames, no probe) is the practical
> choice for local SI/Pong recording and is bit-exact for those games. For big
> batches use the cluster GPU venv (see `cluster.md`).

## Programmatic use

```python
from xai_study.common import record_state as R
traj = R.record("space_invaders", actions=[0]*30, n=30)   # n≤30 stays RAM-exact
ram_exact = traj.ram[:R.RAM_HORIZON_FRAMES]               # bit-exact RAM slice
path = R.save(traj, "tools/xai_study/common/out")
back = R.read(path)                                        # round-trips
```
