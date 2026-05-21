# tools/

Reference-trace generator for the jaxtari / jutari conformance harness. See [`../PORTING_PLAN.md`](../PORTING_PLAN.md) §4 for the role this plays in the porting plan.

## trace_dump

A small C++ program that links against `libxitari.a` and writes one JSON line per Atari frame to stdout. Each line carries the public ALE state for that frame: frame number, episode frame, action, reward, cumulative reward, lives, done flag, the 128 B RAM (hex), and — with `--screen` — the framebuffer (hex).

### Build

Requires `xitari/` cloned at the repo root and built first (xitari produces `libxitari.a` under that directory):

```sh
cd ../xitari && cmake . && make
cd ../tools && make
```

### Run

```sh
# minimal example
./trace_dump --rom /path/to/rom.bin --actions actions.txt --max-frames 5000 > trace.jsonl

# include screen data
./trace_dump --rom /path/to/rom.bin --actions actions.txt --screen > trace.jsonl
```

`actions.txt` is one integer ALE action ID per line; lines starting with `#` are comments. IDs are defined in `xitari/ale_interface.hpp` (`PLAYER_A_NOOP=0`, `PLAYER_A_FIRE=1`, etc.).

`--repeat-last-on-exhaust` makes the runner stay on the last action after the file ends, instead of stopping.

### What this tool covers (and what it does not)

Covers, today, via the public `ALEInterface`:

- frame-level RAM
- screen buffer (with `--screen`)
- reward / cumulative reward / lives / done

Does **not** cover (needed for Phases P1–P5 conformance):

- CPU registers (A, X, Y, SP, PC, P, cycles)
- TIA register file
- RIOT timer / I/O register state
- cartridge bank state
- per-CPU-cycle (not per-frame) snapshots

Exposing those requires patching xitari to add a debug interface that reaches into `Console`, `System`, `M6502`, `TIA`, and `M6532`. That patch is tracked in PORTING_PLAN.md §4.1 as a follow-up to P0.

For Phase P6 (full-game frame matching) the frame-level trace is sufficient.
