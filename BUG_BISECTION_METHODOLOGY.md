# Bug-bisection methodology — how to find difficult jutari/jaxtari↔xitari divergences

This document captures the technique that closed the **breakout
ball-doesn't-die** bug (commit `20b5de0`, 2026-06-03) and the
**pong frame-20 / breakout frame-92** investigations that led there.
Use it whenever a port's RAM diverges from xitari starting at a specific
frame and you can't immediately tell which TIA/RIOT register read
returned a different value.

The technique scales from "single-byte timing drift" (e.g. pong
$3f/$40 swap at frame 20) all the way to "spurious collision latch
cascades into wrong CPU branch into wrong-game-state" (the breakout
case, where frame 92 was 70 scanlines downstream of the actual bug).

---

## Decision tree

Start at the top; descend until you find your case.

```
Is the divergence at frame 0?
├── YES: boot-state issue (RAM seeding, reset vector, P6d random-NOOP
│        reset). Cycle/TIA threading won't help. Compare boot-burn
│        RAM directly. → bug_fix_log "boot" entries for prior examples.
│
└── NO: divergence appears at some specific frame N. Continue ↓

Is the divergence small (≤ 4 b/f, max ≤ 4)?
├── YES: probably timing drift in TIA writes (HMOVE, RES*, PF*).
│        Run `tools/jutari_xitari_ram_diff.py` to confirm frames N-1
│        is bit-exact. Look at TIA register reads around frame N.
│        Most often INTIM / collision-register / floating-bus noise
│        on an INPT read. → Phase 5 RIOT-read threading for INTIM,
│        VBLANK collision-skip for catch-up, Phase 1c collision
│        catch-up beam refinement.
│
└── NO: 6+ bytes diverge with values seemingly unrelated. Probably
        the CPU is on a DIFFERENT CODE PATH from xitari. Continue ↓

Per-bus-op bisection (THE technique that closed breakout):
1. Build / regenerate the jutari trace
2. Build / regenerate the xitari trace
3. Diff event-by-event, find the first SEMANTIC divergence
4. Targeted instrumentation to identify the source
5. Hypothesis, fix, verify
```

---

## The per-bus-op bisection technique

### When to use

- RAM diverges at frame N with 4+ bytes diverging
- Frame N-1 is bit-exact (verified with
  `tools/jutari_xitari_ram_diff.py`)
- You don't already know which TIA/RIOT read returned a different value

### Step 0 — confirm RAM is bit-exact through frame N-1

```bash
cd jaxtari
.venv/bin/python ../tools/jutari_xitari_ram_diff.py \
    --rom ../xitari/roms/breakout.bin \
    --actions ../tools/breakout_video/output/breakout_random_actions.txt \
    --rom-settings breakout \
    --max-frames 300
```

Should show:
```
first divergence: frame N
  RAM[$XX]: xitari=$.. jutari=$..
```

Save the divergent bytes. They're the BREADCRUMBS — at some point
in frame N, an instruction wrote different values to those addresses
in each port.

### Step 1 — generate per-bus-op traces

**jutari (Julia)**:
```bash
julia --project=jutari tools/cpu_tia_cycle_trace.jl \
    --rom xitari/roms/breakout.bin \
    --actions tools/breakout_video/output/breakout_random_actions.txt \
    --rom-settings breakout \
    --boot-noop 60 --boot-reset 4 \
    --max-frames N+1 \
    --out /tmp/breakout_jutari_trace.csv
```

CSV columns: `global_idx, frame, kind, scanline, scanline_cycle,
color_clock, addr, value`.
- `frame` = 1-based action index (= step number)
- `kind ∈ {peek, poke, tick, frame_boundary}`

**xitari** (requires the bus-trace extension in commit `d66b290`):
```bash
tools/trace_dump --rom xitari/roms/breakout.bin \
    --actions tools/breakout_video/output/breakout_random_actions.txt \
    --max-frames N+1 \
    --bus-trace /tmp/breakout_xitari_trace.csv \
    --bus-trace-frames N,N+1   # only emit the frames you need
```

The `--bus-trace-frames LO,HI` filter is important — full-game
traces are 64+ MB. Limiting to 1-2 frames keeps them manageable.

### Step 2 — diff the traces, find first semantic divergence

**Direct (kind, addr, value) diff** will be confused by jutari's NMOS
RMW double-writes and addressing-mode dummy peeks (xitari uses
M6502Low which doesn't emit those). Filter both:

```python
def load_filtered(path, frame):
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            if int(row['frame']) != frame: continue
            if row['kind'] in ('tick', 'frame_boundary'): continue
            a = int(row['addr'], 16) & 0x1FFF
            # Drop cart peeks (ROM bytes match by construction).
            if row['kind'] == 'peek' and (a & 0x1000):
                continue
            rows.append((row['kind'], a, int(row['value'])))
    # Drop RMW dummy pokes (consecutive same-addr pokes).
    out = []
    i = 0
    while i < len(rows):
        if (i + 1 < len(rows)
            and rows[i][0] == 'poke' and rows[i+1][0] == 'poke'
            and rows[i][1] == rows[i+1][1]):
            i += 1
            continue  # skip the OLD-value dummy poke
        out.append(rows[i])
        i += 1
    return out
```

Then compare element-by-element:

```python
n = min(len(xi), len(ju))
for i in range(n):
    if xi[i] != ju[i]:
        print(f'idx {i}: xi={xi[i]} ju={ju[i]}')
        break
```

**What divergence looks like**:

- **Same address, different value** (e.g. `peek CXBLPF $b6` vs
  `peek CXBLPF $36`) → TIA register read differs. The xitari value
  is the "correct" answer. Now investigate why jutari computes
  differently. **This was the breakout case.**
- **Same address+value but at a different scanline/sc** → cycle
  drift. Phase 5 (RIOT-read threading) or Phase 2b (cycle counter)
  may help.
- **jutari has extra peeks/pokes xitari doesn't** → NMOS dummy
  reads that xitari M6502Low doesn't model. Usually harmless to RAM
  state but can pollute floating-bus. Verify with the filter above.

### Step 3 — targeted instrumentation

Once you have a SPECIFIC TIA register + scanline that differs,
add a TEMPORARY logging hook to the jutari handler for that register.
For collision evaluation:

```julia
# In _apply_pixel_collisions! (jutari/src/tia/TIA.jl)
if bl_here && pf_here
    tia.collisions[7] |= 0x80
    # DEBUG — remove after diagnosis
    if isdefined(Main, :_BLPF_DEBUG) && Main._BLPF_DEBUG
        push!(Main._BLPF_LOG,
              (tia.frame, tia.scanline, x, tia.bl_x,
               tia.registers[W_ENABL + 1], tia.enabl_old,
               tia.registers[W_VDELBL + 1]))
    end
end
```

The pattern: a module-level `_BLPF_DEBUG::Bool` gate, a
`_BLPF_LOG::Vector{...}` collector. Drive it from a test script:

```julia
global _BLPF_DEBUG = false
global _BLPF_LOG = ...
for i in 1:N-1; env_step!(env, actions[i]); end   # advance to before divergence
_BLPF_DEBUG = true; empty!(_BLPF_LOG)
env_step!(env, actions[N])                        # step into divergent frame
_BLPF_DEBUG = false
for ev in _BLPF_LOG; println(ev); end
```

You will see EVERY (scanline, color_clock) where the collision bit
was set, with the relevant TIA state at that moment. The bug is
usually visible in the first few events.

### Step 4 — hypothesis + fix + verify

Common bug patterns (so far identified):

1. **Stale shadow latch** — a deferred register write hasn't
   activated yet when a synchronous shadow latch fires. xitari's
   M6502Low does the write immediately, so its shadow captures the
   new value. **Example**: `enabl_old` (commit `20b5de0`). **Fix
   pattern**: at shadow-latch time, scan `pending_writes` for any
   queued write to the read register whose `activation_clock ≤
   current beam_cc`, and use the latest pending value.

2. **Stale catch-up read** — `_object_pixel_sets` reads
   `tia.registers[...]` directly during a mid-instruction collision
   catch-up, ignoring pending writes that would have activated by now.
   **Possible fix pattern**: same as #1 — drain pending writes whose
   `activation_clock ≤ end_cc` before evaluating sets.

3. **Cycle-count mismatch** — bus_op count differs from
   `CYCLE_TABLE[opcode]` for some opcode, causing TIA writes to land
   at the wrong color clock. **Fix pattern**: add `pending_tick!`
   for the implicit internal cycles. Validated by
   `jutari/test/scratch_cycle_audit.jl` / `jaxtari/tests/scratch_cycle_audit.py`.

4. **RIOT-read pre-increment quirk** — xitari's `M6532::peek` uses
   `cycles() - 1` for the timer delta. **Fix pattern**: in jutari's
   `riot_peek!`, use `max(0, pending_extra_cycles - 1)` for the
   intra-instruction extra cycles (Phase 5).

5. **VBLANK collision pipeline** — xitari's
   `TIA::updateFrameScanline` does `memset; return` when `myVBLANK &
   0x02` is set, entirely skipping the per-pixel collision switch.
   jutari/jaxtari were running collisions during VBLANK. **Fix
   pattern**: early-exit the catch-up + render branches when
   `vblank_active`.

**Verification**:
- Re-run `jutari_xitari_ram_diff.py` → expect b/f to drop.
- Add a regression test in `jutari/test/runtests.jl` /
  `jaxtari/tests/test_*.py` so future agents can't silently
  regress this.

---

## Tooling reference (commit pointers)

| Tool | What it does | Commit |
|------|--------------|--------|
| `tools/cpu_tia_cycle_trace.jl` | jutari per-bus-op CSV emitter | `fb72495` |
| `tools/cycle_trace_inspect.py` | grep-like query language for the CSV | `6cdc99a` |
| `tools/jutari_xitari_ram_diff.py` | per-frame RAM diff jutari↔xitari | `0769a5e` |
| `tools/trace_dump --bus-trace` | xitari per-bus-op CSV (System.hxx hook) | `d66b290` |
| `tools/pong_3way_ram_diff.py` | 3-way diff jaxtari↔jutari↔xitari | (older) |
| `jutari/test/scratch_cycle_audit.jl` | per-opcode cycle counter validation | `2aec8c5` |
| `jaxtari/tests/scratch_cycle_audit.py` | same for jaxtari | `a8f2bd7` |

The xitari hook in `xitari/emucore/m6502/src/System.{hxx,cxx}` is
LOCAL-ONLY (xitari is gitignored). Future agents need to re-apply it
manually — see commit `d66b290`'s message for the patch shape.

---

## Examples — when this technique paid off

- **Breakout ball-doesn't-die** (commit `20b5de0`, 2026-06-03):
  RAM diverged at frame 92 by 6 bytes ($37 $5f $61 $65 $67 $6c).
  Per-bus-op diff revealed CXBLPF returning $b6 (collision SET) in
  jutari vs $36 in xitari at scanline 1, sc 15. Targeted
  `_BLPF_LOG` instrumentation pinpointed scn 51-52 with
  `enabl_old=$37` while `ENABL=$00`. Root cause: deferred ENABL
  hadn't activated when GRP1 captured the shadow.

- **Pong RAM 4→0 b/f** (commit `2aec8c5`, 2026-06-02): per-opcode
  cycle audit revealed 21 mismatches between `pending_tia_cycles`
  and `CYCLE_TABLE`. The threading then landed TIA writes at the
  correct color clock.

- **Phase 5 RIOT INTIM `cycles - 1`** (commit `7bd08a3`,
  2026-06-03): xitari's M6532.cxx:161 reads `cycles() - 1` for the
  timer delta. Mirrored in jutari.

---

## Anti-patterns to avoid

- **Don't guess at fixes** without bisecting. The breakout bug had
  several plausible-looking near-misses (Phase 1c collision
  catch-up, VBLANK collision skip, Phase 5 RIOT threading) that
  were semantically correct but didn't move the needle. Only the
  per-bus-op trace narrowed it to the actual cause.
- **Don't compare event indices directly without filtering NMOS
  dummies.** jutari emits ~2× more events than xitari for the same
  instruction stream. Index `i` in jutari is not the same bus op
  as index `i` in xitari.
- **Don't trust "screen looks the same" as evidence of equivalence.**
  Pong's pre-Phase-2b screen residual was 32 px/frame even though
  the RAM was bit-exact — the bug was downstream in the renderer.

---

## Next-target backlog

Order roughly by RAM divergence:

1. **jaxtari pong $04/$3c** — 4 b/f at frame 24+. Apply this
   methodology with a jaxtari per-bus-op trace tool (doesn't exist
   yet — would need to mirror `cpu_tia_cycle_trace.jl` in Python).
2. **seaquest frame-0** — 6 bytes diverge at boot. Different bug
   class (RAM seeding, reset vector). Compare initial RAM directly.
3. **pitfall** (19.8 b/f noop) — start here once the easy ones close.
4. **enduro** (45.0 b/f noop) — biggest gap, last to attack.
