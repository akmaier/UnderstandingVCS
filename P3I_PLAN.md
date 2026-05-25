# P3i implementation plan — beam-accurate TIA rendering

**Goal**: rewrite `render_scanline` so the TIA produces pixels color-clock-by-color-clock, applying register writes at the exact color-clock they happen, instead of flattening to one-scanline-at-end-of-scanline using the latest register state. Matches xitari's `TIA::updateFrame(clock)` and the real 1A05 chip.

This document is the implementation plan the user asked for; the work itself is **not yet started**. The plan is broken into self-contained sub-phases that can land independently so progress is visible per-commit and the PXC1-x / artifact deltas can be measured per-sub-phase.

---

## 1. Why P3i matters

Three downstream items hinge on beam-accuracy and cannot be closed without it:

| Problem | Where it shows up today | Beam-accuracy cure |
|---|---|---|
| Pong's remaining 9-byte PXC1 gap | RAM[$25–$28] off by exactly bit 3; RAM[$33], $3b, $3c off by exactly $48 (= bit 6 + bit 3). Round-4 traced these to TIA-collision-register reads at specific cycle phases — the read happens MID-scanline; we report end-of-scanline collision state, xitari reports the cycle-N-1 state. | Per-pixel collision-latch evaluation, sampled at the read's color clock. |
| Breakout's "split brick stripe" (#55) | The brick rainbow is solid in xitari, split-with-vertical-bars in jaxtari/jutari. The game writes new PF0/PF1/PF2/COLUPF mid-scanline (16 bricks per row × 6 rows, the pattern is encoded in mid-line PF stomping). Our renderer uses end-of-scanline PF, so it draws ONE brick pattern instead of the time-multiplexed pattern. | Per-pixel PF evaluation using the PF register state at each color clock. |
| RES* (sprite re-position) timing precision | We already approximate this via `_resp_position(scanline_cycle)`, but the approximation rounds to the nearest CPU cycle and misses the 3-color-clock sub-cycle component. | Per-pixel RES* application at the exact color clock + xitari's `ourPokeDelayTable`. |

So P3i unblocks PXC1-x's last bytes for Pong AND the visible Breakout artifacts AND general accuracy for any ROM with mid-scanline tricks (most published games).

---

## 2. Current state — what `render_scanline` does today

`jaxtari/jaxtari/tia/system.py` (mirrored in `jutari/src/tia/TIA.jl`):

* `tia_advance(tia, cpu_cycles)` advances the scanline counter, accumulates CPU cycles into `scanline_cycle`, and on scanline boundary calls `render_scanline(tia)` to fill one row of the framebuffer.
* `render_scanline` reads the **current** value of every TIA register (PF0/PF1/PF2/CTRLPF/GRP0/GRP1/ENAM*/ENABL/COLU*/REFP*/NUSIZ*/HMOVE-set positions) and emits 160 pixels in one shot.
* `tia_poke(tia, addr, value)` updates registers immediately. Writes between scanlines naturally affect the next scanline; writes WITHIN a scanline land in registers but are not surfaced until end-of-scanline rendering — they overwrite each other silently.

So **a write at color-clock 50 of scanline N is indistinguishable from a write at color-clock 150 of the same scanline.** Both end up in the register; the rendered scanline uses whichever was last.

---

## 3. Target architecture (post-P3i)

Beam-accuracy means: the TIA processes color clocks one at a time, and at each color clock it evaluates `pixel = composite(register_state_AT_this_color_clock, sprite_positions, …)`. Two design choices:

### Design A — eager per-color-clock loop (xitari's approach)

`tia_advance(tia, cpu_cycles)` runs an internal loop: for each color clock (`= cpu_cycles × 3`), compute the pixel value at the current beam position and write it to the framebuffer. `tia_poke` just updates the register state in place; the next color clock automatically sees the new value.

Maps directly to xitari's `TIA::updateFrame(clock)`. Easy to read; straightforward to test (read framebuffer pixel at any (x, y) and compare to xitari). Performance: ~3 inner-loop iterations per CPU cycle, ~60K iterations per frame, ~3.6M per second — feasible in Python eager mode (the existing scanline loop already does ~228 per scanline = 60K per frame).

### Design B — deferred per-poke timeline + replay-at-end-of-scanline

`tia_poke` appends `(cycle_in_scanline, reg, value)` to a per-scanline event list. `render_scanline` walks the event list, applying writes at their cycle positions and emitting pixels between events.

More functional (matches jaxtari's NamedTuple style), but adds a mutable list to the TIA state. Slightly slower than A in practice (event-list overhead per poke). Easier to JIT later (events form a structured PyTree).

### Recommendation: **Design A**

Closer to xitari (easier to cross-check), no new mutable state, simpler test surface. The per-cycle loop dominates cost only when JIT'd; eager Python is no slower than today's scanline render.

---

## 4. Sub-phases (incremental implementation)

Each sub-phase ships independently with its own tests + PXC1/PXC2/STATUS update. Order is chosen so each sub-phase has measurable closure on existing artifacts.

### P3i-a: Color-clock loop scaffolding (no behavioural change yet)

* Add `color_clock` field to `TIAState`. Update `tia_advance` to advance it by `cpu_cycles × 3` per call, wrapping at 228 (one scanline) and triggering the existing `render_scanline` at the same point as today.
* Add a new entry point `render_pixel(tia, color_clock_in_scanline) -> uint8` that returns the pixel value for a given beam X position using the CURRENT register state. This is the future per-color-clock kernel — at this stage it's only used for tests, not the framebuffer.
* `render_pixel(tia, c)` must equal `render_scanline(tia)[c - HBLANK_CLOCKS]` for every `c` in 68..227. Test this exhaustively.
* **Closure**: zero — this is scaffolding. PXC1-x unchanged. But adds the kernel everything else builds on.
* **Files**: `jaxtari/tia/system.py` (+ jutari mirror), 2 new tests per port.
* **Size**: ~150 lines per port, 1 day each.

### P3i-b: Switch the framebuffer write to per-color-clock

* Change `tia_advance` to call `render_pixel` in a loop over color clocks, writing to the framebuffer at `framebuffer[scanline, color_clock - HBLANK_CLOCKS]` for each visible pixel (68..227 → 0..159).
* Delete the end-of-scanline `render_scanline` framebuffer write. (Keep `render_scanline` itself for the tests that already use it.)
* No new register-write timing yet — `tia_poke` still updates registers immediately. So this sub-phase has the same output as today (every color clock of a scanline still uses the same register state — the latest one when the scanline ends, because nothing has changed when writes apply).
* **Closure**: zero, but the architecture is now ready for P3i-c.
* **Files**: `jaxtari/tia/system.py` (+ jutari mirror), ~10 framebuffer-direct-index tests may need shape touch-ups.
* **Size**: ~100 lines, 0.5 day.

### P3i-c: Per-poke write timing (real beam-accuracy starts here)

* `tia_poke(tia, addr, value)` now first computes `apply_at_clock = current_color_clock + ourPokeDelayTable[addr]` (port xitari's table; most writes apply immediately, some take 1-4 color clocks).
* If `apply_at_clock <= current_color_clock` for the rendering loop, the write applies before the next color-clock pixel render. If `apply_at_clock > current_color_clock`, store the pending write on the TIA state (`pending_writes: tuple[(clock, reg, value), ...]`) and apply when the loop reaches `apply_at_clock`.
* The `tia_advance` color-clock loop now: (1) drain any pending writes whose clock has been reached, (2) call `render_pixel` for the current visible pixel, (3) increment color_clock.
* **Closure expected**: the brick-stripe artifact (#55) should collapse here — Breakout's mid-scanline PF stomping now affects the rendered pixels.
* **Files**: `jaxtari/tia/system.py` (+ jutari mirror), `jaxtari/cpu/m6502.py` may need an interface tweak to surface the current TIA color clock to `tia_poke` (probably via thread-state). Tests: 5+ for poke-delay table, 1+ for the brick stripe, 1+ for mid-scanline COLUPF rainbow.
* **Size**: ~200 lines, 1 day.

### P3i-d: Per-pixel collision latch evaluation

* `_detect_collisions` currently fires once per scanline at end-of-scanline. Move it into `render_pixel` — compute collision pairs from the CURRENT visible-objects mask at this color clock, then OR-merge into the collision latches.
* TIA collision-register READS (CXM0P etc.) already go through `tia_peek` which returns the latch as-is; nothing changes on the read side. But the latch state now reflects collisions at every visible pixel, not just the final state.
* **Closure expected**: Pong's RAM[$25-$28] (bit-3 deltas) and RAM[$33], $3b, $3c ($48 deltas) — these are reading collisions mid-scanline. After this sub-phase those should close.
* **Files**: `jaxtari/tia/system.py` (+ jutari mirror). 5+ new collision-timing tests.
* **Size**: ~150 lines, 0.5 day.

### P3i-e: Exact RES* sub-cycle timing

* `_resp_position(cpu_cycle)` currently returns an X position based on the CPU cycle the write hit. The real chip's RESP* registers latch the position from the CURRENT color clock, which has 3× the resolution of a CPU cycle.
* Replace the lookup-based `_resp_position` with `current_color_clock - HBLANK_CLOCKS` (clamped) — straightforward, but the current TIA tests pin specific RESP* positions, so the test fixtures need updating against xitari ground-truth.
* **Closure expected**: small per-frame sprite-position deltas (sub-pixel) close. Visible improvement in PXC2 conformance on ROMs that use precise sprite positioning (Asteroids, Q*Bert later).
* **Files**: `jaxtari/tia/system.py` (+ jutari mirror), ~5 RES* test fixtures.
* **Size**: ~80 lines, 0.5 day.

### P3i-f: HMOVE per-color-clock application

* HMOVE currently applies all 5 HMP*/HMM*/HMBL deltas to sprite positions at the time of the HMOVE write. xitari applies them gradually over the 8 color clocks following the write.
* Implement xitari's HMOVE state machine: HMOVE write at clock C marks 8 "extra" color clocks; during those, each sprite drifts by its programmed delta (1 pixel per "extra" clock if its HM value is nonzero).
* **Closure expected**: minor pixel-position corrections; closes the cases where games abuse HMOVE for sub-cycle positioning (the "HMOVE comb").
* **Files**: `jaxtari/tia/system.py` (+ jutari mirror), 3+ HMOVE-comb tests.
* **Size**: ~120 lines, 0.5 day.

### Total estimate

~6 sub-phases × 0.5–1 day each = **3–5 working days**, plus ~1 day for the test migration (the ~20 framebuffer-direct-index tests in `test_tia_*.py` will need pixel-position adjustments after P3i-c lands). Realistic **end-to-end: 1 week**.

---

## 5. SOFT path

P3i applies only to the HARD path. The SOFT (differentiable) TIA already uses a much simpler model (`soft_render_scanline` broadcasts one synthesized scanline) that's not affected by mid-scanline timing — SOFT users care about gradient flow, not pixel-level beam accuracy. **No SOFT-side P3i work needed.**

The differentiable gradient through the new HARD `render_pixel` would be relevant only if someone wanted to differentiate through a HARD render — they don't (the soft path is the official differentiable entry point). So we can keep `render_pixel` as plain Python / pure `jnp` without `jit`-friendliness constraints initially. If JIT becomes a need later (e.g. for end-to-end JAX `vmap` over frames), Design B's event-list approach would be a cleaner JIT target.

---

## 6. Risk + open questions

* **Test migration cost**: the ~20 `framebuffer[N, M]` tests in `test_tia_playfield.py`, `test_tia_players.py`, `test_tia_missiles_ball.py`, `test_tia_vsync_vblank.py` may need re-pinned pixel positions when sub-cycle timing changes. Plan a separate "P3i test fixture refresh" commit after P3i-c lands.
* **xitari's `ourPokeDelayTable`** is the source of truth for write delays — porting it is mechanical but tedious (60 entries with idiosyncratic exceptions, e.g. CTRLPF takes 2 color clocks, GRP0 takes 1, RESP0 is instantaneous).
* **Performance regression**: the per-color-clock loop multiplies the inner work by 3. In Python eager mode this adds ~30% wallclock; under JIT it should be neutral or faster (one trace per frame instead of per-scanline). We have `tools/bench_soft_run.py` as a baseline — re-run after each sub-phase.
* **jutari parity**: every sub-phase mirrors directly into `jutari/src/tia/TIA.jl`. The mutable Julia model is actually simpler for per-color-clock work (no NamedTuple replacement overhead). Expect the Julia LOC to be ~70% of the Python LOC.

---

## 7. Definition of done

P3i ships as a single chain of 6 commits (one per sub-phase), each independently green on the existing test suite + adding 3–7 new tests of its own.

End-state acceptance:

1. **PXC1**: pong_noop_10 closes to ≤2 bytes (from 9). Bytes that close cleanly are reflected in the divergence assertion.
2. **PXC2**: re-run all 6 ROMs; record new conformance counts. Expectation: Breakout 3 → 0 (paddle game with mid-scanline tricks); Pong 9 → ≤2; the joystick games (space invaders, pitfall, seaquest, enduro) marginally improve via the collision-timing fix.
3. **#55 (brick-stripe artifact)**: visually disappears in the regenerated comparison videos.
4. **Existing TIA tests**: all pass (re-pinned fixtures where needed).
5. **Both ports mirror**: jaxtari and jutari produce bit-equal framebuffers (PXC2 invariant holds for every frame, not just RAM).
6. **STATUS.md**: P3i entry moves from "deferred" to "✅ shipped"; P3 (HARD TIA) becomes "complete to xitari-bit-exact level."

---

## 8. What to ship FIRST when this work starts

The smallest single commit that buys real visible progress: **P3i-a + P3i-b together** (scaffolding + framebuffer-write rewire). That's ~250 lines per port, half a day, zero behavioural change, and unlocks every subsequent sub-phase as a pure-incremental commit. Anyone picking this up should start there.
