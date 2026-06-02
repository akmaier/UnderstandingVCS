# P3i-g part 2 (third attempt) — CPU↔TIA cycle threading plan

This document is the agent-readable plan for the **deep dive** that
closes the remaining cluster of jutari/jaxtari bugs. It's written so
the work can be picked up across multiple sessions / agents without
losing context. Sequencing decision (user, 2026-06-02): **jutari
first (Phases 1→2→3), then jaxtari (Phase 4)**. Diagnostic harness
ships permanent under `tools/` for future regressions.

---

## The root-cause family

All five remaining "interesting" bugs are downstream of one missing
piece — precise CPU↔TIA sub-instruction cycle accounting:

```
  CPU-cycle ↔ TIA-color-clock threading (1-cycle drift)
                       │
       ┌───────────────┼───────────────┬──────────────────┐
       │               │               │                  │
  pong jutari    breakout      jaxtari pong         the 4-byte/frame
  3 px sites     ball-doesn't  freeze within        jutari↔xitari
  (32 px/frame)  die (both    ~100 frames           pong RAM gap
                  ports)                            (frame-20 FIRE
                                                     STA $3F vs $40)
```

Two prior attempts at P3i-g part 2 have been **reverted**:
- Commit `ed1e498` reverts my session's bus-op-count attempt.
- Another agent's attempt landed differently then was either
  superseded or rolled into `2e44bab` ("P3i-g write-cycle threading
  — fixes Breakout walls + brick artifacts"). The header bug class
  is still open per `bug_fix_log.md`.

**What went wrong both times**: counting "bus ops" undercounts cycles
for instructions whose internal cycles aren't visible as peeks/pokes
(NOPs, register-only ops, branch-not-taken). Cycle table drift built
up across instructions and broke the scanline-cycle unit tests.

**What this plan does differently**: count cycles by EXPLICIT bumps
inside `_step_inner`, and **validate against `CYCLE_TABLE` per opcode
before enabling any threading**.

---

## Phase 1 — diagnostic harness (1 day, no behavioural change)

**Deliverable**: `tools/cpu_tia_cycle_trace.py` + `tools/cpu_tia_cycle_trace.jl`.

Both scripts run a ROM + action stream through the emulator and dump
to CSV (one row per TIA peek + per TIA poke + per WSYNC release):

```
frame,scanline,scanline_cycle,color_clock,event_type,addr,value
```

`event_type ∈ {peek,poke,wsync_release}`. Run on:
- `pong_noop_10` (already bit-exact at RAM level, so this catches
  pure timing drift)
- `pong` with seed-42 action stream for frames 0..25 (covers
  the frame-20 first-FIRE divergence)
- `breakout` with seed-42 action stream for frames 0..25 (same)

A second tool `tools/cycle_trace_diff.py` cross-checks jutari output
against xitari output (xitari trace comes from a small patch to
`tools/trace_dump.cpp` that already exists per
`PXC1-x round 2+` deliverables). The first event whose
`(scanline, scanline_cycle, color_clock)` differs IS the bug entry
point.

**Acceptance**: harness produces deterministic, diff-able CSVs.
Visible output is the "first divergent event" line for each ROM.

**Tasks**: #67.

---

## Phase 2 — jutari precise cycle counter (2-3 days)

Three commits, each shippable independently:

**2a.** Add `bus.cycles_consumed_this_instruction: Int` field. Add
`_step_inner!` instrumentation that bumps it at every CPU cycle —
including the implicit cycles inside opcode handlers (e.g. NOP's
2nd cycle, JSR's internal cycles, branch-taken's extra cycle).
Reset to 0 on each instruction boundary (in `_tia_post_step!`).
NO behavioural change yet — counter is observed-but-unused.

**2b.** Validation test: a new `test/test_cycle_counter.jl` parameterised
over EVERY opcode in `OPCODES`. For each:
- Set up a CPU with that opcode at $F000.
- Call `step(state, bus)`.
- Assert `bus.cycles_consumed_this_instruction == CYCLE_TABLE[opcode]`.

If any opcode fails, fix the counter bump in 2a before proceeding.
**This is the test gate that was missing in both prior attempts.**

**2c.** Wire the threading. Add to `_bus_poke!` for TIA region writes:
```julia
cycles_to_flush = bus.cycles_consumed_this_instruction - bus.tia_advanced_this_instruction
if cycles_to_flush > 0
    tia_advance!(bus.tia, cycles_to_flush)
    bus.tia_advanced_this_instruction = bus.cycles_consumed_this_instruction
end
tia_poke!(bus.tia, addr, value)
```
Same in `_tia_post_step!` for the drain: total CPU cycles for the
instruction (= `state.cycles - pre_cycles`) minus what was already
advanced.

**Acceptance**: 2b passes 151+ opcodes (documented + undocumented).
Existing jutari test suite stays green.

**Tasks**: #68.

---

## Phase 3 — jutari verification (1 day)

**3a.** Re-run Phase 1's diagnostic harness on pong + breakout.
Compare against the `before-threading` snapshot from Phase 1.
First-divergent-event should be GONE or pushed dramatically later.

**3b.** Re-run PXC1 (RAM) for all 6 ROMs. Expect:
- pong, breakout, space_invaders: stay bit-exact (0 bytes)
- seaquest 4→? pitfall 19→? enduro 43→?

**3c.** Re-run PXC-S (screen) for all 6 ROMs. Expect:
- pong 32→near-0 (closes the 3 cosmetic bug sites)
- breakout 0→0 (already bit-exact)
- others: incremental improvements

**3d.** Re-run the breakout ball-death probe (`tools/breakout_ball_death_probe.jl`
to be added). Expect `RAM[$39]` to decrement every ~120 frames
matching xitari.

**3e.** Regenerate the jutari side of pong + breakout comparison videos.

**3f.** Update `bug_fix_log.md` with measured before/after numbers.

**Tasks**: #69. Closes the residuals on tasks #66 + the jutari
breakout ball-death.

---

## Phase 4 — jaxtari mirror (1 day)

Mechanical port of Phase 2 to jaxtari Python:

**4a.** Add `Bus` NamedTuple field `cycles_consumed_this_instruction:
int = 0` + `tia_advanced_this_instruction: int = 0`.

**4b.** Mirror the same `_step_inner` per-cycle bumps + Phase 2b
opcode-table test (`tests/test_cycle_counter.py`).

**4c.** Wire `_bus_poke` TIA-region flush + `_tia_post_step` drain.

**Acceptance**: jaxtari pong freeze closes (game progresses through
500+ frames). jaxtari↔xitari pong gap drops from 15-18 bytes/frame
to single digits.

**Tasks**: #70.

---

## Phase 5 (optional) — TIA-read side threading (1 day)

After Phase 4 lands, measure remaining residual. If non-trivial
(>3 bytes/frame), extend the flush to `_bus_peek` for TIA region —
catch up TIA before reading a collision/INPT register.

**Estimated benefit**: closes the partial-scanline-collision-read
gap that the bug_fix_log "INPT0/INPT1 read at FIRE moment"
diagnosis pointed to.

**Tasks**: #71.

---

## What "done" looks like (acceptance criteria for the whole arc)

After Phase 4 (Phase 5 optional):

| Bug | Before | Target |
|---|---|---|
| jutari pong 32 px/frame residual | 32 | <5 |
| jutari breakout ball-death lives counter | decrements 1× total | every ~120 frames |
| jaxtari pong freeze | freezes within ~100 frames | progresses 500+ frames |
| jaxtari ↔ xitari pong RAM | 15-18 bytes/frame | <5 bytes/frame |
| jutari ↔ xitari pong RAM | 4 bytes/frame | bit-exact or <2 |
| PXC1 RAM 6-ROM scoreboard | pong/breakout/SI bit-exact | unchanged or improved |
| PXC-S screen 6-ROM scoreboard | breakout 0, pong 32 | breakout 0, pong <5 |

---

## Risk register

| Risk | Mitigation |
|---|---|
| Phase 2c breaks scanline-cycle unit tests like prior attempts | Phase 2b's per-opcode CYCLE_TABLE test catches it BEFORE 2c lands |
| jutari and jaxtari have subtly different cycle accounting | Phase 4's mirror test catches it during validation |
| TIA-read threading (Phase 5) breaks collision-register tests | Phase 5 stays optional; only ship if Phase 4 leaves a measurable residual |
| `tia_advanced_this_instruction` accumulates wrong across nested calls | `_tia_post_step` resets to 0; instrumentation in Phase 1 catches drift |
| Some opcode's internal cycles aren't where I think they are | Phase 1's per-event trace makes the actual cycle-by-cycle behaviour observable, removes guesswork |

---

## Pointers for whoever picks this up

- `bug_fix_log.md` "Where we left off" section is the user-facing
  status; this doc is the technical plan that section points to.
- `ed1e498` is the previous revert commit — its diff shows the
  shape of what NOT to do.
- `2e44bab` ("P3i-g write-cycle threading") is the most recent
  surviving threading work; build on top of it rather than
  re-architecting.
- `tools/pong_3way_ram_diff.py` is the existing 3-way RAM diff
  tool; use its output to triangulate.
- xitari source: `xitari/emucore/m6502/src/M6502.cxx` for the
  reference cycle accounting; `xitari/emucore/TIA.cxx::poke`
  for the `updateFrame(clock + delay)` pattern we're matching.
