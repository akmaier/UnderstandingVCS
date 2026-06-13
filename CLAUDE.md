# CLAUDE.md — start here

**Before doing anything in this repo, read [README.md](README.md) in full.**
It is the project's entry point: what UnderstandingVCS is (two differentiable
Atari 2600 VCS ports — `jaxtari` (JAX/Python) and `jutari` (Julia) — validated
bit-for-bit against the `xitari` C++ reference), the current status, the
conformance scoreboards (PXC1 RAM, PXC2 cross-check, PXC-S screen), and the
hand-off notes for where work is picked up.

Then, **for any emulation / conformance / rendering work, read
[bug_fix_log.md](bug_fix_log.md)** — the running history of bugs hunted,
patches landed, and dead-ends ruled out (newest on top). **Append to it**
whenever you fix a bug, rule out a hypothesis, or discover a gotcha, so the
next instance inherits the context. It also carries the live conformance
numbers and the active investigation plan.

Supporting docs: [STATUS.md](STATUS.md) (per-phase commit/test/deferral
ledger), [PORTING_PLAN.md](PORTING_PLAN.md) (design rationale + phase plan),
[BUG_BISECTION_METHODOLOGY.md](BUG_BISECTION_METHODOLOGY.md) (the per-bus-op
trace technique for chasing divergences).

## Operating rules (do not skip)

- **Work on `main`.** This project reviews progress on `main`; never park work
  on a side branch the user can't see. Commit and push after each meaningful
  step (the commit history is the project's developer log).
- **Commit messages** include the triggering user prompt, the model used, and a
  concise what/why. Use a HEREDOC for formatting. End with the
  `Co-Authored-By:` trailer.
- **Verify, don't claim.** Report conformance numbers you actually measured;
  if tests fail, say so with the output.

## Conformance test gates

```bash
# jutari (~20 s)
cd jutari && julia --project=. -e 'using Pkg; Pkg.test()'

# jaxtari full suite (parallel; long)
cd jaxtari && .venv/bin/python -m pytest -q

# PXC-S screen conformance (xitari↔jutari fixtures + xitari↔jaxtari live, ~23 min)
jaxtari/.venv/bin/pytest jaxtari/tests/test_screen_conformance.py
```

## Hard-won methodology (read before chasing a render/screen divergence)

1. **Check state-parity FIRST.** Run `tools/jutari_xitari_ram_diff.py` with the
   exact action stream. RAM bit-exact ⇒ pure render bug. RAM divergent ⇒ a
   CPU/RIOT/boot bug whose screen diff is just downstream — don't chase it as a
   renderer bug.
2. **Check harness parity** — same ROM **RomSettings**, boot, and action stream
   on both sides. (The "pitfall doesn't jump" bug was a *tooling* settings
   mismatch — the screen tools rendered jutari/jaxtari with `GenericRomSettings`,
   omitting Pitfall's `getStartingActions`, while xitari used the real settings.
   The emulator was correct. See bug_fix_log task #95.) The settings maps in
   `tools/jutari_screen_dump.jl`, `tools/jutari_trace_dump.jl`,
   `tools/breakout_video/dump_jutari_frames.jl`, and
   `jaxtari/tests/test_screen_conformance.py` MUST agree.
3. **Never gate debug on `tia.frame`** — it is the VSYNC frame counter (~+65
   from boot), NOT the env-step / video-frame index. Gate on a flag set around
   the exact `env_step`, or on the env-step index.
4. **Bus-trace is the scalpel:** `tools/trace_dump --bus-trace PATH
   --bus-trace-frames LO,HI` (xitari) vs jutari's `Bus.trace_enable!()` /
   `trace_take!()`, aligned by write-sequence index. Identical writes + different
   screen ⇒ render-timing; differing writes ⇒ CPU/state. Note jutari bus ops land
   ~3 color-clocks (1 CPU cycle) before xitari's — expected.
