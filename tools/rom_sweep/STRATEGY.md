# Remaining-ROM conformance evaluation — strategy

**Goal (for the AAAI paper's conformance results):** extend the bit-exact / pixel-exact
conformance paradigm from the 6 deeply-validated games to the **full set of 64
xitari-supported ALE games**, so the paper can report how faithfully jutari/jaxtari
reproduce the reference across the standard Atari benchmark — *with a known ground truth.*

## Paradigm (same as PXC1/PXC2/PXC-S)
For each game, both emulators run the **same ROM bytes** from the **standard ALE boot**
(60 NOOP + 4 RESET) and we compare against the xitari reference:
1. **RAM bit-exactness** — per-frame 128 B RIOT RAM identical to `xitari/tools/trace_dump`
   (NOOP action stream). Primary, fast metric. Tool: `tools/jutari_xitari_ram_diff.py`.
2. **Screen pixel-exactness** — per-frame 210×160 framebuffer identical (PXC-S style).
3. **Videos** — xitari | port | diff side-by-side, for visual inspection.

xitari selects the game purely by ROM **filename** (`buildRomRLWrapper`: basename ==
`RomSettings::rom()`), so `tools/rom_sweep/resolve_roms.py` matches each of the 64
canonical names to a NTSC ROM in the collection and stages it as
`tools/rom_sweep/roms/<game>.bin`. Because both emulators run identical bytes, an
imperfect title match never invalidates the conformance number — it only affects the
per-game label + which xitari RomSettings auto-applies.

## Sequencing (jutari-first, non-blocking)
The slow jaxtari path (~205× jutari) must never block a jutari deliverable, so:

- **Phase 1 — jutari RAM sweep** (`sweep_jutari_ram.py`, RUNNING in background):
  all 64 ROMs, NOOP 30 frames, → `results_jutari_ram.md` (rewritten per ROM).
- **Phase 2 — jutari screen sweep**: per-ROM 210×160 framebuffer diff vs an xitari
  screen reference (extend the PXC-S dumper to the 64 ROMs).
- **Phase 3 — videos (jutari)**: side-by-side comparison videos for a representative
  subset (bit-exact games + a few divergent ones), via the existing
  `tools/breakout_video/render_breakout_compare.py --rom … --skip-jaxtari`.
- **Phase 4 — jaxtari mirror (background, after jutari)**: RAM (PXC2-style) + screen +
  videos for the same 64, scheduled so it trails the jutari phases.

## Interpreting divergence (settings vs. emulation)
For the 4 games with real jutari RomSettings (breakout/pong/pitfall/enduro) both sides
use matching settings → divergence is pure emulation. For the rest, jutari runs
`generic` (no game-specific starting actions) while xitari auto-applies the game's
RomSettings, so some divergence is **starting-action mismatch, not an emulation bug**.
The results table flags `settings = real/generic`; triage divergent generic games by
checking whether frame-0 (boot-only, pre-starting-action) RAM already matches. A clean
emulation story for the paper = "bit-exact at frame 0 across (almost) all 64; residual
later-frame divergence localizes to RomSettings gaps + the known sub-cycle TIA items."

## Status / artifacts
- `rom_names.txt` — 64 canonical `rom()` names.
- `resolve_roms.py` → `manifest.txt` (game → resolved file; 64/64 staged).
- `roms/<game>.bin` — staged ROMs (NTSC-preferred).
- `sweep_jutari_ram.py` → `results_jutari_ram.md` (Phase 1, running).
- TODO: `sweep_jutari_screen.py`, video subset, jaxtari mirror (Phases 2–4).
