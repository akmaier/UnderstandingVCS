# Render-conformance probing plan (jutari screen vs xitari)

Status after the #98 dump-pot fix: **RAM 62/64 bit-exact**, but the new 64-ROM
**screen scoreboard** (`tools/rom_sweep/results_jutari_screen.md`,
`sweep_jutari_screen.py`, 60 frames, breakout_random_actions, correct per-game
RomSettings) shows only **29/64 pixel-exact**. So the render layer — not the
CPU/RAM — is now the frontier. RAM is bit-exact on almost all of these, so the
renderer is producing different pixels from the *same* register/RAM state.

## The unifying signature

Per-pixel probes (`/tmp/renderzoom.py <game> <N>`) of representatives show the
same pattern almost everywhere: **jutari emits a colour where xitari emits black
(0)**, confined to scanline bands — and the rows *outside* the band match
exactly (so it is NOT a global vertical shift like the old #53):

- star_gunner — rows 0-9: ju=184, xi=0  (top band)
- up_n_down   — rows 23-26 & 190-193, ~full width: ju=212, xi=0  (structural)
- asterix     — row 190 col 92: ju=24, xi=0  (single stray pixel, bottom band)

"jutari draws where xitari blanks" ⇒ the prime suspect is **TIA output-blanking
/ the vertical display window**: xitari blanks (black) scanlines that jutari
renders. Candidate mechanisms, in priority order:
1. **VBLANK output-blanking timing** — games drive VBLANK (D1) on/off mid-frame
   to blank HUD/score bands and over/under-scan; if jutari's per-scanline
   `vblank_active` gate is off by a few scanlines (poke-time threading) or
   doesn't track mid-frame toggles the way xitari's `myVBLANK`/`updateFrame`
   does, jutari paints rows xitari blanks. (jutari has frame-level VBLANK
   blanking — the gap is likely sub-frame timing / boundary scanlines.)
2. **Display-window offset** — xitari's `myStartDisplayOffset` /
   `myStopDisplayOffset` (TIA.cxx:235-237) vs jutari's `Y_START` /
   `VISIBLE_HEIGHT` capture window: if jutari captures a few scanlines above the
   top or below the bottom of xitari's window, the edge bands diverge.
3. **HMOVE-blank / object-at-edge** — single stray pixels at band edges
   (asterix 1px) can be an object/missile/ball drawn 1px into a blanked column.

## Buckets (35 non-exact games)

**A. PAL screen height (5) — a missing FEATURE, not a bug. [PARTIALLY DONE — #110]**
air_raid (250h), carnival (214h), journey_escape (230h), pooyan (220h),
surround (250h). xitari renders the PAL display height; jutari was NTSC-only
(210h) so the frames weren't even the same shape (flagged `n/a` in the
scoreboard). Fix = PAL screen-height rendering, gated on `romsettings_pal`.
✅ **Task #110** added this (per-TIA `screen_height_rows`/`scanlines_per_frame`
+ PAL colour-loss + framebuffer 244→312): **air_raid (→24px @ rows 219-223,
genuine PAL-region render residual) and surround (→224px construction-counter
non-bug) are now comparable.** REMAINING: carnival(214)/journey_escape(230)/
pooyan(220) still flag "PAL not matched" — they need their own `RomSettings`
subtypes (`romsettings_pal=true` + `romsettings_screen_height`) AND likely a
**per-game YStart** (currently `Y_START` is a fixed const 34; carnival/pooyan
use YStart=26), so they're a small follow-up, not covered by #110.
Independent of bucket B.

**B. "Draws where xitari blanks" — the VBLANK/window family (largest, shared).**
- B1 top band (rows ~0-11): star_gunner (45), hero (320), solaris (482),
  crazy_climber (1592), bowling (648).
- B2 bottom band (rows ~182-209): asterix (1), amidar (3), centipede (3),
  defender (9), name_this_game (6), atlantis (24), video_pinball (66),
  beam_rider (131), chopper_command (898).
- B3 structural / full-width fills (whole rows ju=colour, xi=0): up_n_down
  (10838 — worst), pacman (3362), private_eye (2888), frostbite (2736),
  robotank (1353), battle_zone (1112).
- B-misc: demon_attack (3 @15), jamesbond (1 @21), kangaroo (9 @2-3),
  wizard_of_wor (3 @164), ice_hockey (5 @87-103), tutankham (80 @103-167),
  berzerk (25, first-div frame 42 = action-driven).

**C. qbert — RAM bit-exact (closed in #106) but screen 7664 px @ rows 34-205.**
Almost certainly a side-effect of the #106 partial-frame / grey-frame model: the
boot→step "sliver"/grey frames have an incomplete framebuffer that renders
differently from xitari's `greyOutFrame`. Probe the framebuffer of the first
2-3 user frames specifically. Distinct from bucket B.

## Probing methodology (per game)

1. `jaxtari/.venv/bin/python /tmp/renderzoom.py <game> 60` → worst frame, the
   diverging rows (by count), and xi/ju pixel values on the worst row.
2. Classify from the values:
   - **ju=colour, xi=0** → "draws where blanked" → bucket B → step 3.
   - **both colour, differ** → wrong object/PF content → object-positioning probe.
   - **shape mismatch** → PAL (bucket A).
3. For bucket B, bus-trace the **VBLANK ($01) pokes** in the diverging band:
   `tools/trace_dump --bus-trace ... --bus-trace-frames F,F` (xitari) vs
   `tools/cpu_tia_cycle_trace.jl` (jutari), filter `addr=01`. Compare the
   scanline at which VBLANK D1 goes 1→0 / 0→1. A scanline mismatch localizes the
   blanking-boundary bug.
4. Cross-check the display window: confirm whether jutari row 0 / row 209
   correspond to the same internal scanline as xitari's
   `myStartDisplayOffset`/`myStopDisplayOffset`.

## Suggested order (highest leverage first)

1. **Probe one B1 (star_gunner) + one B3 (up_n_down) for the VBLANK timing** —
   they are the cleanest "ju=colour / xi=0" cases and likely share ONE root
   cause. A single fix to the per-scanline VBLANK gate could clear most of
   bucket B at once (the way #94/#95 cleared several games together). Gate on
   the screen scoreboard; revert if any 0px game regresses.
2. **qbert (bucket C)** — confirm it's the partial-frame framebuffer; if so,
   render the grey/sliver frame to match xitari's greyOutFrame.
3. **PAL height (bucket A)** — add PAL display height; closes 5 games + makes
   surround/air_raid screen-comparable.
4. Re-run `sweep_jutari_screen.py` after each fix as the gate (mirrors the RAM
   sweep discipline: no 0px game may regress).

## Gating

`jaxtari/.venv/bin/python tools/rom_sweep/sweep_jutari_screen.py --jobs 6
--frames 60` is the render gate. Every fix must keep the 29 pixel-exact games at
0 and improve the target — same revert-on-regression discipline as the RAM
sweep. NOTE: this is a FIRST-PASS depth (60 frames); some render bugs (e.g. the
#98 pong ball blip at f460) only appear later — run the diverging games deeper
(`--frames 500`, single game) once their band is closed.

---

## STATUS UPDATE after #109 (VBLANK black-fill) — screen 29 → 37/64

**Landed:** #109 — `tia_advance!` now writes a BLACK row during VBLANK (was
skipping → stale content). This was the dominant shared cause of bucket B's
"draws-where-blanked": cleared star_gunner, hero, crazy_climber, frostbite,
beam_rider, chopper_command, video_pinball; slashed robotank 1353→241, solaris
482→2. **No regressions** (framebuffer-only change; RAM sweep + Pkg.test green).

**Refined remaining (22 non-PAL) — these are DIVERSE per-game issues, NOT one
shared cause** (post-#109 probes):

- **Boot-set background bands** — up_n_down (10838): rows 23-26 & 190-193 are
  full-width `ju=212 / xi=0`, but frame 1 writes NO VBLANK/COLUBK — the bg is set
  during BOOT and persists. So jutari's boot-end background/PF state differs from
  xitari's at those scanlines (a TIA-register, not RAM, so invisible to the RAM
  sweep). Probe the boot COLUBK/PF/CTRLPF writes and the last value per scanline.
- **Mid-visible COLUBK/PF band** — pacman (3362): rows 165-182 `ju=0 / xi=132` —
  mid-visible (VBLANK is off there in both; jutari VBLANK toggles at sl 33/258
  ≈ xitari's, modulo the trace's per-frame-reset attribution). So this is a
  background/playfield colour or priority divergence at the bottom-middle, not
  VBLANK. battle_zone (1112, cols 0-7) similar (a left-edge object/PF band).
- **Mid-screen HUD-object bottom band** — asterix (1 @190), centipede (3 @193),
  amidar (3 @182), defender (9 @183), name_this_game (6 @188), atlantis (24 @186),
  demon_attack (3 @15), jamesbond (1 @21), wizard_of_wor (3 @164), kangaroo (8 @3),
  bowling (8 @4), solaris (2 @11), ice_hockey (5 @87-103), tutankham (80),
  berzerk (25, action-driven f42), elevator_action (24), ms_pacman (232): small,
  localized, mid-visible — per-game sprite/missile/ball/PF positioning or
  colour edges (object-level, not a blanking issue). Probe each with renderzoom
  + an object-positioning bus-trace (which object set is enabled at that x).
- **qbert (7664)** — RAM bit-exact (#106) but screen-divergent: the partial-frame
  / grey-frame framebuffer. Probe the first 2-3 user frames' framebuffer vs
  xitari greyOutFrame.
- **PAL height (5)** — air_raid/carnival/journey_escape/pooyan/surround: add PAL
  display-height rendering (jutari is NTSC 210h).

**Conclusion:** the single high-leverage render fix (VBLANK black-fill) is done.
The rest is a per-game long tail (object positioning, boot-bg, partial-frame,
PAL) — each needs its own probe + a narrowly-scoped, sweep-gated fix. Recommended
to take them one at a time (renderzoom → classify → fix → re-run
`sweep_jutari_screen.py`), banking each, since they don't share a root cause.
