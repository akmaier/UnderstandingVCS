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

**A. Per-game display window (height + YStart) — a missing FEATURE, not a bug. [CLOSED — #110 + follow-up]**
air_raid (250h), carnival (214h), journey_escape (230h), pooyan (220h),
surround (250h). xitari renders each ROM's `Display.Height`/`Display.YStart`;
jutari was fixed at 210h/YStart=34 so the frames weren't even the same shape
(flagged `n/a`). ✅ **Task #110** added the PAL path (per-TIA
`screen_height_rows`/`scanlines_per_frame` + colour-loss + framebuffer 244→312):
air_raid (→24px @ rows 219-223, PAL-region render residual) + surround (→224px
construction-counter non-bug). ✅ **#110 follow-up** added a per-game **YStart**
(`romsettings_y_start` + per-TIA `y_start_row`, mirror of
`myClockStartDisplay=…+228*myYStart`) and the explicit-height NTSC subtypes
carnival(YStart26/H214)/pooyan(YStart26/H220) + journey_escape(H230). **All 64
games are now screen-comparable** (zero "PAL not matched"). carnival 4px / pooyan
1px are near-exact; journey_escape 325px is a STRUCTURAL render delta (object
X-position, not height/colour-loss → moved to the render long-tail, NOT bucket A).
air_raid 24px likewise a residual render delta. Bucket A (the display-window
feature) is CLOSED; its leftover px are ordinary per-game render deltas.
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

---

## STATUS UPDATE after #110 (PAL + per-game YStart) — deep-dive of the structural divergers

Bucket A (display window) CLOSED — all 64 games screen-comparable, 37/64 px-exact.
Then deep-dived the biggest remaining "structural" divergers (the B3 set). KEY
META-FINDING: **they do NOT share a root cause** — the "structural" label was
misleading. Each was confirmed a GENUINE render bug (re-ran each under an
all-NOOP stream, where RAM is *proven* bit-exact — every divergence persisted
identically, frame 1, so none is an action-driven state desync). Per-game:

- **bowling (8) + kangaroo (8) [CONFIRMED, precise]** — NOT a VBLANK/colour bug.
  The 8px are the LEFT EDGE (cols 0-7 = the HMOVE-blank window) of the FIRST
  VISIBLE scanline. Instrumented render dump: jutari renders that scanline with
  `vblank_active=false` (correct, visible) and the right COLUBK, but
  `hmove_blank_pending=true` → it blanks cols 0-7. xitari does not. ROOT CAUSE:
  jutari's VBLANK render branch deliberately does NOT consume/clear
  `hmove_blank_pending` (TIA.jl ~958, added in #83 so pong's row-0 comb survives
  the last VBLANK line), so an HMOVE strobe several scanlines earlier is carried
  through ALL VBLANK scanlines into the first visible row. xitari clears the comb
  after exactly ONE scanline render (VBLANK lines included). FIX (delicate, in
  tension with #83/pong): clear the comb per VBLANK scanline so it lives exactly
  one line — but verify pong's row-0 comb (frame-top) still renders. Must gate on
  the full screen+RAM sweep; pong/breakout are the regression risk.
- **pacman (3362)** — TWO bands: (a) top rows 0-1 (`xi=0/ju=132`, jutari draws
  where xitari blanks — likely the same first-visible-scanline family) and (b)
  bottom rows 165-182 (the score/HUD region, a colour/PF/object delta). The maze
  (rows 60-120) is PIXEL-EXACT. So pacman is NOT a global shift; it's top-edge +
  bottom-HUD, two separate localized deltas.
- **up_n_down (10838 — worst)** — full-width colour BANDS at rows 23-26 & 190-193
  where xitari shows a HUE GRADIENT (`0x14,0x24,0x34,0x44,0x54` — high-nibble
  ramp = a rainbow done with rapid mid-scanline COLUBK/COLUP writes) and jutari
  shows flat `0x12/0xD4`. COLU* ARE deferred per-color-clock (TIA.jl:607), so the
  bug is in the activation-clock TIMING of those mid-scanline colour writes (the
  rainbow kernel writes COLU every few cycles; a small activation-delay error
  smears the colours). Hardest of the set — a per-color-clock COLU activation
  audit.
- **battle_zone (1112) [FIXED — #111]** — EVERY diff was cols 0-7 (the 8px HMOVE
  comb window) on EVERY visible scanline: jutari armed the comb on every HMOVE
  strobe (battle_zone strobes every line at cc 222), but battle_zone's
  `Emulation.HmoveBlanks` property is "NO" so xitari NEVER arms it. jutari was
  missing the `myAllowHMOVEBlanks` property gate. Added it → **battle_zone 1112→0,
  ms_pacman 232→0** (the only two "NO" games in the set). NOT a missing-object bug.
- **qbert (7664)** — unchanged: RAM bit-exact (#106), the partial-frame/grey-frame
  framebuffer side-effect (bucket C).

RECOMMENDATION: these are 4–5 INDEPENDENT, intricate per-color-clock fixes, each
touching the hot render path with real regression risk to the 37 px-exact + 62
RAM-exact games. Tackle ONE per focused session, gating on the full sweep:
bowling/kangaroo (HMOVE-comb-carry, smallest + most precisely understood, but
pong-risk) → battle_zone (missing-object) → pacman bottom-HUD → up_n_down
(rainbow-kernel COLU timing, hardest). NOT a single shared fix.

---

## STATUS UPDATE after #111/#112 — battle_zone DONE; HMOVE-comb-carry BLOCKED

- ✅ **#111** battle_zone 1112→0 + ms_pacman 232→0 via the `HmoveBlanks=NO`
  property gate (`myAllowHMOVEBlanks`). NOT a missing-object bug. Screen 37→39/64.
- 🔬 **#112** the **beam_sc lead** is battle_zone-only (the `WSYNC;STA HMOVE`
  line-tail idiom: jutari's beam wraps to the next line's cc 0, x=0→comb;
  xitari x=74→no comb). Already moot (fixed by #111's property gate); other
  sl_cyc-73/74 strobers (kangaroo/centipede/asterix) are NOT post-WSYNC so
  jutari handles them right. Foundational WSYNC threading → NOT touched.
- ✅ **bowling/kangaroo (8 px) — SOLVED (#112 round 2).** The real xitari
  mechanism: `updateFrame` returns early for `clock < myClockStartDisplay`
  (TIA.cxx:1708) — it never clears the comb PRE-display (scanline < YStart) — then
  consumes `myHMOVEBlankEnabled` on the FIRST DISPLAY scanline (>= YStart), even
  if VBLANK (invisible). So the comb shows only if the first display scanline is
  visible: pong (VBLANK off < YStart) → row-0 comb; bowling (VBLANK off > YStart)
  → comb consumed invisibly, no comb. FIX: consume the comb in jutari's VBLANK
  branch too, GATED on `scanline >= y_start_row` (the carry from pre-display is
  preserved). bowling 8→0, kangaroo 8→0, berzerk 25→21, elevator_action 24→16;
  pong/pacman unaffected. Screen 39→41/64. (#112 round 1 cleared unconditionally
  incl. pre-display → killed pong's carry → reverted; the gate is the fix.)
- **pacman top rows 0-1** = a SEPARATE "draws-where-blanked" bug (jutari draws
  132 where xitari shows black), NOT the HMOVE comb — still open (part of the
  3362).

---

## STATUS UPDATE after #113 — per-game YStart fixed the "structural" divergers

The big "structural" divergers were a missing per-game `Display.YStart` (vertical
offset vs xitari), NOT sub-cycle render bugs. A full scan of all 64 ROMs' YStart
found up_n_down(30)/pacman(33)/qbert(40) defaulting to 34. Adding the overrides:
- ✅ **pacman 3362→0** (fully fixed — incl. the "top rows 0-1 draws-where-blanked"
  above, which was just the 1-row offset).
- ✅ **up_n_down 10838→221** (98% was the 4-row offset; 221 px/frame genuine
  render residual remains — sprite/PF racing detail, deferred).
- ✅ **qbert total 345224→7664** (steady state EXACT; only frame 2 left = the #106
  grey/partial boot frame vs xitari `greyOutFrame` — a known single-frame artifact).
Screen 41→**42/64**. **LESSON: check per-game `Display.YStart` BEFORE treating a
pervasive "shifted/structural" divergence as a sub-cycle render bug.**

**qbert (7664 @ frame 2) — DIAGNOSED, deferred (#114).** RAM bit-exact + screen-
exact on every frame EXCEPT counted frame 2: a boot→game frame-slice artifact.
jutari's #106 grey-frame budget makes counted-frame-2 a 235-instr sliver (renders
black) where xitari's frame 2 spans the full board; same RAM, boundary lands one
frame off. Fixing means touching the #106 partial-frame slicing (what makes qbert
RAM bit-exact) — deferred for a focused future pass, not worth the risk for one
cosmetic boot frame.

Remaining render long-tail (22 non-exact): journey_escape 325 (object X-position,
structural), robotank 241, surround 224 (construction-counter NON-BUG), up_n_down
221 (sprite/PF racing), qbert 7664 @ frame 2 (#106 grey frame), tutankham 80,
air_raid 24 / atlantis 24 / elevator_action 16 (PAL-region / bottom-band), berzerk
21, defender 9, ice_hockey 5, carnival 4, amidar 3 / centipede 3 / demon_attack 3
/ wizard_of_wor 3, solaris 2, asterix 1 / jamesbond 1 / pooyan 1.
