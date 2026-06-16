# Faithful object-render port (xitari mask model → jutari)

Goal (per README "Conformance philosophy"): make jutari's TIA render the
players/missiles/ball **mechanistically like xitari** — port xitari's actual
per-color-clock mask + reset-when + skip-first model, not a set-based replica.
Breaking superficial fixes / temporary screen-or-RAM score drops are acceptable if
the runtime logic becomes faithful. Gate every step on the full RAM+screen sweep.

This is the deep fix for the G2 cluster (berzerk, robotank, up_n_down, carnival,
defender) + much of the player/missile positioning, and resolves task #93
(skip-first-copy).

## xitari object-render model (reverse-engineered from emucore/TIA.cxx)

### Player draw (updateFrame compositor, TIA.cxx:1673-1693)
Per visible color clock `hpos`: `enabled |= myP0Bit if (myCurrentGRP0 &
myCurrentP0Mask[hpos])`. `myCurrentGRP0` is the VDELP-resolved GRP (myGRP0 or
myDGRP0). `myCurrentP0Mask = &ourPlayerMaskTable[myPOSP0 & 3][skipFirst][NUSIZ0 &
7][160 - (myPOSP0 & 0xFC)]`, read at `[hpos]`.

### ourPlayerMaskTable[align(4)][skipFirst(2)][mode(8)][512]  (computePlayerMaskTable)
- `align = POS & 3`; table for align 0 built then shifted: `table[align][...][x] =
  table[0][...][(x+320-align)%320]`.
- `mode = NUSIZ & 7`. Copies at base offsets per mode: 0:{0}; 1:{0,16}; 2:{0,32};
  3:{0,16,32}; 4:{0,64}; 6:{0,32,64}; each copy is 8px, mask value `0x80>>n` (the
  GRP bit at that pixel). mode 5 (double): single copy, 16px wide, `0x80>>((x-1)/2)`,
  x in (0,16] — **one-pixel delayed** (uses `>` not `>=`). mode 7 (quad): single
  copy 32px, `0x80>>((x-1)/4)`, x in (0,32] — also one-pixel delayed.
- `skipFirst` (the [enable] index): when 1, the FIRST copy (x in [0,8)) is omitted
  (the `enable==0` guard); other copies always drawn. So skipFirst=1 ⇒ draw all
  copies EXCEPT the first.

### skipFirst state machine (when myCurrentP0Mask gets which [enable] index)
- **0 (draw all)**: init; NUSIZ0/1 poke (case 0x04/0x05, 2097/2113); HMOVE
  completeMotion (2749/2751); **END OF EVERY SCANLINE** (1799-1802, the dominant
  reset — comment: "reset at the end of the scanline"); RESP when reset-when==-1.
- **1 (skip first)**: ONLY RESP0/RESP1 (case 0x10/0x11) when reset-when ∈ {0, 1}.
⇒ skipFirst is **per-scanline transient**: starts each scanline at 0, may be set to
  1 by a mid-line RESP, auto-resets to 0 at scanline end. NUSIZ/HMOVE also clear it.

### RESP0/RESP1 (case 0x10/0x11, TIA.cxx:2250-2289)
`newx = hpos<HBLANK ? 3 : ((hpos-HBLANK)+5)%160` (jutari `_resp_player_position`).
`when = ourPlayerPositionResetWhenTable[NUSIZ&7][myPOSP0][newx]`:
- `when==1` (newx inside DISPLAY of an old copy): `updateFrame(clock+11)` FIRST
  (render 11 more clocks at old pos), THEN `myPOSP0=newx`, skipFirst=1.
- `when==0` (neither): `myPOSP0=newx`, skipFirst=1.
- `when==-1` (newx inside the 4-clock DELAY of an old copy): `myPOSP0=newx`,
  skipFirst=0.

### ourPlayerPositionResetWhenTable[mode][oldx][newx]  (computePlayerPositionResetWhenTable)
For each copy at base `b` (per mode's offsets): `newx in [oldx+b, oldx+b+4)` ⇒ -1
(delay); `newx in [oldx+b+4, oldx+b+4+8)` ⇒ 1 (display); else 0. (mode 5: display
window is +4..+4+16; mode 7: +4..+4+32.) Replicate as a function.

### Missile/ball (for later steps)
`ourMissleMaskTable[POSM&3][NUSIZ&7][(NUSIZ&0x30)>>4][160-(POSM&0xFC)]`,
`ourBallMaskTable[POSBL&3][(CTRLPF&0x30)>>4][160-(POSBL&0xFC)]`. Missile width =
`(NUSIZ&0x30)>>4` size code; copies follow the player NUSIZ layout. RESM/RESBL set
`POS = hpos<HBLANK ? 2 : ((hpos-HBLANK)+4)%160` (jutari already does this) + the
HMOVE-relative hardware hacks (Dolphin/Pitfall/Mindmaster).

### HMOVE completeMotion (TIA.cxx:2715-2756)
`myPOSx += ourCompleteMotionTable[x][myHMx]` (wrap 0..159), then recompute ALL
masks at skipFirst=0. jutari already applies the motion (`_hmove_motion`); the
faithful change is that it must run through the same POS→mask path.

## jutari port strategy (incremental, gate each step)

jutari renders per-scanline via `tia_advance!`'s color-clock loop + `cached_sets`
(`_object_pixel_sets`). The port keeps that loop but makes object positioning
faithful:

1. **Player skip-first + reset-when (this step).**
   - Add `p0_skip_first::Bool`, `p1_skip_first::Bool` to TIAState (transient).
   - Add `_player_reset_when(mode, oldx, newx)::Int` (replicate the table logic).
   - Defer RESP0/RESP1 to `pending_writes` (like RESMP #115): at poke time compute
     newx, when, skip_first; activation_clock = beam_cc + (when==1 ? 11 : 0); the
     payload sets `p_x = newx` and `p_skip_first` at activation (new pending kind).
   - `_player_set` / `render_pixel` path: when `skip_first`, omit the first copy.
   - Reset `p0_skip_first=p1_skip_first=false` at scanline start; clear on
     NUSIZ0/1 and HMOVE (jutari recomputes positions there already).
   - HBLANK RESP (beam_cc<68): apply immediately, skip_first=false (xitari's
     end-of-scanline reset means HBLANK strobes never carry skip-first into the
     visible region).
2. Missile/ball faithful mask (RESM/RESBL HMOVE-relative hacks + width) — next.
3. Verify HMOVE completeMotion path matches (positions already move; confirm masks).

**Gate after each step:** `cd jutari && julia --project=. -e 'using Pkg;
Pkg.test()'` + both 64-ROM sweeps (run alone). Expect the G2 games to improve;
some currently-exact games MAY temporarily move — acceptable if the mechanism is
faithful and it's a net move toward xitari (revert only if it's a regression in
*fidelity*, e.g. a faithful-but-incomplete state that's worse than before with no
path forward). RAM must stay 64/64 unless a genuine faithfulness fix requires
otherwise (then re-verify against xitari directly, per the philosophy).
