"""
    TIA

Television Interface Adapter — Atari 2600 video/audio chip.

The CPU talks to the TIA through 64 memory-mapped registers in the
\$0000–\$003F window (with mirrors), some write-only, a subset readable.

Per-frame timing (NTSC):

    228 color clocks    = 1 scanline (= 76 CPU cycles)
    262 scanlines       = 1 frame    (≈ 19,912 CPU cycles)

WSYNC (write \$02) stalls the CPU until the next scanline boundary.

Phase P3a status (this module): register file + scanline/frame timing +
WSYNC. Reads still return 0 stubs for collisions and INPT*. Playfield,
sprites, missiles, ball, collisions, HMOVE positioning, VSYNC-driven
frame ending, and framebuffer rendering land in subsequent sub-phases
(see PORTING_PLAN.md §5 P3).
"""
module TIA

export TIAState, initial_tia_state,
       tia_peek, tia_poke!, tia_advance!, tia_apply_wsync!,
       NTSC_CPU_CYCLES_PER_SCANLINE, NTSC_SCANLINES_PER_FRAME,
       NUM_REGISTERS, SCREEN_WIDTH, SCREEN_HEIGHT,
       W_VSYNC, W_VBLANK, W_WSYNC, W_RSYNC, W_NUSIZ0, W_NUSIZ1,
       W_COLUP0, W_COLUP1, W_COLUPF, W_COLUBK, W_CTRLPF, W_REFP0, W_REFP1,
       W_PF0, W_PF1, W_PF2, W_RESP0, W_RESP1, W_RESM0, W_RESM1, W_RESBL,
       W_AUDC0, W_AUDC1, W_AUDF0, W_AUDF1, W_AUDV0, W_AUDV1,
       W_GRP0, W_GRP1, W_ENAM0, W_ENAM1, W_ENABL,
       W_HMP0, W_HMP1, W_HMM0, W_HMM1, W_HMBL,
       W_VDELP0, W_VDELP1, W_VDELBL, W_RESMP0, W_RESMP1,
       W_HMOVE, W_HMCLR, W_CXCLR

# Write registers ($00..$2C)
const W_VSYNC  = 0x00; const W_VBLANK = 0x01
const W_WSYNC  = 0x02; const W_RSYNC  = 0x03
const W_NUSIZ0 = 0x04; const W_NUSIZ1 = 0x05
const W_COLUP0 = 0x06; const W_COLUP1 = 0x07
const W_COLUPF = 0x08; const W_COLUBK = 0x09
const W_CTRLPF = 0x0A
const W_REFP0  = 0x0B; const W_REFP1  = 0x0C
const W_PF0    = 0x0D; const W_PF1    = 0x0E; const W_PF2    = 0x0F
const W_RESP0  = 0x10; const W_RESP1  = 0x11
const W_RESM0  = 0x12; const W_RESM1  = 0x13; const W_RESBL  = 0x14
const W_AUDC0  = 0x15; const W_AUDC1  = 0x16
const W_AUDF0  = 0x17; const W_AUDF1  = 0x18
const W_AUDV0  = 0x19; const W_AUDV1  = 0x1A
const W_GRP0   = 0x1B; const W_GRP1   = 0x1C
const W_ENAM0  = 0x1D; const W_ENAM1  = 0x1E; const W_ENABL  = 0x1F
const W_HMP0   = 0x20; const W_HMP1   = 0x21
const W_HMM0   = 0x22; const W_HMM1   = 0x23; const W_HMBL   = 0x24
const W_VDELP0 = 0x25; const W_VDELP1 = 0x26; const W_VDELBL = 0x27
const W_RESMP0 = 0x28; const W_RESMP1 = 0x29
const W_HMOVE  = 0x2A; const W_HMCLR  = 0x2B
const W_CXCLR  = 0x2C

# Constants
const NUM_REGISTERS = 64
const NTSC_CPU_CYCLES_PER_SCANLINE = 76
const NTSC_SCANLINES_PER_FRAME = 262
const SCREEN_WIDTH = 160
const SCREEN_HEIGHT = 192

"""
    TIAState

Mutable TIA state. `registers` is the 64-byte register file (last-written
value for each address); `framebuffer` is the rendered output (zeros in
P3a; populated by later sub-phases).
"""
mutable struct TIAState
    registers::Vector{UInt8}
    scanline_cycle::Int
    scanline::Int
    frame::UInt64
    wsync_pending::Bool
    framebuffer::Matrix{UInt8}
end

initial_tia_state() = TIAState(
    zeros(UInt8, NUM_REGISTERS),
    0, 0, UInt64(0), false,
    zeros(UInt8, SCREEN_HEIGHT, SCREEN_WIDTH),
)

"""
    tia_peek(tia, addr) -> UInt8

Read a TIA register. P3a stub: all readable registers (collisions, INPT*)
return 0. Proper values land in P3e (collisions) and a later input phase.
"""
@inline tia_peek(::TIAState, ::Integer) = UInt8(0)

"""
    tia_poke!(tia, addr, value)

Write a TIA register. Stores the byte and applies P3a side-effects
(currently just WSYNC). Other side-effecting writes (HMOVE, RES*, CXCLR,
…) land in later P3 sub-phases.
"""
function tia_poke!(tia::TIAState, addr::Integer, value::Integer)
    reg = Int(addr) & 0x3F                # TIA decodes A0–A5
    tia.registers[reg + 1] = UInt8(Int(value) & 0xFF)
    if reg == W_WSYNC
        tia.wsync_pending = true
    end
    return nothing
end

"""
    tia_advance!(tia, cpu_cycles)

Advance TIA by `cpu_cycles` CPU cycles. Updates `scanline_cycle`,
`scanline`, and `frame`.
"""
function tia_advance!(tia::TIAState, cpu_cycles::Integer)
    total = tia.scanline_cycle + Int(cpu_cycles)
    tia.scanline_cycle = total % NTSC_CPU_CYCLES_PER_SCANLINE
    line_advance = total ÷ NTSC_CPU_CYCLES_PER_SCANLINE
    new_line = tia.scanline + line_advance
    frame_advance = new_line ÷ NTSC_SCANLINES_PER_FRAME
    tia.scanline = new_line % NTSC_SCANLINES_PER_FRAME
    tia.frame += UInt64(frame_advance)
    return nothing
end

"""
    tia_apply_wsync!(tia) -> stall_cycles::Int

If `tia.wsync_pending`, advance TIA to the next scanline boundary and
return the cycle count to be added to `state.cycles`. Otherwise return 0.
"""
function tia_apply_wsync!(tia::TIAState)
    tia.wsync_pending || return 0
    stall = mod(NTSC_CPU_CYCLES_PER_SCANLINE - tia.scanline_cycle,
                NTSC_CPU_CYCLES_PER_SCANLINE)
    if stall > 0
        tia_advance!(tia, stall)
    end
    tia.wsync_pending = false
    return stall
end

end # module
