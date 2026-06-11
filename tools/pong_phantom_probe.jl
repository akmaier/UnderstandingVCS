#!/usr/bin/env julia
# Task #85 probe: dump TIA register state + sprite positions at scanlines
# 69-71 (display rows 35-37) of pong frame 1, where the 16-px phantom
# residual appears. Tells us NUSIZ/ENAM/RESMP/p0_x/m0_x/COLUP* values
# at each scanline so we can localise the phantom source.

using JuTari
using JuTari.Env: StellaEnvironment, env_reset!, env_step!, get_screen
using JuTari.PaddleGames: PongRomSettings

const W_NUSIZ0 = 0x04; const W_NUSIZ1 = 0x05
const W_COLUP0 = 0x06; const W_COLUP1 = 0x07
const W_REFP0  = 0x0B; const W_REFP1  = 0x0C
const W_GRP0   = 0x1B; const W_GRP1   = 0x1C
const W_ENAM0  = 0x1D; const W_ENAM1  = 0x1E
const W_ENABL  = 0x1F
const W_RESMP0 = 0x28; const W_RESMP1 = 0x29

const ROM = read("xitari/roms/pong.bin")
const ACTIONS = parse.(Int, [s for s in strip.(readlines("tools/fixtures/actions/pong_noop_10.txt")) if !isempty(s) && !startswith(s, "#")])

env = StellaEnvironment(ROM, PongRomSettings())
env_reset!(env; boot_noop_steps = 60, boot_reset_steps = 4)

# Step 1 frame (action 0)
env_step!(env, ACTIONS[1])

# Now inspect the TIA state — we don't have per-scanline mid-frame snapshots
# easily, but we can look at the framebuffer at the relevant scanlines.
tia = env.console.bus.tia
# get_screen returns the cropped framebuffer (rows Y_START+1 : Y_START+VISIBLE_HEIGHT).
# That's rows 0..209 in display coords = absolute scanlines 34..243.
# We want display rows 35-37 = absolute scanlines 69-71.
fb_full = tia.framebuffer
screen = get_screen(env)
println("get_screen size: ", size(screen))
println("framebuffer size: ", size(fb_full))
println("After frame 1:")
println("  scanline = ", tia.scanline, "  frame = ", tia.frame)
println("  Y_START = 34, so display row 35 = absolute scanline 69")
println()
println("get_screen at rows 33-39, cols 14-22 (left):")
for r in 33:39
    println("  row ", r, ": ",
            join([string(Int(screen[r + 1, c + 1]), base=16, pad=2) for c in 14:22], " "))
end
println()
println("get_screen at rows 33-39, cols 138-146 (right):")
for r in 33:39
    println("  row ", r, ": ",
            join([string(Int(screen[r + 1, c + 1]), base=16, pad=2) for c in 138:146], " "))
end
println()
println("Current TIA register snapshot (END of frame 1, only valid for last sl):")
println("  NUSIZ0=", string(tia.registers[W_NUSIZ0 + 1], base=16), "  NUSIZ1=", string(tia.registers[W_NUSIZ1 + 1], base=16))
println("  COLUP0=", string(tia.registers[W_COLUP0 + 1], base=16), "  COLUP1=", string(tia.registers[W_COLUP1 + 1], base=16))
println("  ENAM0=", string(tia.registers[W_ENAM0 + 1], base=16), "  ENAM1=", string(tia.registers[W_ENAM1 + 1], base=16))
println("  RESMP0=", string(tia.registers[W_RESMP0 + 1], base=16), "  RESMP1=", string(tia.registers[W_RESMP1 + 1], base=16))
println("  p0_x=", tia.p0_x, "  p1_x=", tia.p1_x, "  m0_x=", tia.m0_x, "  m1_x=", tia.m1_x)
println("  GRP0=", string(tia.registers[W_GRP0 + 1], base=16), "  GRP1=", string(tia.registers[W_GRP1 + 1], base=16))
println("  REFP0=", string(tia.registers[W_REFP0 + 1], base=16), "  REFP1=", string(tia.registers[W_REFP1 + 1], base=16))
