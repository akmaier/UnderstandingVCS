#!/usr/bin/env julia
# Task #85 probe — investigate the space_invaders regression at rows 24-28
# cols 22-23, 54-55, 86-87 (3 missile copies, 2px wide, 32cc apart).
# Inspect the screen + key TIA registers around those positions.

using JuTari
using JuTari.Env: StellaEnvironment, env_reset!, env_step!, get_screen

const ROM = read("xitari/roms/space_invaders.bin")
const ACTIONS = parse.(Int, [s for s in strip.(readlines("tools/fixtures/actions/space_invaders_noop_10.txt")) if !isempty(s) && !startswith(s, "#")])

env = StellaEnvironment(ROM)
env_reset!(env; boot_noop_steps = 60, boot_reset_steps = 4)
env_step!(env, ACTIONS[1])

tia = env.console.bus.tia
screen = get_screen(env)
println("END-of-frame-1 register snapshot:")
println("  NUSIZ0=", string(tia.registers[0x04+1], base=16, pad=2),
        "  NUSIZ1=", string(tia.registers[0x05+1], base=16, pad=2))
println("  COLUP0=", string(tia.registers[0x06+1], base=16, pad=2),
        "  COLUP1=", string(tia.registers[0x07+1], base=16, pad=2))
println("  ENAM0=", string(tia.registers[0x1D+1], base=16, pad=2),
        "  ENAM1=", string(tia.registers[0x1E+1], base=16, pad=2))
println("  RESMP0=", string(tia.registers[0x28+1], base=16, pad=2),
        "  RESMP1=", string(tia.registers[0x29+1], base=16, pad=2))
println("  m0_x=", tia.m0_x, "  m1_x=", tia.m1_x,
        "  p0_x=", tia.p0_x, "  p1_x=", tia.p1_x)

println("\nScreen at display rows 22-30, cols 18-94 (4 cols apart):")
print("       ")
for c in 18:4:94
    print(lpad(string(c), 3))
end
println()
for r in 22:30
    print("row ", lpad(string(r), 2), ": ")
    for c in 18:4:94
        print(lpad(string(Int(screen[r + 1, c + 1]), base=16), 3))
    end
    println()
end

println("\nNon-bg pixels in screen row 24-26 (cols 0-159):")
for r in 24:26
    cols_painted = [c for c in 0:159 if Int(screen[r + 1, c + 1]) != 0]
    println("  row ", r, ": ", length(cols_painted), " painted, sample cols ",
            (length(cols_painted) > 0 ? cols_painted[1:min(end, 30)] : []))
end
