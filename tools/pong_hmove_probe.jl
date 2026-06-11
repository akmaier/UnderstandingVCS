#!/usr/bin/env julia
# Task #83 probe: trace pong HMOVE writes during action 9 (last frame).
# Log every HMOVE write with the resulting hmove_blank_pending state.

using JuTari
using JuTari.Env: StellaEnvironment, env_reset!, env_step!, get_screen
using JuTari.PaddleGames: PongRomSettings

const ROM = read("xitari/roms/pong.bin")
const ACTIONS = parse.(Int, [s for s in strip.(readlines("tools/fixtures/actions/pong_noop_10.txt")) if !isempty(s) && !startswith(s, "#")])

env = StellaEnvironment(ROM, PongRomSettings())
env_reset!(env; boot_noop_steps = 60, boot_reset_steps = 4)

# Step 9 actions
for a in ACTIONS[1:9]
    env_step!(env, a)
end

# Now inject a hook by monkey-patching tia_poke! call.
# Since we can't easily monkey-patch, instead instrument via a wrapper
# step. Simpler: just step ONE more time and capture every poke.
import JuTari.TIA: tia_poke!, W_HMOVE
const orig_poke = tia_poke!

hmove_writes = Vector{NamedTuple{(:scanline, :sc, :beam_cc, :flag), Tuple{Int,Int,Int,Bool}}}()

# Custom step: wrap poke
function instrumented_poke!(tia, addr::Integer, value::Integer, beam_cc=tia.color_clock, beam_sc=tia.scanline_cycle)
    if (addr & 0x3F) == W_HMOVE
        push!(hmove_writes, (
            scanline = tia.scanline, sc = Int(beam_sc), beam_cc = Int(beam_cc),
            flag = false   # will check after
        ))
    end
    orig_poke(tia, addr, value, beam_cc, beam_sc)
    if (addr & 0x3F) == W_HMOVE
        # Update the last entry's flag value
        hmove_writes[end] = (
            scanline = hmove_writes[end].scanline,
            sc = hmove_writes[end].sc,
            beam_cc = hmove_writes[end].beam_cc,
            flag = tia.hmove_blank_pending,
        )
    end
end

# Replace at module level - won't work for already-compiled functions
# Instead, just instrument differently: track via TIA state changes.
# Use a SIMPLER approach: scan framebuffer for HMOVE-blank-affected scanlines.

env_step!(env, ACTIONS[10])

# Now find which scanlines have cols 0-7 = 0 (HMOVE comb affected)
fb = env.console.bus.tia.framebuffer
hmove_combed_scanlines = Int[]
for sl in 0:243
    if all(Int(fb[sl + 1, c + 1]) == 0 for c in 0:7)
        push!(hmove_combed_scanlines, sl)
    end
end
println("jutari scanlines where cols 0-7 == 0 (potential HMOVE comb):")
println("  count: ", length(hmove_combed_scanlines))
println("  first 10: ", hmove_combed_scanlines[1:min(end,10)])
println("  last 10: ", hmove_combed_scanlines[max(end-9,1):end])

println()
println("Scanlines where col 0 != col 8 (any first-8-px asymmetry):")
asym = Int[]
for sl in 0:243
    if Int(fb[sl + 1, 0 + 1]) != Int(fb[sl + 1, 8 + 1])
        push!(asym, sl)
    end
end
println("  count: ", length(asym), " scanlines")
if length(asym) > 0
    sl = asym[1]
    println("  first: sl ", sl, " col 0-15 = ",
        [Int(fb[sl + 1, c + 1]) for c in 0:15])
end
