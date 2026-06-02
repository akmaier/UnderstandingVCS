#!/usr/bin/env julia
# Probe what RIOT/TIA state jutari has just before / just after the FIRE
# action at frame 20 of pong, plus the full RAM dump.
# Compares to xitari's expected state.

using Pkg
Pkg.activate("/Users/maier/Documents/code/UnderstandingVCS/jutari")

using JuTari
using JuTari.Env: StellaEnvironment, env_reset!, env_step!, get_ram
using JuTari.PaddleGames: PongRomSettings

ROM = "/Users/maier/Documents/code/UnderstandingVCS/xitari/roms/pong.bin"
ACTIONS = "/Users/maier/Documents/code/UnderstandingVCS/tools/breakout_video/output/pong_breakout_random_actions.txt"

actions = Int[]
for line in eachline(ACTIONS)
    s = strip(line); (isempty(s) || startswith(s, "#")) && continue
    push!(actions, parse(Int, s))
end

rom = read(ROM)
env = StellaEnvironment(rom, PongRomSettings())
env_reset!(env; boot_noop_steps=60, boot_reset_steps=4)

println("=== jutari pong: state around FIRE (frame 20) ===")
for i in 1:25
    a = actions[i]
    env_step!(env, a)
    if i in (19, 20, 21)
        println("\nframe $(i-1) (action=$a):")
        ram = get_ram(env)
        println("  RAM[\$3e..\$42]: ", [ram[off+1] for off in 0x3e:0x42])
        println("  RAM[\$04..\$10]: ", [ram[off+1] for off in 0x04:0x10])
        # TIA state (paddle resistance, dump_disabled_cycle if accessible)
        tia = env.console.bus.tia
        try
            println("  tia.paddle_resistance: ", tia.paddle_resistance)
            println("  tia.paddle_use_dump_pot: ", tia.paddle_use_dump_pot)
            println("  tia.dump_disabled_cycle: ", tia.dump_disabled_cycle)
            println("  tia.dump_enabled: ", tia.dump_enabled)
            println("  tia.total_cycles: ", tia.total_cycles)
        catch err
            println("  (tia field error: $err)")
        end
        # RIOT state
        riot = env.console.bus.riot
        try
            println("  riot.intim: ", riot.intim)
            println("  riot.swcha_in: ", string(riot.swcha_in; base=16))
            println("  riot.swchb_in: ", string(riot.swchb_in; base=16))
        catch err
            println("  (riot field error: $err)")
        end
    end
end
