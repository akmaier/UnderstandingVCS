#!/usr/bin/env julia
# Task #80 probe: run seaquest boot frame-by-frame, print tia.frame,
# tia.total_cycles, RAM[$01] (the cart's frame counter) at each step.
# Goal: find the frame at which jutari's RAM[$01] first diverges from
# xitari's expected progression (xitari ends at $3e after 60+4 boot,
# jutari at $3f).

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "jutari"))

using JuTari
using JuTari.Env: StellaEnvironment, env_reset!, NOOP
using JuTari.RomSettingsModule: GenericRomSettings, romsettings_reset!
using JuTari.ConsoleModule: console_reset!, run_until_frame!
using JuTari.IO: apply_action!, console_switches!

const ROM = read("xitari/roms/seaquest.bin")

# Mirror env_reset! but step by step so we can introspect after each frame.
env = StellaEnvironment(ROM, GenericRomSettings())
console_reset!(env.console)
romsettings_reset!(env.settings)
env.terminal = false

ram1(env) = Int(env.console.bus.ram[2])    # RAM[$01], Julia 1-indexed
fr(env)   = Int(env.console.bus.tia.frame)
tc(env)   = Int(env.console.bus.tia.total_cycles)

println("post-reset (no frames run yet):")
println("  tia.frame=$(fr(env))  total_cycles=$(tc(env))  RAM[\$01]=$(string(ram1(env), base=16, pad=2))")

# Step 60 NOOP frames
for i in 1:60
    apply_action!(env.console, Int(NOOP))
    run_until_frame!(env.console)
    if i <= 5 || i >= 56 || (i % 10 == 0)
        println("after NOOP $(i): tia.frame=$(fr(env))  RAM[\$01]=$(string(ram1(env), base=16, pad=2))")
    end
end

# Print ram_bytes 0..3 around RESET transitions to see if pattern changes
ram_str(env) = join([string(Int(env.console.bus.ram[k+1]), base=16, pad=2) for k in 0:3], " ")

# Step 4 RESET frames
println("--- pressing RESET ---")
console_switches!(env.console; reset_pressed = true)
for i in 1:4
    apply_action!(env.console, Int(NOOP))
    run_until_frame!(env.console)
    println("after RESET $(i): tia.frame=$(fr(env))  RAM[\$00..\$03]=$(ram_str(env))")
end
console_switches!(env.console; reset_pressed = false)

# Take one extra NOOP after release to see what happens
for i in 1:3
    apply_action!(env.console, Int(NOOP))
    run_until_frame!(env.console)
    println("after release+NOOP $(i): tia.frame=$(fr(env))  RAM[\$00..\$03]=$(ram_str(env))")
end

println()
println("FINAL: tia.frame=$(fr(env))  RAM[\$01]=$(string(ram1(env), base=16, pad=2))")
println("  xitari boot_end has RAM[\$01]=\$3e (= 62 decimal)")
println("  jutari (per bug_fix_log) ends at RAM[\$01]=\$3f (= 63 decimal)")
