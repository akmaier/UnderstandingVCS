#!/usr/bin/env julia
# Bus trace on the EXACT conformance trajectory (mirrors dump_jutari_frames.jl:
# boot 60+4, actions[i] per frame, auto-reset on terminal) so it reproduces the
# real long-horizon divergence. Prints per-frame RAM[$1a],[$1b] to stderr and
# dumps every bus peek/poke for frames [tlo,thi] to a CSV.
#   julia --project=jutari tools/wow_bus_trace.jl <rom> <actions> <out.csv> <n> <tlo> <thi>
using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "jutari"))
using JuTari
using JuTari.Env: StellaEnvironment, env_reset!, env_step!, get_ram
using JuTari.Bus: trace_enable!, trace_disable!, trace_take!
using JuTari.RomSettingsModule: GenericRomSettings
using JuTari.JoystickGames: WizardOfWorRomSettings

function load_actions(p)
    a = Int[]
    for l in eachline(p)
        s = strip(l); (isempty(s) || startswith(s, "#")) && continue
        push!(a, parse(Int, s))
    end
    a
end

rom = read(ARGS[1]); actions = load_actions(ARGS[2]); out = ARGS[3]
n = parse(Int, ARGS[4]); tlo = parse(Int, ARGS[5]); thi = parse(Int, ARGS[6])

env = StellaEnvironment(rom, WizardOfWorRomSettings())
env_reset!(env; boot_noop_steps = 60, boot_reset_steps = 4)

open(out, "w") do io
    println(io, "frame,kind,sl,sc,cc,addr,value")
    for i in 1:n
        env.terminal && env_reset!(env; boot_noop_steps = 60, boot_reset_steps = 4)
        dotrace = tlo <= i <= thi
        dotrace && trace_enable!()
        env_step!(env, actions[i])
        if dotrace
            for (kind, sl, sc, cc, addr, val) in trace_take!()
                println(io, "$i,$kind,$sl,$sc,$cc,$(string(addr, base=16)),$val")
            end
            trace_disable!()
        end
        ram = get_ram(env)
        println(stderr, "frame $i: \$1a=$(ram[0x1a + 1]) \$1b=$(ram[0x1b + 1])")
    end
end
