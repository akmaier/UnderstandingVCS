#!/usr/bin/env julia
# full_instr_trace.jl — dump jutari's per-instruction (PC,A,X,Y,P,SP) for a
# range of frames, to diff against xitari's XICPU trace and pin the FIRST
# instruction whose state diverges. Boots exactly like obj_trace.jl / the sweep.
# INSTR lines ("I pc op A X Y P SP") are emitted to stderr by M6502 when
# TIA._OBJ_TRACE[] is on; this driver toggles it per frame and prints "F <n>"
# frame markers to stderr too, interleaved.
using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "jutari"))
using JuTari
using JuTari.Env: StellaEnvironment, env_reset!, env_step!
using JuTari.JoystickGames: ElevatorActionRomSettings
import JuTari.TIA

function main()
    args = Dict{String,String}(); i = 1
    while i <= length(ARGS); args[ARGS[i]] = ARGS[i+1]; i += 2; end
    rom = read(args["--rom"])
    actions = Int[]
    for line in eachline(args["--actions"])
        s = strip(line); (isempty(s) || startswith(s, "#")) && continue
        push!(actions, parse(Int, s))
    end
    nframes = parse(Int, get(args, "--frames", "10"))
    trace_boot = haskey(args, "--trace-boot")
    if trace_boot
        println(stderr, "F -1")
        TIA._OBJ_TRACE[] = true
    end
    env = StellaEnvironment(rom, ElevatorActionRomSettings())
    env_reset!(env; boot_noop_steps = 60, boot_reset_steps = 4)
    trace_boot && (TIA._OBJ_TRACE[] = false)
    for k in 0:nframes
        println(stderr, "F $k")
        TIA._OBJ_TRACE[] = true
        act = (k + 1) <= length(actions) ? actions[k + 1] : (isempty(actions) ? 0 : actions[end])
        env_step!(env, act)
        TIA._OBJ_TRACE[] = false
    end
    return 0
end

main()
