#!/usr/bin/env julia
#
# cpu_tia_cycle_trace.jl — per-bus-op trace dumper for jutari.
#
# Phase 1 of P3I_G_THREADING_PLAN.md. Runs jutari against a ROM +
# action stream, captures every CPU bus operation (peek, poke,
# internal-cycle tick, WSYNC release), and emits a CSV row per event
# with the TIA's (scanline, scanline_cycle, color_clock) at the moment
# of the operation.
#
# The output is intended to be diffed against the equivalent xitari
# trace (`tools/cpu_tia_cycle_trace_xitari.cpp`) to pin the first event
# whose color_clock differs — that's the bug entry point.
#
# CSV columns:
#   global_idx,frame,kind,scanline,scanline_cycle,color_clock,addr,value
#
# `kind ∈ {peek, poke, tick, wsync_release, frame_boundary}`. The
# `frame_boundary` rows are synthetic markers emitted between frames
# so the diff tool can re-sync if a divergence pushes the trace
# off-by-N events within a frame.
#
# Usage:
#   julia --project=jutari tools/cpu_tia_cycle_trace.jl \
#       --rom xitari/roms/pong.bin \
#       --actions tools/breakout_video/output/pong_breakout_random_actions.txt \
#       --max-frames 25 \
#       --rom-settings pong \
#       --out tools/fixtures/cycle_traces/pong_jutari_25.csv

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "jutari"))

using JuTari
using JuTari.Bus: trace_enable!, trace_disable!, trace_take!
using JuTari.Env: StellaEnvironment, env_reset!, env_step!, frame_number


function _load_actions(path::AbstractString)
    out = Int[]
    for line in eachline(path)
        s = strip(line)
        (isempty(s) || startswith(s, "#")) && continue
        push!(out, parse(Int, s))
    end
    return out
end


function _parse_args(argv::Vector{String})
    rom_path = ""; actions_path = ""; out_path = ""; max_frames = 25
    rom_settings = "generic"
    boot_noop = 60; boot_reset = 4
    i = 1
    while i <= length(argv)
        a = argv[i]
        if     a == "--rom"           && i + 1 <= length(argv); rom_path      = argv[i+1]; i += 2
        elseif a == "--actions"       && i + 1 <= length(argv); actions_path  = argv[i+1]; i += 2
        elseif a == "--out"           && i + 1 <= length(argv); out_path      = argv[i+1]; i += 2
        elseif a == "--max-frames"    && i + 1 <= length(argv); max_frames    = parse(Int, argv[i+1]); i += 2
        elseif a == "--rom-settings"  && i + 1 <= length(argv); rom_settings  = argv[i+1]; i += 2
        elseif a == "--boot-noop"     && i + 1 <= length(argv); boot_noop     = parse(Int, argv[i+1]); i += 2
        elseif a == "--boot-reset"    && i + 1 <= length(argv); boot_reset    = parse(Int, argv[i+1]); i += 2
        else error("unknown arg $a")
        end
    end
    isempty(rom_path) && error("--rom required")
    isempty(out_path) && error("--out required")
    return (; rom_path, actions_path, out_path, max_frames, rom_settings,
            boot_noop, boot_reset)
end


function _make_settings(name::AbstractString)
    nm = lowercase(name)
    nm == "pong"     && return PongRomSettings()
    nm == "breakout" && return BreakoutRomSettings()
    nm == "generic"  && return JuTari.RomSettingsModule.GenericRomSettings()
    error("unknown rom-settings $name — known: pong, breakout, generic")
end


function main(argv::Vector{String} = ARGS)
    args = _parse_args(argv)

    actions = isempty(args.actions_path) ? Int[] : _load_actions(args.actions_path)
    n = max(args.max_frames, 1)

    rom = read(args.rom_path)
    env = StellaEnvironment(rom, _make_settings(args.rom_settings))

    # NB: tracing is ENABLED across the boot-burn too, so divergence
    # during reset is captured. xitari starts its trace at the same
    # point so the two are comparable cycle-for-cycle.
    trace_enable!()
    env_reset!(env; boot_noop_steps = args.boot_noop, boot_reset_steps = args.boot_reset)

    mkpath(dirname(args.out_path))
    open(args.out_path, "w") do io
        write(io, "global_idx,frame,kind,scanline,scanline_cycle,color_clock,addr,value\n")
        idx = 0

        # Flush the boot-burn trace before frame 1 so the per-frame
        # event counts in the CSV match the per-frame action stream
        # exactly (frame 1 = first user step).
        for (kind, sl, sc, cc, addr, val) in trace_take!()
            idx += 1
            write(io, "$idx,0,$kind,$sl,$sc,$cc,$(string(addr, base=16)),$val\n")
        end
        idx += 1
        write(io, "$idx,0,frame_boundary,0,0,0,0,0\n")

        for f in 1:n
            a = f <= length(actions) ? actions[f] : 0
            env_step!(env, a)
            for (kind, sl, sc, cc, addr, val) in trace_take!()
                idx += 1
                write(io, "$idx,$f,$kind,$sl,$sc,$cc,$(string(addr, base=16)),$val\n")
            end
            idx += 1
            write(io, "$idx,$f,frame_boundary,0,0,0,0,0\n")
        end
    end
    trace_disable!()

    sz = stat(args.out_path).size
    println(stderr, "wrote $(args.max_frames) frame(s) of trace to $(args.out_path) ($sz bytes)")
    return 0
end


if abspath(PROGRAM_FILE) == @__FILE__
    exit(main())
end
