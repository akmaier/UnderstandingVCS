#!/usr/bin/env julia
# Run jutari on Breakout with a given action sequence and dump
# per-frame screens as a flat (n_frames, 210, 160) uint8 binary
# file. Matches the layout the Python compositor expects.
#
# Task #53 (vertical-alignment fix): output shape bumped from (n, 192,
# 160) to (n, 210, 160). `get_screen` now returns the ALE-standard
# `Display.YStart=34` / `Display.Height=210` crop (same as xitari),
# so the per-frame screen vertically aligns with `dump_xitari_frames.py`'s
# output.

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", "jutari"))

using JuTari
using JuTari.Env: StellaEnvironment, env_reset!, env_step!, get_screen
using JuTari.PaddleGames: BreakoutRomSettings

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
    rom_path = ""; actions_path = ""; out_path = ""; max_frames = 0
    i = 1
    while i <= length(argv)
        a = argv[i]
        if a == "--rom"        && i + 1 <= length(argv); rom_path     = argv[i+1]; i += 2
        elseif a == "--actions"    && i + 1 <= length(argv); actions_path = argv[i+1]; i += 2
        elseif a == "--out"        && i + 1 <= length(argv); out_path     = argv[i+1]; i += 2
        elseif a == "--max-frames" && i + 1 <= length(argv); max_frames   = parse(Int, argv[i+1]); i += 2
        else
            error("unknown arg $a")
        end
    end
    isempty(rom_path) && error("--rom required")
    isempty(actions_path) && error("--actions required")
    isempty(out_path) && error("--out required")
    max_frames == 0 && error("--max-frames required")
    return (rom_path, actions_path, out_path, max_frames)
end

function main(argv::Vector{String} = ARGS)
    rom_path, actions_path, out_path, max_frames = _parse_args(argv)

    rom = read(rom_path)
    # `BreakoutRomSettings` overrides `romsettings_uses_paddles = true`,
    # so `StellaEnvironment` auto-translates LEFT/RIGHT actions into
    # INPT0 dump-pot paddle-position changes — same shape as xitari's
    # `m_use_paddles` autodetection from stella.pro.
    env = StellaEnvironment(rom, BreakoutRomSettings())
    env_reset!(env; boot_noop_steps = 60, boot_reset_steps = 4)

    actions = _load_actions(actions_path)
    n = min(max_frames, length(actions))

    # Pre-allocate the (n, 210, 160) buffer. Julia is column-major,
    # but the Python reader will reshape from a flat byte stream as
    # (n, 210, 160) so we just need to dump in the (n, 210, 160)
    # row-major order — explicit per-cell write below sidesteps the
    # layout question entirely.
    open(out_path, "w") do io
        for i in 1:n
            env_step!(env, actions[i])
            screen = get_screen(env)            # (210, 160) UInt8
            # Explicit row-major byte write. `screen` in jutari is a
            # 2D Matrix{UInt8} of size (VISIBLE_HEIGHT, SCREEN_WIDTH)
            # = (210, 160). Python reads back as (210, 160) C-order
            # (row-major), so we iterate rows-then-cols and write each
            # byte directly.
            for r in 1:210
                for c in 1:160
                    write(io, UInt8(screen[r, c]))
                end
            end
            if i % 300 == 0
                println(stderr, "  jutari: $i/$n frames")
            end
        end
    end
    println(stderr, "wrote $n frames of shape (210, 160) to $out_path")
    return 0
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main())
end
