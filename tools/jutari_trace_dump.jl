#!/usr/bin/env julia
# jutari_trace_dump.jl
#
# PXC2 helper — replay an xitari trace's action sequence against jutari
# and write a JSONL trace of jutari's per-frame RAM. The output format
# is the same simple subset xitari's `tools/trace_dump.cpp` emits
# ({"frame": N, "action": A, "ram": "<256 hex chars>"} one per line),
# so an independent diff between two such files is a direct
# jaxtari-vs-jutari (or jutari-vs-xitari) cross-check.
#
# We avoid adding a JSON.jl dependency by parsing the few fields we
# need with regex and emitting our own JSON line manually — the format
# is fixed and trivial, so this stays portable across whatever the
# project's Julia env happens to have installed.
#
# Usage:
#   julia --project=jutari tools/jutari_trace_dump.jl \
#       --rom xitari/roms/pong.bin \
#       --trace tools/fixtures/traces/pong_noop_10.jsonl \
#       --out  tools/fixtures/traces/pong_noop_10_jutari.jsonl

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "jutari"))

using JuTari
using JuTari.Env: StellaEnvironment, env_reset!, env_step!, get_ram, frame_number
using JuTari.RomSettingsModule: GenericRomSettings, RomSettings
using JuTari.PaddleGames: BreakoutRomSettings, PongRomSettings

# Per-ROM RomSettings autodetection — mirror of jaxtari `tools/check_trace.py`.
# Activates the dump-pot model + paddle-action handling for paddle
# games so jutari matches xitari's INPT0/INPT1 cycle-dependent reads.
const _SETTINGS_BY_BASENAME = Dict{String,Function}(
    "breakout.bin" => () -> BreakoutRomSettings(),
    "pong.bin"     => () -> PongRomSettings(),
)
_settings_for_rom(rom_path::AbstractString) =
    haskey(_SETTINGS_BY_BASENAME, basename(rom_path)) ?
        _SETTINGS_BY_BASENAME[basename(rom_path)]() :
        GenericRomSettings()

function _hex_of(bytes::AbstractVector{UInt8})
    io = IOBuffer()
    for b in bytes
        print(io, string(b; base = 16, pad = 2))
    end
    return String(take!(io))
end

# Minimal "action" extractor — pulls the integer from `"action": <N>` in
# a JSONL line. The xitari traces have exactly one such field per line.
function _action_in_line(line::AbstractString)
    m = match(r"\"action\"\s*:\s*(-?\d+)", line)
    m === nothing && error("no \"action\" field in line: $line")
    return parse(Int, m.captures[1])
end

function _parse_args(argv::Vector{String})
    rom_path   = ""
    trace_path = ""
    out_path   = ""
    i = 1
    while i <= length(argv)
        a = argv[i]
        if a == "--rom"   && i + 1 <= length(argv); rom_path   = argv[i+1]; i += 2
        elseif a == "--trace" && i + 1 <= length(argv); trace_path = argv[i+1]; i += 2
        elseif a == "--out"   && i + 1 <= length(argv); out_path   = argv[i+1]; i += 2
        else
            error("unknown arg $a")
        end
    end
    isempty(rom_path)   && error("--rom required")
    isempty(trace_path) && error("--trace required")
    isempty(out_path)   && error("--out required")
    return (rom_path, trace_path, out_path)
end

function main(argv::Vector{String} = ARGS)
    rom_path, trace_path, out_path = _parse_args(argv)

    actions = Int[]
    for line in eachline(trace_path)
        isempty(strip(line)) && continue
        push!(actions, _action_in_line(line))
    end

    rom = read(rom_path)
    env = StellaEnvironment(rom, _settings_for_rom(rom_path))
    env_reset!(env; boot_noop_steps = 60, boot_reset_steps = 4)

    open(out_path, "w") do io
        for act_id in actions
            env_step!(env, act_id)
            ram_hex = _hex_of(UInt8.(get_ram(env)))
            println(io, "{\"frame\":", frame_number(env),
                          ",\"action\":", act_id,
                          ",\"ram\":\"", ram_hex, "\"}")
        end
    end
    println(stderr, "OK — wrote $(length(actions)) jutari frames to $out_path")
    return 0
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main())
end
