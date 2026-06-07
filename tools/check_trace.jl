#!/usr/bin/env julia
# PXC1 conformance harness — replay a JSONL trace against jutari.
#
# Mirror of `tools/check_trace.py`. Reads a JSONL trace produced by
# `tools/trace_dump` and replays the same action sequence against
# `JuTari.Env.StellaEnvironment`; compares the per-frame RAM against
# the reference.
#
# Usage
# -----
#
#     julia --project=jutari tools/check_trace.jl \
#         --rom xitari/roms/pong.bin \
#         --trace tools/fixtures/traces/pong_noop_10.jsonl
#
# Exit code: 0 on full match, 1 on divergence (diagnostic on stderr).
#
# Trace parsing
# -------------
# Each line of the trace is a flat JSON object with a fixed shape; we
# extract `action` and `ram` with a regex rather than pull in a JSON
# dependency, since these are the only two fields the conformance check
# consumes.

using JuTari
using JuTari.Env: env_reset!, env_step!, get_ram
using JuTari.RomSettingsModule: GenericRomSettings
using JuTari.PaddleGames: BreakoutRomSettings, PongRomSettings

const _LINE_RE = r"\"action\":(\d+).*\"ram\":\"([0-9a-fA-F]+)\""

# 2026-06-07 (task #75 follow-up): xitari autodetects per-game
# controller wiring from `stella.pro` — pong gets PADDLES with
# SwapPaddles=YES, breakout gets PADDLES too. jutari encodes this
# directly on each RomSettings subclass; check_trace was using the
# generic settings which doesn't trigger the dump-pot model during
# boot, so jutari's pong boot diverged in 4 RAM bytes that depend
# on INPT0/INPT1 dump-pot timing. This map fixes the test setup.
const _SETTINGS_BY_BASENAME = Dict(
    "breakout.bin" => () -> BreakoutRomSettings(),
    "pong.bin"     => () -> PongRomSettings(),
)
_settings_for_rom(p) = haskey(_SETTINGS_BY_BASENAME, basename(p)) ?
    _SETTINGS_BY_BASENAME[basename(p)]() : GenericRomSettings()

"""
    parse_trace_line(line) -> (action::Int, ram_hex::String)
"""
function parse_trace_line(line::AbstractString)
    m = match(_LINE_RE, line)
    m === nothing && error("trace line not parseable: $line")
    return parse(Int, m[1]), String(m[2])
end

"""
    check_trace(rom_path, trace_path) -> Int

Replay the trace against jutari; return the number of frames whose RAM
matched the reference. Throws when a divergence is hit.
"""
function check_trace(rom_path::AbstractString, trace_path::AbstractString)
    rom = read(rom_path)              # Vector{UInt8}
    env = StellaEnvironment(rom, _settings_for_rom(rom_path))
    # Match xitari's ALEInterface::resetGame — see tools/check_trace.py.
    env_reset!(env; boot_noop_steps = 60, boot_reset_steps = 4)

    matched = 0
    open(trace_path) do io
        for line in eachline(io)
            isempty(strip(line)) && continue
            action, ref_hex = parse_trace_line(line)
            env_step!(env, action)
            ram_hex = bytes2hex(get_ram(env))
            if ram_hex != lowercase(ref_hex)
                ref = hex2bytes(ref_hex)
                act = get_ram(env)
                diffs = [(i, ref[i], act[i]) for i in 1:length(ref) if ref[i] != act[i]]
                println(stderr, "DIVERGENCE at trace position $(matched + 1) " *
                                "(action=$action): $(length(diffs)) RAM bytes differ. " *
                                "First 16:")
                for (i, e, a) in first(diffs, 16)
                    println(stderr, "    RAM[\$$(string(i - 1; base = 16, pad = 2))]: " *
                                    "xitari=\$$(string(e; base = 16, pad = 2))  " *
                                    "jutari=\$$(string(a; base = 16, pad = 2))")
                end
                error("RAM divergence at trace position $(matched + 1) " *
                      "($matched matched)")
            end
            matched += 1
        end
    end
    return matched
end

# --- CLI entry point ------------------------------------------------------- #

function _main(argv)
    rom = ""
    trace = ""
    i = 1
    while i <= length(argv)
        a = argv[i]
        if     a == "--rom"   && i + 1 <= length(argv); rom   = argv[i + 1]; i += 2
        elseif a == "--trace" && i + 1 <= length(argv); trace = argv[i + 1]; i += 2
        else;   println(stderr, "check_trace.jl: unknown arg $a"); return 2
        end
    end
    isempty(rom) || isempty(trace) && (println(stderr,
        "usage: julia tools/check_trace.jl --rom <path> --trace <path>"); return 2)
    try
        n = check_trace(rom, trace)
        println(stderr, "OK — $n frame(s) match the xitari reference")
        return 0
    catch e
        println(stderr, sprint(showerror, e))
        return 1
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(_main(ARGS))
end
