#!/usr/bin/env julia
# Dump jutari's per-frame SCREEN (210×160 palette indices) for a noop/action
# trace, gzip-compressed, as a fixture for the screen-conformance tests
# (xitari ↔ jutari frame diff). Mirrors tools/jutari_trace_dump.jl's
# per-ROM RomSettings autodetection + boot convention so the frames line
# up with the xitari `tools/fixtures/screens/*.screen.gz` ground truth.
#
# Usage:
#   julia --project=jutari tools/jutari_screen_dump.jl \
#       --rom xitari/roms/breakout.bin \
#       --actions tools/fixtures/actions/breakout_noop_10.txt \
#       --out tools/fixtures/screens/breakout_noop_10_jutari.screen.gz \
#       --max-frames 10

using JuTari
using JuTari.Env: StellaEnvironment, env_reset!, env_step!, get_screen
using JuTari.RomSettingsModule: GenericRomSettings
using JuTari.PaddleGames: BreakoutRomSettings, PongRomSettings
using JuTari.JoystickGames: PitfallRomSettings, EnduroRomSettings

# Task #95 (2026-06-13): pitfall + enduro were MISSING here, so the
# PXC-S jutari fixtures for those ROMs were generated with
# GenericRomSettings — omitting Pitfall's `getStartingActions`
# (1× PLAYER_A_UP) and Enduro's. The xitari fixtures use the real
# per-game settings, so the screen comparison was measuring a settings
# mismatch on top of any render delta. MUST stay in sync with
# tools/jutari_trace_dump.jl + tools/breakout_video/dump_jutari_frames.jl.
const _SETTINGS_BY_BASENAME = Dict(
    "breakout.bin" => () -> BreakoutRomSettings(),
    "pong.bin"     => () -> PongRomSettings(),
    "pitfall.bin"  => () -> PitfallRomSettings(),
    "enduro.bin"   => () -> EnduroRomSettings(),
)
_settings_for_rom(p) = haskey(_SETTINGS_BY_BASENAME, basename(p)) ?
    _SETTINGS_BY_BASENAME[basename(p)]() : GenericRomSettings()

function main()
    args = Dict{String,String}()
    i = 1
    while i <= length(ARGS)
        args[ARGS[i]] = ARGS[i+1]; i += 2
    end
    rom_path = args["--rom"]; actions_path = args["--actions"]
    out_path = args["--out"]; maxf = parse(Int, get(args, "--max-frames", "10"))

    rom = read(rom_path)
    actions = Int[]
    for line in eachline(actions_path)
        s = strip(line)
        (isempty(s) || startswith(s, "#")) && continue
        push!(actions, parse(Int, s))
    end

    env = StellaEnvironment(rom, _settings_for_rom(rom_path))
    env_reset!(env; boot_noop_steps = 60, boot_reset_steps = 4)
    frames = UInt8[]
    n = 0
    for a in actions
        n >= maxf && break
        env_step!(env, a)
        scr = get_screen(env)          # (210,160) UInt8 (row-major)
        # store row-major to match the xitari capture (np reshape (210,160))
        append!(frames, vec(permutedims(scr)))
        n += 1
    end
    open(out_path, "w") do io          # raw bytes; caller gzips externally
        write(io, frames)
    end
    println("wrote $n jutari screen frames ($(210*160) B each) to $out_path")
end

main()
