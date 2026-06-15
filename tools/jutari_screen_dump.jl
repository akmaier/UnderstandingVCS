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
using JuTari.JoystickGames: PitfallRomSettings, EnduroRomSettings,
    AirRaidRomSettings, AsterixRomSettings, BeamRiderRomSettings,
    DoubleDunkRomSettings, ElevatorActionRomSettings, GopherRomSettings,
    GravitarRomSettings, JourneyEscapeRomSettings, PrivateEyeRomSettings,
    SkiingRomSettings, UpNDownRomSettings, YarsRevengeRomSettings,
    AmidarRomSettings, SurroundRomSettings,
    CarnivalRomSettings, PooyanRomSettings,
    BattleZoneRomSettings, MsPacmanRomSettings,
    PacmanRomSettings, QbertRomSettings

# Task #95/#98 (2026-06-15): full per-ROM RomSettings map — MUST stay in sync
# with tools/jutari_trace_dump.jl. A game booted with the wrong settings (e.g.
# missing its `getStartingActions`) produces a settings-mismatch screen diff on
# top of any genuine render delta — so the screen scoreboard needs the exact
# same per-game boot the xitari `trace_dump --screen` reference uses.
const _SETTINGS_BY_BASENAME = Dict{String,Function}(
    "breakout.bin"        => () -> BreakoutRomSettings(),
    "pong.bin"            => () -> PongRomSettings(),
    "pitfall.bin"         => () -> PitfallRomSettings(),
    "enduro.bin"          => () -> EnduroRomSettings(),
    "air_raid.bin"        => () -> AirRaidRomSettings(),
    "asterix.bin"         => () -> AsterixRomSettings(),
    "beam_rider.bin"      => () -> BeamRiderRomSettings(),
    "double_dunk.bin"     => () -> DoubleDunkRomSettings(),
    "elevator_action.bin" => () -> ElevatorActionRomSettings(),
    "gopher.bin"          => () -> GopherRomSettings(),
    "gravitar.bin"        => () -> GravitarRomSettings(),
    "journey_escape.bin"  => () -> JourneyEscapeRomSettings(),
    "private_eye.bin"     => () -> PrivateEyeRomSettings(),
    "skiing.bin"          => () -> SkiingRomSettings(),
    "up_n_down.bin"       => () -> UpNDownRomSettings(),
    "yars_revenge.bin"    => () -> YarsRevengeRomSettings(),
    "amidar.bin"          => () -> AmidarRomSettings(),
    "surround.bin"        => () -> SurroundRomSettings(),
    "carnival.bin"        => () -> CarnivalRomSettings(),
    "pooyan.bin"          => () -> PooyanRomSettings(),
    "battle_zone.bin"     => () -> BattleZoneRomSettings(),
    "ms_pacman.bin"       => () -> MsPacmanRomSettings(),
    "pacman.bin"          => () -> PacmanRomSettings(),
    "qbert.bin"           => () -> QbertRomSettings(),
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
