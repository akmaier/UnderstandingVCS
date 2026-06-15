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
using JuTari.RomSettingsModule: GenericRomSettings

# ROM basename → RomSettings constructor (mirror of
# tools/jutari_trace_dump.jl's `_settings_for_rom` — they MUST agree, or
# the video panel and the RAM/conformance trace render different games).
# Task #95 (2026-06-13): pitfall + enduro were MISSING here, so the
# pitfall video panel fell through to GenericRomSettings — which omits
# Pitfall's `getStartingActions` (1× PLAYER_A_UP). That single missing
# boot action desynced Harry's whole trajectory from xitari, producing
# the "jutari doesn't jump / jumps late" artifact the user saw in the
# video. The EMULATOR was correct (the RAM tool, which DOES map
# pitfall→PitfallRomSettings, is bit-exact with xitari for 330 frames);
# only this video-tool settings map was wrong. See bug_fix_log #95.
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
    # Per-ROM settings auto-selection by basename. For paddle ROMs
    # (breakout, pong) `romsettings_uses_paddles = true` makes
    # `StellaEnvironment` auto-translate LEFT/RIGHT actions into INPT0
    # dump-pot paddle-position changes — same shape as xitari's
    # `m_use_paddles` autodetection from stella.pro.
    env = StellaEnvironment(rom, _settings_for_rom(rom_path))
    env_reset!(env; boot_noop_steps = 60, boot_reset_steps = 4)

    actions = _load_actions(actions_path)
    n = min(max_frames, length(actions))

    # Pre-allocate the (n, 210, 160) buffer. Julia is column-major,
    # but the Python reader will reshape from a flat byte stream as
    # (n, 210, 160) so we just need to dump in the (n, 210, 160)
    # row-major order — explicit per-cell write below sidesteps the
    # layout question entirely.
    # Auto-reset matches xitari's trace_dump --auto-reset default: when
    # the game declares `gameOver=true` (e.g. breakout after losing all
    # 5 lives), call env_reset! so the next step starts a fresh episode.
    # Mirrors xitari/ale's resetGame() behavior — needed so the
    # comparison video doesn't freeze at game-over while xitari keeps
    # rendering a fresh game.
    open(out_path, "w") do io
        for i in 1:n
            if env.terminal
                # Re-do the boot burn to match xitari resetGame() —
                # see env_reset! signature in StellaEnvironment.jl.
                env_reset!(env; boot_noop_steps = 60, boot_reset_steps = 4)
            end
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
