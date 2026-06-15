#!/usr/bin/env julia
# render_diff.jl — jutari side of the per-scanline render diff harness.
#
# Boots a ROM exactly like jutari_screen_dump.jl (same per-game RomSettings +
# 60-NOOP/4-RESET boot), steps to a target frame, and snapshots jutari's FULL
# render state for one SCREEN ROW: the row→scanline mapping is done HERE from the
# env's actual `y_start_row` (target scanline = y_start_row + row), which is the
# whole point — by-hand mapping is what made earlier inline probes unreliable.
#
# Output: one JSON line on stdout for the orchestrator (tools/render_diff.py):
#   {"y_start":N,"target_scanline":N,"row":R,
#    "regs":"<128 hex>","p0_x":..,"p1_x":..,"m0_x":..,"m1_x":..,"bl_x":..,
#    "pf":[..],"p0":[..],"p1":[..],"m0":[..],"m1":[..],"bl":[..],
#    "screen_row":[160 ints],"render_row":[160 ints]}
#
# Usage:
#   julia --project=jutari tools/render_diff.jl \
#       --rom tools/rom_sweep/roms/tutankham.bin \
#       --actions tools/breakout_video/output/breakout_random_actions.txt \
#       --frame 0 --row 103
using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "jutari"))

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
import JuTari.TIA

# MUST stay in sync with tools/jutari_screen_dump.jl / jutari_trace_dump.jl.
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

_ints(v) = "[" * join(string.(Int.(v)), ",") * "]"

function main()
    args = Dict{String,String}()
    i = 1
    while i <= length(ARGS); args[ARGS[i]] = ARGS[i+1]; i += 2; end
    rom_path = args["--rom"]; actions_path = args["--actions"]
    frame = parse(Int, args["--frame"]); row = parse(Int, args["--row"])

    rom = read(rom_path)
    actions = Int[]
    for line in eachline(actions_path)
        s = strip(line); (isempty(s) || startswith(s, "#")) && continue
        push!(actions, parse(Int, s))
    end

    env = StellaEnvironment(rom, _settings_for_rom(rom_path))
    env_reset!(env; boot_noop_steps = 60, boot_reset_steps = 4)
    ystart = Int(env.console.bus.tia.y_start_row)
    target = ystart + row
    TIA._RENDER_PROBE[] = target
    TIA._RENDER_PROBE_OUT[] = nothing
    for k in 1:(frame + 1)
        act = k <= length(actions) ? actions[k] : (isempty(actions) ? 0 : actions[end])
        env_step!(env, act)
    end
    scr = get_screen(env)
    screen_row = collect(scr[row + 1, :])
    p = TIA._RENDER_PROBE_OUT[]
    TIA._RENDER_PROBE[] = -1   # disarm

    regs_hex = p === nothing ? "" :
        join([string(b; base = 16, pad = 2) for b in p.regs[1:64]])
    io = IOBuffer()
    print(io, "{\"y_start\":", ystart, ",\"target_scanline\":", target,
              ",\"row\":", row, ",\"screen_row\":", _ints(screen_row))
    if p === nothing
        print(io, ",\"probe\":null}")
    else
        print(io, ",\"regs\":\"", regs_hex, "\"",
                  ",\"p0_x\":", p.p0_x, ",\"p1_x\":", p.p1_x,
                  ",\"m0_x\":", p.m0_x, ",\"m1_x\":", p.m1_x, ",\"bl_x\":", p.bl_x,
                  ",\"pf\":", _ints(p.pf), ",\"p0\":", _ints(p.p0),
                  ",\"p1\":", _ints(p.p1), ",\"m0\":", _ints(p.m0),
                  ",\"m1\":", _ints(p.m1), ",\"bl\":", _ints(p.bl),
                  ",\"render_row\":", _ints(p.row),
                  ",\"pending\":[",
                  join(["[$(a),$(r),$(v)]" for (a, r, v) in p.pending], ","), "]}")
    end
    println(String(take!(io)))
    return 0
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main())
end
