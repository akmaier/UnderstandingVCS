#!/usr/bin/env julia
# obj_trace.jl — per-scanline object-position + GRP-shadow trace for jutari.
#
# Boots a ROM exactly like render_diff.jl / jutari_screen_dump.jl, steps to a
# target frame with `TIA._OBJ_TRACE[]` on, and prints one CSV row per completed
# visible scanline of that frame:
#   scanline,p0_x,p1_x,m0_x,m1_x,bl_x,grp0_old,grp1_old,m0_cosmic_line
# Use it to see WHERE a carried object position / GRP shadow diverges from xitari
# (compare against xitari's XI_POKE_DUMP myPOS*/myDGRP*). The render-only
# residuals (robotank ball, elevator_action missile, up_n_down VDELP shadow) set
# the wrong value on an earlier scanline and carry it.
#
# Usage:
#   julia --project=jutari tools/obj_trace.jl --rom tools/rom_sweep/roms/robotank.bin \
#       --actions tools/breakout_video/output/breakout_random_actions.txt --frame 1
using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "jutari"))

using JuTari
using JuTari.Env: StellaEnvironment, env_reset!, env_step!
using JuTari.RomSettingsModule: GenericRomSettings
using JuTari.PaddleGames: BreakoutRomSettings, PongRomSettings
using JuTari.JoystickGames: PitfallRomSettings, EnduroRomSettings,
    AirRaidRomSettings, AsterixRomSettings, BeamRiderRomSettings,
    DoubleDunkRomSettings, ElevatorActionRomSettings, GopherRomSettings,
    GravitarRomSettings, JourneyEscapeRomSettings, PrivateEyeRomSettings,
    SkiingRomSettings, UpNDownRomSettings, YarsRevengeRomSettings,
    AmidarRomSettings, SurroundRomSettings, CarnivalRomSettings, PooyanRomSettings,
    BattleZoneRomSettings, MsPacmanRomSettings, PacmanRomSettings, QbertRomSettings
import JuTari.TIA

const _SETTINGS = Dict{String,Function}(
    "breakout.bin" => () -> BreakoutRomSettings(), "pong.bin" => () -> PongRomSettings(),
    "pitfall.bin" => () -> PitfallRomSettings(), "enduro.bin" => () -> EnduroRomSettings(),
    "air_raid.bin" => () -> AirRaidRomSettings(), "asterix.bin" => () -> AsterixRomSettings(),
    "beam_rider.bin" => () -> BeamRiderRomSettings(), "double_dunk.bin" => () -> DoubleDunkRomSettings(),
    "elevator_action.bin" => () -> ElevatorActionRomSettings(), "gopher.bin" => () -> GopherRomSettings(),
    "gravitar.bin" => () -> GravitarRomSettings(), "journey_escape.bin" => () -> JourneyEscapeRomSettings(),
    "private_eye.bin" => () -> PrivateEyeRomSettings(), "skiing.bin" => () -> SkiingRomSettings(),
    "up_n_down.bin" => () -> UpNDownRomSettings(), "yars_revenge.bin" => () -> YarsRevengeRomSettings(),
    "amidar.bin" => () -> AmidarRomSettings(), "surround.bin" => () -> SurroundRomSettings(),
    "carnival.bin" => () -> CarnivalRomSettings(), "pooyan.bin" => () -> PooyanRomSettings(),
    "battle_zone.bin" => () -> BattleZoneRomSettings(), "ms_pacman.bin" => () -> MsPacmanRomSettings(),
    "pacman.bin" => () -> PacmanRomSettings(), "qbert.bin" => () -> QbertRomSettings(),
)
_settings(p) = haskey(_SETTINGS, basename(p)) ? _SETTINGS[basename(p)]() : GenericRomSettings()

function main()
    args = Dict{String,String}(); i = 1
    while i <= length(ARGS); args[ARGS[i]] = ARGS[i+1]; i += 2; end
    rom = read(args["--rom"])
    actions = Int[]
    for line in eachline(args["--actions"])
        s = strip(line); (isempty(s) || startswith(s, "#")) && continue
        push!(actions, parse(Int, s))
    end
    frame = parse(Int, args["--frame"])
    env = StellaEnvironment(rom, _settings(args["--rom"]))
    env_reset!(env; boot_noop_steps = 60, boot_reset_steps = 4)
    # Run up to (but not into) the target frame with the trace OFF, then capture
    # only the target frame.
    for k in 1:frame
        act = k <= length(actions) ? actions[k] : (isempty(actions) ? 0 : actions[end])
        env_step!(env, act)
    end
    pending = haskey(args, "--pending") ? parse(Int, args["--pending"]) : -1
    empty!(TIA._OBJ_TRACE_LOG); TIA._OBJ_TRACE[] = true
    empty!(TIA._PEND_PROBE_LOG); TIA._PEND_PROBE_SL[] = pending
    act = (frame + 1) <= length(actions) ? actions[frame + 1] : (isempty(actions) ? 0 : actions[end])
    env_step!(env, act)
    TIA._OBJ_TRACE[] = false
    TIA._PEND_PROBE_SL[] = -1
    println("scanline,p0_x,p1_x,m0_x,m1_x,bl_x,grp0_old,grp1_old,m0_cosmic_line,grp0_live,grp1_live,vdelp0,vdelp1")
    for r in TIA._OBJ_TRACE_LOG
        println(join(r, ","))
    end
    if pending >= 0
        println("# PENDING_WRITES (activation_clock, reg, value) for scanline ", pending, ":")
        for (sl, writes) in TIA._PEND_PROBE_LOG
            ws = join(["($(a),$(string(r, base=16)),$(string(v, base=16)))" for (a, r, v) in writes], " ")
            println("# sl", sl, ": ", ws)
        end
    end
    return 0
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main())
end
