# p2_stream_dump.jl — dump jutari's per-frame RAM + screen along the EXACT Paper-2
# shared-testbed action stream, for the jutari↔xitari stream validation
# (tools/xai_study/repro/p2_stream_check.py).
#
# The P2 testbed analyses every scored game at frame f* = prefix (90) and reads its
# outputs at f*+horizon (105), i.e. beyond Paper 1's 30/60-frame validated window.
# This tool replays that stream on the SAME code path the runners use
# (SaliencyAttr.load_env: StellaEnvironment(rom, settings_for(game)) +
# env_reset!(60 NOOP, 4 RESET), then env_step! per action) and writes, per game:
#   <out>/<game>.actions.txt   effective actions (after env_step!'s illegal→NOOP map)
#   <out>/<game>.jutari.bin    frame 0 (boot end) RAM, then per step: RAM(128) + screen(h×w, see meta)
#   <out>/<game>.meta.json     settings type, starting actions skipped vs the game's
#                              registered RomSettings, ROM path, frame count
#
#   julia --project=jutari tools/xai_study/repro/p2_stream_dump.jl --out DIR [--games scored]

using JSON
using JuTari
using JuTari.Env: env_step!, get_screen
using JuTari.RomSettingsModule: romsettings_is_legal_action, romsettings_starting_actions

include(joinpath(@__DIR__, "..", "phaseB_attribution", "saliency.jl"))
using .SaliencyAttr
const SA = SaliencyAttr

function dump_game(game::AbstractString, out::AbstractString)
    env = SA.load_env(; game = game)
    acts = SA.shared_gameplay_actions(; prefix = SA.ST_PREFIX, horizon = SA.ST_HORIZON,
                                       seed = SA.ST_SEED)
    eff = [romsettings_is_legal_action(env.settings, a) ? a : 0 for a in acts]
    open(joinpath(out, "$game.actions.txt"), "w") do io
        println(io, "# P2 shared-testbed stream for $game: seed=$(SA.ST_SEED) prefix=$(SA.ST_PREFIX) horizon=$(SA.ST_HORIZON) (effective actions)")
        foreach(a -> println(io, a), eff)
    end
    hw = size(get_screen(env))          # 210×160, or the game's own height (air_raid 250, …)
    open(joinpath(out, "$game.jutari.bin"), "w") do io
        write(io, UInt8.(env.console.bus.ram))
        for a in acts
            env_step!(env, a)
            write(io, UInt8.(env.console.bus.ram))
            scr = get_screen(env)
            size(scr) == hw || error("$game: screen size changed mid-stream")
            write(io, permutedims(scr))   # row-major h×w, like xitari
        end
    end
    open(joinpath(out, "$game.meta.json"), "w") do io
        JSON.print(io, Dict("game" => game, "settings" => string(typeof(env.settings)),
                            "rom" => SA.rom_path_for(game), "n_steps" => length(acts),
                            "screen_hw" => collect(hw),
                            "starting_actions_applied" => romsettings_starting_actions(env.settings)), 2)
    end
end

function main(args)
    out = args[findfirst(==("--out"), args) + 1]
    i = findfirst(==("--games"), args)
    games = SA.xai_resolve_games(i === nothing ? "scored" : args[i + 1], SA.CORE_GAMES)
    mkpath(out)
    for g in games
        t = @elapsed dump_game(g, out)
        println("[p2_stream_dump] $g  $(round(t, digits = 1)) s"); flush(stdout)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
