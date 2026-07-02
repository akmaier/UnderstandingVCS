# game_sets.jl — the ONE canonical game-set + ROM-root resolver shared by every
# Phase-A/B/C runner (P2). Includable FRAGMENT (not a module), same style as
# common/shared_testbed_impl.jl, so a runner just `include`s it and gets:
#
#   * XAI_LABELED          — the T3-LABELED game set: the 54 games covered by
#                            OCAtari (offset-corrected, render-aligned RAM position
#                            labels), a strict superset of the 22 AtariARI games.
#                            The position-regime study needs verified render-aligned
#                            position labels, so games with NO T3 label can never
#                            contribute — the 10 unlabeled games are dropped. The
#                            machine-readable twin is game_set.json's `labeled_set`.
#   * xai_resolve_games(arg, core)  — the UNIFORM --games expander: "core" → core
#                            set, "labeled" → XAI_LABELED, else a comma list.
#   * xai_rom_roots()      — the ordered ROM search roots: this worktree's
#                            xitari/roms, the primary repo's xitari/roms, and the
#                            canonical ROM store tools/rom_sweep/roms (ALE names),
#                            plus the cluster ROM collection. A runner's rom_path_for
#                            should search these roots (trying both the game's mapped
#                            basename AND its raw ALE name).
#
# WHY here: game_set.json already fixes `core_set`; the broader labeled pool was
# only described in prose. This freezes it as a concrete list the Julia runners
# consume, so a SINGLE invocation form — `--games labeled` — works identically
# across A/B/C. Default stays core (callers pass their own core set through).
#
# The 54 names are ALE-canonical (== tools/rom_sweep/roms/<name>.bin), the store
# loader.resolve_rom() and the cluster both use.

# ---- the 54 T3-labeled games (OCAtari-covered; ALE-canonical names) ----
# EXCLUDED (10 unlabeled): defender elevator_action gravitar journey_escape solaris
# surround tutankham videochess wizard_of_wor zaxxon.
const XAI_LABELED = [
    "air_raid", "alien", "amidar", "assault", "asterix", "asteroids", "atlantis",
    "bank_heist", "battle_zone", "beam_rider", "berzerk", "bowling", "boxing",
    "breakout", "carnival", "centipede", "chopper_command", "crazy_climber",
    "demon_attack", "double_dunk", "enduro", "fishing_derby", "freeway",
    "frostbite", "gopher", "hero", "ice_hockey", "jamesbond", "kangaroo", "krull",
    "kung_fu_master", "montezuma_revenge", "ms_pacman", "name_this_game", "pacman",
    "phoenix", "pitfall", "pong", "pooyan", "private_eye", "qbert", "riverraid",
    "road_runner", "robotank", "seaquest", "skiing", "space_invaders",
    "star_gunner", "tennis", "time_pilot", "up_n_down", "venture", "video_pinball",
    "yars_revenge",
]

# The SCORED battery: the 42 games that both (a) pass the cause-density gate at the
# shared analysis state (seed=0, prefix=90 — >=4 causes with |Δy|>0.5) AND (b) carry
# >=1 causally-verified label that moves a sprite. Every scored game therefore
# contributes to BOTH the all-regime and the position regime. The 12 games excluded
# from XAI_LABELED are documented on the Ground Truth ROMs page and Supplement S3:
#   gate-rejected (too few strong causes): crazy_climber, enduro, pooyan, skiing,
#       time_pilot  (+ asteroids, battle_zone, star_gunner, which are also static);
#   no moving sprite (static at the shared state): amidar, asterix, robotank, up_n_down.
# Derived deterministically from the verified T3 files + the cause-density census.
const XAI_SCORED = [
    "air_raid", "alien", "assault", "atlantis", "bank_heist", "beam_rider", "berzerk",
    "bowling", "boxing", "breakout", "carnival", "centipede", "chopper_command",
    "demon_attack", "double_dunk", "fishing_derby", "freeway", "frostbite", "gopher",
    "hero", "ice_hockey", "jamesbond", "kangaroo", "krull", "kung_fu_master",
    "montezuma_revenge", "ms_pacman", "name_this_game", "pacman", "phoenix", "pitfall",
    "pong", "private_eye", "qbert", "riverraid", "road_runner", "seaquest",
    "space_invaders", "tennis", "venture", "video_pinball", "yars_revenge",
]

"""
    xai_resolve_games(arg, core) -> Vector{String}

The UNIFORM `--games` expander shared by every A/B/C runner:
  * "core"     → `core` (the caller's 6-game core set; default unchanged)
  * "labeled"  → XAI_LABELED (the 54 T3-labeled / OCAtari-covered games)
  * "scored"   → XAI_SCORED (the 42 non-degenerate, moving-sprite games; the battery scope)
  * anything else → a comma-separated explicit list, lowercased+stripped.
"""
function xai_resolve_games(arg::AbstractString, core::AbstractVector{<:AbstractString})
    v = lowercase(strip(String(arg)))
    v == "core"    && return collect(String.(core))
    v == "labeled" && return copy(XAI_LABELED)
    v == "scored"  && return copy(XAI_SCORED)
    return String[strip(String(g)) for g in split(String(arg), ",") if !isempty(strip(String(g)))]
end

"""
    xai_rom_roots(; primary_repo, extra) -> Vector{String}

The ordered ROM search roots every runner's `rom_path_for` should scan:
this worktree's `xitari/roms`, the primary repo's `xitari/roms`, the canonical
ROM store `tools/rom_sweep/roms` (ALE-canonical names — where all games live and
where `loader.resolve_rom` and the cluster look), and the raw cluster ROM
collection. Non-existent roots are harmless (the caller `isfile`-checks each path).
"""
function xai_rom_roots(; primary_repo::AbstractString =
                          get(ENV, "XAI_PRIMARY_REPO",
                              "/Users/maier/Documents/code/UnderstandingVCS"),
                       extra::AbstractVector{<:AbstractString} = String[])
    here = normpath(joinpath(@__DIR__, "..", "..", ".."))
    roots = String[]
    for e in extra; push!(roots, String(e)); end
    for base in (here, primary_repo)
        push!(roots, joinpath(base, "xitari", "roms"))
        push!(roots, joinpath(base, "tools", "rom_sweep", "roms"))
        push!(roots, joinpath(base, "xitari", "games",
                              "Atari-2600-VCS-ROM-Collection", "ROMS"))
    end
    return roots
end

"""
    xai_find_rom(names, roots) -> String

Return the first existing `<root>/<name>.bin` over the given candidate `names`
(e.g. the ROM_BASENAME-mapped stem AND the raw ALE name) × `roots`. Errors if none.
"""
function xai_find_rom(names::AbstractVector{<:AbstractString},
                      roots::AbstractVector{<:AbstractString})
    for base in roots, nm in names
        p = joinpath(base, String(nm) * ".bin")
        isfile(p) && return p
    end
    error("ROM not found: tried names=$(collect(names)) under roots=$(collect(roots))")
end
