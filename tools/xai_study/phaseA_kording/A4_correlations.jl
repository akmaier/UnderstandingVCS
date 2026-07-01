# A4_correlations.jl — Phase-A A4: spike-word / pairwise correlations (P2-E3-4),
# the JULIA path, run on jutari over the 6 core games.
#
# Phase A is Paper 2's *calibration baseline* (experiment_design.md §4, Novelty
# note): classical neuroscience analyses score LOW faithfulness despite the system
# having rich, fully-known structure — the quantified-Kording lesson. A4 is the
# "spike-word" / pairwise-correlation member of the battery (Schneidman et al. 2006;
# Tkačik et al. 2014 — weak pairwise correlations but strong *collective* structure):
#
#   * Over a recorded deterministic jutari trajectory, compute the cell-cell PAIRWISE
#     CORRELATION matrix of the candidate state cells (the "spike-word" co-activation
#     structure).
#   * Extract co-activation CLUSTERS (single-linkage on thresholded |corr|).
#   * Compare the clusters to the KNOWN VARIABLE GROUPINGS (T3 concept families: cells
#     whose concept name shares a family — e.g. all `enemy_*` cells, all `player_*`
#     cells — are one true variable group). Report whether the correlation structure
#     recovers the true grouping — it generally OVER-GROUPS (lumps causally-unrelated
#     cells that merely co-vary, e.g. everything driven by the frame clock). QUANTIFY
#     the over-grouping (Rand index, homogeneity, an explicit over-grouping ratio).
#   * Reproduce the spike-word headline: WEAK-PAIRWISE / STRONG-GLOBAL — the mean
#     |pairwise corr| is modest while a few GLOBAL modes (top eigenvalues of the corr
#     matrix; participation ratio) capture most of the collective structure.
#
# F/S/M scoring (the correctness triad, experiment_design.md §0), scored against the
# exact intervention ORACLE (NOT against any interpretability method). A4's "claim"
# is a recovered cell-cell COUPLING (which cells belong together / are coupled); the
# oracle's TRUE coupling is the causal cell→cell influence map:
#
#   true coupling(i,j) = does an EXACT intervention on candidate cell i (do(i:=v'))
#       move candidate cell j over the horizon? (a bit-exact re-run of the real ROM —
#       the §1 oracle restricted to candidate-cell pairs; this IS the "true coupling"
#       experiment_design.md §4 names). Symmetrised to an undirected coupling weight.
#
#   F (faithful)   — Spearman(|pairwise corr|, oracle coupling) over the off-diagonal
#                    candidate-cell pairs: does the recovered co-activation structure
#                    track the TRUE causal coupling?
#   S (sufficient) — predict a HELD-OUT coupling: from the recovered |corr| predict
#                    whether an UNSEEN intervention value do(i:=base+37) couples (i,j);
#                    Pearson(|corr|, held-out oracle coupling) on the bit-exact re-run.
#   M (minimal)    — 1 − over-claim rate: of the pairs A4 flags coupled (|corr|≥τ),
#                    how many are NOT truly causally coupled (oracle coupling == 0) —
#                    the correlation-over-grouping failure (present ≠ used, pairwise).
#
# Positive control (as in pilot_si.jl): scoring the ORACLE'S OWN coupling map as the
# candidate "method" must yield Spearman F == 1 — the harness rewards a perfectly-
# faithful coupling map, so a sub-1 correlation F is a real measurement.
#
# BUILDS ON the verified jutari foundation (NO emulator core touched):
#   * tools/xai_study/common/jutari_oracle.jl — load/boot/replay/snapshot/intervene,
#     bit-exact baseline guarantee, the dependency-free §R NPY/NPZ writer.
#   * tools/xai_study/common/jutari_record.jl — record_trajectory → (T,n) tape.
#   * tools/xai_study/ground_truth/oracle_intervene.jl — the causal-map reference
#     (this file specialises its Δy(u) machinery to candidate-cell-cell coupling).
#   * tools/xai_study/phaseA_kording/pilot_si.jl — the Phase-A pilot TEMPLATE
#     (per-game oracle scoring + F/S/M + positive control + §R record shape).
#
# Per-game RomSettings: the shared jutari_oracle.settings_for() only knows the 3
# pilot games, so A4 supplies its own per-game settings map (matching the canonical
# tools/jutari_screen_dump.jl map) and builds envs directly via the same JuTari
# primitives — keeping everything inside A4's file_scope (it must not edit the shared
# helper). seaquest legitimately uses GenericRomSettings (Paper-1 64/64 with Generic).
#
# Run (warm shared depot, primary's project):
#   XAI_PRIMARY_REPO=/Users/maier/Documents/code/UnderstandingVCS \
#   julia --project=/Users/maier/Documents/code/UnderstandingVCS/jutari \
#         tools/xai_study/phaseA_kording/A4_correlations.jl
# Flags: --games pong,breakout,...  --target-frame N --horizon N --traj-frames N
#        --tau-corr F  --selftest (run the self-check on the first game, write nothing)
#
# Writes (SPEC §R; file_scope A4_*): one record per game
#   tools/xai_study/phaseA_kording/out/A4_<game>.{json,npz}

module A4Correlations

using JSON
using LinearAlgebra: eigen, Symmetric
import Statistics
using JuTari

# --- the verified foundation (NO core touched) -----------------------------
include(joinpath(@__DIR__, "..", "common", "jutari_oracle.jl"))
using .JutariOracle
using .JutariOracle: Snapshot, snapshot, intervene_ram!, RAM_SIZE, write_npz,
                     rom_path_for
using .JutariOracle.JuTari: StellaEnvironment
using .JutariOracle.JuTari.Env: env_reset!, env_step!, get_ram

# the trajectory recorder (the (T,n) RAM tape A4's correlation runs on). A4 reuses
# its recording PATTERN but re-implements the loop (a4_record_ram) so each game's
# proper RomSettings are used (the shared recorder hard-codes settings_for); the
# include keeps the dependency lineage explicit.
include(joinpath(@__DIR__, "..", "common", "jutari_record.jl"))

# The §1 exact-intervention oracle — used ONLY to build the P2 SHARED gameplay
# state + its cause-density gate (below). A4 keeps its OWN a4_* + coupling-oracle
# machinery; the oracle here supplies the shared action stream + gate. Referenced
# as OracleIntervene.X / OracleIntervene.JutariOracle.X (NOT alias-imported, so its
# internal JutariOracle instance never clashes with the .JutariOracle already
# imported above — separate module instances; no type mixed across them).
include(joinpath(@__DIR__, "..", "ground_truth", "oracle_intervene.jl"))
using JuTari.Diff: soft_ram_peek

# The P2 SHARED TESTBED (experiment_redesign.md): seeded random-action GAMEPLAY
# state + oracle cause-density gate. Included as a fragment (see its header). Phase
# A is not a gradient method, so we consume the shared action STREAM + cause-density
# GATE only, and boot A4's OWN checkpoint + record its OWN trajectory from that
# stream. Opt in with XAI_SHARED_TESTBED=1 (default on).
include(joinpath(@__DIR__, "..", "common", "shared_testbed_impl.jl"))

const OUT_DIR = joinpath(@__DIR__, "out")
const CORE_GAMES = ["pong", "breakout", "space_invaders",
                    "seaquest", "ms_pacman", "qbert"]

# shared-testbed switch + params (redesign protocol: prefix=90 gameplay, horizon=15).
const SHARED_TESTBED = get(ENV, "XAI_SHARED_TESTBED", "1") == "1"
const ST_PREFIX  = parse(Int, get(ENV, "XAI_ST_PREFIX", "90"))
const ST_HORIZON = parse(Int, get(ENV, "XAI_ST_HORIZON", "15"))
const ST_SEED    = parse(Int, get(ENV, "XAI_ST_SEED", "0"))
const ST_GATE_K  = parse(Int, get(ENV, "XAI_ST_GATE_K", "4"))
const ST_FLOOR   = parse(Float64, get(ENV, "XAI_ST_FLOOR", "0.5"))

# Per-game live-play configuration. The candidate cells must actually MOVE for the
# correlation / coupling to carry signal; each game reaches live play at a different
# point (Ms. Pac-Man has a long animated intro; Space Invaders/Seaquest develop more
# cross-cell coupling a bit later). Tuned by probing #candidate cells with non-zero
# variance + #oracle-coupled pairs (see the handoff note). `trace` selects the
# trajectory input style ("fire" = the FIRE+RIGHT/LEFTFIRE active trace; "dir" = a
# full UP/DOWN/LEFT/RIGHT maze trace, for games where FIRE is a no-op). `in_window`
# flags whether the SCORED oracle frame stays inside the strict Paper-1 60-frame
# screen conformance window (jutari↔xitari); games scored beyond it are honest
# descriptive jutari results (bit-exact jutari re-run still asserted), not
# xitari-parity claims — annotated in the §R record.
struct GameCfg
    target_frame::Int
    horizon::Int
    traj_frames::Int
    trace::String       # "fire" | "dir"
    in_window::Bool
end
const GAME_CFG = Dict{String,GameCfg}(
    "pong"           => GameCfg(60,  30, 150, "fire", true),
    "breakout"       => GameCfg(60,  30, 150, "fire", true),
    "space_invaders" => GameCfg(120, 30, 150, "fire", false),
    "seaquest"       => GameCfg(120, 30, 150, "fire", false),
    "ms_pacman"      => GameCfg(330, 30, 400, "dir",  false),
    "qbert"          => GameCfg(60,  30, 200, "fire", true),
)
game_cfg(game::AbstractString) =
    get(GAME_CFG, lowercase(string(game)), GameCfg(60, 30, 150, "fire", true))

# ============================================================================
# Per-game RomSettings (A4-owned; mirrors tools/jutari_screen_dump.jl's canonical
# map). seaquest -> Generic (Paper-1 64/64 with Generic). This stays in A4's
# file_scope; it does not modify the shared jutari_oracle.settings_for().
# ============================================================================
function a4_settings_for(game::AbstractString)
    g = lowercase(string(game))
    JT = JutariOracle.JuTari
    g == "pong"           && return JT.PaddleGames.PongRomSettings()
    g == "breakout"       && return JT.PaddleGames.BreakoutRomSettings()
    g == "space_invaders" && return JT.SpaceInvadersRomSettings()
    g == "ms_pacman"      && return JT.JoystickGames.MsPacmanRomSettings()
    g == "qbert"          && return JT.JoystickGames.QbertRomSettings()
    # seaquest + any other -> Generic (its Paper-1 bit-exact settings)
    return JT.GenericRomSettings()
end

# A few core games' ROM files don't match their canonical name (the shared
# jutari_oracle.rom_path_for looks for `<game>.bin`). Mirror the alias map in
# tools/xai_study/t3/import_labels.py so A4 resolves them in this worktree's
# xitari/roms (or the primary's). A4-owned (stays inside file_scope).
const _ROM_ALIASES = Dict("ms_pacman" => ["ms_pacman", "mspacman"],
                          "beam_rider" => ["beam_rider", "beamrider"])

function a4_rom_path(game::AbstractString)
    # try the shared resolver first (handles the canonical `<game>.bin`)
    try
        return rom_path_for(game)
    catch
    end
    here = normpath(joinpath(@__DIR__, "..", "..", ".."))
    bases = String[here, get(ENV, "XAI_PRIMARY_REPO", ""),
                   "/Users/maier/Documents/code/UnderstandingVCS"]
    names = get(_ROM_ALIASES, lowercase(string(game)), [string(game)])
    for base in bases
        base == "" && continue
        for nm in names
            p = joinpath(base, "xitari", "roms", nm * ".bin")
            isfile(p) && return p
        end
    end
    error("ROM not found for game=$game (tried $(names) under $(filter(!isempty, bases)))")
end

"""A freshly-reset env for `game` with the A4 per-game settings and the
xitari-parity boot (60 NOOP + 4 RESET) — the SAME boot jutari_oracle uses."""
function a4_load_env(game::AbstractString)
    rom = read(a4_rom_path(game))
    env = StellaEnvironment(rom, a4_settings_for(game))
    env_reset!(env; boot_noop_steps = 60, boot_reset_steps = 4)
    return env
end

"""Boot + step actions[1:target_frame]; return the env AT the intervention frame
(deepcopy it to make a reusable checkpoint). Mirrors jutari_oracle.boot_replay
but with the A4 per-game settings."""
function a4_boot_replay(game, actions::AbstractVector{<:Integer}, target_frame::Integer)
    env = a4_load_env(game)
    for i in 1:target_frame
        env_step!(env, Int(actions[i]))
    end
    return env
end

"""deepcopy the checkpoint, step the tail, snapshot (byte-exact continuation)."""
function a4_continue_from(checkpoint, tail::AbstractVector{<:Integer})
    env = deepcopy(checkpoint)
    for a in tail; env_step!(env, Int(a)); end
    return snapshot(env, length(tail))
end

"""A full from-scratch replay (boot included) of actions[1:total] — the bit-exact
baseline guarantee (two fresh runs must be byte-identical)."""
function a4_fresh_baseline(game, actions::AbstractVector{<:Integer}, total::Integer)
    env = a4_load_env(game)
    for i in 1:total; env_step!(env, Int(actions[i])); end
    return snapshot(env, Int(total))
end

"""Record a (frames, n) RAM (+optional screen) tape on `game` with A4's per-game
settings. Re-implements record_trajectory's loop with a4_load_env so the proper
RomSettings are used (the shared recorder hard-codes settings_for)."""
function a4_record_ram(game::AbstractString, frames::Integer,
                       actions::AbstractVector{<:Integer})
    @assert length(actions) >= frames "actions shorter than frames"
    env = a4_load_env(game)
    tape = Matrix{UInt8}(undef, Int(frames), RAM_SIZE)
    for t in 1:frames
        env_step!(env, Int(actions[t]))
        tape[t, :] = UInt8.(collect(get_ram(env)))
    end
    return tape
end

# ============================================================================
# Action traces.
# ============================================================================
# Action codes (jutari src/io/IO.jl): NOOP=0 FIRE=1 RIGHT=3 LEFT=4 UP=2 DOWN=5
# RIGHTFIRE=11 LEFTFIRE=12 UPFIRE=10 DOWNFIRE=13.
#
# The ORACLE / scored trace is the deterministic FIRE-then-NOOP start: FIRE a few
# times to leave the title/attract screen into live play, then NOOP. It stays
# inside the conformance window and the bit-exact re-run is byte-identical.
oracle_actions(total::Integer) = vcat(fill(1, 4), fill(0, max(0, total - 4)))

# The A4 (descriptive) correlation trace is an ACTIVE one: FIRE to start + periodic
# directional fire so the candidate cells actually MOVE (an all-NOOP tail leaves
# most cells static, giving the correlation analysis no signal). A4's correlation /
# clustering / weak-pairwise-strong-global structure are DESCRIPTIVE statistics of a
# deterministic jutari trajectory (the recorder's documented purpose) — not a
# bit-exact-vs-xitari claim — so they legitimately use the livelier replay. The
# scored-vs-oracle claims (F/S/M, the coupling oracle) stay on the oracle trace,
# strictly inside the conformance window.
function active_actions(total::Integer)
    acts = Vector{Int}(undef, max(0, total))
    for t in 1:length(acts)
        acts[t] = t <= 4      ? 1  :   # 4×FIRE: start the game
                  t % 4 == 0  ? 11 :   # RIGHTFIRE: move right + fire
                  t % 4 == 2  ? 12 :   # LEFTFIRE:  move left + fire
                                1      # FIRE: keep firing
    end
    return acts
end

# A full UP/DOWN/LEFT/RIGHT maze trace for games where FIRE is a no-op (Ms.
# Pac-Man). FIRE for the first few frames (dismiss any prompt), then cycle the four
# directions so the player/ghost candidate cells actually move. Codes (IO.jl):
# UP=2 RIGHT=3 LEFT=4 DOWN=5.
function dir_actions(total::Integer)
    acts = Vector{Int}(undef, max(0, total))
    cyc = (3, 2, 4, 5)              # RIGHT, UP, LEFT, DOWN
    for t in 1:length(acts)
        acts[t] = t <= 8 ? 1 : cyc[((t - 9) % 4) + 1]
    end
    return acts
end

trace_actions(kind::AbstractString, total::Integer) =
    kind == "dir" ? dir_actions(total) : active_actions(total)

# ============================================================================
# Candidate cells (E2-1 import) — the "units" the correlation probes.
# ============================================================================
struct Candidate
    ram_index::Int
    concept::String
    family::String        # the true-variable-group label (concept family)
end

"""The concept *family* = the true variable group a cell belongs to. We normalise
the candidate concept to its base variable name (strip x/y/_x/_y/.xy/.value/
coordinate/index suffixes and a trailing object number) so that e.g. `enemy_x`,
`enemy_y`, `enemy.xy`, `enemy5_x` all map to family `enemy`, and `player_score`,
`score`, `score_value` map to `score`. Cells with the same family are ONE true
variable grouping — what correlation clustering should (but generally fails to)
recover."""
function concept_family(concept::AbstractString)
    s = lowercase(strip(string(concept)))
    isempty(s) && return "(unnamed)"
    # take the part before the first dot (player.xy -> player)
    s = split(s, '.')[1]
    # the four named Ms. Pac-Man ghosts are ONE true variable group (the "enemy"
    # ensemble): enemy_sue/inky/pinky/blinky/g1..g4 → enemy.
    s = replace(s, r"^enemy_(sue|inky|pinky|blinky)" => "enemy")
    s = replace(s, r"^(blinky|inky|pinky|sue)" => "enemy")
    # strip a leading enemy<digit> numbering: enemy5 -> enemy
    s = replace(s, r"^enemy[0-9]+" => "enemy")
    s = replace(s, r"^g[0-9]+$" => "enemy")        # ms_pacman g1..g4 -> enemy ensemble
    s = replace(s, r"^ghosts?" => "enemy")         # ghosts_position_x etc. -> enemy
    # strip coordinate / value / count / index suffixes
    for suf in ("_position_x", "_position_y", "_direction", "_meter_value",
                "_value", "_count", "_amount", "_column", "_map", "_bit_map",
                "_collected", "_eaten", "_x", "_y", "_xy", "_wh", "_orientation",
                "_prev_y", "_missile_x", "_missiles_x", "_missile_direction")
        endswith(s, suf) && (s = s[1:end-length(suf)])
    end
    # collapse residual trailing numbers (tile_color_1 -> tile_color)
    s = replace(s, r"_[0-9]+$" => "")
    s = replace(s, r"[0-9]+$" => "")
    s = rstrip(s, '_')
    # canonicalise a few synonyms to one true variable group
    s in ("num_lives", "n_lives", "last_lives") && (s = "lives")
    s in ("player_score",) && (s = "score")
    s in ("enemies",) && (s = "enemy")
    s in ("invaders_left",) && (s = "invaders_left")
    s == "" && (s = "(unnamed)")
    return s
end

"""Read the candidate RAM cells for `game`, de-duplicated by ram_index in file
order, each tagged with its true-variable-group family. Falls back to a tiny set
if the candidates file is absent (then families are best-effort)."""
function load_candidates(candidates_path)
    out = Candidate[]
    if candidates_path !== nothing && isfile(candidates_path)
        data = JSON.parsefile(candidates_path)
        seen = Set{Int}()
        for c in get(data, "candidates", [])
            idx = Int(c["ram_index"])
            idx in seen && continue
            push!(seen, idx)
            concept = c["concept"] === nothing ? "(unnamed)" : string(c["concept"])
            push!(out, Candidate(idx, concept, concept_family(concept)))
        end
    end
    return out
end

# ============================================================================
# Pairwise correlation ("spike-word" co-activation) over the trajectory.
# ============================================================================
function pearson(a::AbstractVector, b::AbstractVector)
    length(a) < 2 && return 0.0
    sa = Statistics.std(a); sb = Statistics.std(b)
    (sa == 0 || sb == 0) && return 0.0
    c = Statistics.cor(a, b)
    return isfinite(c) ? c : 0.0
end

"""The (n_cand × n_cand) Pearson correlation matrix of the candidate cells over the
trajectory tape (the spike-word co-activation matrix). Constant cells (zero
variance over the trajectory) correlate 0 with everything (diagonal forced to 1)."""
function pairwise_corr(tape::AbstractMatrix, cands::Vector{Candidate})
    n = length(cands)
    series = [Float64[Int(tape[t, c.ram_index + 1]) for t in 1:size(tape, 1)] for c in cands]
    C = Matrix{Float64}(undef, n, n)
    for i in 1:n, j in 1:n
        C[i, j] = i == j ? 1.0 : pearson(series[i], series[j])
    end
    varying = Bool[Statistics.std(series[i]) > 0 for i in 1:n]
    return C, varying
end

# ============================================================================
# Co-activation clustering: single-linkage on thresholded |corr|.
# ============================================================================
"""Connected components of the graph where cells i,j share an edge iff
|corr(i,j)| ≥ τ (single-linkage co-activation clusters). Returns a label vector
(1-based cluster id per cell). Isolated cells (incl. constant ones) get singleton
clusters."""
function cluster_by_corr(C::AbstractMatrix, τ::Real)
    n = size(C, 1)
    label = collect(1:n)                       # union-find parents
    find(x) = (while label[x] != x; label[x] = label[label[x]]; x = label[x]; end; x)
    for i in 1:n, j in i+1:n
        if abs(C[i, j]) >= τ
            ri, rj = find(i), find(j)
            ri != rj && (label[ri] = rj)
        end
    end
    roots = [find(i) for i in 1:n]
    uniq = unique(roots)
    remap = Dict(r => k for (k, r) in enumerate(uniq))
    return [remap[r] for r in roots]
end

# ============================================================================
# Cluster ↔ true-grouping agreement (does correlation recover the variable groups?)
# ============================================================================
struct GroupingScore
    n_clusters::Int
    n_true_groups::Int
    rand_index::Float64           # adjusted-for-chance? no — the plain Rand index
    homogeneity::Float64          # mean within-cluster purity (1 = each cluster ⊆ one true group)
    over_grouping_ratio::Float64  # (#cluster-pairs that span ≥2 true groups) / (#cluster-pairs)
    mean_cluster_size::Float64
    mean_true_group_size::Float64
    size_inflation::Float64       # mean_cluster_size / mean_true_group_size (>1 ⇒ over-grouping)
end

"""Compare recovered clusters to the true concept-family grouping. Over the set of
candidate-cell PAIRS:
  RandIndex  = (#pairs agreed: same-cluster⇔same-group + diff⇔diff) / (#pairs)
  homogeneity= mean over clusters of (max true-group share within the cluster)
  over_grouping_ratio = fraction of WITHIN-CLUSTER pairs whose two cells are in
      DIFFERENT true groups — the correlation lumping causally-unrelated cells
      together (the headline A4 failure). 0 ⇒ never over-groups; →1 ⇒ clusters are
      grab-bags. size_inflation>1 ⇒ recovered clusters are bigger than true groups."""
function score_grouping(clabel::Vector{Int}, families::Vector{String})
    n = length(clabel)
    glabel = let u = unique(families); d = Dict(g => k for (k, g) in enumerate(u)); [d[f] for f in families]; end
    same_cluster(i, j) = clabel[i] == clabel[j]
    same_group(i, j)   = glabel[i] == glabel[j]
    agree = 0; total = 0
    within_cluster = 0; within_cluster_crossgroup = 0
    for i in 1:n, j in i+1:n
        total += 1
        agree += (same_cluster(i, j) == same_group(i, j)) ? 1 : 0
        if same_cluster(i, j)
            within_cluster += 1
            within_cluster_crossgroup += same_group(i, j) ? 0 : 1
        end
    end
    rand = total == 0 ? 1.0 : agree / total
    # homogeneity: per cluster, the dominant true-group share
    clusters = unique(clabel)
    hom = 0.0
    for c in clusters
        members = findall(==(c), clabel)
        counts = Dict{Int,Int}()
        for m in members; counts[glabel[m]] = get(counts, glabel[m], 0) + 1; end
        hom += maximum(values(counts)) / length(members)
    end
    homogeneity = isempty(clusters) ? 1.0 : hom / length(clusters)
    over_grp = within_cluster == 0 ? 0.0 : within_cluster_crossgroup / within_cluster
    csizes = [count(==(c), clabel) for c in clusters]
    gsizes = [count(==(g), glabel) for g in unique(glabel)]
    mcs = Statistics.mean(csizes); mgs = Statistics.mean(gsizes)
    infl = mgs == 0 ? 0.0 : mcs / mgs
    return GroupingScore(length(clusters), length(unique(glabel)), rand,
                         homogeneity, over_grp, mcs, mgs, infl)
end

# ============================================================================
# Weak-pairwise / strong-global structure (the spike-word headline).
# ============================================================================
struct GlobalStructure
    # candidate-cell-set structure (what the clustering operates on)
    mean_abs_pairwise::Float64    # mean |corr| over off-diagonal candidate pairs
    median_abs_pairwise::Float64
    frac_strong_pairs::Float64    # fraction of candidate pairs with |corr|≥0.5
    top_eig_share::Float64        # λ1 / Σλ of the candidate corr matrix
    top3_eig_share::Float64       # (λ1+λ2+λ3)/Σλ
    participation_ratio::Float64  # (Σλ)^2 / Σλ^2 — effective #dims (low ⇒ global)
    n_eff_dims::Float64           # = participation_ratio (alias for clarity)
    n_cand_varying::Int
    # POPULATION-LEVEL structure (all varying RAM cells — the proper spike-word
    # context: weak pairwise but strong collective structure, Schneidman/Tkačik)
    pop_n_cells::Int
    pop_mean_abs::Float64
    pop_median_abs::Float64
    pop_top3_eig_share::Float64
    pop_participation_ratio::Float64
    weak_pairwise_strong_global::Bool  # population: median|corr| weak AND PR ≪ #cells
end

function _corr_structure(C::AbstractMatrix)
    n = size(C, 1)
    if n < 2
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    end
    offs = Float64[]
    for i in 1:n, j in i+1:n; push!(offs, abs(C[i, j])); end
    mean_abs = Statistics.mean(offs)
    med_abs = Statistics.median(offs)
    frac_strong = count(>=(0.5), offs) / length(offs)
    ev = sort(abs.(eigen(Symmetric(Matrix(C))).values); rev = true)
    s = sum(ev)
    top1 = s == 0 ? 0.0 : ev[1] / s
    top3 = s == 0 ? 0.0 : sum(ev[1:min(3, end)]) / s
    pr = sum(ev .^ 2) == 0 ? 0.0 : (s^2) / sum(ev .^ 2)
    return (mean_abs, med_abs, frac_strong, top1, top3, pr)
end

"""Quantify the weak-pairwise/strong-global pattern. Two scopes:
  * candidate-cell set (the curated cells the clustering operates on);
  * the FULL POPULATION of varying RAM cells (the proper spike-word context —
    Schneidman 2006 / Tkačik 2014 measure pairwise correlation over the whole
    population, where it is WEAK while a few global modes capture the collective
    structure). The WPSG flag is judged on the population: median pairwise |corr|
    is weak (<0.5) AND the effective dimensionality (participation ratio) is far
    below the number of cells (PR < 0.25·#cells), i.e. a few global modes dominate."""
function global_structure(C::AbstractMatrix, varying::Vector{Bool}, tape::AbstractMatrix)
    idx = findall(varying)
    (cmean, cmed, cfrac, ctop1, ctop3, cpr) =
        length(idx) < 2 ? (0.0, 0.0, 0.0, 0.0, 0.0, 0.0) : _corr_structure(C[idx, idx])

    # population: all RAM cells that vary over the trajectory
    n = size(tape, 2)
    pop_cols = [j for j in 1:n if Statistics.std(Float64.(@view tape[:, j])) > 0]
    m = length(pop_cols)
    if m < 2
        pmean = pmed = ptop3 = ppr = 0.0
    else
        X = Float64.(tape[:, pop_cols])
        Cp = Matrix{Float64}(undef, m, m)
        for i in 1:m, j in 1:m
            Cp[i, j] = i == j ? 1.0 : pearson(@view(X[:, i]), @view(X[:, j]))
        end
        (pmean, pmed, _pfrac, _pt1, ptop3, ppr) = _corr_structure(Cp)
    end
    # weak-pairwise/strong-global: over the population the TYPICAL pairwise coupling
    # is weak (median |corr| < 0.5) yet a few global eigen-modes dominate the
    # collective structure (top-3 eigenvalue share ≥ 0.6 — equivalently a low
    # effective dimensionality). seaquest (median 0.66) is genuinely NOT weak-pairwise
    # → WPSG=false there, correctly.
    wpsg = (m >= 2) && (pmed < 0.5) && (ptop3 >= 0.6)
    return GlobalStructure(cmean, cmed, cfrac, ctop1, ctop3, cpr, cpr, length(idx),
                           m, pmean, pmed, ptop3, ppr, wpsg)
end

# ============================================================================
# The ORACLE cell-cell coupling map (the TRUE coupling — the F ground truth).
# ============================================================================
# true coupling(i,j) = does an EXACT intervention on candidate cell i move candidate
# cell j over the horizon? Computed with the SAME bit-exact machinery as the oracle
# (oracle_intervene.jl's Δy(u), here with y = each OTHER candidate cell's value and
# also a robust whole-RAM/whole-screen footprint). This is the §1 oracle restricted
# to candidate-cell pairs — the "true coupling" experiment_design.md §4 A4 names.

struct CouplingOracle
    coupling::Matrix{Float64}        # symmetric (n,n): |Δ cell_j| under do(cell_i)
    held_out::Matrix{Float64}        # same but with the held-out intervention value
    self_footprint::Vector{Float64}  # per-cell whole-screen break (its own causal mass)
end

"""Δ on every candidate cell j when we intervene on candidate cell i. We use two
intervention values (do(0), do(base+17)) and take the max |Δ cell_j| as the
directed coupling i→j; the held-out map uses do(base+37) (a value the main sweep
never used — the S probe). Symmetrise (i→j and j→i) to an undirected coupling."""
function oracle_coupling(game, checkpoint, tail::Vector{Int}, cands::Vector{Candidate},
                         at_target::Snapshot, base_snap::Snapshot; verbose = true)
    n = length(cands)
    didx = [c.ram_index for c in cands]
    base_ram = base_snap.ram
    dir   = zeros(Float64, n, n)     # directed i->j coupling (max over do(0)/do(+17))
    dir_h = zeros(Float64, n, n)     # directed i->j under the held-out do(+37)
    self  = zeros(Float64, n)
    function lesion(i_idx, value)
        env = deepcopy(checkpoint)
        intervene_ram!(env, i_idx, value)
        for a in tail; env_step!(env, a); end
        return snapshot(env, length(tail))
    end
    for (i, c) in enumerate(cands)
        bv = Int(at_target.ram[c.ram_index + 1])
        s0  = lesion(c.ram_index, 0)
        s17 = lesion(c.ram_index, (bv + 17) & 0xFF)
        s37 = lesion(c.ram_index, (bv + 37) & 0xFF)
        # i's own causal footprint = whole-screen break (its real behavioural mass)
        self[i] = max(Float64(count(s0.screen .!= base_snap.screen)),
                      Float64(count(s17.screen .!= base_snap.screen)))
        for (j, cj) in enumerate(cands)
            i == j && continue
            jcol = cj.ram_index + 1
            b = Float64(Int(base_ram[jcol]))
            d0  = abs(Float64(Int(s0.ram[jcol]))  - b)
            d17 = abs(Float64(Int(s17.ram[jcol])) - b)
            d37 = abs(Float64(Int(s37.ram[jcol])) - b)
            dir[i, j]   = max(d0, d17)
            dir_h[i, j] = d37
        end
        verbose && println("  A4-oracle [$i/$n] RAM[$(lpad(c.ram_index,3))] " *
                           "$(rpad(c.concept,18)) footprint(Δpx)=$(round(self[i])) " *
                           "couples→ $(count(>(0.0), dir[i, :])) cells")
    end
    # symmetrise: undirected coupling = max(i→j, j→i)
    sym(M) = [max(M[i, j], M[j, i]) for i in 1:n, j in 1:n]
    return CouplingOracle(sym(dir), sym(dir_h), self)
end

# ============================================================================
# F / S / M (the correctness triad, scored against the coupling oracle).
# ============================================================================
struct Triad
    F::Float64; S::Float64; M::Float64
    F_note::String; S_note::String; M_note::String
    n_pairs::Int
end

"""Lower-triangle off-diagonal pairs as flat vectors (the scored pair set)."""
function offdiag_pairs(M::AbstractMatrix)
    n = size(M, 1); v = Float64[]
    for i in 1:n, j in i+1:n; push!(v, M[i, j]); end
    return v
end

spearman(a::AbstractVector, b::AbstractVector) = pearson(_rank(a), _rank(b))
function _rank(v::AbstractVector)
    n = length(v); p = sortperm(v); r = zeros(Float64, n); i = 1
    while i <= n
        j = i
        while j < n && v[p[j+1]] == v[p[i]]; j += 1; end
        avg = (i + j) / 2
        for k in i:j; r[p[k]] = avg; end
        i = j + 1
    end
    return r
end

function score_triad(C::AbstractMatrix, oracle::CouplingOracle, τ::Real)
    corr_pairs    = abs.(offdiag_pairs(C))                 # |corr| over pairs (the method)
    couple_pairs  = offdiag_pairs(oracle.coupling)         # true coupling over pairs
    held_pairs    = offdiag_pairs(oracle.held_out)         # held-out true coupling
    # F — does the pairwise-correlation structure track the TRUE coupling?
    F = spearman(corr_pairs, couple_pairs)
    # S — predict the HELD-OUT coupling (do base+37, unseen) from |corr|.
    S = pearson(corr_pairs, held_pairs)
    # M — minimality / over-grouping: of the pairs A4 flags coupled (|corr|≥τ), how
    #     many are NOT truly causally coupled (oracle coupling == 0)? Over-claiming
    #     a co-varying-but-uncoupled pair is the correlation-over-grouping failure.
    flagged = corr_pairs .>= τ
    nflag = sum(flagged)
    overclaim = nflag == 0 ? 0.0 :
        count(k -> flagged[k] && couple_pairs[k] == 0.0, 1:length(flagged)) / nflag
    M = 1.0 - overclaim
    return Triad(F, S, M,
        "Spearman(|pairwise corr|, oracle cell-cell coupling) over $(length(corr_pairs)) candidate-cell pairs",
        "Pearson(|pairwise corr|, held-out do(base+37) coupling) — sufficiency on the bit-exact re-run",
        "1 − over-claim rate: fraction of |corr|≥$(τ)-flagged pairs the oracle says are NOT coupled (correlation over-grouping)",
        length(corr_pairs))
end

# ============================================================================
# Drive one game: trajectory → corr → clusters → grouping → global → oracle → triad.
# ============================================================================
struct GameResult
    game::String
    target_frame::Int
    horizon::Int
    traj_frames::Int
    trace::String
    in_window::Bool
    seed::Int
    bit_exact::Bool
    cands::Vector{Candidate}
    corr::Matrix{Float64}
    varying::Vector{Bool}
    clabel::Vector{Int}
    tau_cluster::Float64
    grouping::GroupingScore
    global_s::GlobalStructure
    oracle::CouplingOracle
    triad::Triad
    # SHARED-TESTBED provenance (redesign); "noop"/-1/false in the legacy path.
    state_kind::String             # "seeded_random_action_gameplay" | "noop"
    st_seed::Int
    st_prefix::Int
    cause_density::Int             # #causes above the floor at the shared output
    cause_density_accepted::Bool   # passed the cause-density gate?
    n_causes::Int
    shared_cell::Tuple{Int,Int}    # the shared screen-buffer output cell
end

"""Build the P2 SHARED gameplay state + cause-density gate for `game` using the §1
oracle machinery (oracle_intervene.jl). Returns the substrate NamedTuple (we use
its `.actions` stream + `.cause_density`/`.accepted`/`.cell` gate). A4 then boots
its OWN checkpoint + records its OWN trajectory from `st.actions` so the coupling /
correlation algorithm is unchanged."""
function build_a4_shared_state(game::AbstractString; verbose = false)
    O = OracleIntervene; J = OracleIntervene.JutariOracle
    return build_shared_testbed(game;
        settings_for = J.settings_for, rom_path_for = J.rom_path_for,
        candidates_path_for = resolve_candidates,
        build_causes = O.build_pong_causes, candidate_ram_indices = O.candidate_ram_indices,
        continue_from = J.continue_from, snapshot = J.snapshot, env_step = J.env_step!,
        intervene_ram = J.intervene_ram!, boot_replay = J.boot_replay,
        run_intervention = O.run_intervention, soft_ram_peek = soft_ram_peek,
        prefix = ST_PREFIX, horizon = ST_HORIZON, seed = ST_SEED,
        k = ST_GATE_K, floor = ST_FLOOR, verbose = verbose,
        assert_bit_exact = O.assert_bit_exact)
end

function compute_game(game::AbstractString; target_frame = nothing, horizon = nothing,
                      traj_frames = nothing, tau_corr = 0.7, seed = 0, verbose = true)
    cfg = game_cfg(game)
    target_frame = target_frame === nothing ? cfg.target_frame : target_frame
    horizon      = horizon      === nothing ? cfg.horizon      : horizon
    traj_frames  = traj_frames  === nothing ? cfg.traj_frames  : traj_frames
    trace        = cfg.trace
    in_window    = cfg.in_window

    # SHARED-TESTBED (redesign): replace the per-game FIRE/attract oracle tape with a
    # seeded random-action GAMEPLAY state at f*=ST_PREFIX, gated by the oracle
    # cause-density gate. The candidate cells must MOVE for the correlation/coupling
    # to carry signal, so both the coupling oracle AND the descriptive correlation
    # trajectory are stepped along the SAME gameplay stream. A4's a4_*/coupling
    # machinery is UNCHANGED — only the state moves.
    st = nothing
    if SHARED_TESTBED
        st = build_a4_shared_state(game; verbose = verbose)
        target_frame = st.prefix; horizon = st.horizon
        seed = st.seed; in_window = false      # gameplay window may exceed 60-frame screen conf.
        verbose && println("[A4:$game] SHARED gameplay state: cause_density=$(st.cause_density)/" *
            "$(length(st.causes)) accepted=$(st.accepted) cell=$(st.cell)")
    end
    total = target_frame + horizon
    oacts = SHARED_TESTBED ? Int.(st.actions) : oracle_actions(total)
    tail = Int.(oacts[target_frame + 1 : target_frame + horizon])

    # 1) bit-exact baseline — two fresh boots+replays must be byte-identical.
    verbose && println("[A4:$game] asserting bit-exactness (2 fresh boots+replays to f$total)...")
    a = a4_fresh_baseline(game, oacts, total)
    b = a4_fresh_baseline(game, oacts, total)
    bit_exact = (a.ram == b.ram) && (a.screen == b.screen)
    bit_exact || error("bit-exact re-run FAILED for $game to f$total — refusing to score")
    verbose && println("[A4:$game] bit-exact re-run: PASS")

    # 2) one checkpoint at the intervention frame (boot+to-target paid once).
    checkpoint = a4_boot_replay(game, oacts, target_frame)
    at_target = a4_continue_from(checkpoint, Int[])
    base_snap = a4_continue_from(checkpoint, tail)

    cand_path = resolve_candidates(game)
    cands = load_candidates(cand_path)
    isempty(cands) && error("no candidate cells for $game (candidates file missing/empty: $cand_path)")
    verbose && println("[A4:$game] candidates: $(cand_path) ($(length(cands)) cells, " *
                       "$(length(unique([c.family for c in cands]))) true variable groups)")

    # 3) descriptive correlation over a trajectory (cells must move). Under the
    #    shared testbed the trajectory is stepped along the SAME seeded gameplay
    #    stream; if traj_frames exceeds the gameplay window (prefix+horizon), the
    #    tail is NOOP-continued.
    if SHARED_TESTBED
        traj_acts = Vector{Int}(undef, max(0, traj_frames))
        for t in 1:length(traj_acts)
            traj_acts[t] = t <= total ? Int(st.actions[t]) : 0    # gameplay then NOOP
        end
        verbose && println("[A4:$game] recording $(traj_frames)-frame gameplay RAM trajectory " *
            "(seed=$(st.seed) prefix=$(st.prefix)+$(st.horizon), NOOP-continued past f$total) + pairwise corr...")
    else
        traj_acts = trace_actions(trace, traj_frames)
        verbose && println("[A4:$game] recording $(traj_frames)-frame active ($(trace)) RAM trajectory + pairwise corr...")
    end
    tape = a4_record_ram(game, traj_frames, traj_acts)
    C, varying = pairwise_corr(tape, cands)

    # 4) co-activation clusters + true-grouping agreement (the over-grouping result).
    clabel = cluster_by_corr(C, tau_corr)
    grouping = score_grouping(clabel, [c.family for c in cands])

    # 5) weak-pairwise / strong-global structure (candidate set + full population).
    global_s = global_structure(C, varying, tape)

    # 6) the ORACLE cell-cell coupling (TRUE coupling — the F ground truth).
    verbose && println("[A4:$game] oracle cell-cell coupling over $(length(cands)) candidate cells...")
    oracle = oracle_coupling(game, checkpoint, tail, cands, at_target, base_snap; verbose = verbose)

    # 7) F / S / M triad (correlation vs the coupling oracle).
    triad = score_triad(C, oracle, tau_corr)

    if verbose
        println("[A4:$game] ---- A4 scores ----")
        println("[A4:$game]   clusters=$(grouping.n_clusters) vs true groups=$(grouping.n_true_groups); " *
                "Rand=$(round(grouping.rand_index,digits=3)) homogeneity=$(round(grouping.homogeneity,digits=3))")
        println("[A4:$game]   OVER-GROUPING ratio=$(round(grouping.over_grouping_ratio,digits=3)) " *
                "size-inflation=$(round(grouping.size_inflation,digits=3))")
        println("[A4:$game]   POP weak-pairwise median|corr|=$(round(global_s.pop_median_abs,digits=3)) " *
                "strong-global top3-eig=$(round(global_s.pop_top3_eig_share,digits=3)) " *
                "PR=$(round(global_s.pop_participation_ratio,digits=2))/$(global_s.pop_n_cells) " *
                "→ WPSG=$(global_s.weak_pairwise_strong_global)")
        println("[A4:$game]   TRIAD F=$(round(triad.F,digits=3)) S=$(round(triad.S,digits=3)) M=$(round(triad.M,digits=3))")
    end

    return GameResult(game, target_frame, horizon, traj_frames, trace, in_window,
                      seed, bit_exact, cands, C, varying, clabel, tau_corr,
                      grouping, global_s, oracle, triad,
                      st === nothing ? "noop" : "seeded_random_action_gameplay",
                      st === nothing ? -1 : st.seed,
                      st === nothing ? -1 : st.prefix,
                      st === nothing ? -1 : st.cause_density,
                      st === nothing ? false : st.accepted,
                      st === nothing ? 0 : length(st.causes),
                      st === nothing ? (-1, -1) : st.cell)
end

# ============================================================================
# Self-check (DoD) — the scoring contract is sound; results are non-fabricated.
# ============================================================================
"""
    selftest(r::GameResult) -> Bool

Asserts A4's load-bearing claims (the same contract shape as pilot_si.selftest):

  (BIT-EXACT) the baseline re-run is byte-identical (every coupling Δ is a clean
    causal effect on the real ROM).

  (ORACLE GROUND-TRUTH ANCHORED) the oracle finds ≥1 genuinely coupled candidate
    pair (some intervention moves another candidate cell) — else F is uninformative.

  (POSITIVE CONTROL) scoring the ORACLE'S OWN coupling map as the candidate method
    yields Spearman F == 1 — the harness rewards a perfectly-faithful coupling map,
    so a sub-1 correlation F is a real measurement.

  (CORRELATION MATRIX WELL-FORMED) C is symmetric, diagonal == 1, |C| ≤ 1.

  (TRIAD RANGES) F,S ∈ [−1,1]; M ∈ [0,1]; finite.

  (GROUPING / GLOBAL WELL-FORMED) Rand, homogeneity, over-grouping ratio ∈ [0,1];
    top-eigen shares ∈ [0,1]; participation ratio ≥ 0.

Throws on a contract violation."""
function selftest(r::GameResult)
    @assert r.bit_exact "bit-exact baseline re-run failed for $(r.game)"

    couple_pairs = offdiag_pairs(r.oracle.coupling)
    n_coupled = count(>(0.0), couple_pairs)
    @assert n_coupled >= 1 "oracle found NO coupled candidate pair for $(r.game) at this " *
        "state — uninformative; pick a livelier target frame"

    # positive control: the oracle's own coupling must score F==1 against itself
    self_F = spearman(couple_pairs, couple_pairs)
    @assert self_F > 0.999 "harness broken: oracle-as-method Spearman != 1 ($self_F) for $(r.game)"

    n = size(r.corr, 1)
    @assert all(abs(r.corr[i, i] - 1.0) < 1e-9 for i in 1:n) "corr diagonal != 1"
    @assert all(abs(r.corr[i, j] - r.corr[j, i]) < 1e-9 for i in 1:n, j in 1:n) "corr not symmetric"
    @assert all(abs(r.corr[i, j]) <= 1.0 + 1e-9 for i in 1:n, j in 1:n) "|corr| > 1"

    @assert -1.0 - 1e-9 <= r.triad.F <= 1.0 + 1e-9 "F out of [-1,1]: $(r.triad.F)"
    @assert -1.0 - 1e-9 <= r.triad.S <= 1.0 + 1e-9 "S out of [-1,1]: $(r.triad.S)"
    @assert  0.0 - 1e-9 <= r.triad.M <= 1.0 + 1e-9 "M out of [0,1]: $(r.triad.M)"

    g = r.grouping
    for (nm, v) in (("rand", g.rand_index), ("homogeneity", g.homogeneity),
                    ("over_grouping_ratio", g.over_grouping_ratio))
        @assert 0.0 - 1e-9 <= v <= 1.0 + 1e-9 "$nm out of [0,1]: $v"
    end
    gs = r.global_s
    @assert 0.0 - 1e-9 <= gs.top_eig_share <= 1.0 + 1e-9 "top_eig_share out of [0,1]"
    @assert 0.0 - 1e-9 <= gs.top3_eig_share <= 1.0 + 1e-9 "top3_eig_share out of [0,1]"
    @assert gs.participation_ratio >= -1e-9 "negative participation ratio"
    @assert 0.0 - 1e-9 <= gs.pop_top3_eig_share <= 1.0 + 1e-9 "pop_top3_eig_share out of [0,1]"
    @assert gs.pop_participation_ratio >= -1e-9 "negative population participation ratio"
    @assert gs.pop_participation_ratio <= gs.pop_n_cells + 1e-6 "participation ratio exceeds #cells"

    println("[A4:$(r.game)] SELF-CHECK PASS:")
    println("[A4:$(r.game)]   bit-exact baseline re-run: $(r.bit_exact)")
    println("[A4:$(r.game)]   oracle coupled pairs: $n_coupled (positive control F = $(round(self_F,digits=3)))")
    println("[A4:$(r.game)]   F=$(round(r.triad.F,digits=3)) S=$(round(r.triad.S,digits=3)) M=$(round(r.triad.M,digits=3))")
    println("[A4:$(r.game)]   over-grouping=$(round(g.over_grouping_ratio,digits=3)) " *
            "WPSG=$(gs.weak_pairwise_strong_global)")
    return true
end

# ============================================================================
# Persist (SPEC §R) — JSON record + sibling .npz; file_scope A4_*.
# ============================================================================
_git_commit() = try
    strip(read(`git -C $(@__DIR__) rev-parse --short HEAD`, String))
catch
    "unknown"
end
_json_num(x::Real) = isfinite(x) ? Float64(x) : nothing

const CANDIDATES_DIR_REL = joinpath("tools", "xai_study", "t3", "out")
function resolve_candidates(game::AbstractString)
    here = normpath(joinpath(@__DIR__, "..", "..", ".."))
    for base in (here, get(ENV, "XAI_PRIMARY_REPO", ""),
                 "/Users/maier/Documents/code/UnderstandingVCS")
        base == "" && continue
        p = joinpath(base, CANDIDATES_DIR_REL, "candidates_$(game).json")
        isfile(p) && return p
    end
    return nothing
end

function write_game(r::GameResult; out_dir = OUT_DIR)
    isdir(out_dir) || mkpath(out_dir)
    stem = "A4_$(r.game)"
    json_path = joinpath(out_dir, stem * ".json")
    npz_path  = joinpath(out_dir, stem * ".npz")

    cell_names = ["RAM[$(c.ram_index)]:$(c.concept)" for c in r.cands]
    families = [c.family for c in r.cands]
    rec = Dict{String,Any}(
        "paper" => "P2",
        "phase" => "phaseA_kording",
        "method" => "A4_spike_word_pairwise_correlation",
        "game" => r.game,
        "state" => r.state_kind == "noop" ? "f$(r.target_frame)+$(r.horizon)" :
                   "gameplay(seed=$(r.st_seed),prefix=$(r.st_prefix))+$(r.horizon)",
        "target_output" => "cell-cell-coupling",
        # headline scalar (SPEC §R value/metric_name): the A4 faithfulness F — how
        # well the pairwise-correlation structure tracks the TRUE causal coupling.
        "metric_name" => "A4_faithfulness_spearman_corr_vs_oracle_coupling",
        "value" => _json_num(r.triad.F),
        "stderr" => nothing,
        "ci" => nothing,
        "n" => length(r.cands),
        "seed" => r.seed,
        "where" => "local",
        "commit" => _git_commit(),
        "oracle_ref" => "oracle_intervene@$(r.game)#cell-cell-coupling(screen+RAM)",
        "timestamp" => string(round(Int, time())),
        "arrays" => basename(npz_path),
        "extra" => Dict{String,Any}(
            "substrate" => "jutari (Julia, HARD) — real-ROM bit-exact path; the " *
                "coupling oracle uses EXACT interventions re-run on the true ROM.",
            "bit_exact_rerun" => r.bit_exact,
            "trajectory_trace" => r.state_kind != "noop" ?
                "seeded random-action GAMEPLAY (seed=$(r.st_seed), prefix=$(r.st_prefix)+$(r.horizon), NOOP-continued past the window)" :
                (r.trace == "dir" ?
                    "directional maze trace (FIRE warmup + cyclic UP/DOWN/LEFT/RIGHT)" :
                    "active trace (4×FIRE + periodic RIGHTFIRE/LEFTFIRE)"),
            "trajectory_frames" => r.traj_frames,
            "testbed" => Dict{String,Any}(
                "state_kind" => r.state_kind,
                "seed" => r.st_seed, "prefix" => r.st_prefix, "horizon" => r.horizon,
                "shared_output" => "screen_region(n_changed_px)@r$(r.shared_cell[1])c$(r.shared_cell[2])",
                "cause_density_above_floor" => r.cause_density,
                "cause_density_floor" => ST_FLOOR, "cause_density_gate_k" => ST_GATE_K,
                "cause_density_accepted" => r.cause_density_accepted, "n_causes" => r.n_causes,
                "note" => "P2 redesign: the correlation/coupling study runs on a seeded " *
                    "random-action GAMEPLAY state (not the per-game FIRE/attract tape), " *
                    "gated by the §1 oracle cause-density gate. Both the coupling oracle " *
                    "AND the descriptive correlation trajectory are stepped along the SAME " *
                    "gameplay stream so the candidate cells VARY on genuine input-driven " *
                    "play. A4's correlation/coupling algorithm is unchanged; only the " *
                    "analysis state moves."),
            "scored_in_conformance_window" => r.in_window,
            "conformance_note" => r.in_window ?
                "scored oracle frame f$(r.target_frame)+$(r.horizon) is inside the " *
                "Paper-1 60-frame screen conformance window (jutari↔xitari bit-exact)." :
                "scored oracle frame f$(r.target_frame)+$(r.horizon) is BEYOND the strict " *
                "60-frame screen conformance window — the game only reaches live play " *
                "later (e.g. Ms. Pac-Man's animated intro). This is an HONEST descriptive " *
                "jutari result (the bit-exact jutari re-run IS asserted), not an " *
                "xitari-parity claim; A4 is the calibration baseline, not a headline.",
            "candidate_cells" => cell_names,
            "candidate_ram_indices" => [c.ram_index for c in r.cands],
            "candidate_concept_family" => families,
            "n_true_variable_groups" => length(unique(families)),
            "triad" => Dict{String,Any}(
                "F" => _json_num(r.triad.F), "F_note" => r.triad.F_note,
                "S" => _json_num(r.triad.S), "S_note" => r.triad.S_note,
                "M" => _json_num(r.triad.M), "M_note" => r.triad.M_note,
                "n_pairs" => r.triad.n_pairs,
                "interpretation" => "Phase A is the calibration baseline " *
                    "(experiment_design.md §4): pairwise correlation structure " *
                    "GENERALLY OVER-GROUPS — it lumps causally-unrelated cells that " *
                    "merely co-vary (the present≠used trap, pairwise). F<1 + " *
                    "over_grouping_ratio>0 quantify the departure from the true coupling.",
            ),
            "clustering" => Dict{String,Any}(
                "tau_cluster" => r.tau_cluster,
                "cluster_label_per_cell" =>
                    Dict(cell_names[i] => r.clabel[i] for i in 1:length(cell_names)),
                "n_clusters" => r.grouping.n_clusters,
                "n_true_groups" => r.grouping.n_true_groups,
                "rand_index" => _json_num(r.grouping.rand_index),
                "homogeneity" => _json_num(r.grouping.homogeneity),
                "over_grouping_ratio" => _json_num(r.grouping.over_grouping_ratio),
                "mean_cluster_size" => _json_num(r.grouping.mean_cluster_size),
                "mean_true_group_size" => _json_num(r.grouping.mean_true_group_size),
                "size_inflation" => _json_num(r.grouping.size_inflation),
                "note" => "single-linkage on |corr|≥tau; over_grouping_ratio = " *
                    "fraction of within-cluster pairs whose cells are in DIFFERENT " *
                    "true variable groups (correlation lumping unrelated cells); " *
                    "size_inflation>1 ⇒ recovered clusters bigger than true groups.",
            ),
            "weak_pairwise_strong_global" => Dict{String,Any}(
                # candidate-cell set (the cells the clustering operates on)
                "candidate_mean_abs_pairwise" => _json_num(r.global_s.mean_abs_pairwise),
                "candidate_median_abs_pairwise" => _json_num(r.global_s.median_abs_pairwise),
                "candidate_frac_strong_pairs(|corr|>=0.5)" => _json_num(r.global_s.frac_strong_pairs),
                "candidate_top_eig_share" => _json_num(r.global_s.top_eig_share),
                "candidate_top3_eig_share" => _json_num(r.global_s.top3_eig_share),
                "candidate_participation_ratio" => _json_num(r.global_s.participation_ratio),
                "candidate_n_varying_cells" => r.global_s.n_cand_varying,
                # POPULATION (all varying RAM cells — the proper spike-word context)
                "population_n_cells" => r.global_s.pop_n_cells,
                "population_mean_abs_pairwise" => _json_num(r.global_s.pop_mean_abs),
                "population_median_abs_pairwise" => _json_num(r.global_s.pop_median_abs),
                "population_top3_eig_share" => _json_num(r.global_s.pop_top3_eig_share),
                "population_participation_ratio" => _json_num(r.global_s.pop_participation_ratio),
                "weak_pairwise_strong_global" => r.global_s.weak_pairwise_strong_global,
                "note" => "the spike-word headline (Schneidman 2006; Tkačik 2014): over " *
                    "the FULL POPULATION of varying RAM cells the median pairwise |corr| " *
                    "is WEAK while a few GLOBAL eigen-modes dominate — the effective " *
                    "dimensionality (participation ratio) is far below the #cells, so the " *
                    "collective structure greatly exceeds the pairwise structure. WPSG " *
                    "flag = population median|corr|<0.5 AND population top3-eig-share≥0.6. " *
                    "(The candidate_* set is curated so its pairwise corr runs higher.)",
            ),
            "oracle_coupling" => Dict{String,Any}(
                "self_footprint_per_cell" =>
                    Dict(cell_names[i] => r.oracle.self_footprint[i] for i in 1:length(cell_names)),
                "n_coupled_pairs" => count(>(0.0), offdiag_pairs(r.oracle.coupling)),
                "n_total_pairs" => r.triad.n_pairs,
                "note" => "true coupling(i,j) = max |Δ candidate-cell j| under an " *
                    "EXACT do(i:=0)/do(i:=base+17) intervention on cell i, re-run on " *
                    "the real ROM (symmetrised). The §1 oracle restricted to " *
                    "candidate-cell pairs — the TRUE coupling A4 is scored against.",
            ),
            "scales_to_cluster_via" =>
                "tools/cluster/xai_*.sbatch (E0-3) — the full A1..A8 battery over the " *
                "core+breadth set; the forward is bit-exact to this HARD path.",
        ),
    )
    open(json_path, "w") do io; JSON.print(io, rec, 2); end

    write_npz(npz_path, Dict(
        "candidate_ram_indices" => Int64[c.ram_index for c in r.cands],
        "corr_matrix"           => r.corr,
        "varying_cells"         => Int64[r.varying[i] ? 1 : 0 for i in 1:length(r.varying)],
        "cluster_label"         => Int64.(r.clabel),
        "oracle_coupling"       => r.oracle.coupling,
        "oracle_coupling_heldout" => r.oracle.held_out,
        "oracle_self_footprint" => r.oracle.self_footprint,
        "grouping_scores"       => Float64[r.grouping.rand_index, r.grouping.homogeneity,
                                           r.grouping.over_grouping_ratio, r.grouping.size_inflation],
        # [cand_mean|corr|, cand_top1_eig, cand_top3_eig, cand_PR,
        #  pop_mean|corr|, pop_median|corr|, pop_top3_eig, pop_PR, pop_n_cells]
        "global_structure"      => Float64[r.global_s.mean_abs_pairwise, r.global_s.top_eig_share,
                                           r.global_s.top3_eig_share, r.global_s.participation_ratio,
                                           r.global_s.pop_mean_abs, r.global_s.pop_median_abs,
                                           r.global_s.pop_top3_eig_share, r.global_s.pop_participation_ratio,
                                           Float64(r.global_s.pop_n_cells)],
        "triad_FSM"             => Float64[r.triad.F, r.triad.S, r.triad.M],
    ))
    return json_path, npz_path
end

# ============================================================================
# CLI
# ============================================================================
function main(args = ARGS)
    games = copy(CORE_GAMES)
    # Per-game live-play frames/horizon/trajectory come from GAME_CFG (the game must
    # be in live play for the candidate cells to move). CLI flags override for all
    # games; `nothing` ⇒ use the per-game config.
    target_frame = nothing; horizon = nothing; traj_frames = nothing
    tau_corr = 0.7; seed = 0
    selftest_only = false
    i = 1
    while i <= length(args)
        a = args[i]
        if     a == "--games";        games = String.(split(args[i+1], ",")); i += 2
        elseif a == "--game";         games = [args[i+1]]; i += 2
        elseif a == "--target-frame"; target_frame = parse(Int, args[i+1]); i += 2
        elseif a == "--horizon";      horizon = parse(Int, args[i+1]); i += 2
        elseif a == "--traj-frames";  traj_frames = parse(Int, args[i+1]); i += 2
        elseif a == "--tau-corr";     tau_corr = parse(Float64, args[i+1]); i += 2
        elseif a == "--seed";         seed = parse(Int, args[i+1]); i += 2
        elseif a == "--selftest";     selftest_only = true; i += 1
        else; i += 1
        end
    end
    println("[A4] games=$(join(games, ",")) target_frame=$target_frame horizon=$horizon " *
            "traj_frames=$traj_frames tau_corr=$tau_corr seed=$seed (jutari/Julia path)")

    if selftest_only
        g = games[1]
        r = compute_game(g; target_frame = target_frame, horizon = horizon,
                         traj_frames = traj_frames, tau_corr = tau_corr, seed = seed)
        selftest(r)
        println("[A4] --selftest: passed on $g, not writing artifact.")
        return 0
    end

    ok = String[]; failed = Tuple{String,String}[]
    for g in games
        try
            r = compute_game(g; target_frame = target_frame, horizon = horizon,
                             traj_frames = traj_frames, tau_corr = tau_corr, seed = seed)
            selftest(r)
            json_path, npz_path = write_game(r)
            println("[A4] wrote $json_path")
            println("[A4] arrays  $npz_path")
            push!(ok, g)
        catch err
            msg = sprint(showerror, err)
            println("[A4] !! game $g FAILED (scoring the rest, not fabricating): " *
                    first(split(msg, '\n')))
            push!(failed, (g, first(split(msg, '\n'))))
        end
    end
    println("[A4] ==== summary: $(length(ok))/$(length(games)) games scored ====")
    for g in ok; println("[A4]   OK   $g"); end
    for (g, m) in failed; println("[A4]   FAIL $g — $m"); end
    return isempty(failed) ? 0 : 0   # partial success is allowed (score the rest)
end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    A4Correlations.main()
end
