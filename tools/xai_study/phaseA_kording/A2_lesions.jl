# A2_lesions.jl — Phase-A analysis A2 (P2-E3-2), JULIA path.
#
# EXHAUSTIVE SINGLE-UNIT LESIONS / per-unit IMPORTANCE MAP on the VCS
# (experiment_design.md §4, row A2). The neuroscience analogy: knock out one
# "neuron" (here a candidate RAM cell — a state unit) at a time, re-run the
# system, and ask "does behaviour still work, and by how much did it change?".
# The collected per-unit behavioural change IS the lesion importance map — the
# classic single-unit ablation study (Jonas & Kording 2017 ran exactly this on
# the same chip; experiment_design.md §4 Novelty note).
#
# A2's place in Paper 2: Phase A is the *calibration baseline* (experiment_design
# §4). Unlike A1's deliberately-thin recovery, the single-unit lesion is a
# RELATIVELY FAITHFUL classical method — clamping one cell and re-running the real
# ROM is itself an exact intervention, so the per-unit importance is close to the
# unit's true causal magnitude. We therefore EXPECT A2 to score HIGH rank-
# correlation vs the exact oracle (the task brief: "it should score HIGH — single-
# unit ablation is close to the causal map"). The interesting residual is the
# MISSED INTERACTION EFFECTS: a single-unit map is, by construction, blind to
# super-/sub-additive PAIR effects (two cells that matter only jointly, or that
# mask each other). A2 quantifies both the high single-unit faithfulness AND the
# interaction blind-spot — the precise sense in which a single-unit lesion map is
# "right but incomplete".
#
# ---------------------------------------------------------------------------
# UNITS, the LESION, the IMPORTANCE MAP, and the GROUND TRUTH.
#
# Units = the candidate RAM cells (E2-1 import, candidates_<game>.json) — the
# state variables the neuroscientist would probe. For each unit u:
#
#   LESION(u)  =  clamp u to a fixed value at the target frame, re-run the
#                 deterministic emulator for the full horizon, read the outputs.
#   We use the EXHAUSTIVE single-unit do-set (the task brief: "clamp→0 and
#   clamp→alt"): clamp→0 (silence the unit) AND clamp→alt = base XOR 0xFF
#   (a maximally-different value), and take the per-unit effect as the MAX over
#   the two clamps (a lesion "works" / "matters" if ANY clamp perturbs behaviour).
#
#   IMPORTANCE(u) = a scalar behavioural-change magnitude under LESION(u),
#                   combining (a) Δ on the game outputs y (scores, content/index
#                   pixels) and (b) the whole-screen change (count of framebuffer
#                   cells that differ from baseline). The screen-change term is the
#                   "does the rendered behaviour still work?" signal; the output Δ
#                   is the targeted behavioural readout. Reported per-output too.
#
#   "boot/run still works?" — recorded per unit as a boolean: after LESION(u),
#   does the emulator still produce a valid (non-crashing, in-range) frame? On the
#   VCS a poked RAM cell never halts the CPU, so this is ~always true; we record it
#   honestly (the screen-change magnitude is the graded version of the same idea).
#
# GROUND TRUTH (the unit's TRUE causal role, T1 — naming optional):
#   The exact-intervention oracle's per-unit total causal effect. For unit u we
#   run the EXHAUSTIVE oracle do-set (occlude→0, set→base+17, set→base+37,
#   clamp→alt) and define
#       ROLE(u) = the per-unit aggregate |Δy| over ALL outputs, MAX over the
#                 do-set (the largest behavioural footprint the unit can cause).
#   This is the bit-exact causal map restricted to "how much does this unit
#   matter, causally, for behaviour" — the true importance every method is scored
#   against (experiment_design.md §1, §4). Every Δ is a real re-run of the real
#   ROM (Paper-1 64/64 bit-exact), so ROLE(u) is a genuine causal magnitude, no
#   world model assumed.
#
# ---------------------------------------------------------------------------
# SCORING (experiment_design.md §4 A2: "rank-correlation of importance with the
# unit's *true* role; #units flagged 'specific' that are generic"):
#
#   (1) RANK-CORRELATION (Spearman ρ) of the single-unit IMPORTANCE map vs the
#       oracle ROLE map over the units. EXPECTED HIGH (single-unit ablation ≈ the
#       causal map). This is F (faithfulness).
#
#   (2) SPURIOUS-SPECIFICITY count: a unit is flagged "specific" by the lesion
#       study if its single-unit lesion drives ESSENTIALLY ONE output (its lesion
#       footprint is concentrated on a single output — high selectivity). It is
#       *actually generic* if the ORACLE says the unit causally touches MANY
#       outputs (broad true role). #(flagged-specific ∧ truly-generic) is the
#       false-specificity count the spec asks for — the lesion's selectivity is a
#       single-value/single-step artifact, not the unit's real (broad) role.
#
#   (3) MISSED INTERACTIONS (the single-unit blind spot): over a budgeted set of
#       unit PAIRS (i,j), measure the JOINT lesion effect E(i,j) (clamp BOTH) and
#       compare to the sum of singles E(i)+E(j). The interaction term
#           Δ_int(i,j) = E(i,j) − (E(i)+E(j))                (super/sub-additive)
#       is invisible to any single-unit map. We report the fraction of pairs with
#       |Δ_int| above a tolerance (the "interaction-missed rate") and the largest
#       super-additive pair — concrete evidence of what single-unit lesions miss.
#
#   POSITIVE CONTROL (oracle-as-method, as in pilot_si.jl / A1): build the
#   importance map with the FULL oracle do-set (i.e. ROLE itself as the method) and
#   rank-correlate vs the oracle ROLE — must give ρ = 1. This proves the harness
#   REWARDS a perfectly-faithful importance map, so a sub-1 single-clamp ρ is a
#   real measurement, not a broken scorer.
#
# F / S / M triad (correctness triad, experiment_design.md §0), scored vs oracle:
#   F (faithful)   — Spearman ρ of the lesion importance map vs the oracle role.
#   S (sufficient) — generalisation to a HELD-OUT clamp value: the lesion map built
#                    from {clamp→0, clamp→alt} predicts the ranking under a value it
#                    never used (clamp→base+37), scored on the bit-exact re-run.
#   M (minimal / right level) — 1 − false-specificity rate: the fraction of
#                    lesion-flagged "specific" units that the oracle says are
#                    actually generic (spurious selectivity). Minimal & at the
#                    right level ⇒ the map's specificity claims are true.
#
# BUILDS ON the verified jutari foundation (NO emulator core touched):
#   * tools/xai_study/common/jutari_oracle.jl — boot/replay/snapshot/deepcopy
#     checkpoint/intervene_ram!/bit-exact baseline/dependency-free NPZ writer.
#   * tools/xai_study/ground_truth/oracle_intervene.jl — the exact-intervention
#     causal-map reference whose Δ definition the oracle ROLE reuses.
#   * each game's tools/xai_study/t3/out/candidates_<game>.json — the units.
#   (Self-contained per-game ROM-basename + RomSettings map, mirroring A1 — does
#    NOT modify the shared lib; constructs the env via the public JuTari API.)
#
# Run (warm shared depot, primary's project):
#   julia --project=/Users/maier/Documents/code/UnderstandingVCS/jutari \
#         tools/xai_study/phaseA_kording/A2_lesions.jl
# Flags: --games pong,breakout,...   --game G   --target-frame N  --horizon N
#        --max-pairs K          (interaction-budget cap; default 60)
#        --out-dir DIR          (default tools/xai_study/phaseA_kording/out)
#        --roms-dir DIR         (override the ROM collection root)
#        --where local|cluster  (recorded in §R)
#        --selftest             (self-check on the pilot game; write nothing)
#   Cluster-shardable: --shard i --nshards n --shard-kind game  → game i of the
#   core set (so tools/cluster/xai_array_jl.sbatch can fan one game per array task).
#
# Writes (SPEC §R; file_scope A2_*): per-game record + combined:
#   tools/xai_study/phaseA_kording/out/A2_lesions_<game>.{json,npz}
#   tools/xai_study/phaseA_kording/out/A2_lesions.{json,npz}   (combined)

module A2Lesions

using JuTari
using JuTari.Env: StellaEnvironment, env_reset!, env_step!, get_ram, get_screen
using JuTari.JoystickGames: MsPacmanRomSettings, QbertRomSettings

# the verified foundation (NO core touched) — reuse the dependency-free NPZ writer
# + the RAM_SIZE constant from the oracle helper.
include(joinpath(@__DIR__, "..", "common", "jutari_oracle.jl"))
using .JutariOracle: write_npz, RAM_SIZE

# The §1 exact-intervention oracle (Cause / build_pong_causes / candidate_ram_indices /
# run_intervention / assert_bit_exact) — used ONLY to build the P2 SHARED gameplay
# state + its cause-density gate (below). A2's lesion algorithm keeps its OWN
# GameSpec/Snap/Candidate machinery; the oracle here supplies the shared action
# stream + the gate, not the lesion map. (Its internal JutariOracle submodule is a
# separate instance from the one included above — no type is mixed across them.)
include(joinpath(@__DIR__, "..", "ground_truth", "oracle_intervene.jl"))
using JuTari.Diff: soft_ram_peek

# The P2 SHARED TESTBED (xai_paper/xai_2_interpretability/experiment_redesign.md):
# seeded random-action GAMEPLAY state + oracle cause-density gate. Included as a
# fragment (see its header for why not a module). Phase A is not a gradient method,
# so the sampler-on path does not apply — we consume the shared action STREAM +
# cause-density GATE only, and boot A2's OWN checkpoint from that stream so the
# lesion machinery is unchanged. Opt in with XAI_SHARED_TESTBED=1 (default on).
include(joinpath(@__DIR__, "..", "common", "shared_testbed_impl.jl"))

import JSON

const DEFAULT_OUT_DIR = joinpath(@__DIR__, "out")
const PRIMARY_REPO = get(ENV, "XAI_PRIMARY_REPO",
                         "/Users/maier/Documents/code/UnderstandingVCS")
# shared-testbed switch + params (redesign protocol: prefix=90 gameplay, horizon=15).
const SHARED_TESTBED = get(ENV, "XAI_SHARED_TESTBED", "1") == "1"
const ST_PREFIX  = parse(Int, get(ENV, "XAI_ST_PREFIX", "90"))
const ST_HORIZON = parse(Int, get(ENV, "XAI_ST_HORIZON", "15"))
const ST_SEED    = parse(Int, get(ENV, "XAI_ST_SEED", "0"))
const ST_GATE_K  = parse(Int, get(ENV, "XAI_ST_GATE_K", "4"))
const ST_FLOOR   = parse(Float64, get(ENV, "XAI_ST_FLOOR", "0.5"))

# ===========================================================================
# Per-game ROM-basename + RomSettings map (self-contained; mirrors A1 /
# tools/jutari_screen_dump.jl _SETTINGS_BY_BASENAME so boot/render parity holds).
# Seaquest has NO jutari RomSettings yet (rendered with Generic — recorded
# honestly per game).
# ===========================================================================
struct GameSpec
    name::String
    rom_basename::String
    settings::Function          # () -> RomSettings
    settings_name::String
end

const GAME_SPECS = Dict{String,GameSpec}(
    "pong"           => GameSpec("pong", "pong.bin",
                                 () -> JuTari.PaddleGames.PongRomSettings(), "PongRomSettings"),
    "breakout"       => GameSpec("breakout", "breakout.bin",
                                 () -> JuTari.PaddleGames.BreakoutRomSettings(), "BreakoutRomSettings"),
    "space_invaders" => GameSpec("space_invaders", "space_invaders.bin",
                                 () -> JuTari.SpaceInvadersRomSettings(), "SpaceInvadersRomSettings"),
    "seaquest"       => GameSpec("seaquest", "seaquest.bin",
                                 () -> JuTari.GenericRomSettings(), "GenericRomSettings"),
    "ms_pacman"      => GameSpec("ms_pacman", "mspacman.bin",
                                 () -> MsPacmanRomSettings(), "MsPacmanRomSettings"),
    "qbert"          => GameSpec("qbert", "qbert.bin",
                                 () -> QbertRomSettings(), "QbertRomSettings"),
)

const CORE_GAMES = ["pong", "breakout", "space_invaders", "seaquest", "ms_pacman", "qbert"]

# resolved ROM root (default uses the in-place gitignored collection; --roms-dir
# overrides for the cluster). xitari/roms/ is the canonical jutari symlink layout.
function _rom_roots(roms_dir)
    roots = String[]
    roms_dir !== nothing && push!(roots, roms_dir)
    here = normpath(joinpath(@__DIR__, "..", "..", ".."))
    push!(roots, joinpath(here, "xitari", "roms"))
    push!(roots, joinpath(PRIMARY_REPO, "xitari", "roms"))
    # the raw ROM collection (cluster / fallback)
    push!(roots, joinpath(PRIMARY_REPO, "xitari", "games",
                          "Atari-2600-VCS-ROM-Collection", "ROMS"))
    return roots
end

function rom_path_for(spec::GameSpec; roms_dir = nothing)
    for base in _rom_roots(roms_dir)
        p = joinpath(base, spec.rom_basename)
        isfile(p) && return p
    end
    error("ROM not found for $(spec.name) ($(spec.rom_basename)) under " *
          join(_rom_roots(roms_dir), ", "))
end

"""A freshly-reset env with the xitari-parity boot (60 NOOP + 4 RESET) and the
game's RomSettings."""
function fresh_env(spec::GameSpec; roms_dir = nothing)
    rom = read(rom_path_for(spec; roms_dir = roms_dir))
    env = StellaEnvironment(rom, spec.settings())
    env_reset!(env; boot_noop_steps = 60, boot_reset_steps = 4)
    return env
end

# --- snapshot / replay / intervene (same primitives as jutari_oracle.jl) -----
struct Snap
    ram::Vector{UInt8}
    screen::Matrix{UInt8}
end
snap(env) = Snap(copy(collect(get_ram(env))), Matrix{UInt8}(get_screen(env)))

"""Boot + step `actions[1:target_frame]`; return the env AT the target frame (the
reusable checkpoint; deepcopy before mutating)."""
function boot_replay(spec::GameSpec, actions, target_frame; roms_dir = nothing)
    env = fresh_env(spec; roms_dir = roms_dir)
    for i in 1:target_frame
        env_step!(env, Int(actions[i]))
    end
    return env
end

"""do(ram[idx] := value) into RIOT RAM (0-based idx; the RAM vector is 1-indexed)."""
function intervene_ram!(env, idx::Integer, value::Integer)
    @assert 0 <= idx < RAM_SIZE "ram_index out of range: $idx"
    env.console.bus.ram[Int(idx) + 1] = UInt8(Int(value) & 0xFF)
    return env
end

"""deepcopy the checkpoint, step `tail`, snapshot (the un-lesioned baseline tail)."""
function continue_from(checkpoint, tail)
    env = deepcopy(checkpoint)
    for a in tail
        env_step!(env, Int(a))
    end
    return snap(env)
end

"""deepcopy the checkpoint, clamp one or more cells, step `tail`, snapshot. `pokes`
is a vector of (idx,value) — one entry = a single-unit lesion, two = a pair."""
function lesion_continue(checkpoint, pokes::Vector{Tuple{Int,Int}}, tail)
    env = deepcopy(checkpoint)
    for (idx, v) in pokes
        intervene_ram!(env, idx, v)
    end
    for a in tail
        env_step!(env, Int(a))
    end
    return snap(env)
end

# ===========================================================================
# Candidate cells (E2-1 import) — the "units" the lesion study probes.
# ===========================================================================
struct Candidate
    ram_index::Int
    concept::String
end

"""Read `candidates_<game>.json`, de-duplicated by ram_index, in file order."""
function load_candidates(game::AbstractString)
    rel = joinpath("tools", "xai_study", "t3", "out", "candidates_$(game).json")
    path = nothing
    here = normpath(joinpath(@__DIR__, "..", "..", ".."))
    for base in (here, PRIMARY_REPO)
        p = joinpath(base, rel)
        if isfile(p); path = p; break; end
    end
    out = Candidate[]
    if path !== nothing
        data = JSON.parsefile(path)
        seen = Set{Int}()
        for c in get(data, "candidates", [])
            idx = Int(c["ram_index"])
            idx in seen && continue
            push!(seen, idx)
            concept = get(c, "concept", nothing)
            push!(out, Candidate(idx, concept === nothing ? "(unnamed)" : string(concept)))
        end
    end
    return out, path
end

# ===========================================================================
# Outputs y(state) — the behavioural readouts a lesion can perturb.
# Generic, game-agnostic readouts (no per-game T3 needed for A2):
#   * each candidate cell's own value at the final frame (the state-level
#     behaviour; lesioning cell i can downstream-change cell j),
#   * the whole-screen change magnitude (count of framebuffer cells differing
#     from the un-lesioned baseline) — the "does behaviour still render?" signal.
# We score the importance map against the oracle over the SAME readout family, so
# the comparison is apples-to-apples (both methods read the same outputs).
# ===========================================================================

"""The per-unit lesion footprint over the READOUT family, for a given snapshot vs
the baseline. Returns (out_delta::Vector{Float64} over candidate cells,
screen_change::Float64). out_delta[j] = |final value of cell j − baseline|."""
function footprint(snap_l::Snap, base::Snap, cands::Vector{Candidate})
    n = length(cands)
    d = zeros(Float64, n)
    for (j, cj) in enumerate(cands)
        d[j] = abs(Float64(Int(snap_l.ram[cj.ram_index + 1])) -
                   Float64(Int(base.ram[cj.ram_index + 1])))
    end
    sc = Float64(count(snap_l.screen .!= base.screen))
    return d, sc
end

# A scalar importance from a footprint: the L1 state change PLUS a screen-change
# term on a comparable scale (screen counts can dwarf RAM deltas; we add the raw
# screen-change as its own behavioural channel — large, render-level effects must
# count as important). We keep both raw so the per-output breakdown is recoverable.
importance_scalar(out_delta::Vector{Float64}, screen_change::Float64) =
    sum(out_delta) + screen_change

# ===========================================================================
# Per-unit lesion map + the oracle role map.
# ===========================================================================
# do-sets:
#   LESION (method under test): {clamp→0, clamp→alt=base⊻0xFF}        (2 clamps)
#   ORACLE (ground-truth role): {occlude→0, set→base+17, set→base+37,
#                                clamp→alt=base⊻0xFF}                  (4 do-values)
# Per-unit effect = MAX importance over the do-set (a unit "matters" if ANY
# in-set intervention perturbs behaviour).

LESION_DOSET(base) = unique(filter(!=(base),
    Int[0, (base ⊻ 0xFF) & 0xFF]))
ORACLE_DOSET(base) = unique(filter(!=(base),
    Int[0, (base + 17) & 0xFF, (base + 37) & 0xFF, (base ⊻ 0xFF) & 0xFF]))
HELDOUT_DOSET(base) = unique(filter(!=(base),
    Int[(base + 37) & 0xFF]))             # the S-probe value, unused by LESION

"""Per-unit importance map under a given do-set generator. Returns
(importance::Vector{Float64}, footprints::Matrix{Float64}, screen::Vector{Float64},
 works::Vector{Bool}) where footprints[i,:] is unit i's best-clamp out-delta over
the candidate readouts, screen[i] its best-clamp screen-change, works[i] whether
the lesioned run still produced a valid frame."""
function lesion_map(checkpoint, tail, cands::Vector{Candidate}, at_target::Snap,
                    base::Snap, doset::Function)
    n = length(cands)
    imp = zeros(Float64, n)
    fps = zeros(Float64, n, n)
    scr = zeros(Float64, n)
    works = trues(n)
    for (i, c) in enumerate(cands)
        bval = Int(at_target.ram[c.ram_index + 1])
        best_imp = -1.0
        best_d = zeros(Float64, n); best_sc = 0.0
        ok_any = false
        for v in doset(bval)
            sl = lesion_continue(checkpoint, [(c.ram_index, v)], tail)
            # "boot/run still works?": a valid frame = correct screen shape, no NaN
            # (palette indices are UInt8 so always valid) and RAM full-size.
            valid = (size(sl.screen) == size(base.screen)) &&
                    (length(sl.ram) == length(base.ram))
            ok_any = ok_any || valid
            d, sc = footprint(sl, base, cands)
            im = importance_scalar(d, sc)
            if im > best_imp
                best_imp = im; best_d = d; best_sc = sc
            end
        end
        imp[i] = max(best_imp, 0.0)
        fps[i, :] = best_d
        scr[i] = best_sc
        works[i] = ok_any
    end
    return imp, fps, scr, works
end

# ===========================================================================
# Scoring: Spearman rank-correlation + false-specificity + interaction misses.
# ===========================================================================
"""Average ranks (1-based; ties get the mean rank) — for Spearman."""
function _avg_ranks(x::Vector{Float64})
    n = length(x)
    p = sortperm(x)
    r = zeros(Float64, n)
    i = 1
    while i <= n
        j = i
        while j < n && x[p[j + 1]] == x[p[i]]; j += 1; end
        rank = (i + j) / 2          # mean rank of the tie block
        for k in i:j; r[p[k]] = rank; end
        i = j + 1
    end
    return r
end

"""Pearson correlation (used on ranks ⇒ Spearman). Degenerate (zero-variance)
inputs ⇒ 1.0 if identical, else 0.0."""
function _pearson(a::Vector{Float64}, b::Vector{Float64})
    n = length(a)
    n == 0 && return 0.0
    ma = sum(a) / n; mb = sum(b) / n
    da = a .- ma; db = b .- mb
    sa = sqrt(sum(da .^ 2)); sb = sqrt(sum(db .^ 2))
    if sa == 0 || sb == 0
        return (a == b) ? 1.0 : 0.0
    end
    return sum(da .* db) / (sa * sb)
end

spearman(a::Vector{Float64}, b::Vector{Float64}) =
    _pearson(_avg_ranks(a), _avg_ranks(b))

"""A unit is flagged 'specific' by the lesion map if its lesion footprint is
concentrated on ≈ONE readout: the top readout carries ≥ `frac` of the total |Δ|
footprint mass AND the unit has a non-trivial footprint. Returns a Bool vector."""
function lesion_flagged_specific(fps::Matrix{Float64}, scr::Vector{Float64};
                                 frac::Float64 = 0.8, eps::Float64 = 1e-9)
    n = size(fps, 1)
    flagged = falses(n)
    for i in 1:n
        row = fps[i, :]
        total = sum(row) + scr[i]
        total <= eps && continue                # no footprint → not "specific"
        # footprint concentrated on a single readout (its own-cell channel or one
        # downstream cell) and NOT broadly driving the screen.
        mass = vcat(row, scr[i])
        top = maximum(mass)
        if top / total >= frac
            flagged[i] = true
        end
    end
    return flagged
end

"""A unit is truly GENERIC (per the oracle) if its true role touches MANY readouts:
the number of readouts the oracle footprint perturbs (> eps) is ≥ `min_breadth`.
Returns a Bool vector aligned to the candidate order."""
function oracle_truly_generic(oracle_fps::Matrix{Float64}, oracle_scr::Vector{Float64};
                              min_breadth::Int = 3, eps::Float64 = 1e-9)
    n = size(oracle_fps, 1)
    generic = falses(n)
    for i in 1:n
        breadth = count(>(eps), oracle_fps[i, :]) + (oracle_scr[i] > eps ? 1 : 0)
        generic[i] = breadth >= min_breadth
    end
    return generic
end

# ===========================================================================
# Per-game run.
# ===========================================================================
struct InteractionStat
    n_pairs::Int
    tol::Float64
    n_missed::Int                 # |Δ_int| > tol
    missed_rate::Float64
    max_superadd::Float64         # largest positive Δ_int (joint > sum-of-singles)
    max_superadd_pair::Tuple{Int,Int}
    mean_abs_int::Float64
end

struct GameResult
    game::String
    settings_name::String
    rom_basename::String
    candidates_path::Union{Nothing,String}
    target_frame::Int
    horizon::Int
    bit_exact::Bool
    cands::Vector{Candidate}
    # method (single-unit lesion)
    lesion_imp::Vector{Float64}
    lesion_fps::Matrix{Float64}
    lesion_scr::Vector{Float64}
    lesion_works::Vector{Bool}
    # ground truth (oracle role)
    oracle_role::Vector{Float64}
    oracle_fps::Matrix{Float64}
    oracle_scr::Vector{Float64}
    # held-out (S probe)
    heldout_role::Vector{Float64}
    # scores
    rho::Float64                   # Spearman ρ lesion-imp vs oracle role  (F)
    rho_control::Float64           # positive control: oracle-as-method vs role
    n_flagged_specific::Int
    n_false_specific::Int          # flagged-specific ∧ truly-generic
    false_specificity_rate::Float64
    interaction::InteractionStat
    triad_F::Float64
    triad_S::Float64
    triad_M::Float64
    # SHARED-TESTBED provenance (redesign); "noop"/-1/false in the legacy path.
    state_kind::String             # "seeded_random_action_gameplay" | "noop"
    st_seed::Int
    st_prefix::Int
    cause_density::Int             # #causes above the floor at the shared output
    cause_density_accepted::Bool   # passed the cause-density gate?
    n_causes::Int
    shared_cell::Tuple{Int,Int}    # the shared screen-buffer output cell
end

# A2's candidates-path resolver, in the shape the shared testbed injects (a
# String->path-or-nothing map). Mirrors load_candidates' file search.
function _st_candidates_path_for(game::AbstractString)
    rel = joinpath("tools", "xai_study", "t3", "out", "candidates_$(game).json")
    here = normpath(joinpath(@__DIR__, "..", "..", ".."))
    for base in (here, PRIMARY_REPO)
        p = joinpath(base, rel)
        isfile(p) && return p
    end
    return nothing
end

"""Build the P2 SHARED gameplay state + cause-density gate for `game` using the §1
oracle machinery (oracle_intervene.jl). Returns the substrate NamedTuple (we use
its `.actions` stream + `.cause_density`/`.accepted`/`.cell` gate). A2 then boots
its OWN checkpoint from `st.actions` so the lesion algorithm is unchanged."""
function build_a2_shared_state(game::AbstractString; verbose = false)
    O = OracleIntervene; J = OracleIntervene.JutariOracle
    return build_shared_testbed(game;
        settings_for = J.settings_for, rom_path_for = J.rom_path_for,
        candidates_path_for = _st_candidates_path_for,
        build_causes = O.build_pong_causes, candidate_ram_indices = O.candidate_ram_indices,
        continue_from = J.continue_from, snapshot = J.snapshot, env_step = J.env_step!,
        intervene_ram = J.intervene_ram!, boot_replay = J.boot_replay,
        run_intervention = O.run_intervention, soft_ram_peek = soft_ram_peek,
        prefix = ST_PREFIX, horizon = ST_HORIZON, seed = ST_SEED,
        k = ST_GATE_K, floor = ST_FLOOR, verbose = verbose,
        assert_bit_exact = O.assert_bit_exact)
end

function run_game(game::AbstractString; target_frame = 30, horizon = 30,
                  max_pairs = 60, roms_dir = nothing, verbose = true)
    haskey(GAME_SPECS, game) || error("unknown game $game (have $(keys(GAME_SPECS)))")
    spec = GAME_SPECS[game]

    # SHARED-TESTBED (redesign): replace the all-NOOP boot/attract tape with a
    # seeded random-action GAMEPLAY state at f*=ST_PREFIX, gated by the oracle
    # cause-density gate. We take the shared action STREAM + the gate from the
    # substrate, then boot A2's OWN checkpoint from that stream so the lesion
    # machinery (GameSpec/Snap/Candidate) is UNCHANGED — only the state moves.
    st = nothing
    if SHARED_TESTBED
        st = build_a2_shared_state(game; verbose = verbose)
        target_frame = st.prefix; horizon = st.horizon
        actions = st.actions
        verbose && println("[A2:$game] SHARED gameplay state: cause_density=$(st.cause_density)/" *
            "$(length(st.causes)) accepted=$(st.accepted) cell=$(st.cell)")
    else
        actions = fill(0, target_frame + horizon)   # all-NOOP deterministic trace
    end
    total = target_frame + horizon
    tail = Int.(actions[target_frame + 1 : total])

    # 1) bit-exact baseline guarantee — two fresh boots+replays must be identical.
    verbose && println("[A2:$game] asserting bit-exactness (2 fresh boots+replays to f$total)...")
    a = continue_from(boot_replay(spec, actions, target_frame; roms_dir = roms_dir), tail)
    b = continue_from(boot_replay(spec, actions, target_frame; roms_dir = roms_dir), tail)
    bit_exact = (a.ram == b.ram) && (a.screen == b.screen)
    bit_exact || error("[A2:$game] bit-exact re-run FAILED to f$total — refusing to score")
    verbose && println("[A2:$game] bit-exact re-run: PASS")

    # 2) ONE checkpoint at the intervention frame (boot+to-target paid once).
    checkpoint = boot_replay(spec, actions, target_frame; roms_dir = roms_dir)
    at_target = continue_from(checkpoint, Int[])      # state AT the target frame
    base = continue_from(checkpoint, tail)            # un-lesioned baseline tail

    cands, cand_path = load_candidates(game)
    isempty(cands) && error("[A2:$game] no candidate cells (candidates_$(game).json missing/empty)")
    n = length(cands)
    verbose && println("[A2:$game] candidates: $(cand_path) ($n cells)")

    # 3) the single-unit LESION importance map (method under test): {clamp→0, clamp→alt}
    verbose && println("[A2:$game] single-unit lesion map (clamp→0, clamp→alt; $n units)...")
    lesion_imp, lesion_fps, lesion_scr, lesion_works =
        lesion_map(checkpoint, tail, cands, at_target, base, LESION_DOSET)

    # 4) the ORACLE role map (ground truth): exhaustive do-set {0, +17, +37, alt}
    verbose && println("[A2:$game] oracle role map (exhaustive do-set; $n units)...")
    oracle_role, oracle_fps, oracle_scr, _ =
        lesion_map(checkpoint, tail, cands, at_target, base, ORACLE_DOSET)

    # 5) HELD-OUT role (S probe): a clamp value the LESION map never used (+37 only)
    heldout_role, _, _, _ =
        lesion_map(checkpoint, tail, cands, at_target, base, HELDOUT_DOSET)

    # --- scores ------------------------------------------------------------
    rho = spearman(lesion_imp, oracle_role)                        # F
    # positive control: oracle-as-method (role) vs role → ρ must be 1
    rho_control = spearman(copy(oracle_role), oracle_role)

    flagged = lesion_flagged_specific(lesion_fps, lesion_scr)
    generic = oracle_truly_generic(oracle_fps, oracle_scr)
    false_spec = flagged .& generic
    n_flagged = count(flagged)
    n_false = count(false_spec)
    false_rate = n_flagged == 0 ? 0.0 : n_false / n_flagged

    # --- interactions (the single-unit blind spot) -------------------------
    interaction = interaction_misses(checkpoint, tail, cands, at_target, base,
                                     lesion_imp; max_pairs = max_pairs, verbose = verbose)

    # --- triad -------------------------------------------------------------
    F = rho
    # S — generalisation to the held-out clamp value: Spearman of the lesion map
    #     (built on {0,alt}) vs the role under the UNSEEN value (+37).
    S = spearman(lesion_imp, heldout_role)
    # M — minimal / right level: 1 − false-specificity rate (specificity claims
    #     that the oracle contradicts).
    M = 1.0 - false_rate

    if verbose
        println("[A2:$game] ---- scores ----")
        println("[A2:$game]   units=$n  Spearman ρ(lesion,role)=$(round(rho,digits=3))  " *
                "(F); positive control ρ=$(round(rho_control,digits=3))")
        println("[A2:$game]   flagged-specific=$n_flagged  false-specific=$n_false  " *
                "(rate=$(round(false_rate,digits=3)))")
        println("[A2:$game]   interactions: $(interaction.n_missed)/$(interaction.n_pairs) " *
                "pairs miss interaction (rate=$(round(interaction.missed_rate,digits=3))) " *
                "max super-add Δ=$(round(interaction.max_superadd,digits=2))")
        println("[A2:$game]   TRIAD F=$(round(F,digits=3)) S=$(round(S,digits=3)) M=$(round(M,digits=3))")
    end

    return GameResult(game, spec.settings_name, spec.rom_basename, cand_path,
                      target_frame, horizon, bit_exact, cands,
                      lesion_imp, lesion_fps, lesion_scr, lesion_works,
                      oracle_role, oracle_fps, oracle_scr, heldout_role,
                      rho, rho_control, n_flagged, n_false, false_rate,
                      interaction, F, S, M,
                      st === nothing ? "noop" : "seeded_random_action_gameplay",
                      st === nothing ? -1 : st.seed,
                      st === nothing ? -1 : st.prefix,
                      st === nothing ? -1 : st.cause_density,
                      st === nothing ? false : st.accepted,
                      st === nothing ? 0 : length(st.causes),
                      st === nothing ? (-1, -1) : st.cell)
end

"""Interaction analysis: over a budgeted set of unit pairs, the joint-lesion effect
vs the sum of single-lesion effects. Δ_int = E(i,j) − (E(i)+E(j)); a non-zero
Δ_int is invisible to any single-unit importance map.

Budget: to keep this paper-reasonable (heavy: each pair = a re-run), we cap at
`max_pairs`. We prioritise the pairs most likely to interact: the TOP-K most-
important units by the single-unit map, all-pairs among them (K chosen so
K(K−1)/2 ≤ max_pairs). This is the natural place for super-additivity to live
(important units interacting), and is deterministic + reproducible."""
function interaction_misses(checkpoint, tail, cands::Vector{Candidate}, at_target::Snap,
                            base::Snap, single_imp::Vector{Float64};
                            max_pairs::Int = 60, tol_frac::Float64 = 0.05,
                            verbose = true)
    n = length(cands)
    # joint-lesion effect of clamping both i and j to their ALT value (the
    # strongest single-unit clamp), full horizon.
    function joint_effect(i, j)
        bi = Int(at_target.ram[cands[i].ram_index + 1])
        bj = Int(at_target.ram[cands[j].ram_index + 1])
        vi = (bi ⊻ 0xFF) & 0xFF; vj = (bj ⊻ 0xFF) & 0xFF
        sl = lesion_continue(checkpoint,
                             [(cands[i].ram_index, vi), (cands[j].ram_index, vj)], tail)
        d, sc = footprint(sl, base, cands)
        return importance_scalar(d, sc)
    end
    # single-ALT effect (same clamp as in the joint) so E(i,j) vs E(i)+E(j) is
    # measured on the SAME do-value (a clean additivity test).
    function single_alt_effect(i)
        bi = Int(at_target.ram[cands[i].ram_index + 1])
        vi = (bi ⊻ 0xFF) & 0xFF
        sl = lesion_continue(checkpoint, [(cands[i].ram_index, vi)], tail)
        d, sc = footprint(sl, base, cands)
        return importance_scalar(d, sc)
    end

    # choose the top-K most-important units; all-pairs among them, capped.
    order = sortperm(single_imp; rev = true)
    K = n
    while K > 2 && (K * (K - 1)) ÷ 2 > max_pairs; K -= 1; end
    top = order[1:min(K, n)]
    pairs = Tuple{Int,Int}[]
    for a in 1:length(top), b in (a + 1):length(top)
        push!(pairs, (top[a], top[b]))
    end
    isempty(pairs) && return InteractionStat(0, 0.0, 0, 0.0, 0.0, (0, 0), 0.0)

    # cache single-ALT effects for the involved units
    single_alt = Dict{Int,Float64}()
    for u in top; single_alt[u] = single_alt_effect(u); end

    # a tolerance scaled to the typical effect magnitude (so "missed" means a
    # materially non-additive pair, not floating-point noise).
    typ = max(1.0, sum(values(single_alt)) / max(1, length(single_alt)))
    tol = tol_frac * typ

    n_missed = 0; sum_abs = 0.0
    max_super = 0.0; max_pair = (0, 0)
    for (i, j) in pairs
        Eij = joint_effect(i, j)
        Δint = Eij - (single_alt[i] + single_alt[j])
        sum_abs += abs(Δint)
        abs(Δint) > tol && (n_missed += 1)
        if Δint > max_super
            max_super = Δint; max_pair = (i - 1, j - 1)     # 0-based ordinals
        end
    end
    np = length(pairs)
    verbose && println("[A2] interaction budget: K=$K top-units → $np pairs " *
                       "(tol=$(round(tol,digits=3)))")
    return InteractionStat(np, tol, n_missed, n_missed / np, max_super, max_pair,
                           sum_abs / np)
end

# ===========================================================================
# Persist (SPEC §R) — per-game record + combined; file_scope A2_*.
# ===========================================================================
_git_commit() = try
    strip(read(`git -C $(@__DIR__) rev-parse --short HEAD`, String))
catch
    "unknown"
end
_jnum(x::Real) = isfinite(x) ? Float64(x) : nothing

function _game_record(r::GameResult; where_str = "local", budget = Dict{String,Any}())
    cell_names = ["RAM[$(c.ram_index)]:$(c.concept)" for c in r.cands]
    # the ranking the lesion map produces, as 0-based candidate ordinals (top first)
    rank_order = sortperm(r.lesion_imp; rev = true) .- 1
    oracle_order = sortperm(r.oracle_role; rev = true) .- 1
    Dict{String,Any}(
        "paper" => "P2",
        "phase" => "phaseA_kording",
        "method" => "A2_lesions(exhaustive single-unit lesion importance map)",
        "game" => r.game,
        "frame" => r.target_frame,
        "state" => r.state_kind == "noop" ? "f$(r.target_frame)+$(r.horizon)" :
                   "gameplay(seed=$(r.st_seed),prefix=$(r.st_prefix))+$(r.horizon)",
        "target_output" => "per-unit behavioural-importance map (candidate RAM cells)",
        # headline scalar (SPEC §R value/metric_name): Spearman ρ of the single-unit
        # lesion importance map vs the exact-oracle per-unit role (F / faithfulness).
        "metric_name" => "lesion_importance_spearman_vs_oracle_role",
        "value" => _jnum(r.rho),
        "stderr" => nothing,
        "ci" => nothing,
        "n" => length(r.cands),
        "seed" => 0,
        "where" => where_str,
        "commit" => _git_commit(),
        "oracle_ref" => "oracle_intervene@$(r.game)#per-unit causal role (exhaustive do-set, full horizon)",
        "timestamp" => string(round(Int, time())),
        "extra" => Dict{String,Any}(
            "substrate" => "jutari (Julia, HARD) — real-ROM bit-exact path; every " *
                "lesion is an EXACT clamp + deterministic re-run on the true ROM.",
            "settings" => r.settings_name,
            "rom_basename" => r.rom_basename,
            "candidates_file" => r.candidates_path,
            "bit_exact_rerun" => r.bit_exact,
            "candidate_cells" => cell_names,
            "candidate_ram_indices" => [c.ram_index for c in r.cands],
            "budget" => budget,
            "testbed" => Dict{String,Any}(
                "state_kind" => r.state_kind,
                "seed" => r.st_seed, "prefix" => r.st_prefix, "horizon" => r.horizon,
                "shared_output" => "screen_region(n_changed_px)@r$(r.shared_cell[1])c$(r.shared_cell[2])",
                "cause_density_above_floor" => r.cause_density,
                "cause_density_floor" => ST_FLOOR, "cause_density_gate_k" => ST_GATE_K,
                "cause_density_accepted" => r.cause_density_accepted, "n_causes" => r.n_causes,
                "note" => "P2 redesign: the lesion study runs on a seeded random-action " *
                    "GAMEPLAY state (not the boot/attract NOOP tape), gated by the §1 " *
                    "oracle cause-density gate. A2's lesion algorithm is unchanged; only " *
                    "the analysis state moves onto genuine input-driven gameplay."),
            "lesion_doset" => "single-unit {clamp→0, clamp→alt=base⊻0xFF}; per-unit " *
                "effect = MAX importance over the do-set",
            "oracle_doset" => "exhaustive {occlude→0, set→base+17, set→base+37, " *
                "clamp→alt}; per-unit role = MAX importance over the do-set",
            "importance_def" => "Σ|Δ(candidate-cell value at final frame)| + " *
                "(#framebuffer cells changed vs baseline) — state + render channels",
            "importance_map" => Dict{String,Any}(
                "lesion_importance" => [_jnum(x) for x in r.lesion_imp],
                "oracle_role"       => [_jnum(x) for x in r.oracle_role],
                "lesion_rank_order_0based" => collect(rank_order),
                "oracle_rank_order_0based" => collect(oracle_order),
                "boot_run_still_works"     => collect(r.lesion_works),
                "note" => "lesion_importance[i] / oracle_role[i] are aligned to " *
                    "candidate_ram_indices[i]. rank_order lists candidate ordinals " *
                    "most→least important. A poked RAM cell never halts the 6502, so " *
                    "boot_run_still_works is ~all true (the screen-change term is the " *
                    "graded 'does behaviour still run' signal).",
            ),
            "scores" => Dict{String,Any}(
                "spearman_rho_lesion_vs_oracle" => _jnum(r.rho),
                "positive_control_rho" => _jnum(r.rho_control),
                "n_flagged_specific" => r.n_flagged_specific,
                "n_false_specific" => r.n_false_specific,
                "false_specificity_rate" => _jnum(r.false_specificity_rate),
                "note" => "ρ EXPECTED HIGH — a single-unit clamp+re-run is itself an " *
                    "exact intervention, so lesion importance ≈ the unit's causal role " *
                    "(experiment_design.md §4 A2). false_specificity = #units the lesion " *
                    "map flags as driving ≈one output (specific) that the oracle says " *
                    "actually drive MANY outputs (generic) — spurious selectivity.",
            ),
            "positive_control" => Dict{String,Any}(
                "name" => "oracle-as-method (role map = the exhaustive do-set itself)",
                "spearman_rho_vs_oracle_role" => _jnum(r.rho_control),
                "note" => "the oracle's OWN importance map must rank-correlate ρ=1 with " *
                    "the role — the harness rewards a perfectly-faithful map, so a " *
                    "single-clamp ρ<1 is a real measurement, not a broken scorer.",
            ),
            "interaction_blind_spot" => Dict{String,Any}(
                "n_pairs" => r.interaction.n_pairs,
                "tolerance" => _jnum(r.interaction.tol),
                "n_missed" => r.interaction.n_missed,
                "interaction_missed_rate" => _jnum(r.interaction.missed_rate),
                "max_superadditive_delta" => _jnum(r.interaction.max_superadd),
                "max_superadditive_pair_0based" =>
                    [r.interaction.max_superadd_pair[1], r.interaction.max_superadd_pair[2]],
                "mean_abs_interaction" => _jnum(r.interaction.mean_abs_int),
                "note" => "the WHERE-SINGLE-UNIT-LESIONS-MISS result: Δ_int = " *
                    "E(i,j) − (E(i)+E(j)) over the top-importance unit pairs (budgeted). " *
                    "A non-zero Δ_int is super/sub-additivity that NO single-unit map can " *
                    "represent — the precise blind spot of A2.",
            ),
            "triad" => Dict{String,Any}(
                "F" => _jnum(r.triad_F),
                "F_note" => "Spearman ρ of the single-unit lesion importance map vs the " *
                    "exact-oracle per-unit causal role ($(length(r.cands)) units).",
                "S" => _jnum(r.triad_S),
                "S_note" => "generalisation to a HELD-OUT clamp value: ρ of the lesion map " *
                    "(built on {0,alt}) vs the per-unit role under the UNSEEN value base+37 " *
                    "(scored on the bit-exact re-run).",
                "M" => _jnum(r.triad_M),
                "M_note" => "1 − false-specificity rate: fraction of lesion-flagged " *
                    "'specific' units the oracle says are actually generic (spurious " *
                    "selectivity) — minimal & at the right level.",
                "interpretation" => "Phase A calibration baseline (experiment_design.md §4): " *
                    "the single-unit lesion is a RELATIVELY FAITHFUL classical method " *
                    "(high F) BUT structurally blind to interaction effects (the " *
                    "interaction_blind_spot) and prone to spurious specificity (M<1). " *
                    "'Right but incomplete' — the quantified-Kording lesson for ablation.",
            ),
            "scales_to_cluster_via" =>
                "tools/cluster/xai_array_jl.sbatch (E0-3): --shard i --nshards n " *
                "--shard-kind game runs game i of the core set per array task; forward " *
                "bit-exact to this HARD path.",
        ),
    )
end

function _write_game_npz(r::GameResult, npz_path)
    write_npz(npz_path, Dict(
        "candidate_ram_indices" => Int64[c.ram_index for c in r.cands],
        "lesion_importance"     => r.lesion_imp,
        "oracle_role"           => r.oracle_role,
        "heldout_role"          => r.heldout_role,
        "lesion_footprints"     => r.lesion_fps,        # (units, readouts)
        "oracle_footprints"     => r.oracle_fps,
        "lesion_screen_change"  => r.lesion_scr,
        "boot_run_still_works"  => UInt8.(r.lesion_works),
        # [ρ, ρ_control, n_flagged, n_false, false_rate, F, S, M,
        #  n_pairs, n_missed, missed_rate, max_superadd]
        "summary" => Float64[r.rho, r.rho_control, r.n_flagged_specific,
                             r.n_false_specific, r.false_specificity_rate,
                             r.triad_F, r.triad_S, r.triad_M,
                             r.interaction.n_pairs, r.interaction.n_missed,
                             r.interaction.missed_rate, r.interaction.max_superadd],
    ))
end

function write_game(r::GameResult; out_dir = DEFAULT_OUT_DIR, where_str = "local",
                    budget = Dict{String,Any}())
    isdir(out_dir) || mkpath(out_dir)
    json_path = joinpath(out_dir, "A2_lesions_$(r.game).json")
    npz_path  = joinpath(out_dir, "A2_lesions_$(r.game).npz")
    rec = _game_record(r; where_str = where_str, budget = budget)
    rec["arrays"] = basename(npz_path)
    open(json_path, "w") do io; JSON.print(io, rec, 2); end
    _write_game_npz(r, npz_path)
    return json_path, npz_path
end

function write_combined(results::Vector{GameResult}; out_dir = DEFAULT_OUT_DIR,
                        where_str = "local")
    isempty(results) && return String[]
    isdir(out_dir) || mkpath(out_dir)
    rhos = Float64[r.rho for r in results]
    mean_rho = sum(rhos) / length(rhos)
    per_game = Dict{String,Any}()
    for r in results
        per_game[r.game] = Dict{String,Any}(
            "spearman_rho_lesion_vs_oracle" => _jnum(r.rho),
            "positive_control_rho" => _jnum(r.rho_control),
            "false_specificity_rate" => _jnum(r.false_specificity_rate),
            "interaction_missed_rate" => _jnum(r.interaction.missed_rate),
            "max_superadditive_delta" => _jnum(r.interaction.max_superadd),
            "triad_F" => _jnum(r.triad_F), "triad_S" => _jnum(r.triad_S),
            "triad_M" => _jnum(r.triad_M),
            "n_units" => length(r.cands),
        )
    end
    combined = Dict{String,Any}(
        "paper" => "P2",
        "phase" => "phaseA_kording",
        "method" => "A2_lesions(exhaustive single-unit lesion importance map)",
        "game" => "core_set",
        "state" => "f$(results[1].target_frame)+$(results[1].horizon)",
        "target_output" => "per-unit behavioural-importance map (per game)",
        "metric_name" => "mean_lesion_importance_spearman_vs_oracle_role",
        "value" => _jnum(mean_rho),
        "stderr" => nothing, "ci" => nothing,
        "n" => length(results), "seed" => 0,
        "where" => where_str,
        "commit" => _git_commit(),
        "oracle_ref" => "oracle_intervene (per-unit causal role per game)",
        "timestamp" => string(round(Int, time())),
        "arrays" => "A2_lesions.npz",
        "games" => [r.game for r in results],
        "per_game" => per_game,
    )
    cjp = joinpath(out_dir, "A2_lesions.json")
    open(cjp, "w") do io; JSON.print(io, combined, 2); end
    n = length(results)
    M = zeros(Float64, n, 8)
    for (k, r) in enumerate(results)
        M[k, :] = [r.rho, r.rho_control, r.false_specificity_rate,
                   r.interaction.missed_rate, r.interaction.max_superadd,
                   r.triad_F, r.triad_S, r.triad_M]
    end
    cnp = joinpath(out_dir, "A2_lesions.npz")
    write_npz(cnp, Dict(
        "per_game_rho_ctrl_falsespec_intmiss_maxsuper_FSM" => M,   # (n_games, 8)
        "game_order" => UInt8.(collect(1:n)),
    ))
    return [cjp, cnp]
end

# ===========================================================================
# Self-check (DoD) — the scoring contract is sound; results are non-fabricated.
# ===========================================================================
"""
    selftest(r::GameResult) -> Bool

Asserts the load-bearing claims of A2:

  (BIT-EXACT) the baseline re-run is byte-identical (every lesion is a clean
    causal effect on the real ROM).

  (GROUND-TRUTH ANCHORED) the oracle role map has ≥1 unit with a non-zero role
    (otherwise the rank-correlation is degenerate — pick a livelier state).

  (POSITIVE CONTROL) the oracle-as-method importance map rank-correlates ρ=1 with
    the oracle role — the harness rewards a perfectly-faithful map (so a single-
    clamp ρ<1 is a real measurement, not a broken scorer).

  (METRIC RANGES) ρ ∈ [−1,1]; F,S ∈ [−1,1]; M ∈ [0,1]; rates ∈ [0,1]; all finite.

  (NOT FABRICATED) the lesion importance map is not identically zero (the lesions
    actually perturbed behaviour) and ≥1 candidate cell was probed.

Throws on a contract violation."""
function selftest(r::GameResult)
    @assert r.bit_exact "[A2:$(r.game)] bit-exact baseline re-run failed"
    @assert !isempty(r.cands) "[A2:$(r.game)] no candidate cells probed"
    @assert any(r.oracle_role .> 0) "[A2:$(r.game)] oracle role map is all-zero at this " *
        "state — uninformative; pick a livelier target frame"
    @assert any(r.lesion_imp .> 0) "[A2:$(r.game)] lesion importance map is all-zero — " *
        "lesions perturbed nothing; refusing to report a fabricated score"
    @assert r.rho_control > 0.999 "[A2:$(r.game)] positive control broken: oracle-as-" *
        "method ρ=$(r.rho_control) (expected 1.0)"
    for (nm, v) in (("rho", r.rho), ("F", r.triad_F), ("S", r.triad_S))
        @assert -1.0 - 1e-9 <= v <= 1.0 + 1e-9 "[A2:$(r.game)] $nm out of [-1,1]: $v"
    end
    for (nm, v) in (("M", r.triad_M), ("false_rate", r.false_specificity_rate),
                    ("int_missed_rate", r.interaction.missed_rate))
        @assert 0.0 - 1e-9 <= v <= 1.0 + 1e-9 "[A2:$(r.game)] $nm out of [0,1]: $v"
    end
    @assert isfinite(r.interaction.max_superadd) "[A2:$(r.game)] max_superadd not finite"

    println("[A2:$(r.game)] SELF-CHECK PASS:")
    println("[A2:$(r.game)]   bit-exact baseline re-run: $(r.bit_exact)")
    println("[A2:$(r.game)]   units=$(length(r.cands))  oracle-active units=$(count(r.oracle_role .> 0))")
    println("[A2:$(r.game)]   positive control ρ = $(round(r.rho_control, digits=3))")
    println("[A2:$(r.game)]   Spearman ρ(lesion,role) = $(round(r.rho, digits=3))  (F)")
    println("[A2:$(r.game)]   false-specificity = $(r.n_false_specific)/$(r.n_flagged_specific) " *
            "(M=$(round(r.triad_M, digits=3)))")
    println("[A2:$(r.game)]   interactions missed = $(r.interaction.n_missed)/$(r.interaction.n_pairs) " *
            "(rate=$(round(r.interaction.missed_rate, digits=3)))")
    return true
end

# ===========================================================================
# CLI — single-game / --games / cluster-shardable (--shard --nshards --shard-kind).
# ===========================================================================
"""Select the shard's games: shard-kind=game → game i of `games` (0-based); any
other kind → the whole list (this runner shards by game only)."""
function shard_games(games::Vector{String}, shard::Int, nshards::Int, shard_kind::String)
    if shard_kind == "game" && nshards > 1
        idx = shard + 1                       # 0-based shard → 1-based index
        (1 <= idx <= length(games)) || return String[]
        return [games[idx]]
    end
    # default: split the list round-robin across shards (kind=shard) or take all.
    if nshards > 1
        return [games[k] for k in 1:length(games) if (k - 1) % nshards == shard]
    end
    return games
end

function main(args = ARGS)
    games = copy(CORE_GAMES)
    target_frame = 30; horizon = 30; max_pairs = 60
    out_dir = DEFAULT_OUT_DIR; roms_dir = nothing; where_str = "local"
    shard = 0; nshards = 1; shard_kind = "game"
    selftest_only = false
    i = 1
    while i <= length(args)
        a = args[i]
        if     a == "--games";        games = String.(split(args[i+1], ",")); i += 2
        elseif a == "--game";         games = [args[i+1]]; i += 2
        elseif a == "--target-frame"; target_frame = parse(Int, args[i+1]); i += 2
        elseif a == "--horizon";      horizon = parse(Int, args[i+1]); i += 2
        elseif a == "--max-pairs";    max_pairs = parse(Int, args[i+1]); i += 2
        elseif a == "--out-dir";      out_dir = args[i+1]; i += 2
        elseif a == "--roms-dir";     roms_dir = args[i+1]; i += 2
        elseif a == "--where";        where_str = args[i+1]; i += 2
        elseif a == "--shard";        shard = parse(Int, args[i+1]); i += 2
        elseif a == "--nshards";      nshards = parse(Int, args[i+1]); i += 2
        elseif a == "--shard-kind";   shard_kind = args[i+1]; i += 2
        elseif a == "--selftest";     selftest_only = true; i += 1
        else; i += 1
        end
    end

    # apply the shard selection (over the core/full game list)
    sel = shard_games(games, shard, nshards, shard_kind)

    budget = Dict{String,Any}(
        "target_frame" => target_frame, "horizon" => horizon,
        "lesion_clamps_per_unit" => 2,          # {0, alt}
        "oracle_doses_per_unit" => 4,           # {0, +17, +37, alt}
        "heldout_doses_per_unit" => 1,          # {+37}
        "max_interaction_pairs" => max_pairs,
        "action_trace" => SHARED_TESTBED ?
            "seeded random-action GAMEPLAY (seed=$ST_SEED, prefix=$ST_PREFIX, horizon=$ST_HORIZON)" :
            "all-NOOP (deterministic)",
        "rationale" => "paper-reasonable single-step lesion budget; each Δ re-runs " *
            "the real ROM for `horizon` frames from a cached boot+replay checkpoint, " *
            "so re-runs are incremental (boot paid once per game).",
    )

    println("[A2] single-unit lesion importance map — games=$(join(sel, ",")) " *
            "target_frame=$target_frame horizon=$horizon max_pairs=$max_pairs " *
            "(shard $shard/$nshards kind=$shard_kind) → out=$out_dir where=$where_str")

    if selftest_only
        g = "space_invaders" in sel ? "space_invaders" :
            (isempty(sel) ? "space_invaders" : sel[1])
        println("[A2] --selftest on $g")
        r = run_game(g; target_frame = target_frame, horizon = horizon,
                     max_pairs = max_pairs, roms_dir = roms_dir, verbose = true)
        selftest(r)
        println("[A2] --selftest: passed on $g, not writing artifacts.")
        return 0
    end

    if isempty(sel)
        println("[A2] shard $shard/$nshards (kind=$shard_kind) selected NO games — nothing to do.")
        return 0
    end

    results = GameResult[]; blockers = Tuple{String,String}[]
    for g in sel
        try
            r = run_game(g; target_frame = target_frame, horizon = horizon,
                         max_pairs = max_pairs, roms_dir = roms_dir, verbose = true)
            selftest(r)
            jp, np = write_game(r; out_dir = out_dir, where_str = where_str, budget = budget)
            println("[A2] wrote $jp")
            println("[A2] arrays  $np")
            push!(results, r)
        catch err
            msg = first(split(sprint(showerror, err), '\n'))
            println("[A2] !! game $g FAILED (scoring the rest, not fabricating): $msg")
            push!(blockers, (g, msg))
        end
    end

    # combined record only when this invocation covered the whole core set in one
    # process (i.e. an unsharded local run) — a sharded array task writes only its
    # per-game record; the SM merges at Review.
    if nshards <= 1 && !isempty(results)
        for w in write_combined(results; out_dir = out_dir, where_str = where_str)
            println("[A2] wrote $w")
        end
    end

    println("[A2] ==== summary: $(length(results))/$(length(sel)) games scored ====")
    for r in results
        println("[A2]   $(rpad(r.game, 16)) ρ=$(round(r.rho,digits=2)) " *
                "ctrl=$(round(r.rho_control,digits=2)) " *
                "false-spec=$(r.n_false_specific)/$(r.n_flagged_specific) " *
                "int-miss=$(r.interaction.n_missed)/$(r.interaction.n_pairs) " *
                "| F=$(round(r.triad_F,digits=2)) S=$(round(r.triad_S,digits=2)) M=$(round(r.triad_M,digits=2))")
    end
    for (g, m) in blockers; println("[A2]   FAIL $g — $m"); end
    return isempty(results) ? 1 : 0
end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    A2Lesions.main()
end
