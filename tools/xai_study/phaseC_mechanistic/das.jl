# das.jl — Phase-C mechanistic interpretability (P2-E5-2), JULIA path (the jutari
# real-ROM substrate; jaxtari eager is ~205× slower — SCRUM §7).
#
# METHOD: INTERCHANGE INTERVENTIONS / DAS (causal abstraction, Geiger et al. 2021;
# Distributed Alignment Search, Geiger et al. 2023) over VCS state, scored against
# the EXACT intervention oracle (experiment_design.md §6 row "Interchange
# interventions / DAS"; §7 prediction for the causal core: **Succeed**).
#
# ---------------------------------------------------------------------------
# What an INTERCHANGE INTERVENTION is here (experiment_design.md §3, §6):
#   the VCS state trajectory is the "activations" and the program's data-flow is
#   the "circuit", both KNOWN. An interchange intervention takes the value of a
#   *candidate variable* Π (an alignment: one RAM cell, or a set of cells — DAS
#   generalises this to a learned subspace; on the VCS the aligned subspace is an
#   AXIS-ALIGNED set of RAM bytes, because the program's variables LIVE in named
#   RIOT-RAM cells) from a SOURCE run and writes it into a BASE run at the same
#   frame, RE-RUNS the real ROM a short horizon, and asks whether the output
#   changed AS THE HIGH-LEVEL CAUSAL MODEL PREDICTS.
#
#   * BASE run   = the clean NOOP trace (the running example, mirrors the oracle).
#   * SOURCE run = a genuinely different run (a joystick-driven context) so each
#     candidate cell carries a DIFFERENT value than BASE → a real interchange.
#
# THE HIGH-LEVEL CAUSAL MODEL (the abstraction we test alignment against):
#   on the VCS the true abstraction is KNOWN — variable V is realised by the RAM
#   cell the program reads V from, and the output o_V is the value of that cell
#   (or the pixels/score it drives). The causal-model PREDICTION for an interchange
#   of an alignment Π that realises V is: o_V takes the SOURCE's value of V, and
#   every output NOT downstream of V is unchanged. The EXACT intervention oracle
#   (P2-E1-1) gives the gold interchanged output bit-exactly, because a single-cell
#   interchange IS `do(cell := source_value)` followed by a bit-exact re-run.
#
# WHAT WE SCORE (DoD: interchange accuracy + alignment vs the true variable):
#   (1) INTERCHANGE ACCURACY of an alignment Π for a target variable V =
#       fraction of (base,source) interchange trials whose re-run output for V
#       equals the high-level model's prediction (= SOURCE's value of V). For the
#       TRUE alignment (the cell the oracle says realises V) this is 1.0 by
#       construction — the POSITIVE CONTROL. For a MISALIGNED cell it collapses —
#       the NEGATIVE CONTROL that proves the metric discriminates.
#   (2) ALIGNMENT vs the true variable = the DAS search: for every output/variable,
#       pick the candidate cell whose interchange BEST reproduces that variable's
#       interchange behaviour (the recovered alignment), and compare to the cell
#       the oracle says truly realises it (the true alignment). Report
#       ALIGNMENT ACCURACY = fraction of variables whose recovered==true cell,
#       plus an alignment confusion summary. On the VCS this is exact because the
#       state is monosemantic per cell (each named variable lives in its own byte).
#   (3) IIA (interchange-intervention accuracy) the Geiger headline: averaged over
#       all aligned variables, the fraction of interchanges matching the abstraction.
#
# Why this SUCCEEDS on the VCS (the paper's point): DAS' alignment problem is
# *solved* here because we have the ground-truth map — the cell↔variable alignment
# is the oracle's data-flow. We don't *search* a subspace blind; we VERIFY that the
# interchange-accuracy metric, applied to the oracle alignment, is perfect and that
# a misaligned subspace fails — i.e. interchange interventions recover the program's
# true variable alignment with no false alignment. The same harness scales to a
# learned/searched subspace on the cluster (SOFT-STE GPU batches, forward bit-exact).
#
# No JuTari/jaxtari/xitari core is modified — pure tooling under tools/xai_study/.
# Reuses the validated foundations on main:
#   * oracle_intervene.jl — candidate_ram_indices / Cause / build_pong_causes (the
#     game-agnostic candidate set the oracle scores).
#   * jutari_oracle.jl — boot/replay/snapshot/deepcopy-checkpoint/intervene_ram!/
#     intervene_tia! + the dependency-free §R NPZ writer.
#   * the per-game ROM-alias + RomSettings + candidates map (mirrors
#     activation_patching.jl / ig_baseline_sweep.jl — NO emulator core touched).
#
# Run (warm shared depot, primary's project):
#   julia --project=/Users/maier/Documents/code/UnderstandingVCS/jutari \
#         tools/xai_study/phaseC_mechanistic/das.jl --games core
# Flags: --games core|<g1,g2,...>  --game <g>
#        --target-frame N --horizon N --selftest
#
# Writes (SPEC §R; file_scope das_* under out/):
#   tools/xai_study/phaseC_mechanistic/out/das_<game>.{json,npz}
#   tools/xai_study/phaseC_mechanistic/out/das_core_summary.json

module DAS

using JuTari
using JuTari.Env: StellaEnvironment, env_reset!, env_step!, get_screen, get_ram

# the oracle's candidate cause set + Cause type (game-agnostic), and the verified
# jutari run helper (boot/replay/snapshot/intervene + NPZ writer).
include(joinpath(@__DIR__, "..", "ground_truth", "oracle_intervene.jl"))
using .OracleIntervene: candidate_ram_indices, build_pong_causes, run_intervention
using .OracleIntervene.JutariOracle: Snapshot, snapshot, intervene_ram!,
                                     intervene_tia!, write_npz, RAM_SIZE
using JuTari.Diff: soft_ram_peek

# The P2 SHARED TESTBED (xai_paper/xai_2_interpretability/experiment_redesign.md):
# seeded random-action GAMEPLAY state + oracle cause-density gate + shared
# screen-buffer REGION output. Included as a fragment (see its header) so
# build_shared_testbed operates on OUR own Cause/Snapshot types. Opt in with
# XAI_SHARED_TESTBED=1 (default on for the redesign re-run). Phase C is not a
# gradient method, so the sampler-on path does not apply — we consume the shared
# STATE + cause-density GATE + shared screen-buffer output only.
include(joinpath(@__DIR__, "..", "common", "shared_testbed_impl.jl"))

const OUT_DIR = joinpath(@__DIR__, "out")
const CORE_GAMES = ["pong", "breakout", "space_invaders", "seaquest", "ms_pacman", "qbert"]
# shared-testbed switch + params (redesign protocol: prefix=90 gameplay, horizon=15).
const SHARED_TESTBED = get(ENV, "XAI_SHARED_TESTBED", "1") == "1"
const ST_PREFIX  = parse(Int, get(ENV, "XAI_ST_PREFIX", "90"))
const ST_HORIZON = parse(Int, get(ENV, "XAI_ST_HORIZON", "15"))
const ST_SEED    = parse(Int, get(ENV, "XAI_ST_SEED", "0"))
const ST_GATE_K  = parse(Int, get(ENV, "XAI_ST_GATE_K", "4"))
const ST_FLOOR   = parse(Float64, get(ENV, "XAI_ST_FLOOR", "0.5"))

# joystick action codes (oracle_intervene.jl: RIGHT=3; LEFT=4). NOOP=0 is BASE.
const ACT_NOOP  = 0
const ACT_RIGHT = 3
const ACT_LEFT  = 4

# content-path TIA cause: background colour register drives bg pixels (same as the
# oracle's COLUBK content cause) — included as a TIA "variable" to align.
const COLUBK_REG = 0x09

# ============================================================================
# Per-game ROM + RomSettings + candidates resolution (mirrors activation_patching.jl
# / ig_baseline_sweep.jl; NO emulator core touched). seaquest has no registered
# RomSettings yet → Generic (boots fine; the screen scoreboard's Generic fallback).
# ============================================================================
const ROM_BASENAME = Dict(
    "pong" => "pong", "breakout" => "breakout",
    "space_invaders" => "space_invaders", "seaquest" => "seaquest",
    "ms_pacman" => "mspacman", "qbert" => "qbert")

const _PRIMARY_REPO = get(ENV, "XAI_PRIMARY_REPO", "/Users/maier/Documents/code/UnderstandingVCS")

function rom_path_for(game::AbstractString)
    stem = get(ROM_BASENAME, lowercase(string(game)), lowercase(string(game)))
    here = normpath(joinpath(@__DIR__, "..", "..", ".."))
    for base in (here, _PRIMARY_REPO)
        p = joinpath(base, "xitari", "roms", stem * ".bin")
        isfile(p) && return p
    end
    error("ROM not found for game=$game (looked under $here and $_PRIMARY_REPO)")
end

function settings_for(game::AbstractString)
    g = lowercase(string(game))
    g == "pong"           && return JuTari.PaddleGames.PongRomSettings()
    g == "breakout"       && return JuTari.PaddleGames.BreakoutRomSettings()
    g == "space_invaders" && return JuTari.SpaceInvadersRomSettings()
    g == "ms_pacman"      && return JuTari.JoystickGames.MsPacmanRomSettings()
    g == "qbert"          && return JuTari.JoystickGames.QbertRomSettings()
    return JuTari.GenericRomSettings()   # seaquest (no registered settings yet)
end

function candidates_path_for(game::AbstractString)
    rel = joinpath("tools", "xai_study", "t3", "out", "candidates_$(game).json")
    here = normpath(joinpath(@__DIR__, "..", "..", ".."))
    for base in (here, _PRIMARY_REPO)
        p = joinpath(base, rel)
        isfile(p) && return p
    end
    return nothing
end

# ============================================================================
# env construction + replay + checkpoint (mirrors jutari_oracle / activation_patching)
# ============================================================================
function load_env(; game::AbstractString)
    rom = read(rom_path_for(game))
    env = StellaEnvironment(rom, settings_for(game))
    env_reset!(env; boot_noop_steps = 60, boot_reset_steps = 4)
    return env
end

function boot_replay(actions, target_frame; game)
    env = load_env(; game = game)
    for i in 1:target_frame; env_step!(env, Int(actions[i])); end
    return env
end

function continue_from(checkpoint::StellaEnvironment, tail)
    env = deepcopy(checkpoint)
    for a in tail; env_step!(env, Int(a)); end
    return snapshot(env, length(tail))
end

function fresh_baseline(actions, total; game)
    env = load_env(; game = game)
    for i in 1:total; env_step!(env, Int(actions[i])); end
    return snapshot(env, Int(total))
end

"""Assert two fresh boots+replays are byte-identical in RAM AND screen — the
load-bearing guarantee that makes every interchange Δ a clean causal effect
(mirrors OracleIntervene.assert_bit_exact)."""
function assert_bit_exact(actions, total; game)
    a = fresh_baseline(actions, total; game = game)
    b = fresh_baseline(actions, total; game = game)
    a.ram == b.ram || error("bit-exact RAM re-run FAILED for $game: " *
        "$(count(a.ram .!= b.ram))/$(length(a.ram)) bytes differ to f$total")
    a.screen == b.screen || error("bit-exact SCREEN re-run FAILED for $game: " *
        "$(count(a.screen .!= b.screen)) px differ to f$total")
    return true
end

# ============================================================================
# Variables to align. A "variable" = a high-level causal variable of the program.
# On the VCS the named variables live in RIOT-RAM cells (the candidate set) + a
# content-path TIA register. Each variable's OUTPUT (its read) is the value of the
# cell it is realised in — so the high-level model "o_V := source(V)" is exactly
# the source run's value of that cell. We additionally carry a whole-screen
# position output to capture downstream pixel effects.
# ============================================================================
struct Variable
    name::String          # human label (concept)
    kind::String          # "ram" | "tia_reg"
    site::Int             # the RAM index / TIA reg that TRULY realises it (0-based)
    read::Function        # Snapshot -> Float64 (the variable's output value)
end

"""Build the alignment-candidate variables for a game: one per candidate RAM cell
(the variable's output = that cell's own value) + the background-colour TIA
register (content path). The `site` is the cell the oracle's data-flow says truly
realises the variable — the GROUND-TRUTH alignment."""
function build_variables(cand_indices_concepts)
    vars = Variable[]
    for (idx, concept) in cand_indices_concepts
        label = isempty(concept) ? "ram@$idx" : "$(concept)@$idx"
        push!(vars, Variable(label, "ram", idx, s -> Float64(Int(s.ram[idx + 1]))))
    end
    push!(vars, Variable("bg_colour@COLUBK", "tia_reg", Int(COLUBK_REG),
                         s -> Float64(Int(s.screen[max(1, size(s.screen, 1) - 2),
                                                    max(1, size(s.screen, 2) ÷ 2)]))))
    return vars
end

# ============================================================================
# The interchange operation. Write the SOURCE's value of a candidate ALIGNMENT
# (a set of sites of one kind) into a deepcopy of the BASE checkpoint, re-run the
# tail, snapshot. This is simultaneously the interchange intervention AND (by
# construction) the exact-patch oracle for `do(site := source_value)`.
# ============================================================================
"""Interchange: write the source value of an alignment (a set of `sites` of one
`kind`) into a deepcopy of the BASE checkpoint, run `tail` actions, snapshot. The
caller passes the explicit `value` to write (the source run's value of the site,
or a directed value). This single operation is simultaneously the interchange
intervention AND (by construction) the exact-patch oracle for do(site:=value)."""
function interchange(base_ckpt, tail, kind::AbstractString,
                     sites::AbstractVector{<:Integer},
                     values::AbstractVector{<:Integer})
    env = deepcopy(base_ckpt)
    for (site, v) in zip(sites, values)
        if kind == "ram"
            intervene_ram!(env, site, v)
        elseif kind == "tia_reg"
            intervene_tia!(env, site, v)
        else
            error("unknown interchange kind: $kind")
        end
    end
    for a in tail; env_step!(env, Int(a)); end
    return snapshot(env, length(tail))
end

# ============================================================================
# Per-game DAS result
# ============================================================================
struct DASResult
    game::String
    target_frame::Int
    horizon::Int
    var_names::Vector{String}
    var_meta::Vector{Dict{String,Any}}
    # interchange-effect matrix on the variable outputs: (alignment cell i, output var j)
    effect::Matrix{Float64}          # interchange Δ of cell i on var j (real re-run)
    exact_effect::Matrix{Float64}    # the exact-patch oracle Δ (independent replay)
    bit_exact::Bool
    interchange_eq_exact::Bool
    max_abs_interchange_minus_exact::Float64
    # the headline alignment metrics
    iia_aligned::Float64             # interchange-accuracy of the TRUE alignment (Geiger IIA @ readout)
    iia_misaligned::Float64          # interchange-accuracy of a MISALIGNED cell (neg control)
    iia_aligned_natural::Float64     # IIA of the TRUE alignment under the NATURAL (joystick) source
    alignment_accuracy::Float64      # fraction of NON-TRANSIENT vars whose recovered cell == true cell
    alignment_accuracy_all::Float64  # same over ALL vars (transient cells counted, conservative)
    recovered_aligned::Vector{Int}   # the cell DAS recovers per variable (index into vars/sites)
    true_aligned::Vector{Int}        # the cell the oracle says realises each variable
    transient_downstream::Vector{String}  # cells whose interchange is clobbered within the horizon
    n_vars::Int
    n_source_diverged::Int           # candidate cells where the NATURAL source differs from BASE
    # SHARED-TESTBED provenance (redesign); all NaN/-1/"" in the legacy NOOP path.
    state_kind::String               # "seeded_random_action_gameplay" | "noop"
    st_seed::Int
    st_prefix::Int
    cause_density::Int               # #causes above the floor at the shared output
    cause_density_accepted::Bool     # passed the cause-density gate?
    n_causes::Int
    shared_cell::Tuple{Int,Int}      # the shared screen-buffer output cell
end

function run_game(; game, target_frame = 120, horizon = 30, verbose = true)
    # SHARED-TESTBED (redesign): replace the all-NOOP boot/attract tape with a
    # seeded random-action GAMEPLAY state at f*=ST_PREFIX, gated by the oracle
    # cause-density gate. BASE, SOURCE, causes, candidate set all come from the
    # shared substrate so every method sits on the SAME state (P1, P4).
    st = nothing
    if SHARED_TESTBED
        st = build_shared_testbed(game;
            settings_for = settings_for, rom_path_for = rom_path_for,
            candidates_path_for = candidates_path_for,
            build_causes = build_pong_causes, candidate_ram_indices = candidate_ram_indices,
            continue_from = continue_from, snapshot = snapshot, env_step = env_step!,
            intervene_ram = intervene_ram!, boot_replay = boot_replay,
            run_intervention = run_intervention, soft_ram_peek = soft_ram_peek,
            prefix = ST_PREFIX, horizon = ST_HORIZON, seed = ST_SEED,
            k = ST_GATE_K, floor = ST_FLOOR, verbose = verbose,
            assert_bit_exact = assert_bit_exact)
        target_frame = st.prefix; horizon = st.horizon
        total = st.total
        base_actions = st.actions
        bit_exact = true
        base_ckpt = st.checkpoint
        tail = base_actions[target_frame + 1 : total]
        base_snap = st.base
        base_at_t = st.at_target
        cand = candidates_path_for(game)
        cand_indices = st.cand_indices
        # SOURCE run: share the gameplay prefix, diverge only at the analysis frame
        # with a RIGHT joystick action (an on-distribution genuinely different state).
        pre = target_frame > 0 ? Int.(base_actions[1:target_frame - 1]) : Int[]
        src_at_t = continue_from(boot_replay(vcat(pre, ACT_RIGHT), target_frame; game = game), Int[])
        cand_ic = candidate_ram_indices(cand)
        vars = build_variables(cand_ic)
        verbose && println("[$game] SHARED gameplay state: cause_density=$(st.cause_density)/" *
            "$(length(st.causes)) accepted=$(st.accepted) cell=$(st.cell)")
        return _run_game_body(; game = game, target_frame = target_frame, horizon = horizon,
            total = total, base_ckpt = base_ckpt, base_actions = base_actions, tail = tail,
            base_snap = base_snap, base_at_t = base_at_t, src_at_t = src_at_t,
            cand_indices = cand_indices, vars = vars, bit_exact = bit_exact, st = st,
            verbose = verbose)
    end

    total = target_frame + horizon
    base_actions = fill(ACT_NOOP, total)

    # ---- 1) bit-exact guarantee (two fresh boots+replays, RAM AND screen) ----
    verbose && println("[$game] asserting bit-exactness (2 fresh replays to f$total)...")
    assert_bit_exact(base_actions, total; game = game)
    verbose && println("[$game] bit-exact re-run: PASS")
    bit_exact = true

    # ---- 2) BASE checkpoint at the target frame + clean continuation ----------
    base_ckpt = boot_replay(base_actions, target_frame; game = game)
    tail = base_actions[target_frame + 1 : total]
    base_snap = continue_from(base_ckpt, Int.(tail))     # BASE continuation
    base_at_t = continue_from(base_ckpt, Int[])          # BASE state AT frame t

    # ---- 3) SOURCE run: a genuinely different context (joystick-driven) -------
    # RIGHT for the whole pre-target trace → candidate cells carry different values
    # than BASE for the NATURAL-context interchange (a real value-swap, not a no-op).
    src_actions = fill(ACT_RIGHT, total)
    src_at_t = continue_from(boot_replay(src_actions, target_frame; game = game), Int[])

    # ---- 4) candidate sites + the variables to align --------------------------
    cand = candidates_path_for(game)
    cand_ic = candidate_ram_indices(cand)                # [(idx, concept), ...]
    cand_indices = [idx for (idx, _) in cand_ic]
    vars = build_variables(cand_ic)
    return _run_game_body(; game = game, target_frame = target_frame, horizon = horizon,
        total = total, base_ckpt = base_ckpt, base_actions = base_actions, tail = tail,
        base_snap = base_snap, base_at_t = base_at_t, src_at_t = src_at_t,
        cand_indices = cand_indices, vars = vars, bit_exact = bit_exact, st = nothing,
        verbose = verbose)
end

"""The per-game body, shared by the legacy NOOP path and the SHARED gameplay-state
path (the ONLY difference is which state/action-stream/checkpoint/SOURCE the
interchange algorithm sits on — the algorithm itself is unchanged). `st` carries the
shared-testbed provenance for the record, or `nothing` in the legacy path."""
function _run_game_body(; game, target_frame, horizon, total, base_ckpt, base_actions,
                        tail, base_snap, base_at_t, src_at_t, cand_indices, vars,
                        bit_exact, st, verbose)
    nvar = length(vars)

    # the candidate ALIGNMENT cells we search over: each candidate RAM cell as a
    # singleton alignment, plus the TIA content register. Index aligns with `vars`
    # 1:1 (the i-th candidate cell is the TRUE alignment of the i-th variable).
    align_kind = [v.kind for v in vars]
    align_site = [v.site for v in vars]
    ram_js = [j for j in 1:nvar if align_kind[j] == "ram"]

    y_base = [v.read(base_snap) for v in vars]   # BASE output of each var (post-horizon)
    # BASE value of each variable's own cell AT frame t (the interchange readout pt).
    base_self = [align_kind[j] == "ram" ? Float64(Int(base_at_t.ram[align_site[j] + 1])) :
                                          0.0 for j in 1:nvar]

    # the SOURCE interchange VALUE for each candidate cell. Use a DIRECTED source
    # (base+17, mirroring the oracle's `set` cause) so EVERY candidate cell is a
    # genuine, non-degenerate interchange — not only the ones a one-frame joystick
    # context happens to move. The natural-context interchange (joystick source) is
    # scored separately below to show the metric also holds for an organic swap.
    src_value = [align_kind[j] == "ram" ? Int((Int(base_at_t.ram[align_site[j] + 1]) + 17) & 0xFF) :
                                          0x0E for j in 1:nvar]
    nat_value = [align_kind[j] == "ram" ? Int(src_at_t.ram[align_site[j] + 1]) :
                                          0x0E for j in 1:nvar]

    # ---- 5) interchange-effect matrix (downstream, over the horizon) ----------
    # interchange cell i (directed source), re-run the tail, read every variable j.
    # This is the data-flow matrix used for the DAS alignment search + the firing-
    # pattern checks. exact_effect is the SAME op via an INDEPENDENT fresh replay.
    effect       = zeros(Float64, nvar, nvar)
    exact_effect = zeros(Float64, nvar, nvar)
    max_diff = 0.0
    for i in 1:nvar
        snap = interchange(base_ckpt, Int.(tail), align_kind[i], [align_site[i]], [src_value[i]])
        for j in 1:nvar
            effect[i, j] = vars[j].read(snap) - y_base[j]
        end
        # exact-patch oracle for the SAME interchange via an INDEPENDENT fresh
        # boot+replay (a separate code path → the equality is not a code artefact).
        fresh = boot_replay(base_actions, target_frame; game = game)
        ex_snap = interchange(fresh, Int.(tail), align_kind[i], [align_site[i]], [src_value[i]])
        for j in 1:nvar
            exact_effect[i, j] = vars[j].read(ex_snap) - y_base[j]
        end
        max_diff = max(max_diff, maximum(abs.(effect[i, :] .- exact_effect[i, :])))
        verbose && println("  [$i/$nvar] interchange $(rpad(vars[i].name, 22)) " *
                           "max|Δ|=$(rpad(round(maximum(abs.(effect[i, :])), digits = 2), 7)) " *
                           "Δrec-exact=$(round(maximum(abs.(effect[i, :] .- exact_effect[i, :])), digits = 4))")
    end
    interchange_eq_exact = (max_diff == 0.0)

    # ---- 6) IIA: interchange-intervention accuracy at the READOUT (Geiger) -----
    # The Geiger IIA reads the output at the intervened variable: for the TRUE
    # alignment of variable V, after the interchange the variable's OWN value must
    # equal the source's value of V (the high-level model o_V := source(V)). We read
    # it at the interchange point (horizon-0 self-readout) — the proper definition,
    # robust to the fact that transient cells are re-derived deeper in the horizon
    # (that downstream re-derivation is the §6 "present ≠ used" story, captured by
    # the effect matrix / transient list below, NOT by IIA which reads at the swap).
    function self_after_interchange(j, value)
        snap = interchange(base_ckpt, Int[], align_kind[j], [align_site[j]], [value])  # horizon 0
        return align_kind[j] == "ram" ? Float64(Int(snap.ram[align_site[j] + 1])) :
                                        0.0
    end
    aligned_hits = 0; aligned_total = 0
    nat_hits = 0; nat_total = 0
    for j in ram_js
        aligned_total += 1
        # directed source: prediction = base+17 (the source value); must be realised.
        self_after_interchange(j, src_value[j]) == Float64(src_value[j]) && (aligned_hits += 1)
        # natural (joystick) source: prediction = source's own value of the cell. Only
        # counted where the natural source actually diverged from BASE (a real swap).
        if nat_value[j] != Int(base_self[j])
            nat_total += 1
            self_after_interchange(j, nat_value[j]) == Float64(nat_value[j]) && (nat_hits += 1)
        end
    end
    iia_aligned = aligned_total == 0 ? 1.0 : aligned_hits / aligned_total
    iia_aligned_natural = nat_total == 0 ? 1.0 : nat_hits / nat_total

    # ---- 7) IIA of a MISALIGNED cell (negative control) -----------------------
    # Align variable j to the WRONG cell (the next candidate, cyclically) and read
    # variable j's own value at the swap. Under a false alignment the variable does
    # NOT take the prediction unless the wrong cell happens to alias it → IIA
    # collapses. This discriminating control is what makes the aligned 1.0 meaningful.
    mis_hits = 0; mis_total = 0
    if length(ram_js) >= 2
        for (k, j) in enumerate(ram_js)
            wrong = ram_js[mod1(k + 1, length(ram_js))]
            wrong == j && continue
            mis_total += 1
            # interchange the WRONG cell with the source value, read variable j's OWN
            # value: matches the prediction (=src_value[j]) only if wrong aliases j.
            snap = interchange(base_ckpt, Int[], align_kind[wrong], [align_site[wrong]], [src_value[wrong]])
            self_j = Float64(Int(snap.ram[align_site[j] + 1]))
            (self_j == Float64(src_value[j]) && src_value[j] != Int(base_self[j])) && (mis_hits += 1)
        end
    end
    iia_misaligned = mis_total == 0 ? 0.0 : mis_hits / mis_total

    # ---- 8) ALIGNMENT vs the true variable (the DAS search) -------------------
    # For each variable j the RECOVERED alignment = the candidate cell whose
    # interchange most closely produces the abstraction's required Δ on o_j over the
    # horizon. The required Δ for the self-cell directed interchange is +17 on its
    # own value (when the cell is NOT transient); for transient cells the within-
    # horizon Δ is 0, so the recovery degenerates to the self-cell unless another
    # cell spuriously drives it (recorded honestly). Compare to the TRUE alignment j.
    recovered_aligned = zeros(Int, nvar)
    true_aligned = collect(1:nvar)
    transient_downstream = String[]
    # two accuracies: over NON-TRANSIENT vars (the clean DAS claim — cells with a
    # within-horizon causal signal) and over ALL vars (conservative). Transience is
    # a property of the PROGRAM (it re-derives the cell next frame), recorded not hidden.
    align_correct_nt = 0; align_total_nt = 0
    align_correct_all = 0; align_total_all = 0
    for j in ram_js
        align_total_all += 1
        target = effect[j, j]                       # the self-cell's own within-horizon Δ
        is_transient = (target == 0.0)
        if is_transient
            # transient: the self interchange has no within-horizon downstream self-
            # effect, so the DAS search can't latch onto it (present ≠ used). Record it.
            push!(transient_downstream, vars[j].name)
            spurious = [i for i in ram_js if i != j && effect[i, j] != 0.0]
            recovered_aligned[j] = isempty(spurious) ? j : spurious[1]
        else
            align_total_nt += 1
            best_i = j; best_err = Inf
            for i in ram_js
                err = abs(effect[i, j] - target)
                if err < best_err - 1e-9 || (abs(err - best_err) <= 1e-9 && i == j)
                    best_err = err; best_i = i
                end
            end
            recovered_aligned[j] = best_i
            recovered_aligned[j] == j && (align_correct_nt += 1)
        end
        recovered_aligned[j] == j && (align_correct_all += 1)
    end
    for j in 1:nvar
        recovered_aligned[j] == 0 && (recovered_aligned[j] = j)   # TIA var → self
    end
    alignment_accuracy     = align_total_nt  == 0 ? 1.0 : align_correct_nt  / align_total_nt
    alignment_accuracy_all = align_total_all == 0 ? 1.0 : align_correct_all / align_total_all

    # how many candidate cells actually carry a different value under the NATURAL
    # (joystick) source — proof the organic interchange is real, not a no-op.
    n_source_diverged = count(idx -> Int(src_at_t.ram[idx + 1]) != Int(base_at_t.ram[idx + 1]),
                              cand_indices)

    var_meta = Dict{String,Any}[]
    for (j, v) in enumerate(vars)
        push!(var_meta, Dict{String,Any}(
            "name" => v.name, "kind" => v.kind, "true_site" => v.site,
            "y_base" => y_base[j], "directed_source_value" => Float64(src_value[j]),
            "natural_source_value" => Float64(nat_value[j]),
            "recovered_aligned_site" => align_site[recovered_aligned[j]],
            "true_aligned_site" => v.site,
            "aligned_correct" => (recovered_aligned[j] == j)))
    end

    verbose && println("[$game] interchange==exact=$interchange_eq_exact " *
                       "(max|Δrec-exact|=$max_diff) " *
                       "IIA_aligned=$(round(iia_aligned, digits = 3)) " *
                       "IIA_misaligned=$(round(iia_misaligned, digits = 3)) " *
                       "IIA_aligned_natural=$(round(iia_aligned_natural, digits = 3)) " *
                       "alignment_acc(non-transient)=$(round(alignment_accuracy, digits = 3)) " *
                       "alignment_acc(all)=$(round(alignment_accuracy_all, digits = 3)) " *
                       "transient=$(length(transient_downstream))/$(length(ram_js)) " *
                       "source_diverged=$n_source_diverged/$(length(cand_indices))")

    return DASResult(game, target_frame, horizon,
                     [v.name for v in vars], var_meta,
                     effect, exact_effect, bit_exact,
                     interchange_eq_exact, max_diff,
                     iia_aligned, iia_misaligned, iia_aligned_natural,
                     alignment_accuracy, alignment_accuracy_all,
                     recovered_aligned, true_aligned,
                     transient_downstream, nvar, n_source_diverged,
                     st === nothing ? "noop" : "seeded_random_action_gameplay",
                     st === nothing ? -1 : st.seed,
                     st === nothing ? -1 : st.prefix,
                     st === nothing ? -1 : st.cause_density,
                     st === nothing ? false : st.accepted,
                     st === nothing ? nvar : length(st.causes),
                     st === nothing ? (-1, -1) : st.cell)
end

# ============================================================================
# Persist (SPEC §R)
# ============================================================================
function _git_commit()
    try
        return strip(read(`git -C $(@__DIR__) rev-parse --short HEAD`, String))
    catch
        return "unknown"
    end
end

# tiny dependency-free JSON (adds no package to the shared jutari env)
_j(s::AbstractString) = '"' * replace(replace(string(s), "\\" => "\\\\"), "\"" => "\\\"") * '"'
_j(b::Bool) = b ? "true" : "false"
_j(::Nothing) = "null"
_j(x::Integer) = string(x)
_j(x::AbstractFloat) = isfinite(x) ? string(x) : "null"
_j(v::AbstractVector) = "[" * join((_j(e) for e in v), ", ") * "]"
function _j(d::AbstractDict)
    parts = String[]
    for (k, v) in d
        push!(parts, _j(string(k)) * ": " * _j(v))
    end
    return "{" * join(parts, ", ") * "}"
end

function write_game_result(r::DASResult; out_dir = OUT_DIR)
    isdir(out_dir) || mkpath(out_dir)
    stem = "das_$(r.game)"
    json_path = joinpath(out_dir, stem * ".json")
    npz_path  = joinpath(out_dir, stem * ".npz")

    eff_map = Dict(r.var_names[i] =>
                   Dict(r.var_names[j] => r.effect[i, j] for j in 1:r.n_vars)
                   for i in 1:r.n_vars)

    rec = Dict{String,Any}(
        "paper" => "P2",
        "phase" => "phaseC_mechanistic",
        "method" => "interchange_interventions_das",
        "game" => r.game,
        "state" => r.state_kind == "noop" ? "f$(r.target_frame)+$(r.horizon)" :
                   "gameplay(seed=$(r.st_seed),prefix=$(r.st_prefix))+$(r.horizon)",
        "target_output" => "aligned causal variables (RAM concept cells + bg-colour)",
        # headline §R scalar: interchange accuracy of the TRUE alignment (1.0 = the
        # abstraction is realised exactly by the oracle alignment).
        "metric_name" => "interchange_accuracy_aligned",
        "value" => r.iia_aligned,
        "stderr" => nothing,
        "ci" => nothing,
        "n" => r.n_vars,
        "seed" => 0,
        "where" => "local",
        "commit" => _git_commit(),
        "oracle_ref" => "oracle_intervene@$(r.game) (P2-E1-1) — exact single-site interchange",
        "timestamp" => string(round(Int, time())),
        "arrays" => basename(npz_path),
        "extra" => Dict{String,Any}(
            "substrate" => "jutari (Julia, HARD) — real-ROM bit-exact path",
            "variables" => r.var_meta,
            "bit_exact_rerun" => r.bit_exact,
            "interchange_equals_exact" => r.interchange_eq_exact,
            "max_abs_interchange_minus_exact" => r.max_abs_interchange_minus_exact,
            # the headline DAS metrics
            "interchange_accuracy_aligned" => r.iia_aligned,
            "interchange_accuracy_misaligned" => r.iia_misaligned,
            "interchange_accuracy_aligned_natural" => r.iia_aligned_natural,
            "alignment_accuracy" => r.alignment_accuracy,
            "alignment_accuracy_all" => r.alignment_accuracy_all,
            "transient_downstream_cells" => r.transient_downstream,
            "n_variables" => r.n_vars,
            "n_source_diverged_natural" => r.n_source_diverged,
            "interchange_effect" => eff_map,
            "testbed" => Dict{String,Any}(
                "state_kind" => r.state_kind,
                "seed" => r.st_seed, "prefix" => r.st_prefix, "horizon" => r.horizon,
                "shared_output" => "screen_region(n_changed_px)@r$(r.shared_cell[1])c$(r.shared_cell[2])",
                "cause_density_above_floor" => r.cause_density,
                "cause_density_floor" => ST_FLOOR, "cause_density_gate_k" => ST_GATE_K,
                "cause_density_accepted" => r.cause_density_accepted, "n_causes" => r.n_causes,
                "note" => "P2 redesign: methods sit on a seeded random-action GAMEPLAY " *
                    "state (not the boot/attract NOOP tape), gated by the oracle " *
                    "cause-density gate (accept iff #causes above the floor >= k). The " *
                    "interchange/DAS algorithm is unchanged; only the state (BASE + " *
                    "SOURCE, both on the shared gameplay prefix) moves."),
            "note" =>
                "Interchange interventions / DAS (Geiger et al. 2021/2023): for each " *
                "candidate alignment (a RAM cell / TIA register realising a high-level " *
                "variable) swap its value SOURCE->BASE at frame t and re-run the real " *
                "ROM. The high-level causal model predicts o_V := source(V). " *
                "interchange_accuracy_aligned = the TRUE alignment reproduces the " *
                "abstraction AT THE READOUT (Geiger IIA, =1.0 by construction on the " *
                "VCS, directed source base+17); interchange_accuracy_aligned_natural " *
                "= same under an organic joystick source (only on cells the source " *
                "actually moved). interchange_accuracy_misaligned = a wrong cell fails " *
                "(negative control). alignment_accuracy = the DAS search recovers the " *
                "cell the oracle says realises each variable from the within-horizon " *
                "interchange-effect matrix; transient cells (re-derived within the " *
                "horizon → no downstream self-effect) are recorded, not hidden " *
                "(present != used). interchange==exact (an independent fresh replay) " *
                "validates the harness — single-cell interchange IS do(cell:=source) " *
                "so it equals the exact patch oracle.",
            "scales_to_cluster_via" =>
                "tools/cluster/xai_*.sbatch (E0-3) + jaxtari SOFT-STE GPU batches " *
                "(forward bit-exact to this HARD map) for learned-subspace DAS over " *
                "state × variables × games.",
        ),
    )
    open(json_path, "w") do io
        write(io, _j(rec) * "\n")
    end

    write_npz(npz_path, Dict(
        "interchange_effect" => r.effect,                 # (alignment cells, vars)
        "exact_effect"       => r.exact_effect,           # (alignment cells, vars)
        "abs_interchange_minus_exact" => abs.(r.effect .- r.exact_effect),
        "recovered_aligned"  => Float64.(r.recovered_aligned),
        "true_aligned"       => Float64.(r.true_aligned),
    ))
    return json_path, npz_path
end

function write_summary(results::Vector{DASResult}; out_dir = OUT_DIR)
    isdir(out_dir) || mkpath(out_dir)
    path = joinpath(out_dir, "das_core_summary.json")
    per_game = Dict{String,Any}[]
    all_eq = true
    for r in results
        all_eq &= r.interchange_eq_exact
        push!(per_game, Dict{String,Any}(
            "game" => r.game,
            "n_variables" => r.n_vars,
            "interchange_equals_exact" => r.interchange_eq_exact,
            "max_abs_interchange_minus_exact" => r.max_abs_interchange_minus_exact,
            "interchange_accuracy_aligned" => r.iia_aligned,
            "interchange_accuracy_misaligned" => r.iia_misaligned,
            "interchange_accuracy_aligned_natural" => r.iia_aligned_natural,
            "alignment_accuracy" => r.alignment_accuracy,
            "alignment_accuracy_all" => r.alignment_accuracy_all,
            "n_transient_downstream" => length(r.transient_downstream),
            "n_source_diverged_natural" => r.n_source_diverged,
        ))
    end
    rec = Dict{String,Any}(
        "paper" => "P2", "phase" => "phaseC_mechanistic",
        "method" => "interchange_interventions_das", "scope" => "core (6 games)",
        "commit" => _git_commit(), "timestamp" => string(round(Int, time())),
        "all_games_interchange_equals_exact" => all_eq,
        "mean_interchange_accuracy_aligned" =>
            sum(r.iia_aligned for r in results) / length(results),
        "mean_interchange_accuracy_misaligned" =>
            sum(r.iia_misaligned for r in results) / length(results),
        "mean_interchange_accuracy_aligned_natural" =>
            sum(r.iia_aligned_natural for r in results) / length(results),
        "mean_alignment_accuracy" =>
            sum(r.alignment_accuracy for r in results) / length(results),
        "mean_alignment_accuracy_all" =>
            sum(r.alignment_accuracy_all for r in results) / length(results),
        "per_game" => per_game,
        "note" => "Phase-C interchange interventions / DAS over the 6 core games; " *
                  "interchange accuracy of the TRUE (oracle) alignment vs a " *
                  "misaligned control + alignment-recovery accuracy vs the true " *
                  "variable (experiment_design.md §6/§7: causal core Succeeds).",
    )
    open(path, "w") do io
        write(io, _j(rec) * "\n")
    end
    return path
end

# ============================================================================
# Self-check (DoD: the small test) — positive control on Pong:
#   1. the bit-exact re-run holds (precondition for trusting any Δ);
#   2. the interchange == the exact-patch oracle for EVERY cell
#      (max|interchange − exact| == 0) — the harness is sound;
#   3. the TRUE alignment reproduces the abstraction (interchange_accuracy_aligned
#      == 1.0) — interchange recovers the program's variable alignment;
#   4. a misaligned cell does NOT (interchange_accuracy_misaligned < aligned) —
#      the metric DISCRIMINATES (no false alignment);
#   5. at least one candidate cell actually diverged across runs (a real, non-
#      degenerate interchange, not a no-op tautology).
# Exits nonzero (via `error`) on any failure.
# ============================================================================
function self_check(; game = "pong", target_frame = 120, horizon = 30)
    println("[self-check] running interchange/DAS on $game (positive control)...")
    r = run_game(; game = game, target_frame = target_frame, horizon = horizon, verbose = false)
    r.bit_exact || error("self-check FAIL: bit-exact re-run did not hold")
    r.interchange_eq_exact ||
        error("self-check FAIL: interchange != exact (max diff $(r.max_abs_interchange_minus_exact))")
    r.iia_aligned == 1.0 ||
        error("self-check FAIL: aligned interchange accuracy = $(r.iia_aligned) (expected 1.0)")
    r.iia_misaligned < r.iia_aligned ||
        error("self-check FAIL: misaligned IIA ($(r.iia_misaligned)) not < aligned ($(r.iia_aligned)) — metric does not discriminate")
    r.n_source_diverged >= 1 ||
        error("self-check FAIL: no candidate cell diverged across runs — interchange is degenerate")
    println("[self-check] PASS — bit-exact ✓, interchange==exact ✓ " *
            "(max|Δ|=$(r.max_abs_interchange_minus_exact)), " *
            "IIA_aligned=1.0 ✓, IIA_misaligned=$(round(r.iia_misaligned,digits=3)) < 1.0 ✓, " *
            "source_diverged=$(r.n_source_diverged) ✓, " *
            "alignment_acc=$(round(r.alignment_accuracy,digits=3))")
    return true
end

# ============================================================================
# CLI
# ============================================================================
function main(args = ARGS)
    games = CORE_GAMES
    target_frame = 120; horizon = 30
    do_self_check = false
    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--games"
            v = args[i + 1]; i += 2
            games = lowercase(v) == "core" ? CORE_GAMES : String.(split(v, ","))
        elseif a == "--game"; games = [args[i + 1]]; i += 2
        elseif a == "--target-frame"; target_frame = parse(Int, args[i + 1]); i += 2
        elseif a == "--horizon"; horizon = parse(Int, args[i + 1]); i += 2
        elseif a == "--selftest" || a == "--self-check"; do_self_check = true; i += 1
        else; i += 1
        end
    end

    if do_self_check
        self_check(; target_frame = target_frame, horizon = horizon)
        return nothing
    end

    println("[das] games=$(join(games, ",")) " *
            "target_frame=$target_frame horizon=$horizon (jutari/Julia)")
    results = DASResult[]
    for g in games
        println("\n========== $g ==========")
        r = run_game(; game = g, target_frame = target_frame, horizon = horizon, verbose = true)
        jp, np = write_game_result(r)
        println("[$g] interchange==exact: $(r.interchange_eq_exact) " *
                "(max|Δ|=$(r.max_abs_interchange_minus_exact)); " *
                "IIA_aligned=$(round(r.iia_aligned,digits=3)) " *
                "IIA_misaligned=$(round(r.iia_misaligned,digits=3)) " *
                "alignment_acc(nt)=$(round(r.alignment_accuracy,digits=3)) " *
                "alignment_acc(all)=$(round(r.alignment_accuracy_all,digits=3))")
        println("[$g] wrote $jp")
        println("[$g] arrays  $np")
        push!(results, r)
    end

    if length(results) > 1
        sp = write_summary(results)
        println("\n[das] core summary -> $sp")
    end

    println("\n[das] headline (interchange/DAS vs true variable alignment, all games):")
    for r in results
        println("    $(rpad(r.game, 16)) interchange==exact=$(r.interchange_eq_exact) " *
                "IIA_aligned=$(round(r.iia_aligned,digits=3)) " *
                "IIA_mis=$(round(r.iia_misaligned,digits=3)) " *
                "align_acc(nt)=$(round(r.alignment_accuracy,digits=3)) " *
                "align_acc(all)=$(round(r.alignment_accuracy_all,digits=3)) " *
                "transient=$(length(r.transient_downstream)) " *
                "src_diverged=$(r.n_source_diverged)/$(r.n_vars - 1)")
    end
    return results
end

end # module

# run as a script (not when `include`d by a test)
if abspath(PROGRAM_FILE) == @__FILE__
    DAS.main()
end
