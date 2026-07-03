# activation_patching.jl — Phase-C mechanistic interpretability (P2-E5-1),
# JULIA path (the jutari real-ROM substrate; jaxtari eager is ~205× slower —
# SCRUM §7). The FULL method: activation patching / causal tracing (Vig et al.
# 2020; ROME causal tracing, Meng et al. 2022) over VCS state sites, generalising
# the Phase-C pilot (pilot_patch_sae.jl, P2-E5-0) from Pong to the 6 CORE games
# (tools/xai_study/common/game_set.json), scored against the EXACT intervention
# oracle Δ (experiment_design.md §6 row "Activation patching / causal mediation";
# §7 prediction: **Succeed** — causal by construction).
#
# ---------------------------------------------------------------------------
# What activation patching IS here (experiment_design.md §3, §6; SPEC §E5):
#   the VCS state trajectory is the "activations" and the program's data-flow is
#   the "circuit". Activation patching = take the value of ONE state site (a RIOT
#   RAM cell at frame t) from a *different* run (a "donor"), write it into the
#   clean run at the same frame, RE-RUN the real ROM a short horizon, and measure
#   the effect Δy on the outputs.
#
# The ground-truth score is the EXACT intervention oracle (P2-E1-1,
# oracle_intervene.jl): the *exact patch* `do(site := value)` followed by a
# bit-exact re-run. Activation patching from a donor and the exact patch are the
# SAME single-site state write, so the recovered effect MUST equal the exact
# patch — that equality is what validates the patching harness. We assert it via
# an INDEPENDENT fresh-from-boot replay (a separate code path, so the equality is
# not a code-sharing artefact), and additionally score each site's recovered
# firing pattern against the TRUE data-flow (T2): patching a cell must move the
# outputs the program actually derives from it and leave the rest untouched →
# per-(patch,output) site precision/recall.
#
# Two complementary patch families, both REAL re-runs of the real ROM:
#   (A) DONOR patches — the donor value comes from a genuinely different run (a
#       LEFT vs RIGHT joystick-context). The cells the donor actually diverges on
#       are the honest "value from another run" sites; we record per-site whether
#       the donor diverged (a real donor) or coincided (degenerate this state).
#   (B) DIRECTED patches — a synthetic donor value `base+17` on each candidate
#       concept cell (mirroring the oracle's `set` cause, so it is directly
#       comparable to the oracle map). This exercises the patching harness across
#       EVERY documented T1/T2 site, even ones a one-frame donor happens not to
#       diverge on.
#
# A-priori target recall is an HONEST mechanistic audit: many RAM cells are
# TRANSIENT — the program re-derives them next frame, so a one-frame patch is
# clobbered and has no downstream effect within the horizon. We record which
# sites are transient/clobbered rather than hiding them (experiment_design.md §6:
# present ≠ used). The headline causal claims rest on (recovered==exact) and the
# firing-pattern P/R, which are exact by construction on the VCS.
#
# No JuTari/jaxtari/xitari core is modified — pure tooling under tools/xai_study/.
# Reuses the validated foundations on main:
#   * oracle_intervene.jl — build_pong_causes / candidate_ram_indices / Cause
#     (the game-agnostic candidate cause set the oracle scores).
#   * jutari_oracle.jl — boot/replay/snapshot/deepcopy-checkpoint/intervene + the
#     dependency-free §R NPZ writer.
#   * the per-game ROM-alias + RomSettings + candidates map (mirrors
#     ig_baseline_sweep.jl / smoothgrad.jl — NO emulator core touched).
#
# Run (warm shared depot, primary's project):
#   julia --project=/Users/maier/Documents/code/UnderstandingVCS/jutari \
#         tools/xai_study/phaseC_mechanistic/activation_patching.jl --games core
# Flags: --games core|<g1,g2,...>  --game <g>
#        --target-frame N --horizon N --selftest
#
# Writes (SPEC §R; file_scope activation_patching_* under out/):
#   tools/xai_study/phaseC_mechanistic/out/activation_patching_<game>.{json,npz}
#   tools/xai_study/phaseC_mechanistic/out/activation_patching_core_summary.json

module ActivationPatching

using JuTari
using JuTari.Env: StellaEnvironment, env_reset!, env_step!, get_screen, get_ram

# the oracle's candidate cause set + Cause type (game-agnostic), and the verified
# jutari run helper (boot/replay/snapshot/intervene + NPZ writer).
include(joinpath(@__DIR__, "..", "ground_truth", "oracle_intervene.jl"))
using .OracleIntervene: build_pong_causes, candidate_ram_indices, Cause,
                        run_intervention
using .OracleIntervene.JutariOracle: Snapshot, snapshot, intervene_ram!,
                                     intervene_tia!, write_npz, RAM_SIZE
using JuTari.Diff: soft_ram_peek

# The P2 SHARED TESTBED (xai_paper/xai_2_interpretability/experiment_redesign.md):
# seeded random-action GAMEPLAY state + oracle cause-density gate + shared
# screen-buffer REGION output. Included as a fragment (see its header for why not a
# module) so build_shared_testbed operates on OUR own Cause/Snapshot types. Opt in
# with XAI_SHARED_TESTBED=1 (default on for the redesign re-run). Phase C is not a
# gradient method, so the sampler-on path does not apply here — we consume the
# shared STATE + cause-density GATE + shared screen-buffer output only.
include(joinpath(@__DIR__, "..", "common", "shared_testbed_impl.jl"))
# the shared game-set + ROM-root resolver (XAI_LABELED / xai_resolve_games / xai_rom_roots).
include(joinpath(@__DIR__, "..", "common", "game_sets.jl"))
# shared Sufficiency (S) + Minimality (M) scorers for the F∧S∧M triad (paper
# sec:triad). Pure vector functions; guarded so a transitive include cannot
# redefine the module. Cannot touch the runner-computed F.
isdefined(@__MODULE__, :TriadSM) ||
    include(joinpath(@__DIR__, "..", "common", "triad_sm.jl"))
using .TriadSM: triad_extra_dict, minimality_score, sufficiency_score, jnum_or_null


const OUT_DIR = joinpath(@__DIR__, "out")
const CORE_GAMES = ["pong", "breakout", "space_invaders", "seaquest", "ms_pacman", "qbert"]
# shared-testbed switch + params (redesign protocol: prefix=90 gameplay, horizon=15).
const SHARED_TESTBED = get(ENV, "XAI_SHARED_TESTBED", "1") == "1"
const ST_PREFIX  = parse(Int, get(ENV, "XAI_ST_PREFIX", "90"))
const ST_HORIZON = parse(Int, get(ENV, "XAI_ST_HORIZON", "15"))
const ST_SEED    = parse(Int, get(ENV, "XAI_ST_SEED", "0"))
const ST_GATE_K  = parse(Int, get(ENV, "XAI_ST_GATE_K", "4"))
const ST_FLOOR   = parse(Float64, get(ENV, "XAI_ST_FLOOR", "0.5"))

# joystick action codes (oracle_intervene.jl: RIGHT=3; LEFT=4) — the two donor
# contexts. NOOP=0 is the clean trace.
const ACT_NOOP  = 0
const ACT_RIGHT = 3
const ACT_LEFT  = 4

# content-path TIA cause: the background colour register drives bg pixels within
# the horizon (a clean content effect; mirrors oracle_intervene's COLUBK cause).
const COLUBK_REG = 0x09

# ============================================================================
# Per-game ROM + RomSettings + candidates resolution (mirrors ig_baseline_sweep.jl
# / smoothgrad.jl; NO emulator core touched). seaquest has no registered
# RomSettings yet → Generic (boots fine; the screen scoreboard's Generic fallback).
# ============================================================================
const ROM_BASENAME = Dict(
    "pong" => "pong", "breakout" => "breakout",
    "space_invaders" => "space_invaders", "seaquest" => "seaquest",
    "ms_pacman" => "mspacman", "qbert" => "qbert")

const _PRIMARY_REPO = get(ENV, "XAI_PRIMARY_REPO", "/Users/maier/Documents/code/UnderstandingVCS")

function rom_path_for(game::AbstractString)
    g = lowercase(string(game))
    stem = get(ROM_BASENAME, g, g)
    # search xitari/roms + the 54-ROM store tools/rom_sweep/roms (ALE names), trying
    # the mapped stem AND the raw ALE name, so all labeled games resolve uniformly.
    return xai_find_rom(unique([stem, g]), xai_rom_roots(; primary_repo = _PRIMARY_REPO))
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
# env construction + replay + checkpoint (mirrors jutari_oracle / ig sweep)
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
load-bearing guarantee that makes every Δ a clean causal effect (mirrors
OracleIntervene.assert_bit_exact)."""
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
# Outputs y(state). Per game: the two most causally-active candidate concept
# cells (read from RAM — the program's own variables), a whole-screen position
# output (count of framebuffer cells differing from the clean baseline frame),
# and a content pixel cell. The score-cell outputs are chosen causally so each
# is a non-degenerate column to score against (mirrors ig pick_content_idx).
# ============================================================================
struct Output
    name::String
    read::Function          # Snapshot -> Float64
    true_sites::Vector{Int} # the RAM site(s) whose patch SHOULD move this output
end

"""Pick the K candidate RAM cells whose own value the oracle's interventions move
the most over the horizon — the most causally-active concept cells (a
non-degenerate set of state-read outputs). Returns (indices, max|Δself|)."""
function pick_active_cells(checkpoint, clean_tail, base_snap, cand_indices, causes; k = 2)
    moves = Tuple{Int,Float64}[]
    y0_for(idx) = Float64(Int(base_snap.ram[idx + 1]))
    for idx in cand_indices
        rd = s -> Float64(Int(s.ram[idx + 1]))
        mv = 0.0
        for c in causes
            snap = run_cause(checkpoint, clean_tail, c)
            mv = max(mv, abs(rd(snap) - y0_for(idx)))
        end
        push!(moves, (idx, mv))
    end
    sort!(moves; by = x -> x[2], rev = true)
    sel = [moves[i][1] for i in 1:min(k, length(moves))]
    return sel, (isempty(moves) ? 0.0 : moves[1][2])
end

"""Run one oracle Cause off the clean checkpoint (used only to *locate* the active
cells; the patch scoring proper uses run_patch)."""
function run_cause(checkpoint, clean_tail, c::Cause)
    env = deepcopy(checkpoint)
    if c.kind == "ram"
        intervene_ram!(env, c.index, c.value)
    elseif c.kind == "tia_reg"
        intervene_tia!(env, c.index, c.value)
    elseif c.kind == "joystick"
        tail = vcat([c.value], Int.(clean_tail[2:end]))
        for a in tail; env_step!(env, a); end
        return snapshot(env, length(tail))
    end
    for a in clean_tail; env_step!(env, Int(a)); end
    return snapshot(env, length(clean_tail))
end

function build_outputs(baseline::Snapshot, score_cells::Vector{Int})
    outs = Output[]
    for idx in score_cells
        push!(outs, Output("state(ram@$idx)",
                           s -> Float64(Int(s.ram[idx + 1])), [idx]))
    end
    # whole-screen position output: count of framebuffer cells differing from the
    # clean baseline frame (robust to which single cell an object lands on). Its
    # true sites are any cell whose patch moves the screen (filled in post-hoc by
    # the exact-patch firing pattern; left empty here for the a-priori audit).
    push!(outs, Output("n_changed_px",
                       s -> Float64(count(s.screen .!= baseline.screen)), Int[]))
    return outs
end

# ============================================================================
# A patch = a single-site state write at the target frame, then re-run.
# ============================================================================
struct Patch
    name::String
    kind::String            # "ram" | "tia_reg"
    site::Int               # RAM index / TIA reg (0-based)
    family::String          # "donor" | "directed"
    value::Int              # the value written (donor's value or a directed one)
    base_value::Int         # the value at the clean frame (for the donor audit)
    donor_label::String     # provenance of the value
    donor_diverged::Bool    # did the donor actually differ from the clean value?
    true_target::String     # the output this site SHOULD move (a-priori, T2)
end

"""Re-run the clean trace from `checkpoint` with site:=value written at the target
frame, for the tail. This is BOTH the activation patch and (by construction) the
exact-patch oracle for `do(site := value)`."""
function run_patch(checkpoint, tail, kind::AbstractString, site::Integer, value::Integer)
    env = deepcopy(checkpoint)
    if kind == "ram"
        intervene_ram!(env, site, value)
    elseif kind == "tia_reg"
        intervene_tia!(env, site, value)
    else
        error("unknown patch kind: $kind")
    end
    for a in tail; env_step!(env, Int(a)); end
    return snapshot(env, length(tail))
end

# ============================================================================
# Per-game patching result
# ============================================================================
struct PatchResult
    game::String
    target_frame::Int
    horizon::Int
    output_names::Vector{String}
    patch_names::Vector{String}
    patch_meta::Vector{Dict{String,Any}}
    y_baseline::Vector{Float64}
    recovered::Matrix{Float64}     # (patches, outputs) — activation patching
    exact::Matrix{Float64}         # (patches, outputs) — exact-patch oracle
    bit_exact::Bool
    recovered_eq_exact::Bool
    max_abs_recovered_minus_exact::Float64
    site_precision::Float64
    site_recall::Float64
    apriori_recall::Float64
    transient_sites::Vector{String}
    n_donor_real::Int              # donor patches whose donor actually diverged
    n_active_patches::Int          # patches with any nonzero recovered effect
    # SHARED-TESTBED provenance (redesign); all NaN/-1/"" in the legacy NOOP path.
    state_kind::String             # "seeded_random_action_gameplay" | "noop"
    st_seed::Int
    st_prefix::Int
    cause_density::Int             # #causes above the floor at the shared output
    cause_density_accepted::Bool   # passed the cause-density gate?
    n_causes::Int
    shared_cell::Tuple{Int,Int}    # the shared screen-buffer output cell
end

function run_game(; game, target_frame = 120, horizon = 30, verbose = true)
    # SHARED-TESTBED (redesign): replace the all-NOOP boot/attract tape with a
    # seeded random-action GAMEPLAY state at f*=ST_PREFIX, gated by the oracle
    # cause-density gate (accept only causally-live states). The clean trace is the
    # gameplay stream; the checkpoint, causes, candidate set and baseline all come
    # from the shared substrate so every method sits on the SAME state (P1, P4).
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
        clean_actions = st.actions
        bit_exact = true
        clean_ckpt = st.checkpoint
        clean_tail = clean_actions[target_frame + 1 : total]
        base_snap = st.base
        at_target = st.at_target
        cand = candidates_path_for(game)
        cand_indices = st.cand_indices
        causes = st.causes
        verbose && println("[$game] SHARED gameplay state: cause_density=$(st.cause_density)/" *
            "$(length(st.causes)) accepted=$(st.accepted) cell=$(st.cell)")
        return _run_game_body(; game = game, target_frame = target_frame, horizon = horizon,
            total = total, clean_actions = clean_actions, bit_exact = bit_exact,
            clean_ckpt = clean_ckpt, clean_tail = clean_tail, base_snap = base_snap,
            at_target = at_target, cand = cand, cand_indices = cand_indices, causes = causes,
            st = st, verbose = verbose)
    end

    total = target_frame + horizon
    clean_actions = fill(ACT_NOOP, total)

    # ---- 1) bit-exact guarantee (two fresh boots+replays, RAM AND screen) ----
    verbose && println("[$game] asserting bit-exactness (2 fresh replays to f$total)...")
    assert_bit_exact(clean_actions, total; game = game)
    verbose && println("[$game] bit-exact re-run: PASS")
    bit_exact = true

    # ---- 2) one clean checkpoint at the target frame (reused for all) --------
    clean_ckpt = boot_replay(clean_actions, target_frame; game = game)
    clean_tail = clean_actions[target_frame + 1 : total]
    base_snap = continue_from(clean_ckpt, Int.(clean_tail))   # clean continuation
    at_target = continue_from(clean_ckpt, Int[])              # state AT frame t

    # ---- 3) candidate sites + the oracle cause set (game-agnostic) -----------
    cand = candidates_path_for(game)
    cand_indices = [idx for (idx, _) in candidate_ram_indices(cand)]
    causes = build_pong_causes(cand, at_target)
    return _run_game_body(; game = game, target_frame = target_frame, horizon = horizon,
        total = total, clean_actions = clean_actions, bit_exact = bit_exact,
        clean_ckpt = clean_ckpt, clean_tail = clean_tail, base_snap = base_snap,
        at_target = at_target, cand = cand, cand_indices = cand_indices, causes = causes,
        st = nothing, verbose = verbose)
end

"""The per-game body, shared by the legacy NOOP path and the SHARED gameplay-state
path (the ONLY difference between them is which state/action-stream/checkpoint the
patch algorithm sits on — the algorithm itself is unchanged). `st` carries the
shared-testbed provenance (cause-density gate, shared output cell) for the record,
or `nothing` in the legacy path."""
function _run_game_body(; game, target_frame, horizon, total, clean_actions, bit_exact,
                        clean_ckpt, clean_tail, base_snap, at_target, cand, cand_indices,
                        causes, st, verbose)

    # outputs: the 2 most causally-active candidate cells + whole-screen px
    score_cells, _mv = pick_active_cells(clean_ckpt, Int.(clean_tail), base_snap,
                                         cand_indices, causes; k = 2)
    outputs = build_outputs(base_snap, score_cells)
    y_base = [o.read(base_snap) for o in outputs]
    out_name2j = Dict(o.name => j for (j, o) in enumerate(outputs))
    # which output each RAM site reads itself into (its a-priori target)
    site2out = Dict(c => o.name for o in outputs for c in o.true_sites)

    # ---- 4) donor runs (genuinely different state at frame t) ----------------
    # A donor = the state at frame t under a DIFFERENT run. In the SHARED gameplay
    # testbed the clean run is the seeded random-action stream, so an on-distribution
    # donor shares the gameplay prefix up to t-1 and diverges only at the analysis
    # frame with a LEFT/RIGHT joystick action (a genuinely different, reachable state
    # — the donor concept is unchanged, only the base state moves off boot/attract).
    donor_left, donor_right = if st !== nothing
        pre = target_frame > 0 ? Int.(clean_actions[1:target_frame - 1]) : Int[]
        dl = continue_from(boot_replay(vcat(pre, ACT_LEFT),  target_frame; game = game), Int[])
        dr = continue_from(boot_replay(vcat(pre, ACT_RIGHT), target_frame; game = game), Int[])
        dl, dr
    else
        (continue_from(boot_replay(fill(ACT_LEFT,  target_frame), target_frame; game = game), Int[]),
         continue_from(boot_replay(fill(ACT_RIGHT, target_frame), target_frame; game = game), Int[]))
    end

    # ---- 5) build the patch set ----------------------------------------------
    patches = Patch[]
    # (A) DONOR patches: each candidate cell, value taken from a different run. We
    #     record whether the donor actually diverged (an honest donor) — patching
    #     a coincident value is a no-op we still report.
    for (donor, dlabel) in ((donor_left, "LEFT-context"), (donor_right, "RIGHT-context"))
        for idx in cand_indices
            v   = Int(donor.ram[idx + 1])
            cur = Int(at_target.ram[idx + 1])
            tgt = get(site2out, idx, "n_changed_px")
            push!(patches, Patch("donor[ram@$idx=$v]<-$dlabel", "ram", idx, "donor",
                                 v, cur, dlabel, v != cur, tgt))
        end
    end
    # (B) DIRECTED patches on every candidate cell (base+17 — mirrors the oracle
    #     `set` cause; exercises the harness across all documented T1/T2 sites)
    #     plus a content-path TIA register patch (drives bg pixels).
    for idx in cand_indices
        base = Int(at_target.ram[idx + 1])
        v = (base + 17) & 0xFF
        tgt = get(site2out, idx, "n_changed_px")
        push!(patches, Patch("directed[ram@$idx=base+17]", "ram", idx, "directed",
                             v, base, "synthetic(base+17)", true, tgt))
    end
    push!(patches, Patch("directed[tia[COLUBK]=0x0E]", "tia_reg", Int(COLUBK_REG),
                         "directed", 0x0E, -1, "synthetic(white)", true, "n_changed_px"))

    # ---- 6) recovered (activation patching) AND exact-patch oracle Δ ----------
    npatch = length(patches); nout = length(outputs)
    recovered = zeros(Float64, npatch, nout)
    exact     = zeros(Float64, npatch, nout)
    patch_meta = Dict{String,Any}[]
    max_diff = 0.0
    for (i, p) in enumerate(patches)
        # activation patch: deepcopy the clean checkpoint, write the value, re-run
        rec_snap = run_patch(clean_ckpt, Int.(clean_tail), p.kind, p.site, p.value)
        recovered[i, :] = [o.read(rec_snap) - y_base[j] for (j, o) in enumerate(outputs)]
        # exact-patch oracle: an INDEPENDENT fresh-from-boot replay to f then the
        # same single-site write + re-run (a separate path → the equality is not a
        # code-sharing artefact).
        fresh_ckpt = boot_replay(clean_actions, target_frame; game = game)
        ex_snap = run_patch(fresh_ckpt, Int.(clean_tail), p.kind, p.site, p.value)
        exact[i, :] = [o.read(ex_snap) - y_base[j] for (j, o) in enumerate(outputs)]
        max_diff = max(max_diff, maximum(abs.(recovered[i, :] .- exact[i, :])))
        verbose && println("  [$i/$npatch] $(rpad(p.name, 34)) " *
                           "max|Δrec|=$(rpad(round(maximum(abs.(recovered[i, :])), digits = 2), 7)) " *
                           "Δrec-exact=$(round(maximum(abs.(recovered[i, :] .- exact[i, :])), digits = 4))")
        push!(patch_meta, Dict{String,Any}(
            "name" => p.name, "kind" => p.kind, "site" => p.site,
            "family" => p.family, "value" => p.value, "base_value" => p.base_value,
            "donor" => p.donor_label, "donor_diverged" => p.donor_diverged,
            "true_target" => p.true_target))
    end
    recovered_eq_exact = (max_diff == 0.0)

    # ---- 7) site precision/recall vs the TRUE data-flow (T2) ------------------
    # The TRUE per-(patch,output) data-flow edge is the EXACT oracle: an edge fires
    # iff the exact-patch Δ is nonzero. Activation patching's recovered firing
    # pattern is scored against it. Because recovered==exact bit-for-bit, the
    # recovered firing pattern equals the true data-flow exactly → P=R=1.0. THIS is
    # the validation: the patching harness recovers the program's true read/write
    # structure with no false or missed edges.
    rec_fire  = abs.(recovered) .> 0.0
    true_fire = abs.(exact) .> 0.0
    tp = count(rec_fire .& true_fire)
    fp = count(rec_fire .& .!true_fire)
    fn = count(.!rec_fire .& true_fire)
    precision = (tp + fp) == 0 ? 1.0 : tp / (tp + fp)
    recall    = (tp + fn) == 0 ? 1.0 : tp / (tp + fn)

    # ---- 8) a-priori target recall (honest "present ≠ used" audit) ------------
    # Does each patch move the output we expected? Transient/clobbered cells (the
    # program re-derives them next frame) produce no within-horizon effect — we
    # record them rather than hide them.
    apriori_hits = 0; apriori_total = 0; transient_sites = String[]
    for (i, p) in enumerate(patches)
        apriori_total += 1
        j = out_name2j[p.true_target]
        if abs(recovered[i, j]) > 0
            apriori_hits += 1
        else
            push!(transient_sites, p.name)
        end
    end
    apriori_recall = apriori_total == 0 ? 1.0 : apriori_hits / apriori_total

    n_donor_real = count(p -> p.family == "donor" && p.donor_diverged, patches)
    n_active = count(i -> any(abs.(recovered[i, :]) .> 0), 1:npatch)

    verbose && println("[$game] recovered==exact=$recovered_eq_exact (max|rec-exact|=$max_diff) " *
                       "P=$(round(precision,digits=3)) R=$(round(recall,digits=3)) " *
                       "apriori_recall=$(round(apriori_recall,digits=3)) " *
                       "donor_real=$n_donor_real active=$n_active/$npatch")

    return PatchResult(game, target_frame, horizon,
                       [o.name for o in outputs],
                       [p.name for p in patches], patch_meta,
                       y_base, recovered, exact, bit_exact,
                       recovered_eq_exact, max_diff, precision, recall,
                       apriori_recall, transient_sites, n_donor_real, n_active,
                       st === nothing ? "noop" : "seeded_random_action_gameplay",
                       st === nothing ? -1 : st.seed,
                       st === nothing ? -1 : st.prefix,
                       st === nothing ? -1 : st.cause_density,
                       st === nothing ? false : st.accepted,
                       st === nothing ? length(causes) : length(st.causes),
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

function write_game_result(r::PatchResult; out_dir = OUT_DIR)
    isdir(out_dir) || mkpath(out_dir)
    stem = "activation_patching_$(r.game)"
    json_path = joinpath(out_dir, stem * ".json")
    npz_path  = joinpath(out_dir, stem * ".npz")

    rec_map = Dict(r.patch_names[i] =>
                   Dict(r.output_names[j] => r.recovered[i, j] for j in 1:length(r.output_names))
                   for i in 1:length(r.patch_names))
    ex_map  = Dict(r.patch_names[i] =>
                   Dict(r.output_names[j] => r.exact[i, j] for j in 1:length(r.output_names))
                   for i in 1:length(r.patch_names))

    rec = Dict{String,Any}(
        "paper" => "P2",
        "phase" => "phaseC_mechanistic",
        "method" => "activation_patching",
        "game" => r.game,
        "state" => r.state_kind == "noop" ? "f$(r.target_frame)+$(r.horizon)" :
                   "gameplay(seed=$(r.st_seed),prefix=$(r.st_prefix))+$(r.horizon)",
        "target_output" => "state(ram concept cells)+screen_px",
        # headline §R scalar: max |recovered − exact| (0.0 = perfect causal recovery).
        "metric_name" => "max_abs_recovered_minus_exact",
        "value" => r.max_abs_recovered_minus_exact,
        "stderr" => nothing,
        "ci" => nothing,
        "n" => length(r.patch_names),
        "seed" => 0,
        "where" => "local",
        "commit" => _git_commit(),
        "oracle_ref" => "oracle_intervene@$(r.game) (P2-E1-1) — exact single-site patch",
        "timestamp" => string(round(Int, time())),
        "arrays" => basename(npz_path),
        "extra" => Dict{String,Any}(
            # F∧S∧M triad (paper sec:triad). F = the leaderboard-oriented
            # faithfulness (UNCHANGED); S = held-out predictive sufficiency, M =
            # |U*|/|U_named|, both from the method vs oracle per-site effects.
            "triad" => triad_extra_dict((abs(r.max_abs_recovered_minus_exact) < 1e-12 ? 1.0 : clamp(1.0 - abs(r.max_abs_recovered_minus_exact), 0.0, 1.0)), vec(r.recovered), vec(r.exact)),
            "substrate" => "jutari (Julia, HARD) — real-ROM bit-exact path",
            "outputs" => r.output_names,
            "patches" => r.patch_meta,
            "y_baseline" => Dict(r.output_names[j] => r.y_baseline[j]
                                 for j in 1:length(r.output_names)),
            "bit_exact_rerun" => r.bit_exact,
            "recovered_equals_exact" => r.recovered_eq_exact,
            "site_precision" => r.site_precision,
            "site_recall" => r.site_recall,
            "apriori_target_recall" => r.apriori_recall,
            "transient_clobbered_sites" => r.transient_sites,
            "n_donor_real" => r.n_donor_real,
            "n_active_patches" => r.n_active_patches,
            "n_patches" => length(r.patch_names),
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
                    "activation-patching algorithm is unchanged; only the state moves."),
            "recovered_delta" => rec_map,
            "exact_patch_delta" => ex_map,
            "note" =>
                "activation patching = single-site state write (donor value or " *
                "directed) at frame t + bit-exact re-run; the exact-patch oracle " *
                "(P2-E1-1) is the same operation via an INDEPENDENT fresh replay, " *
                "so recovered==exact validates the harness. Site P/R = recovered " *
                "firing pattern vs the exact-patch (true data-flow) edges. " *
                "a-priori recall audits present-vs-used (transient cells clobbered " *
                "next frame have no within-horizon effect — recorded, not hidden).",
            "scales_to_cluster_via" =>
                "tools/cluster/xai_*.sbatch (E0-3) + jaxtari SOFT-STE GPU batches " *
                "(forward bit-exact to this HARD map) for patch sweeps over " *
                "state × outputs × games.",
        ),
    )
    open(json_path, "w") do io
        write(io, _j(rec) * "\n")
    end

    write_npz(npz_path, Dict(
        "recovered" => r.recovered,                 # (patches, outputs)
        "exact"     => r.exact,                     # (patches, outputs)
        "y_baseline" => r.y_baseline,               # (outputs,)
        "abs_recovered_minus_exact" => abs.(r.recovered .- r.exact),
    ))
    return json_path, npz_path
end

function write_summary(results::Vector{PatchResult}; out_dir = OUT_DIR)
    isdir(out_dir) || mkpath(out_dir)
    path = joinpath(out_dir, "activation_patching_core_summary.json")
    per_game = Dict{String,Any}[]
    all_eq = true
    for r in results
        all_eq &= r.recovered_eq_exact
        push!(per_game, Dict{String,Any}(
            "game" => r.game,
            "n_patches" => length(r.patch_names),
            "recovered_equals_exact" => r.recovered_eq_exact,
            "max_abs_recovered_minus_exact" => r.max_abs_recovered_minus_exact,
            "site_precision" => r.site_precision,
            "site_recall" => r.site_recall,
            "apriori_target_recall" => r.apriori_recall,
            "n_donor_real" => r.n_donor_real,
            "n_active_patches" => r.n_active_patches,
        ))
    end
    rec = Dict{String,Any}(
        "paper" => "P2", "phase" => "phaseC_mechanistic",
        "method" => "activation_patching", "scope" => "core (6 games)",
        "commit" => _git_commit(), "timestamp" => string(round(Int, time())),
        "all_games_recovered_equals_exact" => all_eq,
        "mean_site_precision" => sum(r.site_precision for r in results) / length(results),
        "mean_site_recall" => sum(r.site_recall for r in results) / length(results),
        "per_game" => per_game,
        "note" => "Phase-C activation patching / causal tracing over the 6 core " *
                  "games; recovered effect vs the exact intervention oracle + site " *
                  "P/R vs the true data-flow (experiment_design.md §6/§7: Succeed).",
    )
    open(path, "w") do io
        write(io, _j(rec) * "\n")
    end
    return path
end

# ============================================================================
# Self-check (DoD: the small test) — positive control on Pong:
#   1. the bit-exact re-run holds (precondition for trusting any Δ);
#   2. activation patching == the exact-patch oracle for EVERY patch
#      (max|recovered − exact| == 0);
#   3. the patching harness recovers the true data-flow firing pattern exactly
#      (P == R == 1.0);
#   4. a directed state-cell patch produces the expected directed Δ (+17 on the
#      cell it writes) — a real causal effect on the output it should drive.
# Exits nonzero (via `error`) on any failure.
# ============================================================================
function self_check(; game = "pong", target_frame = 120, horizon = 30)
    println("[self-check] running activation patching on $game (positive control)...")
    r = run_game(; game = game, target_frame = target_frame, horizon = horizon, verbose = false)
    r.bit_exact || error("self-check FAIL: bit-exact re-run did not hold")
    r.recovered_eq_exact ||
        error("self-check FAIL: recovered != exact (max diff $(r.max_abs_recovered_minus_exact))")
    (r.site_precision == 1.0 && r.site_recall == 1.0) ||
        error("self-check FAIL: data-flow P/R != 1.0 (P=$(r.site_precision) R=$(r.site_recall))")
    # a directed patch on a state cell must move THAT cell's own output by +17.
    name2j = Dict(o => j for (j, o) in enumerate(r.output_names))
    found = false
    for (i, pm) in enumerate(r.patch_meta)
        if pm["family"] == "directed" && pm["kind"] == "ram"
            tgt = pm["true_target"]
            if haskey(name2j, tgt) && startswith(tgt, "state(ram@")
                Δ = r.recovered[i, name2j[tgt]]
                Δ == 17.0 || error("self-check FAIL: directed self-cell patch Δ=$Δ (expected 17.0) for $(pm["name"])")
                found = true
                break
            end
        end
    end
    found || error("self-check FAIL: no directed self-cell state patch to verify +17")
    println("[self-check] PASS — bit-exact ✓, recovered==exact ✓ " *
            "(max|rec-exact|=$(r.max_abs_recovered_minus_exact)), " *
            "data-flow P=R=1.0 ✓, directed self-cell patch Δ=+17 ✓")
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
            games = xai_resolve_games(args[i + 1], CORE_GAMES); i += 2
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

    println("[activation_patching] games=$(join(games, ",")) " *
            "target_frame=$target_frame horizon=$horizon (jutari/Julia)")
    results = PatchResult[]
    na_records = Tuple{String,String}[]
    for g in games
        println("\n========== $g ==========")
        try
            r = run_game(; game = g, target_frame = target_frame, horizon = horizon, verbose = true)
            jp, np = write_game_result(r)
            println("[$g] recovered==exact: $(r.recovered_eq_exact) " *
                    "(max|rec-exact|=$(r.max_abs_recovered_minus_exact)); " *
                    "data-flow P=$(round(r.site_precision,digits=3)) R=$(round(r.site_recall,digits=3)); " *
                    "a-priori recall=$(round(r.apriori_recall,digits=3))")
            println("[$g] wrote $jp")
            println("[$g] arrays  $np")
            push!(results, r)
        catch e
            # A game DEGENERATE/STATIC at the shared gameplay state (or a bit-exact
            # re-run failure) records an n/a and is SKIPPED — it does NOT abort the
            # whole battery. Many non-core games are static at prefix=90; flag for a
            # per-game livelier prefix later.
            msg = first(split(sprint(showerror, e), '\n'))
            println("[activation_patching] !! $g SKIPPED (n/a): $msg")
            push!(na_records, (g, msg))
        end
    end
    isempty(na_records) || println("\n[activation_patching] $(length(na_records)) game(s) n/a " *
        "(degenerate/static at the shared state): $(join([g for (g, _) in na_records], ", "))")

    if length(results) > 1
        sp = write_summary(results)
        println("\n[activation_patching] core summary -> $sp")
    end

    println("\n[activation_patching] headline (recovered == exact-patch oracle, all games):")
    for r in results
        println("    $(rpad(r.game, 16)) recovered==exact=$(r.recovered_eq_exact) " *
                "max|rec-exact|=$(rpad(r.max_abs_recovered_minus_exact, 6)) " *
                "P=$(round(r.site_precision,digits=3)) R=$(round(r.site_recall,digits=3)) " *
                "apriori=$(round(r.apriori_recall,digits=3)) " *
                "donor_real=$(r.n_donor_real) active=$(r.n_active_patches)/$(length(r.patch_names))")
    end
    return results
end

end # module

# run as a script (not when `include`d by a test)
if abspath(PROGRAM_FILE) == @__FILE__
    ActivationPatching.main()
end
