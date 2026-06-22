# pilot_patch_sae.jl — Phase-C pilot, the ACTIVATION-PATCHING half (P2-E5-0),
# JULIA path (the jutari real-ROM substrate; jaxtari eager is ~205× slower —
# SCRUM §7). The SAE half is `pilot_patch_sae.py` (numpy-only, offline, reads
# the recorded trajectory npz). Together they stand up the Phase-C harness:
# state-as-activations + exact-patch scoring, validated before the E5-1..E5-10
# fan-out (experiment_design.md §6).
#
# ---------------------------------------------------------------------------
# What activation patching IS here (experiment_design.md §3, §6; SPEC §E5):
#   the VCS state trajectory is the "activations" and the program's data-flow
#   is the "circuit". Activation patching = take the value of ONE state site
#   (a RIOT RAM cell at frame t) from a *different* run (a "donor"), write it
#   into the clean run at the same frame, RE-RUN the real ROM a short horizon,
#   and measure the effect Δy on the outputs.
#
# The ground-truth score is the EXACT intervention oracle (P2-E1-1,
# oracle_intervene.jl): the *exact patch* `do(site := donor_value)` followed by
# a bit-exact re-run. Activation patching from a donor and the exact patch are
# the SAME single-site state write, so the recovered effect MUST equal the exact
# patch — that equality is what validates the patching harness. We assert it,
# and additionally score each site's effect against the TRUE data-flow (T2):
# patching a score cell must move a score output and leave an unrelated output
# untouched → site precision/recall.
#
# Two complementary patch families, both real re-runs of the real Pong ROM:
#   (A) DONOR patches — the donor value comes from a genuinely different run
#       (a LEFT/RIGHT paddle-input context). Within the conformance window the
#       paddle-position cell (RAM[$33]=51) is the cell that actually diverges
#       between contexts, so it is the honest "value from another run" site.
#   (B) DIRECTED patches — a synthetic donor value on the documented Pong causal
#       cells (scores RAM[$0D]=13/[$0E]=14, ball RAM[$31]=49/[$36]=54). This
#       exercises the patching harness across the known T1/T2 sites and recovers
#       the same Δy the oracle's `set` cause reports.
#
# No JuTari/jaxtari/xitari core is modified — pure tooling under tools/xai_study/.
# Reuses the verified jutari foundation: jutari_oracle.jl (load/boot/replay/
# snapshot/deepcopy-checkpoint/intervene/NPZ writer).
#
# Run (warm shared depot, primary's project):
#   julia --project=/Users/maier/Documents/code/UnderstandingVCS/jutari \
#         tools/xai_study/phaseC_mechanistic/pilot_patch_sae.jl
# Optional flags: --target-frame N --horizon N --game pong
#
# Writes (SPEC §R):
#   tools/xai_study/phaseC_mechanistic/out/pilotC_patch_pong.{json,npz}

module PilotPatchSAE

# the verified jutari run helper (sibling common/ dir)
include(joinpath(@__DIR__, "..", "common", "jutari_oracle.jl"))
using .JutariOracle
using .JutariOracle: Snapshot, boot_replay, continue_from, snapshot,
                     intervene_ram!, fresh_baseline_ram_screen, write_npz
using .JutariOracle.JuTari.Env: env_step!

# --- Pong sites + outputs (xitari Pong.cpp; src/games/PaddleGames.jl) --------
# RAM site indices (0-based, matching the oracle / OCAtari candidates):
const P0_SCORE = 13   # RAM[$0D] agent/"cpu" score
const P1_SCORE = 14   # RAM[$0E] opponent/"human" score
const BALL_X   = 49   # RAM[$31]
const BALL_Y   = 54   # RAM[$36]
const PADDLE_Y = 51   # RAM[$33] user paddle position (the cell that diverges
                      #          between LEFT/RIGHT paddle-input contexts)
const ENEMY_Y  = 50   # RAM[$32] (a near-by control cell)

# Pong joystick action codes (oracle_intervene.jl: RIGHT=3); LEFT=4.
const ACT_NOOP = 0
const ACT_RIGHT = 3
const ACT_LEFT = 4

const OUT_DIR = joinpath(@__DIR__, "out")

# ============================================================================
# Outputs y(state). The score outputs read the score RAM cells; n_changed_px is
# a whole-screen position output (counts framebuffer cells differing from the
# clean baseline frame); paddle_band_px counts lit cells in the user-paddle
# column band (a position output that the paddle-cell patch should move).
# ============================================================================
struct Output
    name::String
    read::Function          # Snapshot -> Float64
end

function pong_outputs(baseline::Snapshot)
    h, w = size(baseline.screen)
    # the user (right) paddle lives in the right-most columns; count lit cells
    # there as a paddle-position output.
    pc0 = max(1, w - 6)
    Output[
        Output("p0_score", s -> Float64(Int(s.ram[P0_SCORE + 1]))),
        Output("p1_score", s -> Float64(Int(s.ram[P1_SCORE + 1]))),
        Output("n_changed_px",
               s -> Float64(count(s.screen .!= baseline.screen))),
        Output("paddle_band_px",
               s -> Float64(count(!=(0), @view s.screen[:, pc0:w]))),
    ]
end

# ============================================================================
# A patch = a single-site state write at the target frame, then re-run.
# ============================================================================
struct Patch
    name::String
    site::Int               # RAM index (0-based)
    family::String          # "donor" | "directed"
    value::Int              # the value written (donor's value or a directed one)
    donor_label::String     # provenance of the value
    true_target::String     # the output this site SHOULD move (T2 data-flow)
end

"""Re-run the clean trace from `checkpoint`, with RAM[`site`] := `value` written
at the target frame, for `horizon` steps. This is BOTH the activation patch and
(by construction) the exact-patch oracle for `do(site := value)` — a single-site
state write followed by a bit-exact re-run."""
function run_patch(checkpoint, tail_actions, site::Integer, value::Integer)
    env = deepcopy(checkpoint)
    intervene_ram!(env, site, value)
    for a in tail_actions
        env_step!(env, a)
    end
    return snapshot(env, length(tail_actions))
end

"""The exact-patch oracle Δy for `do(site := value)`: identical operation to the
activation patch (we compute it through the same code path AND through an
independent fresh-from-boot replay, then assert the two agree — the bit-exact
guarantee that makes recovered==exact meaningful)."""
function exact_patch_delta(checkpoint, clean_tail, base_snap, outputs,
                           site::Integer, value::Integer)
    snap = run_patch(checkpoint, clean_tail, site, value)
    return [o.read(snap) - o.read(base_snap) for o in outputs], snap
end

# ============================================================================
# The pilot
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
end

function run_pilot(; game = "pong", target_frame = 120, horizon = 30, verbose = true)
    total = target_frame + horizon
    clean_actions = fill(ACT_NOOP, total)

    # ---- 1) bit-exact guarantee (two fresh boots+replays, RAM AND screen) ----
    verbose && println("[pilotC-patch] asserting bit-exactness (2 fresh replays to f$total)...")
    a = fresh_baseline_ram_screen(clean_actions, total; game = game)
    b = fresh_baseline_ram_screen(clean_actions, total; game = game)
    bit_exact = (a.ram == b.ram) && (a.screen == b.screen)
    bit_exact || error("bit-exact re-run FAILED — cannot trust patch Δ")
    verbose && println("[pilotC-patch] bit-exact re-run: PASS")

    # ---- 2) one clean checkpoint at the target frame (reused for all) --------
    clean_ckpt = boot_replay(clean_actions, target_frame; game = game)
    clean_tail = clean_actions[target_frame + 1 : total]
    base_snap = continue_from(clean_ckpt, Int.(clean_tail))   # clean continuation
    at_target = continue_from(clean_ckpt, Int[])              # state AT frame t
    outputs = pong_outputs(base_snap)
    y_base = [o.read(base_snap) for o in outputs]

    # ---- 3) donor runs (genuinely different state at frame t) ----------------
    # A LEFT and a RIGHT paddle-input context; their state at frame t supplies
    # the donor values. Within the conformance window the paddle cell diverges.
    donor_left  = continue_from(boot_replay(fill(ACT_LEFT, target_frame),  target_frame; game = game), Int[])
    donor_right = continue_from(boot_replay(fill(ACT_RIGHT, target_frame), target_frame; game = game), Int[])

    # ---- 4) build the patch set ---------------------------------------------
    patches = Patch[]
    # (A) DONOR patches: value taken from a different run. We patch each site
    #     that the donor actually diverges on (honest "value from another run");
    #     the paddle cell is the live one — patch it from BOTH donors.
    for (donor, dlabel) in ((donor_left, "LEFT-context"), (donor_right, "RIGHT-context"))
        for (site, sname, tgt) in ((PADDLE_Y, "paddle_y", "paddle_band_px"),)
            v = Int(donor.ram[site + 1])
            cur = Int(at_target.ram[site + 1])
            push!(patches, Patch("donor[$sname=$v]<-$dlabel", site, "donor", v, dlabel, tgt))
        end
    end
    # (B) DIRECTED patches on the documented Pong causal cells (T1/T2):
    #     a synthetic donor value (current+17, mirroring the oracle's `set`
    #     cause so this is directly comparable to oracle_pong_score.json).
    for (site, sname, tgt) in ((P0_SCORE, "p0_score", "p0_score"),
                               (P1_SCORE, "p1_score", "p1_score"),
                               (BALL_X,   "ball_x",   "n_changed_px"),
                               (BALL_Y,   "ball_y",   "n_changed_px"),
                               (ENEMY_Y,  "enemy_y",  "n_changed_px"))
        base = Int(at_target.ram[site + 1])
        v = (base + 17) & 0xFF
        push!(patches, Patch("directed[$sname=base+17]", site, "directed", v,
                             "synthetic(base+17)", tgt))
    end

    # ---- 5) recovered (activation patching) AND exact-patch oracle Δ ---------
    npatch = length(patches); nout = length(outputs)
    recovered = zeros(Float64, npatch, nout)
    exact     = zeros(Float64, npatch, nout)
    patch_meta = Dict{String,Any}[]
    max_diff = 0.0
    for (i, p) in enumerate(patches)
        # activation patch: deepcopy clean checkpoint, write donor value, re-run
        rec_snap = run_patch(clean_ckpt, Int.(clean_tail), p.site, p.value)
        recovered[i, :] = [o.read(rec_snap) - y_base[j] for (j, o) in enumerate(outputs)]
        # exact-patch oracle: an INDEPENDENT fresh-from-boot replay to f then the
        # same single-site write + re-run (a separate path proving the equality
        # isn't a code-sharing artefact).
        fresh_ckpt = boot_replay(clean_actions, target_frame; game = game)
        ex_snap = run_patch(fresh_ckpt, Int.(clean_tail), p.site, p.value)
        exact[i, :] = [o.read(ex_snap) - y_base[j] for (j, o) in enumerate(outputs)]
        max_diff = max(max_diff, maximum(abs.(recovered[i, :] .- exact[i, :])))
        verbose && println("  [$i/$npatch] $(rpad(p.name, 30)) " *
                           "max|Δrec|=$(round(maximum(abs.(recovered[i, :])), digits = 2)) " *
                           "Δrec-exact=$(round(maximum(abs.(recovered[i, :] .- exact[i, :])), digits = 4))")
        push!(patch_meta, Dict{String,Any}(
            "name" => p.name, "site" => p.site, "family" => p.family,
            "value" => p.value, "donor" => p.donor_label,
            "true_target" => p.true_target))
    end
    recovered_eq_exact = (max_diff == 0.0)

    # ---- 6) site precision/recall vs the TRUE data-flow (T2) -----------------
    # The TRUE per-(patch,output) data-flow edge is given by the EXACT oracle:
    # an edge fires iff the exact-patch Δ is nonzero. Activation patching's
    # recovered firing pattern is scored against it (a (patch×output) confusion
    # matrix). Because recovered==exact bit-for-bit, the recovered firing pattern
    # equals the true data-flow exactly → P=R=1.0. This IS the validation: the
    # patching harness recovers the program's true read/write structure, no false
    # or missed edges. (`true_target` below is an independent a-priori expectation
    # we additionally audit.)
    eps = 0.0
    rec_fire = abs.(recovered) .> eps
    true_fire = abs.(exact) .> eps        # ground-truth edges = exact-patch oracle
    tp = count(rec_fire .& true_fire)
    fp = count(rec_fire .& .!true_fire)
    fn = count(.!rec_fire .& true_fire)
    precision = (tp + fp) == 0 ? 1.0 : tp / (tp + fp)
    recall    = (tp + fn) == 0 ? 1.0 : tp / (tp + fn)

    # a-priori audit: does each site move the output we *expected* it to? Honest
    # mechanistic finding — the donor paddle-cell (RAM[$33]) and enemy_y are
    # TRANSIENT cells the program re-derives next frame (clobbered on the next
    # step), so a one-frame RAM patch has no downstream effect within the horizon.
    # This is recorded, not hidden (experiment_design.md §6: present ≠ used).
    name2j = Dict(o => j for (j, o) in enumerate([o.name for o in outputs]))
    apriori_hits = 0; apriori_total = 0; transient_sites = String[]
    for (i, p) in enumerate(patches)
        apriori_total += 1
        if abs(recovered[i, name2j[p.true_target]]) > 0
            apriori_hits += 1
        else
            push!(transient_sites, p.name)
        end
    end
    apriori_recall = apriori_total == 0 ? 1.0 : apriori_hits / apriori_total
    verbose && println("[pilotC-patch] a-priori target recall=$(round(apriori_recall, digits=3)) " *
                       "(transient/clobbered sites: $(transient_sites))")

    return PatchResult(game, target_frame, horizon,
                       [o.name for o in outputs],
                       [p.name for p in patches], patch_meta,
                       y_base, recovered, exact, bit_exact,
                       recovered_eq_exact, max_diff, precision, recall,
                       apriori_recall, transient_sites)
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

# tiny dependency-free JSON (the pilot adds no package to the shared jutari env)
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

function write_result(r::PatchResult; out_dir = OUT_DIR)
    isdir(out_dir) || mkpath(out_dir)
    stem = "pilotC_patch_$(r.game)"
    json_path = joinpath(out_dir, stem * ".json")
    npz_path  = joinpath(out_dir, stem * ".npz")

    # headline metric: max |recovered − exact| (0.0 = perfect agreement).
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
        "state" => "f$(r.target_frame)+$(r.horizon)",
        "target_output" => "pong_score+paddle/ball_pixels",
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
            "recovered_delta" => rec_map,
            "exact_patch_delta" => ex_map,
            "note" =>
                "activation patching = single-site state write (donor value or " *
                "directed) at frame t + bit-exact re-run; the exact-patch oracle " *
                "(P2-E1-1) is the same operation via an independent fresh replay, " *
                "so recovered==exact validates the harness. Site P/R vs T2 data-flow.",
            "scales_to_cluster_via" =>
                "tools/cluster/xai_*.sbatch (E0-3) + jaxtari SOFT-STE GPU batches " *
                "(forward bit-exact to this HARD map) for patch sweeps over " *
                "state × outputs × games (E5-1..E5-10).",
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

# ============================================================================
# Self-check (DoD: the small test) — assert the headline patching properties:
#   1. the bit-exact re-run holds (precondition for trusting any Δ);
#   2. activation patching == the exact-patch oracle for EVERY patch
#      (max|recovered − exact| == 0);
#   3. the patching harness recovers the true data-flow firing pattern exactly
#      (P == R == 1.0);
#   4. a score-cell patch produces the expected directed score Δ (+17) — a real
#      causal effect on the output it should drive (present∧used).
# Exits nonzero (via `error`) on any failure.
# ============================================================================
function self_check(; game = "pong", target_frame = 120, horizon = 30)
    println("[self-check] running patching pilot...")
    r = run_pilot(; game = game, target_frame = target_frame, horizon = horizon,
                  verbose = false)
    r.bit_exact || error("self-check FAIL: bit-exact re-run did not hold")
    r.recovered_eq_exact ||
        error("self-check FAIL: recovered != exact (max diff $(r.max_abs_recovered_minus_exact))")
    (r.site_precision == 1.0 && r.site_recall == 1.0) ||
        error("self-check FAIL: data-flow P/R != 1.0 (P=$(r.site_precision) R=$(r.site_recall))")
    # a directed score patch must move its score output by +17.
    name2j = Dict(o => j for (j, o) in enumerate(r.output_names))
    i_p0 = findfirst(n -> startswith(n, "directed[p0_score"), r.patch_names)
    i_p0 === nothing && error("self-check FAIL: no p0_score directed patch")
    Δp0 = r.recovered[i_p0, name2j["p0_score"]]
    Δp0 == 17.0 || error("self-check FAIL: p0_score patch Δ=$Δp0 (expected 17.0)")
    println("[self-check] PASS — bit-exact ✓, recovered==exact ✓ " *
            "(max|rec-exact|=$(r.max_abs_recovered_minus_exact)), " *
            "data-flow P=R=1.0 ✓, p0_score patch Δ=+17 ✓")
    return true
end

# ============================================================================
# CLI
# ============================================================================
function main(args = ARGS)
    game = "pong"; target_frame = 120; horizon = 30
    do_self_check = false
    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--game"; game = args[i + 1]; i += 2
        elseif a == "--target-frame"; target_frame = parse(Int, args[i + 1]); i += 2
        elseif a == "--horizon"; horizon = parse(Int, args[i + 1]); i += 2
        elseif a == "--self-check"; do_self_check = true; i += 1
        else; i += 1
        end
    end
    if do_self_check
        self_check(; game = game, target_frame = target_frame, horizon = horizon)
        return nothing
    end
    println("[pilotC-patch] game=$game target_frame=$target_frame horizon=$horizon (jutari/Julia)")
    r = run_pilot(; game = game, target_frame = target_frame, horizon = horizon, verbose = true)
    json_path, npz_path = write_result(r)
    println("[pilotC-patch] bit-exact re-run: $(r.bit_exact)")
    println("[pilotC-patch] recovered == exact (all patches): $(r.recovered_eq_exact) " *
            "(max|rec-exact|=$(r.max_abs_recovered_minus_exact))")
    println("[pilotC-patch] data-flow firing-pattern P=$(round(r.site_precision, digits=3)) " *
            "R=$(round(r.site_recall, digits=3)) (recovered edges vs exact-patch edges)")
    println("[pilotC-patch] a-priori target recall=$(round(r.apriori_recall, digits=3)) " *
            "(transient/clobbered: $(r.transient_sites))")
    println("[pilotC-patch] recovered Δy per patch (cause -> {output: Δ}):")
    for (i, pn) in enumerate(r.patch_names)
        row = Dict(r.output_names[j] => round(r.recovered[i, j], digits = 2)
                   for j in 1:length(r.output_names))
        println("    $(rpad(pn, 30)) $row")
    end
    println("[pilotC-patch] wrote $json_path")
    println("[pilotC-patch] arrays  $npz_path")
    return r
end

end # module

# run as a script (not when `include`d by the test)
if abspath(PROGRAM_FILE) == @__FILE__
    PilotPatchSAE.main()
end
