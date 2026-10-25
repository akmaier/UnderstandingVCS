# e2e_gradient.jl — Phase-B attribution, TRUE END-TO-END GRADIENT arm (JULIA path).
#
# WHY THIS ARM EXISTS. Every other gradient-family runner (saliency, Grad×Input,
# SmoothGrad, IG, EG, guided backprop) differentiates a one-line surrogate taken
# AT the intervention frame f*: y = soft_ram_peek(ram_f*, idx), whose gradient is
# e_idx by construction (or the bilinear sampler for the position output). No CPU
# instruction and no frame of the horizon is ever differentiated. Here the
# gradient is taken THROUGH THE MACHINE: the oracle's output is read at the END of
# the horizon, f*+H, and differentiated back to the RAM tape at f* through every
# 6507 instruction the real ROM executes in between, via the full-system SOFT step
# (`JuTari.Diff.soft_system_step`, Theorem 1: forward bit-exact to the hard
# console, gradient = the soft handlers' surrogate, joined by the STE).
#
#     g(u) = ∂ y(f*+H) / ∂ ram_f*[u]      through  Φ_S^{N}  (N ≈ H × instr/frame)
#
# WHAT THE GRADIENT CAN AND CANNOT SEE (stated, not hidden). soft_system_step keeps
# TIA / RIOT / cartridge state in the hard console: reads of those are replayed as
# constants and carry no gradient; non-RAM writes are dropped from the soft path.
# So g flows through registers, RAM, ALU ops, loads/stores and the ROM — the CPU
# program — but NOT through the renderer or the timer. Discrete addressing (an
# index taken from a RAM byte) and branch decisions carry no gradient either (the
# STE hard path decides; the soft handler sees the decided address). Hence:
#   * CONTENT output  y = RAM[content_idx] at f*+H   — fully inside the gradient's
#     reach (no renderer between the cause and y). The headline of this arm.
#   * POSITION output (the shared screen region)    — the renderer sits between
#     RAM and the pixel, so we chain the CPU/RAM rollout with the SAME bilinear
#     sampler the sampler-on arm uses, applied to RAM at f*+H (the sprite drawn at
#     the column its position byte holds at the end of the horizon). Labelled
#     "e2e rollout ∘ sampler"; the sampler, not the TIA, supplies ∂pixel/∂ram.
#
# SCORED AGAINST THE SAME ORACLE, SAME STATE, SAME CAUSES, SAME METRICS as
# saliency.jl (this file includes it and reuses build_shared_testbed,
# pick_content_idx, oracle_abs_delta, deletion_insertion_auc and the scorers
# verbatim). Two attributions from the one end-to-end gradient:
#   * e2e saliency      |g|
#   * e2e Grad×Input    |g ⊙ ram_f*|
# Path methods (IG / EG) are NOT run end-to-end: they need the hard machine at
# interpolated (non-integer) RAM, which has no meaning; the STE gradient exists
# only along the actual hard trajectory.
#
# COMPUTATION. Reverse mode over ~10^5 instructions in one Zygote tape is too
# large (Zygote's cost per instruction grows with tape length), so the backward
# pass is checkpointed every 200 instructions: the hard console is stored at each
# chunk start, and chunk by chunk, last first, the map (cpu,ram)→(cpu',ram') is
# re-run under Zygote.pullback from its stored console and the cotangent pulled
# back through it. Exact chain rule; one chunk of tape in memory. Validity gates
# (asserted): the env_step!-mirror reproduces env_step!, and every chunk's soft
# forward equals the hard registers + RAM at the chunk's end, bit-for-bit.
#
# Run (warm depot, primary's project):
#   julia --project=<repo>/jutari tools/xai_study/phaseB_attribution/e2e_gradient.jl --games core
# Flags: --games core|<g1,g2,...>  --game <g>  --topk K  --selftest
#        --merge (fold out/e2e_summary_<g>.json of --games into e2e_core_summary.json)
#        XAI_ST_PREFIX / XAI_ST_HORIZON / XAI_ST_SEED as in saliency.jl (90 / 15 / 0).
#
# Writes (SPEC §R; file_scope e2e_*):
#   out/e2e_saliency_<game>_{content,position}.{json,npz}
#   out/e2e_gradxinput_<game>_{content,position}.{json,npz}
#   out/e2e_core_summary.json

module E2EGradient

using JSON
import Zygote
import Statistics

using JuTari
const JE = JuTari.Env
const JD = JuTari.Diff

include(joinpath(@__DIR__, "saliency.jl"))
using .SaliencyAttr
const SA = SaliencyAttr

const OUT_DIR = joinpath(@__DIR__, "out")
const CORE_GAMES = SA.CORE_GAMES
const _BUDGET = 25000                      # xitari execute(25000), as run_until_frame!

# ============================================================================
# The env_step! mirror: action application, the framebuffer swap, reward and
# terminal bookkeeping run on the hard env exactly as env_step! does; the CPU
# loop of run_until_frame! is split into checkpointed chunks.
# ============================================================================
const _NSTATE = 11
_state_vec(s) = [s.A, s.X, s.Y, s.SP, s.PC, s.P, s.cycles, s.P_N, s.P_Z, s.P_C, s.P_V]
_state_of(v) = JD.SoftCPUState(v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[9], v[10], v[11])

function _pre_frame!(env, action::Int)
    env.terminal && return false
    a = JE.romsettings_is_legal_action(env.settings, action) ? action : 0
    up = JE.romsettings_uses_paddles(env.settings)
    up && JE._apply_paddle_action!(env, a)
    JE.apply_action!(env.console, a; paddle_mode = up,
                     swap_paddles = JE.romsettings_swap_paddles(env.settings),
                     player = JE.romsettings_agent_player(env.settings))
    tia = env.console.bus.tia
    if tia.buffer_swap_pending
        tia.framebuffer, tia.framebuffer_prev = tia.framebuffer_prev, tia.framebuffer
        tia.buffer_swap_pending = false
    end
    return true
end

function _post_frame!(env)
    JE.romsettings_get_reward(env.settings, env.console)
    env.terminal = JE.romsettings_is_terminal(env.settings, env.console)
    return nothing
end

"""
    hard_chunks(checkpoint, tail; chunk) -> (chunks, env_T)

Run the horizon on a copy of `checkpoint` exactly as `env_step!` does (the
same `_pre_frame!` / CPU loop / `_post_frame!` sequence), saving a deepcopy of
the console every `chunk` instructions. A chunk never straddles a frame
boundary, so action application between frames happens on the hard machine
only (no soft state is touched by it). Returns the ordered `(console, n)`
list and the env at f*+H."""
function hard_chunks(checkpoint, tail::Vector{Int}; chunk::Int)
    env = deepcopy(checkpoint)
    chunks = Tuple{Any,Int}[]
    for a in tail
        _pre_frame!(env, a) || continue
        con = env.console
        f0 = con.bus.tia.frame
        n_total = 0; done = false
        while !done && n_total < _BUDGET
            snap = deepcopy(con); n = 0
            while n < chunk && n_total < _BUDGET
                JuTari.ConsoleModule.console_step!(con); n += 1; n_total += 1
                if con.bus.tia.frame != f0
                    done = true; break
                end
            end
            push!(chunks, (snap, n))
        end
        _post_frame!(env)
    end
    return chunks, env
end

# One chunk of full-system soft instructions on a private console copy.
function soft_chunk(svec, ram, rom, con, n::Int)
    st = _state_of(svec); bus = JD.SoftBus(ram, rom)
    for _ in 1:n
        st, bus = JD.soft_system_step(st, bus, con)
    end
    return _state_vec(st), bus.ram
end

_hard_regs(con) = (c = con.cpu; Float32[c.A, c.X, c.Y, c.SP, c.PC, c.P])

"""
    e2e_vjp(checkpoint, tail, cot_fns; chunk=200) -> (grads, info)

Checkpointed reverse pass over the horizon. `cot_fns` is a vector of functions
`ram_T -> ∂y/∂ram_T` (the gradient of each end-of-horizon read). Every chunk's
map (cpu,ram)→(cpu',ram') is re-run under `Zygote.pullback` from its stored
console, last chunk first, and each output's cotangent is pulled back through
it — the exact chain rule, with one chunk of tape in memory. Each chunk's soft
forward is checked against the hard state at the chunk's end (the next
checkpoint, or f*+H): `info.exact` is true only if every chunk matched
bit-for-bit (registers + RAM). ROM reads are constants in reverse mode
(`with_rom_grad(false)`), since only ∂/∂RAM is scored."""
function e2e_vjp(checkpoint, tail::Vector{Int}, cot_fns; chunk::Int = 200)
    rom = Float32.(checkpoint.console.bus.cart.rom)
    chunks, env_T = hard_chunks(checkpoint, tail; chunk = chunk)
    # bit-exactness of the env_step!-mirror itself vs the real env_step!
    ref = deepcopy(checkpoint)
    for a in tail; JE.env_step!(ref, a); end
    mirror_exact = ref.console.bus.ram == env_T.console.bus.ram &&
                   _hard_regs(ref.console) == _hard_regs(env_T.console)
    ram_T = Float32.(collect(env_T.console.bus.ram))
    dS = [zeros(Float32, _NSTATE) for _ in cot_fns]
    dR = [Float32.(f(ram_T)) for f in cot_fns]
    exact = mirror_exact
    n_instr = 0
    JD.with_rom_grad(false) do
        for j in length(chunks):-1:1
            (length(chunks) - j) % 50 == 0 &&
                (println("[e2e]   chunk $(length(chunks) - j + 1)/$(length(chunks))"); flush(stdout))
            con0, n = chunks[j]
            n_instr += n
            con = deepcopy(con0)
            st0, bus0 = JD.soft_system_init(con)
            (sv, rm), back = Zygote.pullback((s, r) -> soft_chunk(s, r, rom, con, n),
                                             _state_vec(st0), copy(bus0.ram))
            nxt = j < length(chunks) ? chunks[j + 1][1] : env_T.console
            exact &= Float32.(collect(nxt.bus.ram)) == rm && _hard_regs(nxt) == sv[1:6]
            for k in eachindex(cot_fns)
                gs, gr = back((dS[k], dR[k]))
                dS[k] = gs === nothing ? zeros(Float32, _NSTATE) : Float32.(collect(gs))
                dR[k] = gr === nothing ? zeros(Float32, 128) : Float32.(collect(gr))
            end
        end
    end
    return dR, (exact = exact, mirror_exact = mirror_exact, ram_T = ram_T,
                n_instr = n_instr, n_chunks = length(chunks))
end

# ============================================================================
# Scoring — the saliency.jl contract with an externally supplied attribution.
# ============================================================================
function score(; game, method, output_kind, output_name, content_idx, attr_full,
               checkpoint, actions, tf, hz, causes, read_y, topk, seed)
    odelta = SA.oracle_abs_delta(checkpoint, actions, tf, hz, causes, read_y)
    attr = SA.ig_attribution_per_cause(attr_full, causes)
    pr = SA.pearson(attr, odelta); sp = SA.spearman(attr, odelta)
    pak = SA.precision_at_k(attr, odelta, topk)
    order = sortperm(attr; rev = true)
    del_auc, ins_auc, del_curve, ins_curve =
        SA.deletion_insertion_auc(checkpoint, actions, tf, hz, causes, order, read_y)
    or_order = sortperm(odelta; rev = true)
    or_del, or_ins, _, _ = SA.deletion_insertion_auc(checkpoint, actions, tf, hz,
                                                     causes, or_order, read_y)
    return (game = game, method = method, output_kind = output_kind, output = output_name,
            content_idx = content_idx, cause_names = [c.name for c in causes],
            odelta = odelta, attr = attr, attr_full = Float64.(attr_full),
            pearson = pr, spearman = sp, precision_at_k = pak,
            deletion_auc = del_auc, insertion_auc = ins_auc,
            del_curve = del_curve, ins_curve = ins_curve,
            oracle_self_pearson = SA.pearson(odelta, odelta),
            oracle_self_precision_at_k = SA.precision_at_k(odelta, odelta, topk),
            oracle_self_deletion_auc = or_del, oracle_self_insertion_auc = or_ins,
            degenerate = Statistics.std(odelta) == 0, topk = topk, seed = seed,
            tf = tf, hz = hz)
end

function compute_game(game; topk, seed, verbose = true)
    st = SA.build_shared_testbed(game;
        settings_for = SA.settings_for, rom_path_for = SA.rom_path_for,
        candidates_path_for = SA.candidates_path_for,
        build_causes = SA.build_pong_causes, candidate_ram_indices = SA.candidate_ram_indices,
        continue_from = SA.continue_from, snapshot = SA.snapshot, env_step = JE.env_step!,
        intervene_ram = SA.intervene_ram!, boot_replay = SA.boot_replay,
        run_intervention = SA.run_intervention, soft_ram_peek = JD.soft_ram_peek,
        prefix = SA.ST_PREFIX, horizon = SA.ST_HORIZON, seed = SA.ST_SEED,
        k = SA.ST_GATE_K, floor = SA.ST_FLOOR, verbose = verbose,
        assert_bit_exact = SA.assert_bit_exact)
    actions = st.actions; checkpoint = st.checkpoint; causes = st.causes
    tf = st.prefix; hz = st.horizon
    content_idx, _ = SA.pick_content_idx(checkpoint, actions, tf, hz, causes, st.cand_indices)
    tail = Int.(actions[tf + 1 : tf + hz])

    # end-of-horizon cotangents: content one-hot; position = sampler at ram_T with
    # the sprite's baseline column re-anchored to the baseline RAM at f*+H.
    cot_content = ram_T -> (g = zeros(Float32, 128); g[content_idx + 1] = 1f0; g)
    geomT = nothing
    if st.geom !== nothing
        pidx, _, offs, py0, px0 = st.geom
        geomT = (pidx, Int(st.base.ram[pidx + 1]), offs, py0, px0)
    end
    cot_pos = ram_T -> geomT === nothing ? zeros(Float32, 128) :
        Float32.(something(Zygote.gradient(r -> SA._st_sampler_position_read(r, geomT, st.cell;
                 soft_ram_peek = JD.soft_ram_peek), ram_T)[1], zeros(Float32, 128)))

    verbose && (println("[e2e] $game: content=RAM[$content_idx]  backward over $hz frames ..."); flush(stdout))
    t = @elapsed (grads, info) = e2e_vjp(checkpoint, tail, [cot_content, cot_pos])
    info.exact || error("soft rollout NOT bit-exact to env_step! for $game " *
                        "(mirror_exact=$(info.mirror_exact)) — refusing to score")
    g_content, g_pos = grads
    ram_now = st.ram_now
    verbose && println("[e2e] $game: forward exact ✓ ($(info.n_instr) instructions, " *
        "$(info.n_chunks) chunks)  backward $(round(t, digits=1)) s  " *
        "‖g_content‖₁=$(round(sum(abs, g_content), sigdigits=3)) nnz=$(count(!iszero, g_content))  " *
        "‖g_pos‖₁=$(round(sum(abs, g_pos), sigdigits=3)) nnz=$(count(!iszero, g_pos))")

    read_content = s -> Float64(Int(s.ram[content_idx + 1]))
    outs = Any[]
    common = (; game, checkpoint, actions, tf, hz, causes, topk, seed)
    for (method, f) in (("e2e_saliency", g -> abs.(g)), ("e2e_gradxinput", g -> abs.(g .* ram_now)))
        push!(outs, score(; common..., method, output_kind = "content",
            output_name = "content(ram@$content_idx, f*+$hz)", content_idx,
            attr_full = f(g_content), read_y = read_content))
        push!(outs, score(; common..., method, output_kind = "position",
            output_name = "screen_region(n_changed_px)@r$(st.cell[1])c$(st.cell[2])",
            content_idx = -1, attr_full = f(g_pos), read_y = st.read_y))
    end
    meta = (seconds = t, n_instr = info.n_instr, g_content = g_content, g_pos = g_pos, geom = st.geom,
            cell = st.cell, cause_density = st.cause_density, accepted = st.accepted,
            prefix = st.prefix, horizon = st.horizon, seed = st.seed)
    return outs, meta
end

# ============================================================================
# Persist (SPEC §R).
# ============================================================================
function write_record(r, meta; out_dir = OUT_DIR)
    isdir(out_dir) || mkpath(out_dir)
    stem = "$(r.method)_$(r.game)_$(r.output_kind)"
    jp = joinpath(out_dir, stem * ".json"); np = joinpath(out_dir, stem * ".npz")
    gl1 = Float64(sum(abs, r.output_kind == "content" ? meta.g_content : meta.g_pos))
    rec = Dict{String,Any}(
        "paper" => "P2", "phase" => "phaseB_attribution", "method" => r.method,
        "game" => r.game,
        "state" => "gameplay(seed=$(meta.seed),prefix=$(meta.prefix))+$(meta.horizon)",
        "target_output" => r.output,
        "metric_name" => "pearson_corr_with_oracle", "value" => r.pearson,
        "stderr" => nothing, "ci" => nothing, "n" => length(r.cause_names),
        "seed" => r.seed, "where" => "local", "commit" => SA._git_commit(),
        "oracle_ref" => "oracle_intervene@$(r.game)#$(r.output)",
        "timestamp" => string(round(Int, time())), "arrays" => basename(np),
        "extra" => Dict{String,Any}(
            "method_ref" => r.method == "e2e_saliency" ?
                "vanilla gradient (Simonyan et al. 2014), TRUE END-TO-END: |∂y(f*+H)/∂ram_f*| through the full-system SOFT rollout" :
                "Gradient×Input (Shrikumar et al. 2016), TRUE END-TO-END: |∂y(f*+H)/∂ram_f* ⊙ ram_f*|",
            "substrate" => "jutari Diff.soft_system_step over every instruction of the horizon " *
                "(forward asserted bit-exact to env_step!); Zygote VJP checkpointed every 200 instructions. " *
                "Gradient flows through CPU registers, RAM, ALU and ROM; TIA/RIOT reads are " *
                "constants; addressing/branch decisions carry no gradient.",
            "gradient_path" => r.output_kind == "content" ?
                "content: y = RAM[idx] at f*+H, differentiated through the CPU rollout" :
                "position: CPU rollout ∘ bilinear sampler at f*+H (sampler supplies ∂pixel/∂ram_T; no TIA gradient)",
            "output_kind" => r.output_kind, "content_ram_index" => r.content_idx,
            "grad_l1" => gl1,
            "grad_nnz" => count(!iszero, r.output_kind == "content" ? meta.g_content : meta.g_pos),
            "backward_seconds" => meta.seconds,
            "n_instructions_differentiated" => meta.n_instr,
            "metrics" => Dict{String,Any}(
                "pearson_corr" => r.pearson, "spearman_corr" => r.spearman,
                "precision_at_k" => r.precision_at_k, "topk" => r.topk,
                "deletion_auc" => SA._json_num(r.deletion_auc),
                "insertion_auc" => SA._json_num(r.insertion_auc)),
            "triad" => SA.triad_extra_dict(r.pearson, r.attr, r.odelta; topk = r.topk, seed = r.seed),
            "harness_positive_control" => Dict{String,Any}(
                "pearson_corr" => r.oracle_self_pearson,
                "precision_at_k" => r.oracle_self_precision_at_k,
                "deletion_auc" => SA._json_num(r.oracle_self_deletion_auc),
                "insertion_auc" => SA._json_num(r.oracle_self_insertion_auc)),
            "oracle_column_degenerate" => r.degenerate,
            "cause_names" => r.cause_names,
            "attr_per_cause" => Dict(r.cause_names[i] => r.attr[i] for i in eachindex(r.cause_names)),
            "oracle_abs_delta_per_cause" => Dict(r.cause_names[i] => r.odelta[i] for i in eachindex(r.cause_names)),
            "testbed" => Dict{String,Any}(
                "state_kind" => "seeded_random_action_gameplay",
                "prefix" => meta.prefix, "horizon" => meta.horizon, "seed" => meta.seed,
                "cell" => collect(meta.cell),
                "position_byte_ram_index" => meta.geom === nothing ? -1 : meta.geom[1],
                "cause_density_above_floor" => meta.cause_density,
                "cause_density_accepted" => meta.accepted)))
    open(jp, "w") do io; JSON.print(io, rec, 2); end
    SA.write_npz(np, Dict(
        "attr_per_cause" => r.attr, "oracle_abs_delta" => r.odelta,
        "attr_over_ram" => r.attr_full,
        "deletion_curve" => r.del_curve, "insertion_curve" => r.ins_curve,
        "scalars" => Float64[r.pearson, r.spearman, r.precision_at_k,
                             r.deletion_auc, r.insertion_auc, gl1]))
    return jp
end

function merge_summaries(games)
    results = Any[]; na = Any[]; parts = String[]
    for g in games
        p = joinpath(OUT_DIR, "e2e_summary_$g.json")
        if !isfile(p)
            push!(na, Dict{String,Any}("game" => g, "status" => "n/a", "reason" => "no per-game summary"))
            continue
        end
        d = JSON.parsefile(p); push!(parts, basename(p))
        append!(results, d["results"]); append!(na, d["na_games"])
    end
    sp = joinpath(OUT_DIR, "e2e_core_summary.json")
    open(sp, "w") do io
        JSON.print(io, Dict{String,Any}("paper" => "P2", "phase" => "phaseB_attribution",
            "method" => "e2e_gradient (saliency + Grad×Input)", "games" => games,
            "merged_from" => parts, "commit" => SA._git_commit(),
            "timestamp" => string(round(Int, time())),
            "n_games_scored" => length(games) - length(na), "na_games" => na,
            "results" => results), 2)
    end
    println("[e2e] merged $(length(parts)) per-game summaries -> $sp")
    return 0
end

function main(args = ARGS)
    games = CORE_GAMES; topk = 3; seed = 0; selftest_only = false
    i = 1
    while i <= length(args)
        a = args[i]
        if     a == "--games";    games = SA.xai_resolve_games(args[i+1], CORE_GAMES); i += 2
        elseif a == "--game";     games = [args[i+1]]; i += 2
        elseif a == "--topk";     topk = parse(Int, args[i+1]); i += 2
        elseif a == "--selftest"; selftest_only = true; i += 1
        elseif a == "--merge";    return merge_summaries(games)
        else; i += 1
        end
    end
    println("[e2e] TRUE end-to-end gradient arm vs the oracle — games=$games topk=$topk " *
            "prefix=$(SA.ST_PREFIX) horizon=$(SA.ST_HORIZON) seed=$(SA.ST_SEED)")
    summary = Dict{String,Any}[]; na = Dict{String,Any}[]
    for game in games
        println("\n[e2e] ===== $game =====")
        try
            outs, meta = compute_game(game; topk = topk, seed = seed)
            for r in outs
                r.output_kind == "content" && r.degenerate &&
                    error("content oracle column degenerate for $game")
                if r.output_kind == "content"
                    @assert r.oracle_self_pearson > 0.999 "harness positive control broken [$game]"
                end
                println("[e2e]   $(rpad(r.method, 15)) $(rpad(r.output_kind, 9)) " *
                    "corr=$(round(r.pearson, digits=3)) spear=$(round(r.spearman, digits=3)) " *
                    "p@$topk=$(round(r.precision_at_k, digits=3)) " *
                    "del=$(round(r.deletion_auc, digits=3)) ins=$(round(r.insertion_auc, digits=3))")
                selftest_only || println("[e2e]   wrote ", write_record(r, meta))
                push!(summary, Dict{String,Any}("game" => game, "method" => r.method,
                    "output_kind" => r.output_kind, "output" => r.output,
                    "content_ram_index" => r.content_idx,
                    "pearson" => r.pearson, "spearman" => r.spearman,
                    "precision_at_k" => r.precision_at_k,
                    "deletion_auc" => SA._json_num(r.deletion_auc),
                    "insertion_auc" => SA._json_num(r.insertion_auc),
                    "grad_l1" => Float64(sum(abs, r.attr_full)),
                    "backward_seconds" => meta.seconds))
            end
        catch e
            msg = first(split(sprint(showerror, e), '\n'))
            println("[e2e] !! $game SKIPPED (n/a): $msg")
            push!(na, Dict{String,Any}("game" => game, "status" => "n/a", "reason" => msg))
        end
    end
    selftest_only && return 0
    # one process per game is the practical way to run this arm (each game is ~1 h
    # of reverse mode); single-game runs write a per-game summary, `--merge`
    # folds them into e2e_core_summary.json.
    sp = joinpath(OUT_DIR, length(games) == 1 ? "e2e_summary_$(only(games)).json" :
                                                 "e2e_core_summary.json")
    open(sp, "w") do io
        JSON.print(io, Dict{String,Any}("paper" => "P2", "phase" => "phaseB_attribution",
            "method" => "e2e_gradient (saliency + Grad×Input)", "games" => games,
            "topk" => topk, "seed" => seed, "commit" => SA._git_commit(),
            "timestamp" => string(round(Int, time())),
            "n_games_scored" => length(games) - length(na), "na_games" => na,
            "results" => summary), 2)
    end
    println("\n[e2e] wrote summary $sp")
    return 0
end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    E2EGradient.main()
end
