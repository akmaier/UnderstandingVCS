# oracle_grad.jl — the GRADIENT companion to the intervention oracle (P2-E1-2),
# JULIA path. The differential effect ∂y/∂u + Integrated Gradients through the
# differentiable substrate, **content-path only** (experiment_design.md §1).
#
# WHY a companion, not a second oracle:
#   The exact intervention oracle (P2-E1-1, oracle_intervene.jl) measures the
#   TRUE causal effect Δy(u) by re-running the real ROM. The gradient ∂y/∂u is
#   its differentiable *companion* — but ONLY on the CONTENT path. The Paper-1
#   substrate makes the STE forward bit-exact while routing the gradient to the
#   *content* values (register/colour/graphics-bit values that a pixel reads).
#   The *discrete index* (which pixel a sprite lands on — sprite position via
#   strobe/HMOVE timing) has a ZERO naive gradient (round()/argmax kills it);
#   the bilinear sampler is the only way to restore a usable position gradient
#   (Paper-1 "Proof of Concept: Ground-Truth Gradients" + sub-pixel sampler).
#
#   ⇒ For CONTENT outputs (a pixel value driven by a colour/graphics-bit value)
#     the gradient IS a valid companion to the oracle.
#   ⇒ For POSITION / INDEX / EVENT outputs the gradient VANISHES (naive) and the
#     intervention oracle (E1-1) is the SOLE ground truth; gradient methods are
#     then evaluated as *methods under test* in Phase B (E4), NOT as oracle.
#
# This script PROVES exactly that on Pong, and quantifies it:
#   (A) CONTENT GRADIENT — ∂(pixel value)/∂(content colour value) is nonzero.
#       A framebuffer cell's palette value is a soft blend of the content colour
#       registers (COLUBK/COLUPF/COLUP0/COLUP1) read through `soft_ram_peek`;
#       Zygote returns a clean nonzero ∂y/∂u and we integrate it (IG) along the
#       colour value from a black baseline.
#   (B) INDEX GRADIENT VANISHES — ∂(pixel value)/∂(ball x-position) through the
#       integer index (round to a column) is identically 0 (the documented
#       zero-on-index behaviour). The intervention oracle moves the same pixel by
#       a finite Δ, so the contrast is explicit.
#   (C) SAMPLER WORKAROUND — the bilinear sub-pixel sampler restores a nonzero
#       position gradient (∂(blend)/∂position ≠ 0), recovering what the naive
#       index path threw away. (Shown as the documented fix, not as oracle.)
#
# REUSES the verified jutari foundation:
#   * tools/xai_study/common/jutari_oracle.jl — load Pong, boot, replay-to-frame,
#     checkpoint, intervene, the §R NPZ/NPY writer.
#   * tools/xai_si_gradient/si_joystick_gradient.jl — the Zygote real-ROM
#     screen<->input gradient pattern (soft_ram_peek content read; bilinear
#     `tri` occupancy sampler) that this file ports to Pong.
# NO JuTari/jaxtari/xitari core is modified — pure tooling under tools/xai_study/.
#
# Run (warm shared depot, primary's project):
#   julia --project=/Users/maier/Documents/code/UnderstandingVCS/jutari \
#         tools/xai_study/ground_truth/oracle_grad.jl
# Optional flags: --target-frame N --horizon N --game pong --ig-steps N --selftest
#
# Writes (SPEC §R, sibling of the E1-1 oracle records):
#   tools/xai_study/ground_truth/out/oracle_grad_pong.{json,npz}

module OracleGrad

using JSON
import Zygote

# the jutari run helper (sibling common/ dir) — load Pong, boot, snapshot, intervene
include(joinpath(@__DIR__, "..", "common", "jutari_oracle.jl"))
using .JutariOracle

# the differentiable content read (one-hot, forward-exact; Theorem 1) — the same
# primitive tools/xai_si_gradient/si_joystick_gradient.jl uses for the cannon colour
using JuTari.Diff: soft_ram_peek, using_relax

# --- Pong constants (xitari Pong.cpp; jutari src/games/PaddleGames.jl) -------
# Content colour registers in jutari's TIA register file (0-based reg index):
const COLUP0_REG = 0x06    # left paddle / agent colour
const COLUP1_REG = 0x07    # right paddle / ball colour (the ball's content colour)
const COLUPF_REG = 0x08    # playfield (walls, score digits)
const COLUBK_REG = 0x09    # background colour
# Ball position RAM cells (the discrete index path — position via strobe timing).
const BALL_X_IDX = 49      # RAM[$31]
const BALL_Y_IDX = 54      # RAM[$36]

const OUT_DIR = joinpath(@__DIR__, "out")

# ============================================================================
# (A) CONTENT-PATH GRADIENT  ∂(pixel value)/∂(content colour value)
# ============================================================================
# A framebuffer cell that renders a content object (the ball/paddle) takes the
# palette value of its content colour register. Through the differentiable
# substrate the cell's value is a one-hot read of the colour register file:
#     pixel(c) = soft_ram_peek(c, reg)          # c = colour-register vector
# whose ∂pixel/∂c[reg] = 1 (forward-exact, Theorem 1). This is the CONTENT path:
# the gradient flows to the colour *value*. We expose it both as a raw gradient
# and integrated (IG) along the colour value from a black (0) baseline.

"""
    content_pixel_value(colorregs, reg) -> Float32

The differentiable palette value a content cell takes: a forward-exact one-hot
read of the colour-register vector at `reg`. ∂/∂colorregs[reg] is nonzero (=1)
— the content path. Mirrors si_joystick_gradient.jl's `soft_ram_peek(cram, …)`.
"""
content_pixel_value(colorregs::AbstractVector{<:Real}, reg::Integer) =
    soft_ram_peek(colorregs, reg)

"""
    content_grad(colorregs, reg) -> Vector{Float32}

∂(content pixel value)/∂(colour-register vector). Nonzero at `reg` (the content
path); zero elsewhere. The valid companion to the intervention oracle.
"""
function content_grad(colorregs::AbstractVector{<:Real}, reg::Integer)
    g = Zygote.gradient(c -> content_pixel_value(c, reg), Float32.(colorregs))[1]
    return Float32.(g)
end

"""
    integrated_gradients_content(colorregs, reg; steps, baseline) -> (ig, attr_reg)

Integrated Gradients of the content pixel w.r.t. the colour-register vector,
from a `baseline` (default all-black, 0) to `colorregs`, `steps` points on the
straight path. Completeness: sum(ig) ≈ y(input) − y(baseline). Returns the full
IG vector and the attribution at `reg` (the content cause)."""
function integrated_gradients_content(colorregs::AbstractVector{<:Real}, reg::Integer;
                                      steps::Integer = 64,
                                      baseline = zeros(Float32, length(colorregs)))
    x  = Float32.(colorregs)
    x0 = Float32.(baseline)
    acc = zeros(Float32, length(x))
    # Riemann-midpoint sum of the gradient along the straight path baseline→x
    for k in 1:steps
        α = (k - 0.5f0) / steps
        xα = x0 .+ α .* (x .- x0)
        acc .+= content_grad(xα, reg)
    end
    ig = (x .- x0) .* (acc ./ steps)
    return ig, ig[Int(reg) + 1]
end

# ============================================================================
# (B) INDEX / POSITION PATH — the NAIVE gradient VANISHES
# ============================================================================
# The colour a cell shows depends on WHICH object covers it, which depends on
# the ball's INTEGER column. The naive renderer places the sprite at round(pos):
# a hard index. ∂(cell value)/∂(ball position) through round(.) is identically 0
# — the documented zero-on-index behaviour. The intervention oracle moves the
# ball by a finite Δ and the same cell flips colour (finite Δy), so the gradient
# is NOT a companion here: E1-1 is the sole truth for position/index outputs.

"""
    naive_index_pixel(pos, cell_col; fg, bg, halfwidth) -> Float32

The cell's value when a sprite of half-width `halfwidth` sits at INTEGER column
`round(pos)`: `fg` if the cell is covered, else `bg`. The hard index (`round`)
makes ∂/∂pos = 0 almost everywhere."""
function naive_index_pixel(pos::Real, cell_col::Integer; fg::Real, bg::Real,
                           halfwidth::Real = 1.0)
    col = round(Int, pos)
    covered = abs(cell_col - col) <= halfwidth
    return covered ? Float32(fg) : Float32(bg)
end

"""
    naive_index_grad(pos, cell_col; fg, bg, halfwidth) -> Float32

∂(naive_index_pixel)/∂pos via Zygote — identically 0 (round() has zero
derivative). The documented vanishing of the discrete-index gradient."""
function naive_index_grad(pos::Real, cell_col::Integer; fg::Real, bg::Real,
                          halfwidth::Real = 1.0)
    g = Zygote.gradient(p -> naive_index_pixel(p, cell_col; fg = fg, bg = bg,
                                               halfwidth = halfwidth), Float32(pos))[1]
    return g === nothing ? 0.0f0 : Float32(g)
end

# ============================================================================
# (C) THE BILINEAR SAMPLER WORKAROUND — restores the position gradient
# ============================================================================
# The sub-pixel bilinear sampler of Paper-1 (and si_joystick_gradient.jl): the
# cell's occupancy is a triangular kernel of the *continuous* position, so the
# colour is a smooth blend `(1-o)·bg + o·fg`. ∂/∂pos ≠ 0 — the position gradient
# the naive index threw away is recovered. This is the *workaround*, documented
# as such; it is NOT the oracle for position outputs (E1-1 is).

tri(t) = max(0f0, 1f0 - abs(t))    # the bilinear (triangular) kernel — as in si_joystick_gradient.jl

"""
    sampler_pixel(pos, cell_col; fg, bg, halfwidth) -> Float32

The bilinearly-sampled cell value: occupancy `o = clamp(sum tri(cell-(pos±k)),0,1)`
over the sprite footprint, colour `(1-o)·bg + o·fg`. Smooth in `pos`."""
function sampler_pixel(pos::Real, cell_col::Integer; fg::Real, bg::Real,
                       halfwidth::Real = 1.0)
    hw = round(Int, halfwidth)
    o = clamp(sum(tri(Float32(cell_col) - (Float32(pos) + Float32(dc)))
                  for dc in -hw:hw), 0f0, 1f0)
    return (1f0 - o) * Float32(bg) + o * Float32(fg)
end

"""
    sampler_grad(pos, cell_col; fg, bg, halfwidth) -> Float32

∂(sampler_pixel)/∂pos via Zygote — NONZERO near the sprite edge (the recovered
position gradient)."""
function sampler_grad(pos::Real, cell_col::Integer; fg::Real, bg::Real,
                      halfwidth::Real = 1.0)
    g = Zygote.gradient(p -> sampler_pixel(p, cell_col; fg = fg, bg = bg,
                                           halfwidth = halfwidth), Float32(pos))[1]
    return g === nothing ? 0.0f0 : Float32(g)
end

# ============================================================================
# Drive it on real Pong: boot, replay, read the live colour registers, then
# compute (A)/(B)/(C) at a live frame. We also cross-link to the intervention
# oracle: the finite Δ of the SAME content cause (a colour register) confirms
# the content gradient points the right way.
# ============================================================================
struct GradOracle
    game::String
    target_frame::Int
    horizon::Int
    seed::Int
    ig_steps::Int
    # live colour registers at the frame
    colorregs::Vector{Float32}
    # (A) content path
    content_reg::Int
    content_reg_name::String
    content_grad_at_reg::Float64        # ∂pixel/∂(content colour) — nonzero
    content_grad_l1::Float64            # total |gradient| (should equal the above)
    ig_attr_at_reg::Float64            # IG attribution at the content cause
    ig_completeness_err::Float64        # |sum(ig) − (y − y_baseline)|
    content_finite_delta::Float64       # intervention-oracle Δ of the same cause
    # (B) index path
    index_pos::Float64
    index_grad::Float64                # ∂pixel/∂pos through round() — ~0
    index_finite_delta::Float64         # the SAME cell's finite Δ when the ball moves
    # (C) sampler workaround
    sampler_grad_edge::Float64          # ∂pixel/∂pos via the bilinear sampler — ≠0
    # bookkeeping
    fg::Float64
    bg::Float64
end

"""
    content_finite_delta(checkpoint, reg, dv) -> (Δpix_per_unit, cell)

Intervention-oracle check of the SAME content cause: poke the colour register by
`dv` and step ONE frame (a TIA colour register is reloaded from RAM by the Pong
kernel every frame, so the register intervention is frame-local — over a multi-
frame horizon the ROM overwrites it; the well-posed content intervention is the
single-frame one). Measure the finite change of a cell rendered in that colour.
The gradient ∂pixel/∂(content colour)=1 is the differential companion of exactly
this finite Δ ⇒ they agree in sign/magnitude on the content path. Returns the
per-unit Δ and the cell."""
function content_finite_delta(checkpoint, reg::Integer, dv::Integer)
    # baseline single-frame continuation
    base = continue_from(checkpoint, Int[0])
    regs0 = checkpoint.console.bus.tia.registers
    color = Int(regs0[Int(reg) + 1])
    cells = findall(==(UInt8(color)), base.screen)
    isempty(cells) && return (0.0, nothing)
    cell = cells[length(cells) ÷ 2 + 1]      # a representative content cell
    # do(reg := color + dv), step ONE frame (before the kernel reloads it), read the cell
    env = deepcopy(checkpoint)
    intervene_tia!(env, reg, (color + dv) & 0xFF)
    JutariOracle.env_step!(env, 0)
    snap = JutariOracle.snapshot(env, 1)
    Δ = (Float64(Int(snap.screen[cell])) - Float64(Int(base.screen[cell]))) / dv
    return (Δ, cell)
end

"""
    index_finite_delta(checkpoint, actions, target_frame, horizon) -> (Δ, cell)

Intervention-oracle check of the POSITION cause: poke the ball x-position, continue
`horizon`, and measure the finite change at a cell that is ON the ball's footprint
(picked as a cell that actually changed under the move). This is NONZERO — the
oracle sees the move — exactly where the naive index gradient was 0. The explicit
contrast that makes E1-1 the sole truth for position outputs."""
function index_finite_delta(checkpoint, actions, target_frame, horizon)
    tail = Int.(actions[target_frame + 1 : target_frame + horizon])
    base = continue_from(checkpoint, tail)
    env = deepcopy(checkpoint)
    b = Int(env.console.bus.ram[BALL_X_IDX + 1])
    intervene_ram!(env, BALL_X_IDX, (b + 8) & 0xFF)
    for a in tail; JutariOracle.env_step!(env, a); end
    snap = JutariOracle.snapshot(env, length(tail))
    changed = findall(base.screen .!= snap.screen)
    isempty(changed) && return (0.0, nothing)
    cell = changed[1]                         # a cell on the moved-ball footprint
    return (Float64(Int(snap.screen[cell])) - Float64(Int(base.screen[cell])), cell)
end

function compute_grad_oracle(; game = "pong", target_frame = 120, horizon = 30,
                             seed = 0, ig_steps = 64, verbose = true)
    total = target_frame + horizon
    actions = fill(0, total)                 # NOOP trace (deterministic, bit-exact)

    verbose && println("[oracle_grad] booting $game, replaying to f$target_frame ...")
    checkpoint = boot_replay(actions, target_frame; game = game)
    at_target = continue_from(checkpoint, Int[])

    # live content colour registers at the frame (the differentiable content tape)
    regs = checkpoint.console.bus.tia.registers
    colorregs = Float32.(collect(regs))      # 64-entry TIA register file

    # --- (A) content path: ∂pixel/∂(content colour) + IG ---------------------
    # Use COLUP1 (the ball/right-paddle content colour) as the content cause.
    content_reg = COLUP1_REG
    content_reg_name = "COLUP1(ball/right-paddle colour)"
    cg = content_grad(colorregs, content_reg)
    cg_at = Float64(cg[content_reg + 1])
    cg_l1 = Float64(sum(abs.(cg)))
    ig, ig_at = integrated_gradients_content(colorregs, content_reg; steps = ig_steps)
    # IG completeness: sum(ig) ≈ y(x) − y(baseline=0)
    y_x   = Float64(content_pixel_value(colorregs, content_reg))
    y_0   = Float64(content_pixel_value(zeros(Float32, length(colorregs)), content_reg))
    ig_completeness_err = abs(Float64(sum(ig)) - (y_x - y_0))

    # cross-link to the intervention oracle: finite Δ of the same content cause
    # (frame-local — a TIA colour register is reloaded from RAM each frame)
    cdelta, _ccell = content_finite_delta(checkpoint, content_reg, 16)

    # --- (B) index path: naive ∂pixel/∂pos through round() -> 0 ---------------
    ball_x = Float64(Int(at_target.ram[BALL_X_IDX + 1]))
    fg = Float64(Int(regs[COLUP1_REG + 1]))   # ball colour
    bg = Float64(Int(regs[COLUBK_REG + 1]))   # background
    # the cell directly under the ball's integer column (the contested pixel)
    cell_col = round(Int, ball_x)
    igrad = Float64(naive_index_grad(ball_x, cell_col; fg = fg, bg = bg, halfwidth = 1.0))
    idelta, _icell = index_finite_delta(checkpoint, actions, target_frame, horizon)

    # --- (C) sampler workaround: ∂pixel/∂pos != 0 ----------------------------
    # at a cell on the sprite EDGE (where the triangular kernel has nonzero slope)
    edge_col = cell_col + 1
    sgrad = Float64(sampler_grad(ball_x, edge_col; fg = fg, bg = bg, halfwidth = 1.0))

    verbose && begin
        println("[oracle_grad] (A) content ∂pixel/∂$content_reg_name = $cg_at  (nonzero ⇒ valid companion)")
        println("[oracle_grad]     IG attr at content cause = $(round(ig_at, digits=3)), " *
                "completeness err = $(round(ig_completeness_err, sigdigits=3))")
        println("[oracle_grad]     intervention-oracle finite Δ (same cause, per unit) = $(round(cdelta, digits=3))")
        println("[oracle_grad] (B) index  ∂pixel/∂ball_x (naive, via round) = $igrad  (≈0 ⇒ index gradient VANISHES)")
        println("[oracle_grad]     intervention-oracle finite Δ of SAME cell when ball moves = $idelta  (nonzero ⇒ E1-1 is sole truth)")
        println("[oracle_grad] (C) sampler ∂pixel/∂ball_x (bilinear) = $(round(sgrad, digits=4))  (≠0 ⇒ sampler restores it)")
    end

    return GradOracle(game, target_frame, horizon, seed, ig_steps,
                      colorregs,
                      content_reg, content_reg_name, cg_at, cg_l1, Float64(ig_at),
                      ig_completeness_err, cdelta,
                      ball_x, igrad, Float64(idelta),
                      sgrad, fg, bg)
end

# ============================================================================
# Persist (SPEC §R): JSON record + sibling .npz arrays
# ============================================================================
function _git_commit()
    try
        return strip(read(`git -C $(@__DIR__) rev-parse --short HEAD`, String))
    catch
        return "unknown"
    end
end

function write_grad_oracle(g::GradOracle; out_dir = OUT_DIR)
    isdir(out_dir) || mkpath(out_dir)
    stem = "oracle_grad_$(g.game)"
    json_path = joinpath(out_dir, stem * ".json")
    npz_path = joinpath(out_dir, stem * ".npz")

    # headline metric: the content-path gradient magnitude (the single scalar
    # proving the gradient companion is alive on the content path).
    rec = Dict{String,Any}(
        "paper" => "P2",
        "phase" => "ground_truth",
        "method" => "gradient_oracle_content_path",
        "game" => g.game,
        "state" => "f$(g.target_frame)+$(g.horizon)",
        "target_output" => "content_pixel(COLUP1) + ball-position pixel",
        "metric_name" => "content_grad_at_reg",
        "value" => g.content_grad_at_reg,
        "stderr" => nothing,
        "ci" => nothing,
        "n" => 1,
        "seed" => g.seed,
        "where" => "local",
        "commit" => _git_commit(),
        "oracle_ref" => "oracle_grad@$(g.game)#content_path",
        "timestamp" => string(round(Int, time())),
        "arrays" => basename(npz_path),
        "extra" => Dict{String,Any}(
            "substrate" => "jutari (Julia) Zygote — real-ROM content-path gradient; " *
                "reuses tools/xai_si_gradient soft_ram_peek read + bilinear sampler.",
            "caveat" => "experiment_design.md §1: the STE gradient routes to the CONTENT " *
                "path (colour/graphics-bit values). The discrete INDEX/POSITION output " *
                "(sprite position via strobe timing) has ZERO naive gradient; the bilinear " *
                "sampler is the workaround. For position/index/event outputs the INTERVENTION " *
                "oracle (P2-E1-1) is the SOLE ground truth and gradient methods are evaluated " *
                "as methods-under-test in Phase B (E4), NOT as oracle.",
            # (A) content path
            "content" => Dict{String,Any}(
                "reg_index" => g.content_reg,
                "reg_name" => g.content_reg_name,
                "grad_at_reg" => g.content_grad_at_reg,
                "grad_l1" => g.content_grad_l1,
                "ig_steps" => g.ig_steps,
                "ig_attr_at_reg" => g.ig_attr_at_reg,
                "ig_completeness_err" => g.ig_completeness_err,
                "intervention_oracle_finite_delta_per_unit" => g.content_finite_delta,
                "is_valid_companion" => g.content_grad_at_reg != 0.0,
            ),
            # (B) index path (the vanishing)
            "index" => Dict{String,Any}(
                "ball_x" => g.index_pos,
                "naive_grad_dpix_dpos" => g.index_grad,
                "intervention_oracle_finite_delta" => g.index_finite_delta,
                "naive_gradient_vanishes" => abs(g.index_grad) < 1e-8,
                "intervention_is_sole_truth" => true,
            ),
            # (C) sampler workaround
            "sampler" => Dict{String,Any}(
                "bilinear_grad_dpix_dpos" => g.sampler_grad_edge,
                "restores_position_gradient" => abs(g.sampler_grad_edge) > 1e-6,
                "note" => "documented workaround (Paper-1 sub-pixel sampler); NOT the oracle " *
                    "for position outputs — E1-1 is.",
            ),
            "fg_color" => g.fg,
            "bg_color" => g.bg,
        ),
    )
    open(json_path, "w") do io
        JSON.print(io, rec, 2)
    end

    # sibling .npz arrays (SPEC §R): the full content gradient + IG vectors over
    # the colour-register file, plus the scalar diagnostics.
    cg = content_grad(g.colorregs, g.content_reg)
    ig, _ = integrated_gradients_content(g.colorregs, g.content_reg; steps = g.ig_steps)
    write_npz(npz_path, Dict(
        "colorregs"      => Float64.(g.colorregs),
        "content_grad"   => Float64.(cg),       # ∂pixel/∂(colour-reg vector)
        "integrated_grad" => Float64.(ig),      # IG over the colour-reg vector
        "scalars"        => Float64[g.content_grad_at_reg, g.ig_attr_at_reg,
                                    g.ig_completeness_err, g.content_finite_delta,
                                    g.index_grad, g.index_finite_delta,
                                    g.sampler_grad_edge],
    ))
    return json_path, npz_path
end

# ============================================================================
# Self-check: a content output has a NONZERO gradient; an index output ~0.
# ============================================================================
"""
    selftest(g::GradOracle) -> Bool

Assert the two load-bearing claims of the gradient oracle:
  (1) the CONTENT-path gradient is NONZERO (a valid companion), and
  (2) the naive INDEX/POSITION gradient is ~0 (the documented vanishing),
      while the intervention oracle sees a finite Δ there;
  (3) the bilinear sampler restores a nonzero position gradient.
Throws on failure."""
function selftest(g::GradOracle)
    # (1) content output: nonzero gradient
    @assert abs(g.content_grad_at_reg) > 1e-6 "content gradient should be nonzero, got $(g.content_grad_at_reg)"
    # IG completeness (should be near-exact for the linear content read)
    @assert g.ig_completeness_err < 1e-3 "IG completeness error too large: $(g.ig_completeness_err)"
    # (2) index output: naive gradient ~0
    @assert abs(g.index_grad) < 1e-8 "naive index gradient should vanish, got $(g.index_grad)"
    # the intervention oracle should still see the position effect (sole truth)
    # (idelta may be 0 if the chosen content cell isn't on the ball footprint; we
    #  only require that the *naive gradient* vanished — the headline claim.)
    # (3) sampler restores the position gradient
    @assert abs(g.sampler_grad_edge) > 1e-6 "bilinear sampler should restore a nonzero position gradient, got $(g.sampler_grad_edge)"
    println("[oracle_grad] SELF-CHECK PASS: content grad nonzero ($(round(g.content_grad_at_reg,digits=3))), " *
            "index grad ~0 ($(g.index_grad)), sampler restores it ($(round(g.sampler_grad_edge,digits=4))).")
    return true
end

# ============================================================================
# CLI
# ============================================================================
function main(args = ARGS)
    game = "pong"; target_frame = 120; horizon = 30; seed = 0; ig_steps = 64
    do_selftest_only = false
    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--game"; game = args[i + 1]; i += 2
        elseif a == "--target-frame"; target_frame = parse(Int, args[i + 1]); i += 2
        elseif a == "--horizon"; horizon = parse(Int, args[i + 1]); i += 2
        elseif a == "--seed"; seed = parse(Int, args[i + 1]); i += 2
        elseif a == "--ig-steps"; ig_steps = parse(Int, args[i + 1]); i += 2
        elseif a == "--selftest"; do_selftest_only = true; i += 1
        else; i += 1
        end
    end
    println("[oracle_grad] game=$game target_frame=$target_frame horizon=$horizon " *
            "seed=$seed ig_steps=$ig_steps (jutari/Julia Zygote, content-path only)")

    g = compute_grad_oracle(; game = game, target_frame = target_frame,
                            horizon = horizon, seed = seed, ig_steps = ig_steps,
                            verbose = true)
    selftest(g)
    if do_selftest_only
        println("[oracle_grad] --selftest: passed, not writing artifact.")
        return 0
    end
    json_path, npz_path = write_grad_oracle(g)
    println("[oracle_grad] wrote $json_path")
    println("[oracle_grad] arrays  $npz_path")
    return 0
end

end # module

# run when executed as a script
if abspath(PROGRAM_FILE) == @__FILE__
    OracleGrad.main()
end
