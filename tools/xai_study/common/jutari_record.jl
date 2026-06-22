# jutari_record.jl — the jutari state/trajectory recorder (P2-E0-2j), the JULIA
# counterpart of the (jaxtari) E0-2 recorder. It deterministically replays N
# frames of a real ROM via `jutari_oracle.jl` and stacks the per-frame state into
# a (T, n) array — by default the 128-byte RIOT RAM tape, optionally the 64-byte
# TIA register file and/or the flattened cropped framebuffer.
#
# This is the shared trajectory substrate the downstream pilots import:
#   * Phase-A tuning curves / dim-reduction (E3-0) consume the RAM tape;
#   * the Phase-C SAE activation capture (E5-0) consumes per-frame state vectors.
#
# It is a THIN wrapper: all the load/boot/replay/snapshot machinery already lives
# in `JutariOracle` (the verified Paper-1 bit-exact path — 64/64 games). We only
# add the per-frame stacking + a self-describing artifact (`.npz` + a JSON
# sidecar) at the SPEC §R out/ path, using the oracle's dependency-free NumPy
# writer so no package is added to the shared jutari env.
#
# No JuTari/jaxtari/xitari core is modified — pure tooling under tools/xai_study/.
#
# Run (default = Pong, 60 frames, RAM tape):
#   julia --project=/Users/maier/Documents/code/UnderstandingVCS/jutari \
#         tools/xai_study/common/jutari_record.jl
# or with options:
#   julia --project=.../jutari tools/xai_study/common/jutari_record.jl \
#         --game pong --frames 60 --fields ram,tia

module JutariRecord

include(joinpath(@__DIR__, "jutari_oracle.jl"))
using .JutariOracle
using .JutariOracle: RAM_SIZE, load_pong_env, snapshot, write_npz
using .JutariOracle.JuTari.Env: env_step!, get_ram, get_screen

export Trajectory, record_trajectory, write_trajectory, FIELDS

const TIA_SIZE = 64

# The fields a trajectory may stack, in their canonical column order. Each maps a
# live env to its flat per-frame UInt8 vector (so the on-disk tape is byte-exact
# to the emulator's observable state, no float rounding).
const FIELDS = ("ram", "tia", "screen")

# RIOT RAM: the 128-byte tape (the default — the variable the agent reads).
_field_ram(env) = UInt8.(collect(get_ram(env)))
# TIA register file: the 64-byte write-register snapshot (COLUPx, GRPx, ...).
_field_tia(env) = copy(env.console.bus.tia.registers)
# Cropped framebuffer (210×160 palette indices), flattened ROW-MAJOR so a numpy
# reshape to (rows, cols) round-trips the screen exactly.
function _field_screen(env)
    scr = get_screen(env)                 # (rows, cols) Julia (column-major)
    return vec(permutedims(scr, (2, 1)))  # row-major flatten
end

_field_fn(name::AbstractString) =
    name == "ram"    ? _field_ram    :
    name == "tia"    ? _field_tia    :
    name == "screen" ? _field_screen :
    error("unknown field $(name) (choose from $(FIELDS))")

_field_width(name::AbstractString, env) =
    name == "ram"    ? RAM_SIZE :
    name == "tia"    ? TIA_SIZE :
    name == "screen" ? length(get_screen(env)) :
    error("unknown field $(name)")

"""
    Trajectory

A recorded state trajectory. `tape` is the `(T, n)` UInt8 matrix (row t = the
flattened state after the t-th post-boot action step, for t in 1:T); `frame` is
the matching `1:T` index vector; `fields` lists the stacked field names in column
order; `widths` their per-field column counts (so `tape` can be sliced back into
fields); `game` / `actions` record how it was produced.
"""
struct Trajectory
    game::String
    frames::Int
    fields::Vector{String}
    widths::Vector{Int}
    tape::Matrix{UInt8}
    frame::Vector{Int}
    actions::Vector{Int}
end

"""
    record_trajectory(game; frames=60, fields=["ram"], actions=nothing) -> Trajectory

Deterministically replay `frames` post-boot action steps of `game` (xitari-parity
boot via `JutariOracle.load_pong_env`) and stack the requested per-frame `fields`
into a `(frames, n)` UInt8 matrix. The default `actions` is the all-NOOP trace
(action 0), matching the oracle's deterministic baseline; pass a vector of length
`>= frames` to record a custom input trace.

`fields` is any subset/order of `$(FIELDS)`; the default `["ram"]` is the 128-byte
RAM tape. The concatenated per-frame vector is laid out field-by-field in the
given order (column widths recorded in the returned `Trajectory`).
"""
function record_trajectory(game::AbstractString;
                           frames::Integer = 60,
                           fields = ["ram"],
                           actions::Union{Nothing,AbstractVector{<:Integer}} = nothing)
    frames = Int(frames)
    @assert frames >= 1 "frames must be >= 1, got $frames"
    flds = String[lowercase(string(f)) for f in fields]
    @assert !isempty(flds) "fields must be non-empty"
    for f in flds
        f in FIELDS || error("unknown field $(f) (choose from $(FIELDS))")
    end
    acts = actions === nothing ? fill(0, frames) : Int.(collect(actions))
    @assert length(acts) >= frames "actions has length $(length(acts)) < frames=$frames"

    env = load_pong_env(; game = game)
    fns = [_field_fn(f) for f in flds]
    widths = [_field_width(f, env) for f in flds]
    n = sum(widths)

    tape = Matrix{UInt8}(undef, frames, n)
    frame_idx = Vector{Int}(undef, frames)
    for t in 1:frames
        env_step!(env, acts[t])
        col = 1
        for fn in fns
            v = fn(env)
            tape[t, col:col + length(v) - 1] = v
            col += length(v)
        end
        frame_idx[t] = t
    end

    return Trajectory(string(game), frames, flds, widths, tape, frame_idx, acts)
end

# --- §R artifact (npz + JSON sidecar) --------------------------------------
const OUT_DIR = normpath(joinpath(@__DIR__, "out"))

_git_commit() = try
    strip(read(`git -C $(@__DIR__) rev-parse --short HEAD`, String))
catch
    "unknown"
end

# A tiny dependency-free JSON serializer for the flat sidecar dict (the recorder
# must add no package to the shared jutari env). Handles the value types we emit:
# String, Integer, AbstractFloat, Bool, Nothing, and Vector{<:Union{String,Int}}.
_json(s::AbstractString) = '"' * replace(replace(string(s), "\\" => "\\\\"), "\"" => "\\\"") * '"'
_json(b::Bool)           = b ? "true" : "false"
_json(::Nothing)         = "null"
_json(x::Integer)        = string(x)
_json(x::AbstractFloat)  = isfinite(x) ? string(x) : "null"
_json(v::AbstractVector) = "[" * join((_json(e) for e in v), ", ") * "]"
function _json(d::AbstractDict)
    parts = ["  " * _json(string(k)) * ": " * _json_inline(v) for (k, v) in d]
    return "{\n" * join(parts, ",\n") * "\n}\n"
end
# inline (no leading indent) for nested values
_json_inline(v) = _json(v)

"""
    write_trajectory(traj; out_dir=OUT_DIR) -> (npz_path, json_path)

Persist a `Trajectory` to the SPEC §R sibling-array layout:
`out/traj_<game>.npz` (numpy-loadable: keys `tape`, `frame`, `widths`) + a tiny
`out/traj_<game>.json` sidecar (`game, frames, fields, shape, widths, ...`).
"""
function write_trajectory(traj::Trajectory; out_dir = OUT_DIR)
    isdir(out_dir) || mkpath(out_dir)
    stem = "traj_$(traj.game)"
    npz_path = joinpath(out_dir, stem * ".npz")
    json_path = joinpath(out_dir, stem * ".json")

    write_npz(npz_path, Dict(
        "tape"   => traj.tape,                  # (T, n) UInt8
        "frame"  => Int64.(traj.frame),         # (T,) 1-based step index
        "widths" => Int64.(traj.widths),        # (n_fields,) per-field column widths
    ))

    rec = Dict{String,Any}(
        "paper"         => "P2",
        "phase"         => "common",
        "method"        => "trajectory_recorder",
        "game"          => traj.game,
        "frames"        => traj.frames,
        "fields"        => traj.fields,
        "widths"        => traj.widths,
        "shape"         => collect(size(traj.tape)),
        "dtype"         => "uint8",
        "target_output" => "state_trajectory",
        "metric_name"   => "n_frames",
        "value"         => traj.frames,
        "n"             => traj.frames,
        "seed"          => 0,
        "where"         => "local",
        "commit"        => _git_commit(),
        "oracle_ref"    => "jutari_oracle@$(traj.game) (bit-exact replay)",
        "timestamp"     => string(round(Int, time())),
        "arrays"        => basename(npz_path),
        "substrate"     => "jutari (Julia, HARD) — real-ROM bit-exact path",
    )
    open(json_path, "w") do io
        write(io, _json(rec))
    end
    return npz_path, json_path
end

# --- CLI -------------------------------------------------------------------
function _parse_args(args)
    game = "pong"; frames = 60; fields = ["ram"]
    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--game";   game = args[i + 1]; i += 2
        elseif a == "--frames"; frames = parse(Int, args[i + 1]); i += 2
        elseif a == "--fields"; fields = String.(split(args[i + 1], ",")); i += 2
        else; error("unknown arg $(a)"); end
    end
    return game, frames, fields
end

function main(args = ARGS)
    game, frames, fields = _parse_args(args)
    fieldstr = join(fields, ",")
    println("[jutari_record] recording game=$game frames=$frames fields=$fieldstr")
    traj = record_trajectory(game; frames = frames, fields = fields)
    npz_path, json_path = write_trajectory(traj)
    println("[jutari_record] tape shape = $(size(traj.tape))  (T, n)")
    println("[jutari_record] wrote $npz_path")
    println("[jutari_record] wrote $json_path")
    return traj
end

end # module

# Run as a script (not when `include`d by the test)
if abspath(PROGRAM_FILE) == @__FILE__
    JutariRecord.main()
end
