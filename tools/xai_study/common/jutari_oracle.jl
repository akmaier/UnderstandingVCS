# jutari_oracle.jl — the minimal jutari run helper for the P2 intervention
# oracle (P2-E1-1), the JULIA path (substrate pivot off the ~205× slower jaxtari
# eager harness). It does exactly what the oracle needs and nothing more:
#
#   * load a real ROM and reset it with the xitari-parity boot (60 NOOP + 4
#     RESET), using the game's RomSettings;
#   * deterministically replay an action trace to a target frame;
#   * snapshot RAM + screen (byte-exact copies);
#   * intervene on a candidate cause `u` — a RAM byte, a TIA register, or a
#     do(action) joystick/paddle input — and continue;
#   * a checkpoint = a `deepcopy` of the booted+replayed env, so the expensive
#     boot+to-target replay is paid ONCE for the whole sweep and every cause /
#     the baseline continue from a byte-identical copy of it;
#   * a tiny dependency-free NumPy `.npy`/`.npz` writer so the §R sibling-array
#     artifact is produced without adding a package to the shared jutari env.
#
# No JuTari/jaxtari/xitari core is modified — this is pure tooling under
# tools/xai_study/. The emulator is bit-exact under a fixed action trace
# (Paper-1 64/64), which is what makes every Δy a clean causal effect; the
# oracle asserts that re-run determinism before trusting any Δ.

module JutariOracle

using JuTari
using JuTari.Env: StellaEnvironment, env_reset!, env_step!, get_screen, get_ram

export Snapshot, load_pong_env, boot_replay, snapshot, continue_from,
       intervene_ram!, intervene_tia!, fresh_baseline_ram_screen,
       rom_path_for, settings_for, RAM_SIZE,
       write_npy, write_npz

const RAM_SIZE = 128

# --- ROM + settings resolution ---------------------------------------------
# The conformance ROMs live in the PRIMARY checkout (gitignored — used in place,
# never committed). tools/xai_si_gradient/* uses the same `xitari/roms/<game>.bin`
# layout; we resolve it relative to the repo root (parents of this file) and fall
# back to the primary worktree path if this is run from an isolated worktree whose
# xitari/roms is absent.
const _PRIMARY_REPO = "/Users/maier/Documents/code/UnderstandingVCS"

# HARNESS PARITY (CLAUDE.md rule #2): this is the LOWEST-level shared boot helper
# — every P2 oracle/Phase-B path (oracle_intervene.jl, common/gameplay_state.jl,
# and each phaseB runner via its own map) resolves ROMs+settings the same way.
# We therefore carry the FULL core-6 map HERE so `assert_bit_exact` (which routes
# through `load_pong_env` → `settings_for`/`rom_path_for`) boots ms_pacman/qbert
# under the SAME MsPacman/Qbert joystick settings the checkpoint uses, not the
# Generic fallback (which diverges bit-exactness). The ROM-basename alias
# (ms_pacman → mspacman.bin) MUST live here too, or the ROM lookup fails.
const ROM_BASENAME = Dict(
    "pong" => "pong", "breakout" => "breakout",
    "space_invaders" => "space_invaders", "seaquest" => "seaquest",
    "ms_pacman" => "mspacman", "qbert" => "qbert")

"""
    rom_path_for(game) -> String

Absolute path to the real ROM for `game` (e.g. "pong"). Applies the core-6
ROM-basename alias (ms_pacman → mspacman.bin) then searches the repo root that
contains this file first, then the known primary checkout.
"""
function rom_path_for(game::AbstractString)
    stem = get(ROM_BASENAME, lowercase(string(game)), lowercase(string(game)))
    here = normpath(joinpath(@__DIR__, "..", "..", ".."))   # repo root of this worktree
    for base in (here, _PRIMARY_REPO)
        p = joinpath(base, "xitari", "roms", stem * ".bin")
        isfile(p) && return p
    end
    error("ROM not found for game=$game (looked under $(here) and $(_PRIMARY_REPO))")
end

"""
    settings_for(game) -> RomSettings

The per-game RomSettings (so paddle/joystick routing, scoring, terminal logic
match xitari). Carries the full core-6 map — pong/breakout (paddles),
space_invaders (terminal), ms_pacman/qbert (joystick); seaquest has no
registered settings yet → Generic (boots fine; matches the screen scoreboard's
Generic fallback for seaquest). This map MUST agree with
common/gameplay_state.jl and every phaseB runner's local `settings_for`
(CLAUDE.md rule #2)."""
function settings_for(game::AbstractString)
    g = lowercase(string(game))
    g == "pong"           && return JuTari.PaddleGames.PongRomSettings()
    g == "breakout"       && return JuTari.PaddleGames.BreakoutRomSettings()
    g == "space_invaders" && return JuTari.SpaceInvadersRomSettings()
    g == "ms_pacman"      && return JuTari.JoystickGames.MsPacmanRomSettings()
    g == "qbert"          && return JuTari.JoystickGames.QbertRomSettings()
    return JuTari.GenericRomSettings()   # seaquest (no registered settings yet)
end

# --- env construction + boot ------------------------------------------------
"""
    load_pong_env(; game="pong") -> StellaEnvironment

A freshly-reset env with the xitari-parity boot (60 NOOP + 4 RESET). Booted but
NOT yet stepped into the action trace.
"""
function load_pong_env(; game::AbstractString = "pong")
    rom = read(rom_path_for(game))
    env = StellaEnvironment(rom, settings_for(game))
    env_reset!(env; boot_noop_steps = 60, boot_reset_steps = 4)
    return env
end

# --- snapshots --------------------------------------------------------------
"""
    Snapshot

A byte-exact copy of the emulator's observable output at a frame: `ram` (128 B)
and `screen` (the cropped framebuffer, palette indices). `frame` is the number of
post-boot action steps taken to reach it.
"""
struct Snapshot
    frame::Int
    ram::Vector{UInt8}
    screen::Matrix{UInt8}
end

"""
    snapshot(env, frame) -> Snapshot

Copy RAM + screen out of `env` (independent buffers; `get_screen` returns a view,
`get_ram` the live vector, so we `copy`/`Matrix` to freeze them)."""
snapshot(env::StellaEnvironment, frame::Integer) =
    Snapshot(Int(frame), copy(collect(get_ram(env))), Matrix{UInt8}(get_screen(env)))

# --- replay + checkpoint ----------------------------------------------------
"""
    boot_replay(actions, target_frame; game="pong") -> StellaEnvironment

Boot + deterministically step `actions[1:target_frame]`; return the env AT the
intervention frame. `deepcopy` the result to make a reusable checkpoint."""
function boot_replay(actions::AbstractVector{<:Integer}, target_frame::Integer;
                     game::AbstractString = "pong")
    env = load_pong_env(; game = game)
    for i in 1:target_frame
        env_step!(env, Int(actions[i]))
    end
    return env
end

"""
    continue_from(checkpoint, tail) -> Snapshot

`deepcopy` the checkpoint env, step the `tail` actions, snapshot. The deepcopy is
byte-identical to the checkpoint, so the continuation reproduces a fresh replay
exactly — the boot + to-target replay is never re-paid."""
function continue_from(checkpoint::StellaEnvironment, tail::AbstractVector{<:Integer})
    env = deepcopy(checkpoint)
    for a in tail
        env_step!(env, Int(a))
    end
    return snapshot(env, length(tail))
end

# --- interventions (do(u := v')) -------------------------------------------
"""
    intervene_ram!(env, ram_index, value)

Write `value & 0xFF` into RIOT RAM at 0-based `ram_index` (RAM[\$0D] → index 13).
The RAM vector is 1-indexed in Julia, so we hit `ram[ram_index + 1]`."""
function intervene_ram!(env::StellaEnvironment, ram_index::Integer, value::Integer)
    @assert 0 <= ram_index < RAM_SIZE "ram_index out of range: $ram_index"
    env.console.bus.ram[Int(ram_index) + 1] = UInt8(Int(value) & 0xFF)
    return env
end

"""
    intervene_tia!(env, reg_index, value)

Write `value & 0xFF` into the TIA register file at 0-based `reg_index` (e.g.
COLUP1=0x07, COLUBK=0x09). The register file is 64 B, 1-indexed in Julia."""
function intervene_tia!(env::StellaEnvironment, reg_index::Integer, value::Integer)
    regs = env.console.bus.tia.registers
    @assert 0 <= reg_index < length(regs) "tia reg out of range: $reg_index"
    regs[Int(reg_index) + 1] = UInt8(Int(value) & 0xFF)
    return env
end

# --- bit-exact baseline (the load-bearing guarantee) ------------------------
"""
    fresh_baseline_ram_screen(actions, total; game="pong") -> Snapshot

A FULL from-scratch replay (boot included) of `actions[1:total]`. Used by the
oracle's bit-exact assertion: two such fresh runs must be byte-identical."""
function fresh_baseline_ram_screen(actions::AbstractVector{<:Integer}, total::Integer;
                                   game::AbstractString = "pong")
    env = load_pong_env(; game = game)
    for i in 1:total
        env_step!(env, Int(actions[i]))
    end
    return snapshot(env, Int(total))
end

# ===========================================================================
# Dependency-free NumPy writers (.npy / .npz) — so the §R sibling array file is
# produced without adding a package to the shared jutari env. Format per the
# NumPy spec: a magic string + version + a header dict, then the raw buffer in
# little-endian C order. `.npz` is an *uncompressed* ZIP of `.npy` members
# (numpy.load reads it transparently).
# ===========================================================================

const _NPY_MAGIC = UInt8[0x93, UInt8('N'), UInt8('U'), UInt8('M'),
                         UInt8('P'), UInt8('Y')]

_npy_dtype(::Type{Float64}) = "<f8"
_npy_dtype(::Type{Float32}) = "<f4"
_npy_dtype(::Type{Int64})   = "<i8"
_npy_dtype(::Type{Int32})   = "<i4"
_npy_dtype(::Type{UInt8})   = "|u1"

# numpy shapes are row-major (C order). A Julia array is column-major; we write
# `permutedims` to reverse-axis-order so the on-disk C-order buffer matches the
# Julia logical shape. For vectors this is a no-op.
function _npy_bytes(arr::AbstractArray{T}) where {T}
    shape = size(arr)
    shapestr = length(shape) == 1 ? "($(shape[1]),)" :
               "(" * join(shape, ", ") * ")"
    header = "{'descr': '$(_npy_dtype(T))', 'fortran_order': False, 'shape': $shapestr, }"
    # header must be padded so total (magic+2+2+len) is a multiple of 64, ending '\n'
    base = length(_NPY_MAGIC) + 2 + 2 + length(header) + 1
    pad = (64 - (base % 64)) % 64
    header *= " "^pad * "\n"
    io = IOBuffer()
    write(io, _NPY_MAGIC)
    write(io, UInt8(1)); write(io, UInt8(0))                 # version 1.0
    write(io, UInt16(length(header)))                        # little-endian header len
    write(io, codeunits(header))
    # raw data in C order
    cdata = ndims(arr) <= 1 ? arr : permutedims(arr, reverse(1:ndims(arr)))
    write(io, reinterpret(UInt8, vec(collect(cdata))))
    return take!(io)
end

"""
    write_npy(path, arr)

Write a single array as a `.npy` file (numpy-loadable)."""
function write_npy(path::AbstractString, arr::AbstractArray)
    open(path, "w") do io; write(io, _npy_bytes(arr)); end
    return path
end

# Minimal uncompressed-ZIP (.npz) writer: one local-file record + central
# directory per member. CRC-32 (IEEE) computed in-line.
const _CRC_TABLE = let t = Vector{UInt32}(undef, 256)
    for n in 0:255
        c = UInt32(n)
        for _ in 1:8
            c = (c & 0x1) != 0 ? (0xedb88320 ⊻ (c >> 1)) : (c >> 1)
        end
        t[n + 1] = c
    end
    t
end
function _crc32(data::Vector{UInt8})
    c = 0xffffffff
    @inbounds for b in data
        c = _CRC_TABLE[((c ⊻ b) & 0xff) + 1] ⊻ (c >> 8)
    end
    return c ⊻ 0xffffffff
end

"""
    write_npz(path, arrays::Dict{String,<:AbstractArray})

Write an uncompressed `.npz` (a ZIP of `<name>.npy` members). Loadable by
`numpy.load(path)` → keys are the dict names."""
function write_npz(path::AbstractString, arrays::AbstractDict)
    members = Tuple{String,Vector{UInt8}}[]
    for (k, v) in arrays
        push!(members, (string(k) * ".npy", _npy_bytes(v)))
    end
    open(path, "w") do io
        offsets = Int[]
        crcs = UInt32[]
        # local file headers + data
        for (name, data) in members
            push!(offsets, position(io))
            push!(crcs, _crc32(data))
            nm = codeunits(name)
            write(io, UInt32(0x04034b50))     # local file header sig
            write(io, UInt16(20))             # version needed
            write(io, UInt16(0))              # flags
            write(io, UInt16(0))              # method 0 = stored
            write(io, UInt16(0)); write(io, UInt16(0))   # mod time/date
            write(io, _crc32(data))
            write(io, UInt32(length(data)))   # compressed size
            write(io, UInt32(length(data)))   # uncompressed size
            write(io, UInt16(length(nm)))     # name len
            write(io, UInt16(0))              # extra len
            write(io, nm)
            write(io, data)
        end
        # central directory
        cd_start = position(io)
        for (i, (name, data)) in enumerate(members)
            nm = codeunits(name)
            write(io, UInt32(0x02014b50))     # central dir sig
            write(io, UInt16(20)); write(io, UInt16(20))
            write(io, UInt16(0)); write(io, UInt16(0))
            write(io, UInt16(0)); write(io, UInt16(0))
            write(io, crcs[i])
            write(io, UInt32(length(data)))
            write(io, UInt32(length(data)))
            write(io, UInt16(length(nm)))
            write(io, UInt16(0)); write(io, UInt16(0))   # extra, comment len
            write(io, UInt16(0)); write(io, UInt16(0))   # disk no, int attrs
            write(io, UInt32(0))                          # ext attrs
            write(io, UInt32(offsets[i]))                 # local header offset
            write(io, nm)
        end
        cd_end = position(io)
        # end of central directory
        write(io, UInt32(0x06054b50))
        write(io, UInt16(0)); write(io, UInt16(0))
        write(io, UInt16(length(members))); write(io, UInt16(length(members)))
        write(io, UInt32(cd_end - cd_start))
        write(io, UInt32(cd_start))
        write(io, UInt16(0))                  # comment len
    end
    return path
end

end # module
