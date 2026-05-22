# ROM-as-weights — make cartridge bytes differentiable parameters.
#
# Wraps a float32 view of the ROM bytes. `peek(rom::RomTensor, addr)`
# computes a one-hot dot product with the ROM, so the returned scalar
# differentiates back to the ROM array.
#
# This is the PORTING_PLAN.md §6.2 "ROM-as-weights" primitive — the
# building block for SOFT execution mode and the eventual XAI work.

"""
    RomTensor

A cartridge ROM image as a `Vector{Float32}`, with a differentiable
peek. Construct from any byte vector; ints are cast to Float32.
"""
struct RomTensor
    rom::Vector{Float32}
end

RomTensor(bytes::AbstractVector) = RomTensor(Vector{Float32}(bytes))

Base.length(r::RomTensor) = length(r.rom)
Base.size(r::RomTensor)   = (length(r.rom),)

"""
    peek(rom::RomTensor, addr::Integer) -> Float32

Differentiable single-byte read. Returns `one_hot(addr) · rom.rom`.
The result is bit-exact-equal to `rom.rom[addr + 1]`; the dot-product
formulation gives the autodiff system a clean Jacobian.

The one-hot vector is built by broadcasting a comparison rather than
allocate-then-`setindex!` — array mutation is invisible to Zygote, so
the broadcast form is what makes `Zygote.gradient(peek, …)` work
(P7e).
"""
function peek(rom::RomTensor, addr::Integer)
    n = length(rom.rom)
    one_hot = Float32.((0:n - 1) .== Int(addr))
    return _dot(one_hot, rom.rom)
end

"""
    peek_many(rom::RomTensor, addrs) -> Vector{Float32}

Differentiable batched read. Returns the values at each address in
`addrs` as a `Vector{Float32}` of matching length. The selection
matrix is built by a broadcast comparison (Zygote-friendly — see
`peek`).
"""
function peek_many(rom::RomTensor, addrs)
    n = length(rom.rom)
    idxs = Int.(addrs)
    A = Float32.(idxs .== (0:n - 1)')        # m×n one-hot rows
    return A * rom.rom
end
