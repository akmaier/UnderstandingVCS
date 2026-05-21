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
"""
function peek(rom::RomTensor, addr::Integer)
    n = length(rom.rom)
    one_hot = zeros(Float32, n)
    one_hot[Int(addr) + 1] = 1f0
    return _dot(one_hot, rom.rom)
end

"""
    peek_many(rom::RomTensor, addrs) -> Vector{Float32}

Differentiable batched read. Returns the values at each address in
`addrs` as a `Vector{Float32}` of matching length.
"""
function peek_many(rom::RomTensor, addrs)
    n = length(rom.rom)
    m = length(addrs)
    A = zeros(Float32, m, n)
    @inbounds for (i, addr) in enumerate(addrs)
        A[i, Int(addr) + 1] = 1f0
    end
    return A * rom.rom
end
