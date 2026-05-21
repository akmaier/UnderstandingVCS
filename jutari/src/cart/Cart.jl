"""
    Cart

Cartridge ROM + bank-switching state.

Supported in P5:
  2K       2 KB ROM, mirrored across the 4 KB cart window.
  4K       4 KB ROM, plain mapping.
  F8       8 KB / 2 banks, hotspots \$1FF8 / \$1FF9, boots in bank 1.
  F6      16 KB / 4 banks, hotspots \$1FF6..\$1FF9, boots in bank 3.
  F4      32 KB / 8 banks, hotspots \$1FF4..\$1FFB, boots in bank 7.

Bank-switching hotspots fire on ANY access (read or write), so
`cart_peek` may mutate `current_bank` as a side-effect. The mutable
struct makes this natural; the surrounding `BusState` already uses
mutable semantics in jutari.

Deferred to a P5 follow-up: SC variants of F8/F6/F4 (add 128 B of
on-cart RAM), E0, FE, 3F, 3E, MB, MC, AR, DPC.
"""
module Cart

export CartState, make_cart, cart_peek, cart_poke!,
       KIND_2K, KIND_4K, KIND_F8, KIND_F6, KIND_F4

const KIND_2K = 0
const KIND_4K = 1
const KIND_F8 = 2
const KIND_F6 = 3
const KIND_F4 = 4

const _SIZE_TO_KIND = Dict{Int, Int}(
    2048  => KIND_2K,
    4096  => KIND_4K,
    8192  => KIND_F8,
    16384 => KIND_F6,
    32768 => KIND_F4,
)

const _DEFAULT_BANK = Dict{Int, Int}(
    KIND_2K => 0, KIND_4K => 0,
    KIND_F8 => 1, KIND_F6 => 3, KIND_F4 => 7,
)

# Hotspot address → target bank for each switched format.
const F8_HOTSPOTS = Dict{UInt16, Int}(UInt16(0x1FF8) => 0, UInt16(0x1FF9) => 1)
const F6_HOTSPOTS = Dict{UInt16, Int}(
    UInt16(0x1FF6) => 0, UInt16(0x1FF7) => 1,
    UInt16(0x1FF8) => 2, UInt16(0x1FF9) => 3,
)
const F4_HOTSPOTS = Dict{UInt16, Int}(
    UInt16(0x1FF4) => 0, UInt16(0x1FF5) => 1,
    UInt16(0x1FF6) => 2, UInt16(0x1FF7) => 3,
    UInt16(0x1FF8) => 4, UInt16(0x1FF9) => 5,
    UInt16(0x1FFA) => 6, UInt16(0x1FFB) => 7,
)

"""
    CartState

Mutable cartridge state: ROM bytes, kind tag, and the currently-mapped
bank index for bank-switched carts.
"""
mutable struct CartState
    kind::Int
    rom::Vector{UInt8}
    current_bank::Int
end

"""
    make_cart(rom) -> CartState

Auto-detect the cart kind from the ROM size and build a `CartState`.
Throws `ArgumentError` for unrecognised sizes.
"""
function make_cart(rom)
    rom_v = Vector{UInt8}(rom)
    n = length(rom_v)
    haskey(_SIZE_TO_KIND, n) || throw(ArgumentError(
        "unrecognised ROM size $n bytes. P5 supports " *
        "$(sort(collect(keys(_SIZE_TO_KIND))))."))
    kind = _SIZE_TO_KIND[n]
    return CartState(kind, rom_v, _DEFAULT_BANK[kind])
end

"""
    cart_peek(cart, addr) -> UInt8

Read a byte. Any access to a hotspot address triggers a bank switch AFTER
the byte has been read (so the returned value comes from the bank that
was current at the time of the access).
"""
function cart_peek(cart::CartState, addr::Integer)
    a = UInt16(Int(addr) & 0x1FFF)
    if cart.kind == KIND_2K
        return cart.rom[(Int(a) & 0x07FF) + 1]
    elseif cart.kind == KIND_4K
        return cart.rom[(Int(a) & 0x0FFF) + 1]
    end
    bank_offset = cart.current_bank * 0x1000
    value = cart.rom[bank_offset + (Int(a) & 0x0FFF) + 1]
    _maybe_switch_bank!(cart, a)
    return value
end

"""
    cart_poke!(cart, addr, value)

Writes to ROM are dropped, but hotspot accesses still fire.
"""
function cart_poke!(cart::CartState, addr::Integer, ::Integer)
    _maybe_switch_bank!(cart, UInt16(Int(addr) & 0x1FFF))
    return nothing
end

@inline function _maybe_switch_bank!(cart::CartState, masked_addr::UInt16)
    if cart.kind == KIND_F8
        haskey(F8_HOTSPOTS, masked_addr) &&
            (cart.current_bank = F8_HOTSPOTS[masked_addr])
    elseif cart.kind == KIND_F6
        haskey(F6_HOTSPOTS, masked_addr) &&
            (cart.current_bank = F6_HOTSPOTS[masked_addr])
    elseif cart.kind == KIND_F4
        haskey(F4_HOTSPOTS, masked_addr) &&
            (cart.current_bank = F4_HOTSPOTS[masked_addr])
    end
    return nothing
end

end # module
