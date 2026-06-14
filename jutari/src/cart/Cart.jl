"""
    Cart

Cartridge ROM + bank-switching state.

Supported:
  2K       2 KB ROM, mirrored across the 4 KB cart window.
  4K       4 KB ROM, plain mapping.
  F8       8 KB / 2 banks, hotspots \$1FF8 / \$1FF9, boots in bank 1.
  F6      16 KB / 4 banks, hotspots \$1FF6..\$1FF9, boots in bank 3.
  F4      32 KB / 8 banks, hotspots \$1FF4..\$1FFB, boots in bank 7.
  E0       8 KB Parker Bros, 4 × 1 KB slots (3 mutable + 1 fixed at slice 7),
           hotspots \$1FE0..\$1FF7 (task #100).

Bank-switching hotspots fire on ANY access (read or write), so
`cart_peek` may mutate cart state as a side-effect.

**Task #100 — content-based auto-detection.** The 8 KB formats F8/E0(/FE)
are size-identical, so size alone can't pick the mapper. `make_cart` now
mirrors xitari `Cartridge::autodetectType`: an 8 KB image is scanned for
E0 bankswitch signatures (`isProbablyE0`) and routed to E0 if matched,
else F8. (Parker Bros titles — Tutankham, Montezuma's Revenge, James Bond
— are E0 and were misdetected as F8 by the old size-only map.)

Deferred: SC variants (on-cart RAM) and FE (8K JSR-driven banking, used by
Robot Tank/Decathlon — the most exotic mapper; 1 ALE game). 3F/3E/MB/MC/AR/
DPC also deferred.
"""
module Cart

export CartState, make_cart, cart_peek, cart_poke!,
       KIND_2K, KIND_4K, KIND_F8, KIND_F6, KIND_F4, KIND_E0

const KIND_2K = 0
const KIND_4K = 1
const KIND_F8 = 2
const KIND_F6 = 3
const KIND_F4 = 4
const KIND_E0 = 5    # task #100 — Parker Bros 8K, 4 × 1K slice slots

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
    KIND_E0 => 0,                  # placeholder; E0 uses `slice_slots`
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

# E0: $1FE0..$1FE7 → slot 0 slice, $1FE8..$1FEF → slot 1, $1FF0..$1FF7 → slot 2.
# Fourth slot ($1C00-$1FFF) is hard-wired to slice 7 (holds the reset vector).
const E0_HOTSPOT_BASE = (UInt16(0x1FE0), UInt16(0x1FE8), UInt16(0x1FF0))
const E0_FIXED_SLICE  = 7

"""
    CartState

Mutable cartridge state: ROM bytes, kind tag, currently-mapped bank, and
(E0 only) the three mutable 1 KB slice-slot indices.
"""
mutable struct CartState
    kind::Int
    rom::Vector{UInt8}
    current_bank::Int
    slice_slots::Vector{Int}
    CartState(kind::Int, rom::Vector{UInt8}, bank::Int) =
        new(kind, rom, bank, kind == KIND_E0 ? Int[0, 0, 0] : Int[])
end

# --------------------------------------------------------------------------- #
# Content-based mapper auto-detection (xitari Cartridge::autodetectType)
# --------------------------------------------------------------------------- #

# Mirror of xitari `Cartridge::searchForBytes`: true if `sig` occurs at least
# once in `rom`.
function _search_for_bytes(rom::Vector{UInt8}, sig::NTuple{N,UInt8}) where {N}
    n = length(rom)
    @inbounds for i in 0:(n - N - 1)
        ok = true
        for j in 1:N
            if rom[i + j] != sig[j]
                ok = false
                break
            end
        end
        ok && return true
    end
    return false
end

# xitari `Cartridge::isProbablyE0` — known E0 bankswitch signatures (MESS).
const _E0_SIGNATURES = (
    (UInt8(0x8D), UInt8(0xE0), UInt8(0x1F)),   # STA $1FE0
    (UInt8(0x8D), UInt8(0xE0), UInt8(0x5F)),   # STA $5FE0
    (UInt8(0x8D), UInt8(0xE9), UInt8(0xFF)),   # STA $FFE9
    (UInt8(0xAD), UInt8(0xE9), UInt8(0xFF)),   # LDA $FFE9
    (UInt8(0xAD), UInt8(0xED), UInt8(0xFF)),   # LDA $FFED
    (UInt8(0xAD), UInt8(0xF3), UInt8(0xBF)),   # LDA $BFF3
)

_is_probably_e0(rom::Vector{UInt8}) =
    any(sig -> _search_for_bytes(rom, sig), _E0_SIGNATURES)

function _autodetect_kind(rom::Vector{UInt8})
    n = length(rom)
    haskey(_SIZE_TO_KIND, n) || throw(ArgumentError(
        "unrecognised ROM size $n bytes. Supported " *
        "$(sort(collect(keys(_SIZE_TO_KIND))))."))
    # 8 KB is ambiguous (F8 vs E0 vs FE) — distinguish by content like xitari.
    if n == 8192 && _is_probably_e0(rom)
        return KIND_E0
    end
    return _SIZE_TO_KIND[n]
end

"""
    make_cart(rom) -> CartState

Build a `CartState`, auto-detecting the mapper (size + content). Throws
`ArgumentError` for unrecognised sizes.
"""
function make_cart(rom)
    rom_v = Vector{UInt8}(rom)
    kind = _autodetect_kind(rom_v)
    return CartState(kind, rom_v, _DEFAULT_BANK[kind])
end

# --------------------------------------------------------------------------- #
# Peek / poke
# --------------------------------------------------------------------------- #

"""
    cart_peek(cart, addr) -> UInt8

Read a byte. Any access to a hotspot address triggers a bank/slice switch
AFTER the byte has been read.
"""
function cart_peek(cart::CartState, addr::Integer)
    a = UInt16(Int(addr) & 0x1FFF)
    if cart.kind == KIND_2K
        return cart.rom[(Int(a) & 0x07FF) + 1]
    elseif cart.kind == KIND_4K
        return cart.rom[(Int(a) & 0x0FFF) + 1]
    elseif cart.kind == KIND_E0
        value = cart.rom[_e0_offset(cart, a) + 1]
        _maybe_switch_e0!(cart, a)
        return value
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
    a = UInt16(Int(addr) & 0x1FFF)
    if cart.kind == KIND_E0
        _maybe_switch_e0!(cart, a)
    else
        _maybe_switch_bank!(cart, a)
    end
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

# E0: translate cart-window addr $1000-$1FFF → absolute ROM offset, honouring
# the three mutable slice slots + the fixed-slice-7 fourth slot.
@inline function _e0_offset(cart::CartState, masked_addr::UInt16)
    slot = (Int(masked_addr) - 0x1000) >> 10        # 0..3
    in_slice = Int(masked_addr) & 0x03FF
    slice_idx = slot < 3 ? cart.slice_slots[slot + 1] : E0_FIXED_SLICE
    return slice_idx * 0x0400 + in_slice
end

@inline function _maybe_switch_e0!(cart::CartState, masked_addr::UInt16)
    for slot_idx in 0:2
        base = E0_HOTSPOT_BASE[slot_idx + 1]
        if base <= masked_addr <= base + UInt16(7)
            cart.slice_slots[slot_idx + 1] = Int(masked_addr - base)
            return nothing
        end
    end
    return nothing
end

end # module
