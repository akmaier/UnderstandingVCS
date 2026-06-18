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

export CartState, make_cart, cart_peek, cart_peek_pure, cart_poke!, cart_reset!,
       KIND_2K, KIND_4K, KIND_F8, KIND_F6, KIND_F4, KIND_E0, KIND_FE,
       KIND_F8SC, KIND_F6SC, KIND_F4SC

const KIND_2K = 0
const KIND_4K = 1
const KIND_F8 = 2
const KIND_F6 = 3
const KIND_F4 = 4
const KIND_E0 = 5    # task #100 — Parker Bros 8K, 4 × 1K slice slots
const KIND_FE = 6    # task #102 — Activision 8K (Robot Tank / Decathlon)
const KIND_F8SC = 7  # task #103 — F8 + 128 B on-cart RAM (Superchip); e.g. elevator_action
const KIND_F6SC = 8  # task #103 — F6 + 128 B Superchip RAM
const KIND_F4SC = 9  # task #103 — F4 + 128 B Superchip RAM

# F8SC/F6SC/F4SC carry a 128 B on-cart Superchip RAM: write at $1000-$107F,
# read at $1080-$10FF (both alias the same buffer via addr & 0x7F). xitari's
# autodetect picks these (isProbablySC) BEFORE F8/F6/F4. Mirror of jaxtari.
const _SC_KINDS = (KIND_F8SC, KIND_F6SC, KIND_F4SC)
const _SC_RAM_BYTES   = 128
const _SC_READ_BASE   = 0x1080
const _SC_WRITE_BASE  = 0x1000
const _SC_AREA_MASK    = 0x1F80
const _SC_OFFSET_MASK  = 0x007F

# xitari's CartF8SC/F6SC/F4SC ctor fills the 128 B Superchip RAM with
# `Random::next()` — the ale Random LCG `v = (v*2416+374441) % 1771875`, returning
# `v & 0xFF`, seeded from `ourSeed` (ALE `random_seed`). For DETERMINISTIC
# conformance the suite pins `random_seed=0` (xitari Defaults.cpp), so jutari
# mirrors xitari's exact seed-0 sequence here instead of zero-filling. Games that
# initialise their SC RAM overwrite this; elevator_action's attract demo reads it
# UNINITIALISED as a cheap RNG, so matching the init makes elevator deterministic +
# pixel-exact (bug_fix_log #122). `reset()` does NOT re-init (xitari CartF8SC::reset
# only does bank(1)), so this runs once at construction.
function _sc_ram_lcg_init()
    ram = Vector{UInt8}(undef, _SC_RAM_BYTES)
    v = 0
    for i in 1:_SC_RAM_BYTES
        v = (v * 2416 + 374441) % 1771875
        ram[i] = UInt8(v & 0xFF)
    end
    return ram
end

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
    KIND_FE => 0,                  # FE has no dynamic bank (static upper 4K)
    KIND_F8SC => 1, KIND_F6SC => 3, KIND_F4SC => 7,   # task #103 — same as F8/F6/F4
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
    sc_ram::Vector{UInt8}          # task #103 — 128 B Superchip RAM (SC kinds only)
    CartState(kind::Int, rom::Vector{UInt8}, bank::Int) =
        new(kind, rom, bank,
            kind == KIND_E0 ? Int[0, 0, 0] : Int[],
            kind in _SC_KINDS ? _sc_ram_lcg_init() : UInt8[])
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

# xitari `Cartridge::isProbablyFE` — FE always includes a 'JSR $xxxx' (MESS).
const _FE_SIGNATURES = (
    (UInt8(0x20), UInt8(0x00), UInt8(0xD0), UInt8(0xC6), UInt8(0xC5)),  # JSR $D000;DEC $C5
    (UInt8(0x20), UInt8(0xC3), UInt8(0xF8), UInt8(0xA5), UInt8(0x82)),  # JSR $F8C3;LDA $82
    (UInt8(0xD0), UInt8(0xFB), UInt8(0x20), UInt8(0x73), UInt8(0xFE)),  # BNE;JSR $FE73
    (UInt8(0x20), UInt8(0x00), UInt8(0xF0), UInt8(0x84), UInt8(0xD6)),  # JSR $F000;STY $D6
)

_is_probably_fe(rom::Vector{UInt8}) =
    any(sig -> _search_for_bytes(rom, sig), _FE_SIGNATURES)

# xitari `Cartridge::isProbablySC` — a Superchip cart fills its 128 B RAM area
# (the first 256 bytes of EACH 4 KB bank) with a single constant byte in the
# ROM image. xitari checks this BEFORE F8/F6/F4 (task #103, elevator_action).
function _is_probably_sc(rom::Vector{UInt8})
    n = length(rom)
    (n >= 4096 && n % 4096 == 0) || return false
    banks = n ÷ 4096
    @inbounds for i in 0:(banks - 1)
        first = rom[i * 4096 + 1]
        for j in 0:255
            rom[i * 4096 + j + 1] == first || return false
        end
    end
    return true
end

function _autodetect_kind(rom::Vector{UInt8})
    n = length(rom)
    haskey(_SIZE_TO_KIND, n) || throw(ArgumentError(
        "unrecognised ROM size $n bytes. Supported " *
        "$(sort(collect(keys(_SIZE_TO_KIND))))."))
    # Content-based disambiguation, matching xitari's autodetectType order.
    # SC (Superchip on-cart RAM) is checked FIRST at each banked size, then the
    # 8 KB E0/FE special mappers, else the plain F8/F6/F4.
    if n == 8192
        _is_probably_sc(rom) && return KIND_F8SC
        _is_probably_e0(rom) && return KIND_E0
        _is_probably_fe(rom) && return KIND_FE
    elseif n == 16384
        _is_probably_sc(rom) && return KIND_F6SC
    elseif n == 32768
        _is_probably_sc(rom) && return KIND_F4SC
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
    elseif cart.kind == KIND_FE
        # Task #102: xitari CartFE returns myImage[(addr&0x0FFF) +
        # ((addr&0x2000)==0 ? 4096 : 0)] — the bank is A13 of the FULL CPU
        # address (Stella uses the emulator's 16-bit PC: $Fxxx→A13=1→lower
        # bank, $Dxxx→A13=0→upper bank). The bus passes the un-masked
        # `addr` here so A13 survives. Robot Tank / Decathlon.
        return cart.rom[(Int(addr) & 0x0FFF) +
                        ((Int(addr) & 0x2000) == 0 ? 0x1000 : 0) + 1]
    end
    # task #103: F8SC/F6SC/F4SC Superchip RAM read window $1080-$10FF.
    # The write window $1000-$107F falls through to ROM/open-bus on reads
    # (mirror of jaxtari / xitari). Hotspots fire only outside the RAM area.
    if cart.kind in _SC_KINDS && (Int(a) & _SC_AREA_MASK) == _SC_READ_BASE
        return cart.sc_ram[(Int(a) & _SC_OFFSET_MASK) + 1]
    end
    bank_offset = cart.current_bank * 0x1000
    value = cart.rom[bank_offset + (Int(a) & 0x0FFF) + 1]
    _maybe_switch_bank!(cart, a)
    return value
end

"""
    cart_peek_pure(cart, addr) -> UInt8

Side-effect-free cart read: returns the byte at `addr` in the CURRENT bank
WITHOUT firing any bank-switch hotspot. Used by the (default-off) CPU relaxation
read-blend for neighbour reads, which must not switch banks or advance any clock.
Mirrors `cart_peek` minus the `_maybe_switch_*!` calls.
"""
function cart_peek_pure(cart::CartState, addr::Integer)
    a = UInt16(Int(addr) & 0x1FFF)
    if cart.kind == KIND_2K
        return cart.rom[(Int(a) & 0x07FF) + 1]
    elseif cart.kind == KIND_4K
        return cart.rom[(Int(a) & 0x0FFF) + 1]
    elseif cart.kind == KIND_E0
        return cart.rom[_e0_offset(cart, a) + 1]
    elseif cart.kind == KIND_FE
        return cart.rom[(Int(addr) & 0x0FFF) +
                        ((Int(addr) & 0x2000) == 0 ? 0x1000 : 0) + 1]
    end
    if cart.kind in _SC_KINDS && (Int(a) & _SC_AREA_MASK) == _SC_READ_BASE
        return cart.sc_ram[(Int(a) & _SC_OFFSET_MASK) + 1]
    end
    return cart.rom[cart.current_bank * 0x1000 + (Int(a) & 0x0FFF) + 1]
end

"""
    cart_poke!(cart, addr, value)

Writes to ROM are dropped, but hotspot accesses still fire.
"""
# Reset the cart's bank-switch mapping to its power-on default — xitari's
# Cartridge::reset(), which System::reset() invokes on every system reset
# (jutari's console_reset! mirrors System::reset). The cart is a device on the
# bus, so a bank the game switched into during the 60-frame construction probe
# must NOT leak across the post-probe reset. elevator_action (F8SC) ends the
# probe in bank 0, so without resetting here console_reset! read the reset vector
# + init code (CLD;SEI;LDX #FF;TXS) from the wrong bank, giving a divergent boot
# register state (Y) that only surfaced as a 16 px missile mis-position. RAM
# (RIOT + Superchip) is preserved; only the bank / E0 slice mapping resets.
function cart_reset!(cart::CartState)
    cart.current_bank = _DEFAULT_BANK[cart.kind]
    cart.kind == KIND_E0 && fill!(cart.slice_slots, 0)
    return cart
end

function cart_poke!(cart::CartState, addr::Integer, value::Integer)
    a = UInt16(Int(addr) & 0x1FFF)
    # task #103: F8SC/F6SC/F4SC Superchip RAM write window $1000-$107F.
    if cart.kind in _SC_KINDS && (Int(a) & _SC_AREA_MASK) == _SC_WRITE_BASE
        cart.sc_ram[(Int(a) & _SC_OFFSET_MASK) + 1] = UInt8(Int(value) & 0xFF)
        return nothing
    end
    if cart.kind == KIND_E0
        _maybe_switch_e0!(cart, a)
    else
        _maybe_switch_bank!(cart, a)
    end
    return nothing
end

@inline function _maybe_switch_bank!(cart::CartState, masked_addr::UInt16)
    # task #103: SC variants share the F8/F6/F4 hotspot banking.
    if cart.kind == KIND_F8 || cart.kind == KIND_F8SC
        haskey(F8_HOTSPOTS, masked_addr) &&
            (cart.current_bank = F8_HOTSPOTS[masked_addr])
    elseif cart.kind == KIND_F6 || cart.kind == KIND_F6SC
        haskey(F6_HOTSPOTS, masked_addr) &&
            (cart.current_bank = F6_HOTSPOTS[masked_addr])
    elseif cart.kind == KIND_F4 || cart.kind == KIND_F4SC
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
