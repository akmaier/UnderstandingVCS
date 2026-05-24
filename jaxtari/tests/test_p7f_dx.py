"""P7f-dx tests — collision-register dispatch via `_bus_read`.

P7f-d shipped `soft_collision_registers` as a standalone function:
SOFT code that wanted collision data had to call it from outside the
trace. P7f-dx wires the same function into `_bus_read`, so a SOFT
program doing `LDA $30` (CXM0P) now gets the live collision latch
instead of whatever happened to be sitting in `bus.ram[0x30]`.

That matters for XAI: a real Atari ROM constantly reads collision
registers to drive game logic (Pong's score detection, Combat's
tank-hit handling). Without P7f-dx those reads return register-file
garbage from the SOFT collapse, so any attribution attempt past the
first collision read would be meaningless.
"""

import jax
import jax.numpy as jnp

from jaxtari.diff.soft_state import SoftBus, initial_soft_cpu_state
from jaxtari.diff.soft_step import soft_step, _bus_read


# Object position / pattern register offsets — match jaxtari.tia.system.
W_NUSIZ0 = 0x04
W_NUSIZ1 = 0x05
W_GRP0   = 0x1B
W_GRP1   = 0x1C
W_RESP0  = 0x10
W_RESP1  = 0x11
W_RESM0  = 0x12
W_RESM1  = 0x13
W_RESBL  = 0x14
W_ENAM0  = 0x1D
W_ENAM1  = 0x1E
W_ENABL  = 0x1F
W_PF1    = 0x0E

# Read-side collision register addresses.
R_CXM0P  = 0x30
R_CXM1P  = 0x31
R_CXP0FB = 0x32
R_CXP1FB = 0x33
R_CXM0FB = 0x34
R_CXM1FB = 0x35
R_CXBLPF = 0x36
R_CXPPMM = 0x37


def _bus_with(*pairs) -> SoftBus:
    """Build a SoftBus whose RAM cells are populated from `pairs`.
    Each pair is `(ram_offset, value)`. ROM is empty (collision reads
    don't need it)."""
    ram = jnp.zeros((128,), dtype=jnp.float32)
    for off, val in pairs:
        ram = ram.at[off].set(jnp.float32(val))
    return SoftBus(ram=ram, rom=jnp.zeros((256,), dtype=jnp.float32))


# --------------------------------------------------------------------------- #
# Direct `_bus_read` dispatch — collision reads return the live latch
# --------------------------------------------------------------------------- #

def test_bus_read_cxppmm_zero_when_objects_not_overlapping():
    """P0 at col 0, P1 at col 100 — no overlap, CXPPMM bit 7 (P0-P1) = 0."""
    bus = _bus_with(
        (W_GRP0, 0xFF), (W_RESP0, 0),
        (W_GRP1, 0xFF), (W_RESP1, 100),
    )
    val = int(_bus_read(bus, R_CXPPMM))
    assert val == 0


def test_bus_read_cxppmm_set_when_p0_p1_overlap():
    """P0 and P1 both at column 50 — CXPPMM bit 7 (P0-P1) latches to 1."""
    bus = _bus_with(
        (W_GRP0, 0xFF), (W_RESP0, 50),
        (W_GRP1, 0xFF), (W_RESP1, 50),
    )
    val = int(_bus_read(bus, R_CXPPMM))
    assert val & 0x80 == 0x80         # P0-P1 hit


def test_bus_read_cxp0fb_set_when_p0_overlaps_playfield():
    """P0 at col 16; PF1 bit 7 lights playfield pixels 16..19. Their
    overlap sets CXP0FB bit 7 (P0-PF)."""
    bus = _bus_with(
        (W_GRP0, 0xFF), (W_RESP0, 16),
        (W_PF1, 0x80),
    )
    val = int(_bus_read(bus, R_CXP0FB))
    assert val & 0x80 == 0x80


def test_bus_read_cxm0p_set_when_missile0_overlaps_p1():
    """M0 enabled at col 50; P1 at col 50 — CXM0P bit 7 (M0-P1) latches."""
    bus = _bus_with(
        (W_ENAM0, 0x02), (W_RESM0, 50), (W_NUSIZ0, 0x00),    # size 1
        (W_GRP1, 0xFF),  (W_RESP1, 50),
    )
    val = int(_bus_read(bus, R_CXM0P))
    assert val & 0x80 == 0x80


def test_bus_read_collision_mirror_at_30_to_37():
    """All eight collision addresses ($30-$37) are decoded — pick one
    each from a setup that produces a single hit pattern."""
    bus = _bus_with(
        (W_GRP0, 0xFF), (W_RESP0, 50),
        (W_GRP1, 0xFF), (W_RESP1, 50),
    )
    # CXPPMM should hit, the other 7 should be 0.
    assert int(_bus_read(bus, R_CXPPMM)) == 0x80
    assert int(_bus_read(bus, R_CXM0P))  == 0
    assert int(_bus_read(bus, R_CXM1P))  == 0
    assert int(_bus_read(bus, R_CXP0FB)) == 0
    assert int(_bus_read(bus, R_CXP1FB)) == 0
    assert int(_bus_read(bus, R_CXM0FB)) == 0
    assert int(_bus_read(bus, R_CXM1FB)) == 0
    assert int(_bus_read(bus, R_CXBLPF)) == 0


# --------------------------------------------------------------------------- #
# End-to-end — a SOFT program does `LDA $30` and sees the collision
# --------------------------------------------------------------------------- #

def test_soft_program_lda_cxppmm_loads_collision_into_A():
    """The point of P7f-dx: a SOFT program that polls a collision
    register through the bus gets the live latched value, not the
    register-file collapse garbage. After LDA $30, A == 0x80 (P0-P1 hit)."""
    # Bus setup: P0 + P1 colliding at column 50.
    ram = (jnp.zeros((128,), dtype=jnp.float32)
           .at[W_GRP0].set(0xFF).at[W_RESP0].set(50.0)
           .at[W_GRP1].set(0xFF).at[W_RESP1].set(50.0))

    # ROM: `LDA $30` at offset 0 (PC=$F000).
    rom = (jnp.zeros((256,), dtype=jnp.float32)
           .at[0].set(0xA5)       # LDA zp opcode
           .at[1].set(0x30))      # operand = $30 (CXPPMM mirror — but
    # actually $30 maps to CXM0P (low 3 bits = 0). Let me use $37
    # instead to read CXPPMM.
    rom = rom.at[1].set(0x37)

    bus = SoftBus(ram=ram, rom=rom)
    state = initial_soft_cpu_state(pc=0xF000)
    state2, bus2 = soft_step(state, bus)
    assert int(state2.A) == 0x80


def test_soft_program_reads_zero_collision_when_no_overlap():
    """Same program, no overlap — A reads 0."""
    ram = (jnp.zeros((128,), dtype=jnp.float32)
           .at[W_GRP0].set(0xFF).at[W_RESP0].set(0.0)
           .at[W_GRP1].set(0xFF).at[W_RESP1].set(100.0))
    rom = (jnp.zeros((256,), dtype=jnp.float32)
           .at[0].set(0xA5).at[1].set(0x37))    # LDA $37 (CXPPMM)
    bus = SoftBus(ram=ram, rom=rom)
    state = initial_soft_cpu_state(pc=0xF000)
    state2, _ = soft_step(state, bus)
    assert int(state2.A) == 0


# --------------------------------------------------------------------------- #
# Non-collision TIA reads stay on the RAM-collapse path (no regression)
# --------------------------------------------------------------------------- #

def test_bus_read_tia_non_collision_still_returns_ram():
    """TIA addresses outside $30-$37 keep the SOFT RAM collapse — a
    program that does LDA $1B (which is W_GRP0, a *write* address;
    real hardware returns garbage but our SOFT collapse returns
    whatever was last written there) gets the stored byte."""
    bus = _bus_with((W_GRP0, 0x42))
    val = int(_bus_read(bus, W_GRP0))
    assert val == 0x42


# --------------------------------------------------------------------------- #
# Gradient — the structural collision *bit* breaks the gradient (the
# `> 0` test in soft_collision_registers is non-differentiable), so a
# read from $30-$37 carries no gradient back to the ROM. This test
# pins that down so any future "soft collision" change is intentional.
# --------------------------------------------------------------------------- #

def test_collision_read_gradient_is_zero():
    """∂(state.A after LDA $37)/∂ROM is zero across the board — the
    collision latch is a hard threshold (`sum(mask * mask) > 0`)."""
    ram = (jnp.zeros((128,), dtype=jnp.float32)
           .at[W_GRP0].set(0xFF).at[W_RESP0].set(50.0)
           .at[W_GRP1].set(0xFF).at[W_RESP1].set(50.0))
    rom = (jnp.zeros((256,), dtype=jnp.float32)
           .at[0].set(0xA5).at[1].set(0x37))    # LDA $37

    def collision_into_A(r):
        bus = SoftBus(ram=ram, rom=r)
        state = initial_soft_cpu_state(pc=0xF000)
        state2, _ = soft_step(state, bus)
        return state2.A

    g = jax.grad(collision_into_A)(rom)
    assert float(jnp.sum(jnp.abs(g))) == 0.0
