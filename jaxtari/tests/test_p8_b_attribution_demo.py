"""P8-b — XAI attribution on the differentiable VCS: three semantics,
the same SOFT-mode kernel, the discrete-opcode finding documented.

PORTING_PLAN.md's P8 criterion is *"Integrated Gradients on ROM bytes
recovers a known sprite-defining region in Pong."* Running on real
Pong needs PXC1-x first (the bit-exact gap); this file delivers the
attribution claim on a hand-written kernel where the ground truth is
exact, and along the way records an important finding about IG on
discrete-opcode emulators.

The demos
---------

  1. **Plain `jax.grad` attribution** — gradient of a framebuffer
     pixel w.r.t. the ROM is one-hot at the byte that supplied its
     value. Answers: *"which byte is the **source** of this value?"*

  2. **Occlusion attribution** — replace each ROM byte with 0 and
     measure the output change. Answers: *"which bytes are
     **necessary** for this output?"* — typically a wider set, since
     occluding an opcode breaks the program even if the opcode itself
     doesn't carry the output's value.

  3. **Smart-baseline IG** — IG with the baseline = the target ROM
     with the *operand byte* zeroed. The path interpolates only that
     operand, so opcodes stay valid and the gradient is meaningful
     along the whole path. Answers: *"how much does this byte
     contribute, with magnitude?"* — completeness is recovered.

The naive-IG finding
--------------------

Zero-baseline IG on a SOFT-mode ROM is **misleading** for opcode
bytes. The path `0 → target` samples programs at α=0.5, α=0.25, …
where the opcode bytes are *different opcodes* than the target's —
mostly unhandled, mostly zero-gradient. The IG sum collapses to zero
even for clearly-critical bytes. This is the discrete-input pathology
the IG paper itself notes (Sundararajan et al. 2017 §6); on emulators
the fix is to use a *smart baseline* (the same program with the
attributed operand zeroed) or to switch to occlusion. This file
records the failure mode in `test_naive_zero_baseline_ig_collapses`
so the lesson stays on the record.

Each demo prints (with `pytest -s`) the top-3 attributions for human
inspection — the project's first concrete XAI output.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from jaxtari.diff import (
    SoftBus,
    initial_soft_cpu_state,
    soft_render_scanline,
    soft_run,
)
from jaxtari.xai import (
    assert_completeness,
    integrated_gradients,
    occlusion_attribution,
)

# TIA register offsets (= SOFT-bus RAM cells; see P7f-a/b/c).
_COLUBK = 0x09


def _rom_with(bytes_: list[int], size: int = 256) -> jnp.ndarray:
    rom = jnp.zeros((size,), dtype=jnp.float32)
    for i, b in enumerate(bytes_):
        rom = rom.at[i].set(jnp.float32(b))
    return rom


def _top_attributions(attr: jnp.ndarray, k: int = 3):
    """Return the indices and signed values of the top-k attributions
    by absolute magnitude — for human inspection in test output."""
    mags = jnp.abs(attr)
    order = jnp.argsort(-mags)
    return [(int(i), float(attr[int(i)])) for i in order[:k]]


# A small reusable kernel:
#   LDA #colour / STA COLUBK   — 4 ROM bytes
# rom[1] is the colour immediate; rom[0/2/3] are opcodes / zp address.
_KERNEL_BYTES = [0xA9, 0x1E, 0x85, _COLUBK]


def _background_kernel_simulator(rom_arr):
    """Run the LDA #imm / STA COLUBK kernel and return the colour of
    background pixel 0 after rendering."""
    bus = SoftBus(ram=jnp.zeros((128,), dtype=jnp.float32), rom=rom_arr)
    state = initial_soft_cpu_state()
    state, bus = soft_run(state, bus, n_steps=2)
    return soft_render_scanline(bus)[0]


# --------------------------------------------------------------------------- #
# Demo 1 — Plain `jax.grad` attribution
# --------------------------------------------------------------------------- #

def test_demo_1_plain_gradient_attributes_to_colour_byte(capsys):
    """`jax.grad(pixel)(rom)` is one-hot at the colour immediate —
    "which byte is the **source** of this value?". The differentiable
    VCS's `jax.grad` answer to the project's headline question."""
    rom = _rom_with(_KERNEL_BYTES)
    grad = jax.grad(_background_kernel_simulator)(rom)

    # Localisation: gradient is exactly 1.0 at rom[1] (the immediate
    # propagates linearly into A, then RAM[COLUBK], then the pixel).
    assert float(grad[1]) == pytest.approx(1.0)
    other = float(jnp.sum(jnp.abs(grad))) - abs(float(grad[1]))
    assert other == pytest.approx(0.0, abs=1e-5)

    with capsys.disabled():
        print(f"\n  Demo 1 — plain gradient attribution:")
        print(f"    pixel value:  {float(_background_kernel_simulator(rom)):.0f}")
        print(f"    top 3 |grad|: {_top_attributions(grad)}")


# --------------------------------------------------------------------------- #
# Demo 2 — Occlusion attribution
# --------------------------------------------------------------------------- #

def test_demo_2_occlusion_highlights_all_necessary_bytes(capsys):
    """Occlusion answers a *different* question: 'which bytes are
    necessary?'. Zeroing ANY of the four kernel bytes breaks the
    program (turns an opcode into BRK / changes the operand), so all
    four show large occlusion attribution — a wider, but truer,
    answer to 'what's load-bearing in this ROM?'."""
    rom = _rom_with(_KERNEL_BYTES)
    attr = occlusion_attribution(_background_kernel_simulator, rom)

    # Bytes 0..3 are all part of the kernel — each is necessary.
    # The pixel value is 0x1E = 30 with the full program, 0 if any
    # critical byte is zeroed (opcode → BRK, or operand → black).
    for i in range(4):
        assert float(attr[i]) == pytest.approx(0x1E, abs=1e-3), (
            f"occluding kernel byte {i} should collapse the pixel to 0; "
            f"got attribution {float(attr[i])}"
        )
    # Bytes beyond the kernel are unused — occlusion is zero.
    for i in range(4, 16):
        assert abs(float(attr[i])) < 1e-5, (
            f"byte {i} is not part of the kernel; should have zero occlusion"
        )

    with capsys.disabled():
        print(f"\n  Demo 2 — occlusion attribution:")
        print(f"    top 3 |Δ|:    {_top_attributions(attr)}")
        print(f"    interpretation: rom[0..3] are ALL necessary "
              f"(opcodes + operand + zp addr); zeroing any breaks the program")


# --------------------------------------------------------------------------- #
# Demo 3 — Smart-baseline IG (works)
# --------------------------------------------------------------------------- #

def test_demo_3_smart_baseline_ig_recovers_quantified_attribution(capsys):
    """With baseline = target ROM with the colour operand zeroed, IG
    interpolates ONLY rom[1] along the path. The opcodes stay valid
    throughout, the simulator runs the real (if colour-shifted) program
    at every alpha, the gradient is meaningful — and IG returns
    `rom[1]` (the colour value) as the attribution, satisfying
    completeness."""
    rom = _rom_with(_KERNEL_BYTES)
    baseline = rom.at[1].set(0.0)        # same program, colour zeroed

    ig = integrated_gradients(_background_kernel_simulator, rom,
                              baseline=baseline, steps=16)

    # Only rom[1] is on the path → only rom[1] has non-zero attribution.
    assert float(ig[1]) == pytest.approx(0x1E, abs=1e-3)
    other = float(jnp.sum(jnp.abs(ig))) - abs(float(ig[1]))
    assert other == pytest.approx(0.0, abs=1e-5)

    # Completeness: sum(IG) = f(target) - f(baseline) = 0x1E - 0 = 0x1E.
    assert_completeness(_background_kernel_simulator, rom, ig,
                        baseline=baseline, atol=1e-3)

    with capsys.disabled():
        print(f"\n  Demo 3 — smart-baseline IG:")
        print(f"    top 3 |IG|:   {_top_attributions(ig)}")
        print(f"    completeness: sum(IG) = "
              f"{float(jnp.sum(ig)):.2f} = pixel value")


# --------------------------------------------------------------------------- #
# Documented finding — naive IG breaks on discrete opcodes
# --------------------------------------------------------------------------- #

def test_naive_zero_baseline_ig_collapses_on_opcode_path(capsys):
    """**This is a recorded finding, not a bug.** Naive zero-baseline
    IG on a SOFT-mode ROM produces zero attribution even for clearly
    critical bytes — the path interpolates the opcodes to garbage
    values, the simulator's gradient along the path is essentially
    zero, and the IG sum collapses. Use smart-baseline IG (Demo 3) or
    occlusion (Demo 2) instead. Documented so the lesson stays in the
    repo."""
    rom = _rom_with(_KERNEL_BYTES)
    ig_naive = integrated_gradients(_background_kernel_simulator, rom, steps=32)

    # Both the gradient demos above show real signal at the kernel bytes.
    # Naive IG collapses to ~0 even there.
    assert abs(float(ig_naive[1])) < 1e-3, (
        "naive zero-baseline IG was expected to collapse; if it now "
        "returns real attribution this finding has changed and the test "
        "(and the P8-b docstring) should be updated"
    )

    with capsys.disabled():
        print(f"\n  Documented finding — naive zero-baseline IG:")
        print(f"    IG[rom[1]] = {float(ig_naive[1]):.3e}  (collapsed to ~0)")
        print(f"    sum |IG|   = {float(jnp.sum(jnp.abs(ig_naive))):.3e}")
        print(f"    reason: at α<1 the opcodes interpolate to garbage opcodes,")
        print(f"    the simulator just BRKs/no-ops, gradient is zero along the path.")
