"""P4c tests — paddle position helper.

`set_paddle(tia, paddle, value)` drives INPT0..INPT3 directly with
the requested wheel position byte. Faithful capacitor-charge
"dump-pot" timing — where real INPT D7 is a timing function rather
than a stored value — is a separate deferral (STATUS.md P4c); this
helper gives RL / XAI callers a usable analog paddle interface
without the timing model.
"""

import jax.numpy as jnp
import pytest

from jaxtari.tia.system import (
    R_INPT0,
    R_INPT1,
    R_INPT2,
    R_INPT3,
    initial_tia_state,
    set_paddle,
    tia_peek,
)


def test_set_paddle_0_stores_in_inpt0():
    tia = initial_tia_state()
    tia = set_paddle(tia, 0, 0x42)
    assert int(tia.inpt[0]) == 0x42


def test_set_paddle_1_2_3_independent():
    tia = initial_tia_state()
    tia = set_paddle(tia, 0, 0x10)
    tia = set_paddle(tia, 1, 0x20)
    tia = set_paddle(tia, 2, 0x30)
    tia = set_paddle(tia, 3, 0x40)
    assert int(tia.inpt[0]) == 0x10
    assert int(tia.inpt[1]) == 0x20
    assert int(tia.inpt[2]) == 0x30
    assert int(tia.inpt[3]) == 0x40


def test_set_paddle_preserves_triggers():
    """INPT4/5 (the fire buttons) must not be touched by paddle writes."""
    tia = initial_tia_state()
    before_4 = int(tia.inpt[4])
    before_5 = int(tia.inpt[5])
    tia = set_paddle(tia, 0, 0xFF)
    tia = set_paddle(tia, 3, 0x00)
    assert int(tia.inpt[4]) == before_4
    assert int(tia.inpt[5]) == before_5


def test_set_paddle_value_masked_to_byte():
    """Values outside 0..255 are silently masked (uint8 semantics)."""
    tia = initial_tia_state()
    tia = set_paddle(tia, 0, 0x142)            # 322 & 0xFF = 0x42
    assert int(tia.inpt[0]) == 0x42


def test_set_paddle_rejects_invalid_index():
    tia = initial_tia_state()
    for bad in (-1, 4, 5, 99):
        with pytest.raises(ValueError):
            set_paddle(tia, bad, 0)


def test_tia_peek_inpt0_returns_paddle_value():
    """A program doing `LDA $08` (INPT0 read) should see whatever
    `set_paddle(0, ...)` wrote."""
    tia = initial_tia_state()
    tia = set_paddle(tia, 0, 0x77)
    # `tia_peek($08)` returns INPT0 via the standard read decode.
    assert tia_peek(tia, R_INPT0) == 0x77


def test_paddle_initial_value_is_0x80_centred():
    """The pre-P4c default — paddles default to "centred" = $80 in the
    INPT array — must still hold so existing tests / boot defaults
    don't shift."""
    tia = initial_tia_state()
    assert int(tia.inpt[0]) == 0x80
    assert int(tia.inpt[1]) == 0x80
    assert int(tia.inpt[2]) == 0x80
    assert int(tia.inpt[3]) == 0x80


def test_paddle_address_constants_match_tia_layout():
    assert R_INPT0 == 0x08
    assert R_INPT1 == 0x09
    assert R_INPT2 == 0x0A
    assert R_INPT3 == 0x0B
