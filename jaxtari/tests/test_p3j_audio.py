"""P3j tests — TIA audio register storage.

The Atari 2600's TIA has 6 audio registers:
  AUDC0 / AUDC1  — audio control (tone selector, 4 bits each)
  AUDF0 / AUDF1  — audio frequency (5 bits each)
  AUDV0 / AUDV1  — audio volume (4 bits each)

In jaxtari they are *stored* in the TIA register file but **not yet
synthesised** — the TIASnd / TIATables polynomial-counter logic that
xitari uses to drive an audio buffer isn't ported. P3j locks in the
"stored but inert" contract so:

  1. A program can write to AUDC0 / AUDF0 / AUDV0 / etc. and read the
     same byte back (via the SOFT bus collapse, since real TIA reads
     of AUD* return open bus on hardware).
  2. The register file holds the right bytes after a sequence of
     writes, so an attribution test that wants to back-prop through
     "volume controls" has something to work with.

Real audio synthesis (TIASnd) is deferred — see STATUS.md P3j.
"""

from __future__ import annotations

import jax.numpy as jnp

from jaxtari.tia.system import (
    W_AUDC0, W_AUDC1, W_AUDF0, W_AUDF1, W_AUDV0, W_AUDV1,
    initial_tia_state, tia_poke,
)


# --------------------------------------------------------------------------- #
# Canonical addresses — pin them down so any future renumber regression fires
# --------------------------------------------------------------------------- #

def test_audio_register_offsets_match_canonical_tia_layout():
    assert W_AUDC0 == 0x15
    assert W_AUDC1 == 0x16
    assert W_AUDF0 == 0x17
    assert W_AUDF1 == 0x18
    assert W_AUDV0 == 0x19
    assert W_AUDV1 == 0x1A


# --------------------------------------------------------------------------- #
# Storage — writes land in the register file and round-trip
# --------------------------------------------------------------------------- #

def test_audc0_write_stored_in_register_file():
    tia = initial_tia_state()
    tia = tia_poke(tia, W_AUDC0, 0x0F)         # tone selector 0xF (white noise)
    assert int(tia.registers[W_AUDC0]) == 0x0F


def test_audf0_write_stored_in_register_file():
    tia = initial_tia_state()
    tia = tia_poke(tia, W_AUDF0, 0x1F)         # max frequency (5-bit)
    assert int(tia.registers[W_AUDF0]) == 0x1F


def test_audv0_write_stored_in_register_file():
    tia = initial_tia_state()
    tia = tia_poke(tia, W_AUDV0, 0x0F)         # max volume (4-bit)
    assert int(tia.registers[W_AUDV0]) == 0x0F


def test_audc1_audf1_audv1_independent():
    """Writes to the *1 set don't bleed into *0."""
    tia = initial_tia_state()
    tia = tia_poke(tia, W_AUDC0, 0x42)
    tia = tia_poke(tia, W_AUDC1, 0x84)
    tia = tia_poke(tia, W_AUDF0, 0x11)
    tia = tia_poke(tia, W_AUDF1, 0x22)
    tia = tia_poke(tia, W_AUDV0, 0x03)
    tia = tia_poke(tia, W_AUDV1, 0x07)
    assert int(tia.registers[W_AUDC0]) == 0x42
    assert int(tia.registers[W_AUDC1]) == 0x84
    assert int(tia.registers[W_AUDF0]) == 0x11
    assert int(tia.registers[W_AUDF1]) == 0x22
    assert int(tia.registers[W_AUDV0]) == 0x03
    assert int(tia.registers[W_AUDV1]) == 0x07


def test_audio_register_initial_value_is_zero():
    """After RESET, all six audio registers start at 0 (silence)."""
    tia = initial_tia_state()
    for off in (W_AUDC0, W_AUDC1, W_AUDF0, W_AUDF1, W_AUDV0, W_AUDV1):
        assert int(tia.registers[off]) == 0


def test_audio_register_overwrite():
    """Last-write-wins — a second write replaces the first."""
    tia = initial_tia_state()
    tia = tia_poke(tia, W_AUDV0, 0x0F)
    tia = tia_poke(tia, W_AUDV0, 0x00)
    assert int(tia.registers[W_AUDV0]) == 0


# --------------------------------------------------------------------------- #
# Inertness — writing to audio registers doesn't disturb other TIA state
# (no synthesis side effects yet).
# --------------------------------------------------------------------------- #

def test_audio_write_does_not_advance_scanline():
    tia = initial_tia_state()
    before = (int(tia.scanline_cycle), int(tia.scanline), int(tia.frame))
    tia = tia_poke(tia, W_AUDC0, 0x0F)
    tia = tia_poke(tia, W_AUDV0, 0x0F)
    after = (int(tia.scanline_cycle), int(tia.scanline), int(tia.frame))
    assert before == after


def test_audio_write_does_not_request_wsync():
    """WSYNC is set by writes to its own register, not by AUD*."""
    tia = initial_tia_state()
    tia = tia_poke(tia, W_AUDC0, 0xFF)
    tia = tia_poke(tia, W_AUDF0, 0xFF)
    tia = tia_poke(tia, W_AUDV0, 0xFF)
    assert tia.wsync_pending is False


# --------------------------------------------------------------------------- #
# Bus-level — `tia_poke` is what `_bus_poke` calls for TIA writes, so the
# storage path is reachable from a real CPU program.
# --------------------------------------------------------------------------- #

def test_bus_write_lands_in_audio_register_via_tia_poke():
    """End-to-end: a `tia_poke($19, val)` (which is what STA $19 from a
    program decodes to) stores the byte at registers[$19]."""
    tia = initial_tia_state()
    for val in (0x00, 0x05, 0x08, 0x0F):
        tia = tia_poke(tia, W_AUDV0, val)
        assert int(tia.registers[W_AUDV0]) == val
