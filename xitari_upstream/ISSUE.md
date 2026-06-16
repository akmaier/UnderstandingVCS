# Superchip (on-cart RAM) is initialised non-deterministically, ignoring `random_seed`

## Summary

Cartridge types with on-cart RAM — `CartridgeF8SC`, `CartridgeF6SC`,
`CartridgeF4SC` (and the same pattern in `CartridgeFASC`, `CartridgeE7`,
`CartridgeCV`) — fill their RAM in the **constructor** with `ale::Random::next()`.
Because of the seeding order, that RAM ends up seeded from **`time(NULL)`** even
when the user sets a fixed `random_seed`. As a result, the on-cart RAM is
**non-deterministic across runs**, and games that read it before initialising it
render **differently on every run** — breaking ALE's reproducibility guarantee for
those titles.

## Affected code

- `emucore/CartF8SC.cxx` (ctor): fills `myRAM[0..127]` with `random.next()`.
  `CartridgeF8SC::reset()` only does `bank(1)` — it does **not** re-initialise RAM.
  Same construction pattern in `CartF6SC.cxx`, `CartF4SC.cxx`, `CartFASC.cxx`,
  `CartE7.cxx`, `CartCV.cxx`.
- `emucore/Random.cxx`: `ourSeeded` defaults to **`false`**, so the first `Random`
  instance constructed falls back to `ourSeed = (uInt32)time(0)` (the ctor's
  "random seed" branch).
- `common/ale_interface.cpp` / `emucore/Console.cxx`: the cartridge is constructed
  during `createConsole(...)` **before** the configured `random_seed` is applied to
  `ale::Random` (`Console::Console` calls `Random::seed(settings.getString(
  "random_seed"))`, but the on-cart RAM has already been filled by then — and the
  default `random_seed` is `"time"` anyway, see `common/Defaults.cpp`).

## Reproduction

With a Superchip ROM (e.g. **Elevator Action**, an `F8SC` cart) and a **fixed**
`random_seed`:

```cpp
ALEInterface ale(rom);                 // random_seed pinned to a constant
// dump the on-cart RAM read window $1080–$10FF right after construction:
ale::System &sys = ale.osystem().console().system();
for (int i = 0; i < 128; ++i) printf("%02x ", sys.peek(0x1080 + i));
```

Running this twice (across a one-second boundary) prints **different** RAM, despite
the fixed `random_seed`. Equivalently, stepping Elevator Action's attract-mode
demo ~40 frames and dumping the framebuffer yields **different screens** on
different runs: the demo reads uninitialised Superchip RAM (`LDA $F0D2`, i.e. SC
byte `0x52`) and uses it as a cheap RNG.

A within-second re-run *matches* (the `time(0)` seed is stable to the second),
which is the tell-tale of a `time(NULL)`-based seed.

## Root cause

Two compounding issues:

1. **`Random::ourSeeded` defaults to `false`** → the first `Random()` constructed
   (the cartridge RAM init) seeds itself from `time(NULL)`.
2. **Seeding order:** the cartridge (and thus its RAM init) is constructed before
   `random_seed` is applied to `ale::Random`. So even a fixed `random_seed` does
   not make the on-cart RAM deterministic.

The 128-byte console (RIOT) RAM is unaffected, so this is **invisible to
RAM-state checks** — it only surfaces in rendered frames (and any agent-visible
behaviour) for the subset of ROMs that read uninitialised Superchip RAM.

## Impact

- **Reproducibility:** `random_seed` does not fully determinise emulation for
  Superchip titles. Two runs with the same seed can produce different observations.
- **Conformance / re-implementations:** we hit this while validating an independent
  differentiable re-implementation bit-for-bit against xitari (see paper, arXiv:
  `<ARXIV-ID-TBD>`). Every other ALE game we tested is deterministic; Elevator
  Action was the lone exception, traced to this.

## Proposed fix

Make the on-cart RAM init deterministic and honour `random_seed`. See the
accompanying pull request (`PULL_REQUEST.md`): apply the configured `random_seed`
to `ale::Random` **before** the cartridge is constructed, and/or default
`Random::ourSeeded = true` so the pre-`Console` init uses a fixed `ourSeed` rather
than `time(NULL)`. (Real hardware leaves this RAM in an undefined state, so any
*fixed, documented* seed is an improvement over wall-clock time for an emulator
that advertises `random_seed`-based reproducibility.)
