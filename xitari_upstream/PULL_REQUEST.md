# Make Superchip (on-cart RAM) initialisation deterministic / honour `random_seed`

Fixes #`<ISSUE-NUMBER-TBD>` (Superchip RAM is initialised non-deterministically,
ignoring `random_seed`).

## What

On-cart RAM carts (`F8SC`/`F6SC`/`F4SC`, and the same pattern in `FASC`/`E7`/`CV`)
fill their RAM in the constructor with `ale::Random::next()`, which — because of
the seeding order and `Random::ourSeeded` defaulting to `false` — is seeded from
`time(NULL)` even when a fixed `random_seed` is configured. This makes those carts
non-deterministic across runs. This PR makes the on-cart RAM init reproducible.

## Why

ALE advertises `random_seed`-based reproducibility. Today it does not hold for
Superchip titles: e.g. Elevator Action's attract demo reads uninitialised on-cart
RAM as an RNG and renders differently on every run, even with a pinned seed. The
console RIOT RAM is unaffected, so the bug is invisible to RAM-state comparisons
and only shows up in rendered frames / agent observations. (Found while validating
an independent re-implementation bit-for-bit against xitari — paper: arXiv
`<ARXIV-ID-TBD>`.)

## The change (recommended: honour `random_seed`)

Apply the configured `random_seed` to `ale::Random` **before** the cartridge is
constructed, so on-cart RAM is seeded from the user's seed (deterministic, and
user-controllable). Concretely: move the `Random::seed(settings.getString(
"random_seed"))` logic so it runs before `Cartridge::create(...)` /
`createConsole(...)`, rather than partway through `Console::Console`.

## Minimal alternative (what we apply locally)

If a behaviour change to `random_seed` semantics is undesirable, the smallest fix
is to give the static RNG a deterministic default so the pre-`Console` init no
longer uses `time(NULL)`:

```diff
--- a/emucore/Random.cxx
+++ b/emucore/Random.cxx
-bool Random::ourSeeded = false;
+bool Random::ourSeeded = true;   // deterministic default: pre-Console Random
+                                 // (on-cart RAM init) uses ourSeed (0), not time(NULL)
```

and (optionally) change the default seed from wall-clock to a fixed value:

```diff
--- a/common/Defaults.cpp
+++ b/common/Defaults.cpp
-    settings.setString("random_seed", "time");
+    settings.setString("random_seed", "0");
```

(Real hardware leaves on-cart RAM undefined at power-on, so a fixed, documented
seed is a strictly better default than wall-clock time for a reproducible
emulator.)

## Verification

- With the fix, dumping the on-cart RAM read window (`$1080–$10FF`) right after
  construction is **stable across runs** (and follows `random_seed`).
- Elevator Action's attract demo becomes reproducible frame-for-frame.
- No change to non-Superchip ROMs, and no change to the 128-byte console RIOT RAM
  for any ROM (the seed only feeds on-cart RAM init here). In our independent
  conformance suite, fixing this took the full 64-game ROM set from 63/64 to
  **64/64 pixel-exact** with zero regressions, and made the suite fully
  reproducible.

## Alternatives considered

- *Zero-fill on-cart RAM:* deterministic, but diverges from the existing
  random-fill convention and from any tooling that depends on it; honouring
  `random_seed` is the smaller behavioural change.
- *Re-init RAM in `reset()`:* doesn't address the root (default seed +
  construction order) and would change reset semantics.
