# Phase A — Jonas & Kording battery, scored against ground truth (spec)

Replay each neuroscience method on our **architectural state** and report both the
Kording-style qualitative result and a **quantitative agreement with the known
mechanism**. Games: Donkey Kong, Space Invaders, Pitfall (their set) + a few of
our 64. Parallel track: the Visual6502 transistor netlist for a direct head-to-head
(scope decided after pilots).

State variables ("units"): RAM bits (RIOT 128 B), CPU registers (A,X,Y,S,P,PC bits),
TIA/RIOT registers, opcodes, ROM bytes, framebuffer. We log full per-step state
trajectories (the "processor activity map").

| ID | Analysis | Our implementation | Ground-truth score |
|---|---|---|---|
| A1 | Connectomics | static read/write graph per opcode | recovered vs true data-flow graph (P/R) |
| A2 | Lesions | clamp each unit; boot/run? | lesion specificity vs the unit's *known* role |
| A3 | Tuning curves | toggle-rate vs pixel luminance | spurious-tuning rate vs true role |
| A4 | Correlations | across RAM/register bits | weak-pairwise/strong-global vs true coupling |
| A5 | LFP | pooled toggle activity + spectra | "rhythms" are the frame/scanline clocks (known) |
| A6 | Granger causality | CPU/TIA/RIOT subsystems | false-edge rate vs true data-flow |
| A7 | Dim-reduction (NMF/PCA) | full state tensor | components vs known signals (clock/RW/vsync) |
| A8 | Whole-state recording | full RAM+register map over time | descriptive baseline |

**Pilot** (`pilot_si.{py,jl}`, local): A2 + A3 + A7 on Space Invaders with
ground-truth scoring. Reuses Paper-1 state-dump tooling.
**Scale-out** (cluster, CPU array jobs): full lesion sweeps over all units × games.

Outputs: per-analysis `out/A<k>_<game>.*` + a scores table (method → fidelity).
