# Understanding VCS: XAI for the Atari Simulator via Differentiable Emulation

## Project Overview

This project asks: **can modern XAI methods produce a hierarchical, mechanistic understanding of the Atari 2600 VCS — a system whose ground truth we fully possess?** It is directly inspired by Jonas & Kording's "Could a Neuroscientist Understand a Microprocessor?" (2017), but inverts the usual XAI experiment.

The conventional approach uses XAI to probe a black-box neural network (e.g., a DQN agent). This project flips the target: **the simulator itself becomes the XAI subject**. The DQN is, at most, a behaviour policy that drives the simulator into interesting states — we are not trying to explain the DQN.

To make XAI methods applicable to a classical hand-written C++ emulator, we are building **two differentiable, end-to-end ports of xitari** — one in **JAX (Python)** and one in **Julia**. Once the simulator is differentiable, gradient-based attribution, concept probing, mechanistic interpretability, and ablation analyses can be turned on it directly. Because we already know the true hierarchy (6507 CPU → TIA → RIOT → cartridge bank-switching → console), every XAI claim is testable against ground truth.

### Conceptual framing of the differentiable simulator

- **ROM as a hardwired neural network.** A cartridge ROM is a fixed bit pattern that the CPU "executes." Read as a tensor, it is the weight matrix of a network whose forward pass is one machine cycle. Backpropagating into ROM bytes asks "which bits explain this pixel?"
- **RAM as a Neural-Turing-Machine-like tape.** The 128 B of VCS RAM is small enough to carry as a differentiable state vector with soft (attention-style) read/write addressing. Unlike a vanilla NTM, the addressing is sometimes hard (direct, indexed) — we relax those into convex combinations only where gradients must flow.
- **Branches and case-selects as soft switches.** `if flag then PC+=offset` and opcode-indexed dispatch become gated mixtures (softmax / Gumbel-softmax / straight-through estimators). Forward behaviour stays bit-exact when the gates are saturated; gradients flow when they are relaxed.

This is the lever that lets us aim XAI tools at a system we already understand bit-for-bit.

---

## Current status

Both ports run a full VCS — CPU + Bus + RAM + TIA + RIOT + cart (2K, 4K, F8, F6, F4, F8SC, F6SC, F4SC, E0) — wrapped in an ALE-style `StellaEnvironment` (`reset` / `step(action)` / `get_screen` / `get_ram`). All **151 documented NMOS 6502 opcodes** + USBC + **37 common undocumented opcodes** (NOP/LAX/SAX) are implemented in both ports and in both execution modes (**HARD** = bit-exact uint8 dispatch, **SOFT** = differentiable float32 dispatch). The TIA renders playfield, both player sprites with NUSIZ multi-copy + 2×/4× scaling + VDELP shadow, both missiles, the ball with VDELBL, and all 8 collision latches; drives the framebuffer through VSYNC / VBLANK / VBLANK-output-blanking; supports HMOVE positioning + the floating-bus quirk on TIA reads. RIOT has the timer (4 prescalers, with INTIM-read-clears-flag P4d semantics) + I/O ports (SWCHA/SWCHB + DDRs, 2-player joystick wiring). Differentiability primitives (`RomTensor`, `soft_select`, `soft_memory_read`, `soft_branch`, straight-through round/clamp) are integrated end-to-end via `soft_step` / `soft_run` / `soft_run_scan`, with full Zygote support in jutari via the functional `_FUNC_HANDLERS` table.

**~1950 tests green** across the two ports (jaxtari ~780 + jutari ~1170). **PXC2** (jaxtari ≡ jutari cross-check) covers 6 ROMs — pong, breakout, space invaders, pitfall, seaquest, enduro — with byte-identical RAM between the two ports at every frame. **PXC1** (jaxtari↔xitari RAM, `*_noop_10` last frame): **pong, breakout, space_invaders bit-exact (0 bytes)**, seaquest 4, pitfall 19, enduro 43 (= the pre-P3i-g parent baseline; the original "29" pin was stale). **PXC-S** (a new test the P3i-g pt4 work added — per-frame *screen* diff vs xitari): breakout **0 px (BIT-EXACT)** (was 8 before pt6 defer-ENAM/ENABL), pong **568 px** (was 920 before pt5 SCOREMODE), others 1700–3940 px (pre-existing rendering-accuracy gaps now pinned and chased one by one). See [bug_fix_log.md](bug_fix_log.md) for the scoreboards and the running history of fixes.

For the per-phase commit ledger, what each port can do today, and the complete list of deferrals, see **[STATUS.md](STATUS.md)**. For the design rationale and the still-pending phase plan, see **[PORTING_PLAN.md](PORTING_PLAN.md)**.

> **Agents working on emulation / conformance:** read **[bug_fix_log.md](bug_fix_log.md)** first — it's the running history of bugs hunted, patches landed, dead-ends ruled out (e.g. the `bisect.py` stdlib-shadow gotcha), and ideas still open (e.g. enduro collision-timing convergence). **Append to it** whenever you fix a bug or rule out a hypothesis, so the next agent inherits the context. It also carries the live jaxtari↔xitari conformance scoreboard.

---

## Repository Structure

```
UnderstandingVCS/
├── README.md              # This file
├── PORTING_PLAN.md        # Phase plan + design rationale
├── STATUS.md              # Per-phase commit/test/deferral ledger
├── bug_fix_log.md         # Running bug/patch history + open debugging ideas (agents: read & update)
├── .gitignore             # Excludes papers/, dqn/, xitari/ (external deps) and .DS_Store
├── literature/            # AI-readable markdown versions of papers with BibTeX (13 papers)
├── jaxtari/               # JAX port — see jaxtari/README.md
├── jutari/                # Julia port — see jutari/README.md
├── tools/                 # trace_dump.cpp sketch (xitari conformance helper, not built yet)
├── papers/                # PDF downloads (excluded from git, reproducible via DOIs)
├── dqn/                   # DeepMind DQN repository clone (excluded — used as black-box agent)
└── xitari/                # DeepMind Xitari (ALE fork) — the bit-exact reference (excluded)
```

`xitari/`, `dqn/`, and `papers/` are external dependencies, cloned/downloaded locally and not version-controlled here. `jaxtari/` and `jutari/` are the primary deliverables of this project and **are** version-controlled.

---

## Rules / Working Setup

### Developer Log Book via Commits

**Every command, change, or action performed in this project is committed and pushed to GitHub immediately after each turn.** The commit history serves as a complete developer log.

### Commit Message Format

Each commit message MUST include:
- **The full user prompt** that triggered the changes
- **The AI model used** (e.g., `claude-opus-4-7[1m]`)
- **A concise summary** of what was changed and why

This ensures reproducibility and full traceability.

### Literature Management

1. **papers/**: Downloaded PDFs. Excluded from git (binary, reproducible via URLs).
2. **literature/**: Markdown conversions with full text and a closing BibTeX block. Tracked in git.

### External Dependencies

The following live locally but are **excluded from git**:
- `dqn/` — DeepMind's DQN implementation (Lua/Torch). Used here as an *optional* black-box action source.
- `xitari/` — DeepMind's ALE fork. The **reference oracle** that the differentiable ports must match cycle-by-cycle.

---

## Key Papers and Their Role

### 1. Could a Neuroscientist Understand a Microprocessor? (Jonas & Kording, 2017)

**DOI**: [10.1371/journal.pcbi.1005268](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005268)

**Core insight**: Standard neuroscience methods applied to a fully simulated MOS 6502 — with perfect data and ground truth — failed to recover the processor's hierarchical organisation. The bottleneck was the methods, not the data.

**Relevance**: This is our foundational paper. Where Jonas & Kording asked whether neuroscience methods could understand a microprocessor running an Atari game, we ask whether **XAI methods can understand the whole VCS** (CPU + TIA + RIOT + cart + ROM) once we expose it through gradients. The 6507 / VCS is our "known artifact." Any failure of XAI here is informative regardless of outcome.

### 2. Human-level Control through Deep Reinforcement Learning (Mnih et al., 2015)

**DOI**: [10.1038/nature14236](https://www.nature.com/articles/nature14236) · **arXiv**: [1312.5602](https://arxiv.org/abs/1312.5602)

**Relevance to the *new* project goal**: The DQN is **not** the XAI target. We use it (or any other agent) as a *policy* that drives the simulator into game-relevant trajectories, so that the XAI we run on the simulator is conditioned on realistic state distributions rather than random play. The original "explain the DQN" framing is preserved in the bibliography for context, but is no longer the research question.

---

## DQN and Xitari Breakdown

### DQN (Deep Q-Network)

**Source**: `https://github.com/google-deepmind/dqn`. Lua/Torch implementation of the Nature 2015 DQN. Treated here as a black-box action source. We do not modify or instrument it.

### Xitari (Arcade Learning Environment fork)

**Source**: `https://github.com/google-deepmind/xitari`. C++ Atari 2600 emulator based on Stella, with an ALE-style RL interface.

Top-level layout that the ports mirror:

| xitari path | Role | Port target |
|---|---|---|
| `emucore/m6502/src/` | 6502/6507 CPU (M6502, M6502Hi/Low, System) | `jaxtari/cpu/`, `jutari/src/cpu/` |
| `emucore/TIA.{cxx,hxx}` | Television Interface Adapter (video + audio) | `jaxtari/tia/`, `jutari/src/tia/` |
| `emucore/M6532.{cxx,hxx}` | RIOT: 128 B RAM + I/O + timer | `jaxtari/riot/`, `jutari/src/riot/` |
| `emucore/Cart*.{cxx,hxx}` | Cartridge / bank-switching types | `jaxtari/cart/`, `jutari/src/cart/` |
| `emucore/Console.{cxx,hxx}` | Top-level VCS console wiring | `jaxtari/console.py`, `jutari/src/Console.jl` |
| `emucore/Control.cxx`, `Joystick`, `Paddles`, `Switches` | Controllers + console switches | `jaxtari/io/`, `jutari/src/io/` |
| `environment/` | ALE-style RL wrapper, phosphor blend | `jaxtari/env/`, `jutari/src/env/` |
| `games/` | Per-game ROM scoring/termination rules | `jaxtari/games/`, `jutari/src/games/` |
| `controllers/` | Agent IPC (fifo, internal, rlglue) | Out of scope for ports (use direct API) |
| `agents/` | Sample agents | Out of scope for ports |

The full module-by-module plan, including the differentiability layer, the bit-exact test harness against xitari, and milestone phasing, is in **[PORTING_PLAN.md](PORTING_PLAN.md)**.

### Why Xitari is the right reference

Xitari is deterministic given a ROM, an action stream, and an RNG seed. That makes it a perfect oracle: every CPU register, every RAM byte, every TIA register, and every frame buffer the JAX/Julia ports produce can be compared byte-for-byte against xitari's trace for the same inputs. Disagreement is unambiguously a port bug.

---

## Project Plan and Hypotheses

### Core Hypothesis

> **A complete, differentiable port of the Atari VCS — where ROMs act as weights, RAM acts as a soft tape, and control flow acts as gated switches — lets us turn modern XAI methods on a system whose true mechanism is fully known. The mismatch between XAI explanations and the verified ground-truth hierarchy will quantify the methods' limits more sharply than any biological or neural-network target can.**

### Specific Questions

1. Can gradient-based attribution (Integrated Gradients, Grad-CAM analogues) localise the ROM bytes / RAM cells / TIA registers that "explain" a given pixel or score change?
2. Do concept probes (CAVs, network dissection analogues) recover known hardware concepts — bank-switch bits, sprite registers, collision latches, timer reloads?
3. Does mechanistic / circuit-style interpretability reconstruct the true hierarchy (instruction fetch → decode → ALU → bus → TIA/RIOT → frame) when the gates are relaxed?
4. How do the JAX and Julia ports compare on (a) throughput vs. xitari and (b) gradient-pass cost? Are the two languages' AD systems equivalent for this workload?
5. Where do soft-switch relaxations leak — i.e., for which instructions/branches does the differentiable version diverge from the bit-exact reference, and by how much?

### Methodology

1. **Port xitari to JAX and Julia, bit-exactly**, validated against per-cycle xitari traces. (See PORTING_PLAN.md.)
2. **Layer differentiability on top**, with a flag that toggles "hard / bit-exact" mode vs. "soft / differentiable" mode.
3. **Drive trajectories** with either random play, scripted actions, or a pre-trained DQN.
4. **Apply XAI methods** to ROM bytes, RAM cells, TIA registers, and intermediate CPU state.
5. **Compare to ground truth**: 6502 datasheet, TIA documentation, disassembly of the target ROM, known cartridge bank-switching schemes.

### Expected Outcomes

- **Best case**: XAI methods cleanly recover the documented VCS hierarchy on at least one game, validating the methods on a transparent target.
- **Likely case**: Methods recover *parts* of the hierarchy (e.g., correctly attribute a sprite pixel to a TIA player register) but miss higher-level structure (e.g., the game's score logic in ROM).
- **Worst / most informative case**: XAI produces confident, plausible, but **wrong** explanations on a system we can verify — a stronger version of Jonas & Kording's negative result.

---

## Bibliography

```bibtex
@article{Jonas2017Could,
  author    = {Jonas, Eric and Kording, Konrad Paul},
  title     = {Could a Neuroscientist Understand a Microprocessor?},
  journal   = {PLOS Computational Biology},
  year      = {2017},
  volume    = {13},
  number    = {1},
  pages     = {e1005268},
  doi       = {10.1371/journal.pcbi.1005268},
  issn      = {1553-7358},
  publisher = {Public Library of Science},
  url       = {https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005268}
}

@article{Mnih2015Human,
  author    = {Mnih, Volodymyr and Kavukcuoglu, Koray and Silver, David and Rusu, Andrei A. and Veness, Joel and Bellemare, Marc G. and Graves, Alex and Riedmiller, Martin and Fidjeland, Andreas K. and Ostrovski, Georg and Petersen, Stig and Beattie, Charles and Sadik, Amir and Antonoglou, Ioannis and King, Helen and Kumaran, Dharshan and Wierstra, Daan and Legg, Shane and Hassabis, Demis},
  title     = {Human-level control through deep reinforcement learning},
  journal   = {Nature},
  year      = {2015},
  volume    = {518},
  number    = {7540},
  pages     = {529--533},
  doi       = {10.1038/nature14236},
  publisher = {Nature Publishing Group},
  url       = {https://www.nature.com/articles/nature14236}
}

@article{Bellemare2013Arcade,
  author    = {Bellemare, Marc G. and Naddaf, Yavar and Veness, Joel and Bowling, Michael},
  title     = {The {Arcade} {Learning} {Environment}: An Evaluation Platform for General Agents},
  journal   = {Journal of Artificial Intelligence Research},
  year      = {2013},
  volume    = {47},
  pages     = {253--279}
}

@article{Graves2014NTM,
  author    = {Graves, Alex and Wayne, Greg and Danihelka, Ivo},
  title     = {Neural {Turing} {Machines}},
  journal   = {arXiv preprint arXiv:1410.5401},
  year      = {2014},
  url       = {https://arxiv.org/abs/1410.5401}
}
```

---

*This project is developed with AI assistance. Every change is committed and pushed to GitHub with full prompt and model documentation.*
