# Understanding VCS: XAI for Atari Game Neural Networks

## Project Overview

This project explores the limits of Explainable Artificial Intelligence (XAI) methods in understanding neural networks trained to play Atari games on the 6502 microprocessor. Inspired by the seminal paper "Could a Neuroscientist Understand a Microprocessor?" (Jonas & Kording, 2017), we examine whether modern XAI techniques can provide meaningful insights into deep reinforcement learning agents, or whether they fall short of true understanding—just as neuroscience methods failed to reveal the hierarchical structure of the 6502 processor.

Our goal is to test the hypothesis that **having unlimited data and state-of-the-art XAI tools is insufficient to achieve genuine understanding of a neural network's internal computations**. By applying various XAI methods (feature attribution, concept-based explanations, mechanistic interpretability, etc.) to a Deep Q-Network (DQN) trained on classic Atari games, we aim to identify the gaps between current XAI capabilities and the kind of understanding that would allow us to truly "fix" or redesign the network.

---

## Repository Structure

```
UnderstandingVCS/
├── README.md              # This file
├── .gitignore             # Git ignore rules (excludes papers/, dqn/, xitari/)
├── literature/            # AI-readable markdown versions of papers with BibTeX
│   ├── jonas_kording_2017_pcbi.md
│   └── mnih_2015_nature.md
├── papers/                # PDF downloads (excluded from git)
│   ├── jonas_kording_2017_pcbi.pdf
│   └── mnih_2015_nature.pdf
├── dqn/                   # DeepMind DQN repository clone (excluded from git)
└── xitari/                # DeepMind Xitari (ALE fork) repository clone (excluded from git)
```

---

## Rules / Instructions / Working Setup

### Developer Log Book via Commits

**Every command, change, or action performed in this project is committed and pushed to GitHub immediately after each turn.** The commit history serves as a complete developer log book.

### Commit Message Format

Each commit message MUST include:
- **The full user prompt** that triggered the changes
- **The AI model used** (e.g., `moonshotai/Kimi-K2.6`)
- **A concise summary** of what was changed and why

This ensures reproducibility and full traceability of all development steps.

### Literature Management

1. **papers/**: Downloaded PDFs of academic papers. **Excluded from git** (binary files, large, reproducible via URLs).
2. **literature/**: Markdown conversions of papers for AI readability.
   - Each `.md` file includes the full paper text
   - **A BibTeX citation block at the end** for easy referencing
   - Preserves figures and tables as best as possible
   - These files ARE tracked in git

### External Dependencies (Cloned Repositories)

The following repositories are cloned locally but **excluded from git**:
- `dqn/` - DeepMind's DQN implementation (Lua/Torch)
- `xitari/` - DeepMind's fork of the Arcade Learning Environment (ALE)

These are external dependencies; we reference them but do not modify or version them.

---

## Key Papers and Their Role

### 1. Could a Neuroscientist Understand a Microprocessor? (Jonas & Kording, 2017)

**DOI**: [10.1371/journal.pcbi.1005268](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005268)

**Core Insight**: The authors applied standard neuroscience analysis methods to a fully simulated MOS 6502 microprocessor running Atari games. They found that while these methods revealed interesting statistical structure in the data, they **failed to produce a meaningful, hierarchical understanding** of the processor's information processing. The paper argues that having perfect data and ground truth knowledge is insufficient—**the methods themselves are the bottleneck**.

**Relevance to Our Project**: This is our foundational paper. We ask: if neuroscience methods failed to understand a microprocessor, can XAI methods succeed in understanding a neural network running on that same microprocessor? The 6502 serves as our "known artifact" with ground truth. If XAI cannot meaningfully explain a DQN controlling an Atari game on the 6502, what does that tell us about XAI's limitations for understanding biological brains or more complex AI systems?

**Key Methods Tested in the Paper**:
- Connectomics (circuit motif analysis)
- Single-transistor lesion studies
- Single-unit tuning curves (pixel luminance)
- Pairwise correlation analysis / spike-word statistics
- Local field potential (LFP) analysis and spectral power
- Granger causality for functional connectivity
- Dimensionality reduction (NMF) on whole-processor recordings

**Finding**: All these methods, when applied naively, failed to reveal the true hierarchical organization (instruction fetcher → decoder → registers → ALU → memory I/O).

---

### 2. Human-level Control through Deep Reinforcement Learning (Mnih et al., 2015)

**DOI**: [10.1038/nature14236](https://www.nature.com/articles/nature14236)

**arXiv preprint**: [1312.5602](https://arxiv.org/abs/1312.5602)

**Core Insight**: DeepMind's DQN agent learns to play Atari 2600 games directly from raw pixel inputs, achieving human-level performance across 49 games using a single algorithm and architecture. The network uses **experience replay** and **target Q-networks** to stabilize training of a deep convolutional network with Q-learning.

**Relevance to Our Project**: This is the RL agent we will analyze with XAI methods. The DQN architecture is the "black box" we want to understand. Because we know both the network architecture AND the underlying 6502 processor it controls, we can compare XAI-derived explanations against ground truth.

**DQN Architecture Summary**:
- **Input**: 84×84×4 grayscale image stack (4 most recent frames)
- **Layer 1**: Conv2D(16 filters, 8×8, stride 4) + ReLU
- **Layer 2**: Conv2D(32 filters, 4×4, stride 2) + ReLU
- **Layer 3**: Fully connected (256 units) + ReLU
- **Output**: Linear layer with one output per valid action (Q-values)

**Training Algorithm** (Deep Q-learning with Experience Replay):
1. Initialize replay memory D (capacity N)
2. Initialize Q-network with random weights
3. For each episode:
   - Initialize state sequence s₁ = {x₁}, preprocess φ₁ = φ(s₁)
   - For each timestep t:
     - Select action via ε-greedy policy
     - Execute action, observe reward rₜ and next image xₜ₊₁
     - Store transition (φₜ, aₜ, rₜ, φₜ₊₁) in D
     - Sample random minibatch from D
     - Compute target: yⱼ = rⱼ for terminal states, else rⱼ + γ max Q(φⱼ₊₁, a'; θ⁻)
     - Perform gradient descent on (yⱼ - Q(φⱼ, aⱼ; θ))²
     - Periodically update target network weights θ⁻ ← θ

---

## DQN and Xitari Breakdown

### DQN (Deep Q-Network) Repository

**Source**: `https://github.com/google-deepmind/dqn`

**What it is**: The official implementation of the Nature 2015 DQN paper. Written in **Lua** using the **Torch 7** deep learning framework.

**Architecture** (from `dqn/dqn/convnet.lua`):
```lua
net = nn.Sequential()
net:add(nn.Reshape(input_dims))        -- reshape input
net:add(SpatialConvolution(...))       -- conv layer 1
net:add(ReLU())
net:add(SpatialConvolution(...))       -- conv layer 2
net:add(ReLU())
net:add(nn.Reshape(nel))               -- flatten
net:add(nn.Linear(nel, n_hid))         -- FC hidden layer
net:add(ReLU())
net:add(nn.Linear(n_hid, n_actions))   -- output Q-values
```

**Why DQN is suited for XAI analysis**:
- The network architecture is **fully known and relatively small**
- We can extract **intermediate activations** from any layer
- The input-output mapping is well-defined (pixels → Q-values)
- We know the **training objective** (Q-learning loss)
- We can perform **ablation studies** by modifying specific layers
- Feature attribution methods (Grad-CAM, SHAP, LIME) can be directly applied
- We can inspect **what the network "looks at"** in the input frame
- We can analyze **which features activate for specific game states**
- The network's behavior is measurable (action selection, Q-value predictions)

**DQN is a neural network artifact that XAI methods CAN analyze**.

---

### Xitari (Arcade Learning Environment Fork)

**Source**: `https://github.com/google-deepmind/xitari`

**What it is**: A fork of the Arcade Learning Environment (ALE) v0.4, modified by DeepMind for DQN experiments. It is an Atari 2600 emulator based on **Stella**, the open-source Atari emulator.

**Architecture** (from examining the source):
- **emucore/**: Core emulation engine (C++ classes for 6507 CPU, TIA graphics, RIOT I/O, cartridge types)
- **games/**: Supported game ROM metadata
- **controllers/**: Agent interface (fifo, internal, named pipes)
- **environment/**: ALE environment wrapper
- **common/**: Shared utilities
- **main.cpp**: CLI entry point

**Why Xitari is NOT suited for XAI analysis**:

Unlike DQN, Xitari is **not a neural network**—it is a **classical software emulator** with the following characteristics that make it unsuitable for standard XAI methods:

1. **No Learned Representations**: Xitari is a hand-written C++ emulator implementing the 6502/6507 CPU, TIA chip, and Atari hardware. It contains **no trainable parameters, no weights, no gradients**.

2. **Deterministic Logic**: The emulator follows fixed hardware rules (CPU instruction cycles, memory maps, video timing). There is no "prediction" or "inference" to explain—behavior is fully specified by the hardware design.

3. **No Feature Hierarchy**: While the 6502 has a logical hierarchy (ALU, registers, decoder), these are **explicit hardware modules**, not learned feature detectors. XAI methods like Grad-CAM or SHAP rely on analyzing how a model's internal representations contribute to predictions—Xitari has no such representations.

4. **Binary State Space**: The emulator operates on discrete binary states (transistors on/off, memory values, instruction opcodes). XAI methods designed for continuous neural activations do not apply.

5. **Ground Truth is Manual Inspection**: To "understand" Xitari, one reads the C++ source code or the hardware datasheets. There is no black-box model to explain.

6. **No Input-Output Mapping to Explain**: Xitari takes actions as input and produces video frames as output, but this is a direct forward simulation of hardware, not a learned mapping.

7. **Different Abstraction Level**: XAI targets "why did the neural network choose action A?" Xitari answers "because the CPU executed instruction X at cycle Y, which wrote value Z to memory address W." The explanation is at the execution trace level, not the representation level.

**In summary**: Xitari is the **environment/simulator**, not the **agent**. XAI explains the agent (DQN), not the environment. To "explain" Xitari, one uses reverse engineering, static analysis, and code inspection—methods from computer engineering, not XAI.

---

## Project Plan and Hypotheses

### Core Hypothesis

> **Modern XAI methods, despite their sophistication, will fail to produce a satisfying hierarchical understanding of the DQN's internal computations, just as neuroscience methods failed for the 6502 processor.**

### Specific Questions

1. Can feature attribution methods (Grad-CAM, Integrated Gradients) reveal what the DQN "pays attention to"? Do these explanations generalize across game states?

2. Can we identify "concepts" in the DQN's hidden layers (e.g., "ball location", "paddle position") using concept activation vectors (CAVs) or network dissection?

3. Can mechanistic interpretability methods (circuit tracing, attribution patching) reveal the computational graph the DQN uses to choose actions?

4. Do XAI explanations align with what we know about the game structure (e.g., does the DQN really use the ball's trajectory to predict future rewards)?

5. Can we "break" the DQN in a controlled way and see if XAI predicts the failure? (E.g., lesion specific channels and check if XAI predicted their importance.)

6. How much does the explanation quality depend on the amount of training data? (Jonas & Kording had unlimited data for the 6502; does more data help XAI?)

### Proposed Methodology

1. **Train or load a pre-trained DQN** on one or more Atari games
2. **Apply diverse XAI methods**:
   - Input attribution: Grad-CAM, SHAP, LIME
   - Hidden layer analysis: Network dissection, CAVs
   - Mechanistic: Circuit tracing, attention rollout
   - Counterfactual: Input manipulation, adversarial examples
3. **Compare explanations to ground truth**:
   - What does the game require? (e.g., tracking moving objects)
   - What does the DQN actually do? (layer activation patterns)
   - What do XAI methods say? (saliency maps, concept rankings)
4. **Evaluate if any XAI method provides a satisfying hierarchical understanding** of how pixels → features → Q-values → actions

### Expected Outcomes

- **Best case**: Some XAI method reveals clear, hierarchical, verifiable structure in the DQN (e.g., early layers detect sprites, middle layers track trajectories, late layers estimate value)
- **Likely case**: XAI reveals interesting but incomplete structure—some useful insights, but no full understanding
- **Worst case (but most informative)**: XAI produces plausible-sounding but misleading explanations, highlighting a fundamental gap between correlation-based explanation and true mechanistic understanding

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
```

---

*This project is developed with AI assistance. Every change is committed and pushed to GitHub with full prompt and model documentation.*
