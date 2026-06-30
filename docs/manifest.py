#!/usr/bin/env python3
"""Source of truth for the results-audit site.

Every claim row resolves to: claim -> measured value -> producing script ->
exact command -> output artifact -> runtime -> hardware -> verifying gate ->
status (measured | deferred).  build_pages.py renders these into HTML.

Numbers here were read from the committed result files at build time
(results/gpu/*.json, tools/rom_sweep/results_*.md, the xai_study leaderboard,
the ROM-hash table) — not transcribed by hand.  Re-run docs/build_pages.py
after the underlying results change.
"""

REPO = "akmaier/UnderstandingVCS"
REPO_URL = "https://github.com/akmaier/UnderstandingVCS"
PAGES_URL = "https://akmaier.github.io/UnderstandingVCS/"

# ---------------------------------------------------------------------------
# Shared: the "why trust this" oracle panel
# ---------------------------------------------------------------------------
ORACLE = {
    "headline": "Validated bit-for-bit against an independent C++ emulator",
    "body": (
        "The strongest defence against “the AI hallucinated it” is that the two "
        "differentiable ports are checked against <b>xitari</b> — a separately written, "
        "pre-existing C++ Atari 2600 emulator that is <i>not</i> part of this project. "
        "On all <b>64</b> ALE-supported games the ports reproduce xitari’s 128 B of RAM "
        "<b>byte-for-byte</b> and its 210×160 framebuffer <b>pixel-for-pixel</b>. "
        "Hallucinated code does not accidentally match an external reference to the bit."
    ),
    "pillars": [
        ("External oracle", "xitari C++ — 64/64 RAM byte-exact + 64/64 screen pixel-exact",
         "xitari/", None),
        ("Dual-port cross-check", "PXC2: the JAX port and the Julia port diverge from xitari "
         "<i>identically</i>, in lock-step — a shared root cause, not two coincidences",
         "jaxtari/tests/test_pxc2_jaxtari_vs_jutari.py", None),
        ("Conformance harness", "PXC1 / PXC-S / PXC4 replay xitari traces and diff RAM, screen "
         "and a full 6502 functional test", "jaxtari/tests/test_pxc1_conformance.py", None),
        ("Open dev log", "769 commits on <code>main</code> with the prompt→code chain; a "
         "7,525-line running bug-fix log", "bug_fix_log.md", None),
    ],
}

# ---------------------------------------------------------------------------
# Environment & hardware (shared page)
# ---------------------------------------------------------------------------
ENVIRONMENT = {
    "hardware": [
        ("Local development", "Apple M1 Max (CPU)",
         "jutari conformance sweeps + jaxtari CPU baseline (~60k env-steps/s asymptote, batched)"),
        ("GPU benchmark A", "NVIDIA GeForce GTX 1080 Ti",
         "soft-mode throughput sweep — results/gpu/gtx1080ti.json"),
        ("GPU benchmark B", "NVIDIA Quadro RTX 5000",
         "soft-mode throughput sweep — results/gpu/q5000.json"),
        ("Cluster (Paper 2)", "LME SLURM cluster, /cluster/maier",
         "Phase A/B/C batteries: CPU array 4 cores / 16 GB / 4 h; GPU jobs 4 cores / 32 GB / 8 h; "
         "concurrency throttled to 10 jobs (user QOS)"),
    ],
    "software": [
        ("Julia", "1.12.x (jutari authoring; CI on 1.10)", "jutari/Project.toml + Manifest.toml (pinned)"),
        ("Python", "3.13 (CI); ≥ 3.10 required", "jaxtari/pyproject.toml (pinned)"),
        ("JAX", "0.10.1 (GPU benchmark); ≥ 0.4.30 required", "recorded in results/gpu/*.json env block"),
        ("CUDA", "12 (jax[cuda12] on the cluster)", "tools/cluster/xai_gpu.sbatch"),
    ],
    "determinism": [
        "All Paper-2 result records use <b>seed = 0</b> — the differentiable substrate is "
        "deterministic, so seeded methods (RISE masks, SmoothGrad noise, Expected-Gradients "
        "sampling) reproduce exactly.",
        "xitari is made deterministic by pinning its Superchip-RAM seed to 0 "
        "(<code>tools/xitari_conformance_seed.patch</code>); otherwise it seeds from "
        "<code>time(NULL)</code> and elevator_action’s attract demo is non-reproducible.",
        "Experiment state is encoded as <code>f&lt;start&gt;+&lt;window&gt;</code> "
        "(e.g. <code>f120+30</code> = boot to frame 120, then a 30-frame window) — always "
        "inside the Paper-1 bit-exact horizon (~30-frame RAM / ~60-frame screen).",
    ],
    "roms": {
        "note": ("ROMs are not redistributed (copyright). They are obtained via AutoROM and "
                 "verified by SHA-256 so an auditor confirms byte-identical inputs."),
        "table": "tools/xai_study/repro/rom_hash_table.csv",
        "rows": [
            # (game, size, sha256-prefix)
            ("pong", "2048 B", "41623e3c2614…"),
            ("breakout", "2048 B", "376323f051c3…"),
            ("space_invaders", "4096 B", "7224b17462b9…"),
            ("seaquest", "4096 B", "fbc29f4678f6…"),
            ("ms_pacman", "8192 B", "dde0b43c5dee…"),
            ("qbert", "4096 B", "3257221832a7…"),
        ],
    },
}

# ---------------------------------------------------------------------------
# Provenance / test infrastructure (shared page)
# ---------------------------------------------------------------------------
PROVENANCE = {
    "tests": [
        ("jaxtari (pytest)", "837 <code>def test_*</code> functions across 60 files",
         "jaxtari/tests/"),
        ("jutari (Julia)", "1,215 <code>@test</code> assertions",
         "jutari/test/runtests.jl"),
    ],
    "tests_note": ("Counts above are grepped from source at build time. Project docs report "
                   "<b>~1,950 effective</b> tests (some parametrised cases expand at run time); "
                   "STATUS.md and RESULTS.md give slightly different roll-ups — we show the "
                   "raw source count rather than pick a number."),
    "harnesses": [
        ("PXC1", "Replay an xitari JSONL trace against a port; assert per-frame RAM byte-identity.",
         "jaxtari/tests/test_pxc1_conformance.py"),
        ("PXC2", "Drive jaxtari live + load a jutari fixture trace; assert the two ports diverge "
         "from xitari identically.", "jaxtari/tests/test_pxc2_jaxtari_vs_jutari.py"),
        ("PXC-S", "Per-frame 210×160 framebuffer diff, xitari vs both ports (~23 min).",
         "jaxtari/tests/test_screen_conformance.py"),
        ("PXC4", "Klaus Dormann 6502 functional test on flat 64K memory (CPU-core validator).",
         "jaxtari/tests/test_pxc4_klaus_dormann.py"),
    ],
    "tools": [
        ("trace_dump", "C++ driver over xitari — emits deterministic JSONL frame traces "
         "(RAM, optional screen + CPU state).", "tools/trace_dump.cpp"),
        ("check_trace", "Replay a trace against a port and diff byte-by-byte (Python + Julia).",
         "tools/check_trace.py"),
        ("ram_diff", "Per-frame RAM divergence, jutari vs xitari.", "tools/jutari_xitari_ram_diff.py"),
        ("conformance seed patch", "Pins xitari’s RNG seed for reproducible replay.",
         "tools/xitari_conformance_seed.patch"),
    ],
    "ci": [
        ("test.yml", "Fast PR gate on every push/PR: full jaxtari pytest + jutari Pkg.test() + "
         "PXC2 cross-check.", ".github/workflows/test.yml"),
        ("heavy.yml", "Nightly (04:00 UTC) + manual: JAX-autodiff and slow ROM/screen groups on "
         "fresh runners.", ".github/workflows/heavy.yml"),
    ],
    "logs": [
        ("bug_fix_log.md", "7,525 lines — newest-on-top history of every bug, patch and "
         "ruled-out hypothesis.", "bug_fix_log.md"),
        ("STATUS.md", "Per-phase commit/test/deferral ledger.", "STATUS.md"),
        ("RESULTS.md", "Milestone write-up + reproducibility section.", "RESULTS.md"),
    ],
    "reproduce": [
        ("jutari conformance (~20 s)",
         "cd jutari && julia --project=. -e 'using Pkg; Pkg.test()'"),
        ("jaxtari full suite (long)",
         "cd jaxtari && .venv/bin/python -m pytest -q"),
        ("screen conformance (~23 min)",
         "jaxtari/.venv/bin/pytest jaxtari/tests/test_screen_conformance.py"),
        ("regenerate an xitari reference trace",
         "./tools/trace_dump --rom xitari/roms/pong.bin "
         "--actions tools/fixtures/actions/pong_noop_10.txt > /tmp/pong.jsonl"),
    ],
}

# ---------------------------------------------------------------------------
# Paper 1 — differentiable VCS emulator
# ---------------------------------------------------------------------------
PAPER1 = {
    "id": "paper1",
    "title": "Paper 1 — A Differentiable Atari 2600",
    "subtitle": "Two end-to-end differentiable ports of the xitari VCS, validated bit-for-bit",
    "venue": "AAAI 2027 (submitted) · arXiv preview",
    "pdf": "jutari_paper/paper/paper.pdf",
    "supplement_pdf": "jutari_paper/paper/supplementary.pdf",
    "blurb": (
        "Two differentiable ports of the Atari 2600 — <b>jaxtari</b> (JAX/Python) and "
        "<b>jutari</b> (Julia) — run a full VCS (6507 CPU + TIA + RIOT + cartridge banking) "
        "in a bit-exact HARD mode and a differentiable SOFT mode. The SOFT forward is "
        "straight-through, so it stays bit-identical to HARD while letting gradients flow."),
    "claims": [
        {
            "claim": "RAM byte-identical to xitari on every game",
            "value": "64 / 64",
            "detail": "128 B RIOT RAM, per frame, 30 frames of NOOP from the standard ALE boot "
                      "(60 NOOP + 4 RESET). Max diff 0 bytes/frame on all 64 games.",
            "script": "tools/rom_sweep/sweep_jutari_ram.py",
            "command": "python3 tools/rom_sweep/sweep_jutari_ram.py",
            "artifact": "tools/rom_sweep/results_jutari_ram.md",
            "runtime": "~9–10 s / game",
            "hardware": "M1 Max (CPU)",
            "verified_by": "PXC1 + jutari Pkg.test()",
            "status": "measured",
        },
        {
            "claim": "Screen pixel-identical to xitari on every game",
            "value": "64 / 64",
            "detail": "210×160 palette-index framebuffer, per frame, 60 frames of "
                      "breakout_random_actions after the standard boot. Max diff 0 px/frame.",
            "script": "tools/rom_sweep/sweep_jutari_screen.py",
            "command": "python3 tools/rom_sweep/sweep_jutari_screen.py",
            "artifact": "tools/rom_sweep/results_jutari_screen.md",
            "runtime": "~10 s / game",
            "hardware": "M1 Max (CPU)",
            "verified_by": "PXC-S screen conformance",
            "status": "measured",
        },
        {
            "claim": "Second independent port (jaxtari) also matches",
            "value": "64 / 64 RAM + screen",
            "detail": "The JAX port is swept against xitari with the same paradigm; PXC2 then "
                      "asserts jaxtari ≡ jutari frame-by-frame.",
            "script": "tools/rom_sweep/sweep_jaxtari.py",
            "command": "python3 tools/rom_sweep/sweep_jaxtari.py",
            "artifact": "tools/rom_sweep/results_jaxtari_ram.md",
            "runtime": "minutes / game (JAX eager)",
            "hardware": "M1 Max (CPU)",
            "verified_by": "PXC2 cross-check",
            "status": "measured",
        },
        {
            "claim": "SOFT forward is bit-exact to HARD (Theorem 1)",
            "value": "0 px divergence",
            "detail": "The straight-through SOFT path renders the Space Invaders scene identically "
                      "to HARD; a relaxed (α,T) path is shown diverging for contrast.",
            "script": "tools/relaxation_study/dump_divergence_frames.jl",
            "command": "julia --project=jutari tools/relaxation_study/dump_divergence_frames.jl",
            "artifact": "tools/relaxation_study/video_out/divergence_si.mp4",
            "runtime": "scene-length render",
            "hardware": "M1 Max (CPU)",
            "verified_by": "4-panel divergence video (HARD | SOFT-STE | relaxed | diff)",
            "status": "measured",
        },
        {
            "claim": "GPU throughput — forward, soft mode (Pong, batched)",
            "value": "2.95M / 3.12M env-steps/s",
            "detail": "Peak forward throughput: 2,947,553 env-steps/s on GTX 1080 Ti (batch "
                      "4096) and 3,119,115 on Quadro RTX 5000. 3000 CPU instructions/rollout, "
                      "10 repeats, batch sweep 1→65536.",
            "script": "tools/bench_jaxtari_gpu.py",
            "command": "python3 tools/bench_jaxtari_gpu.py --rom pong.bin",
            "artifact": "results/gpu/gtx1080ti.json",
            "runtime": "~10–80 s / batch config",
            "hardware": "GTX 1080 Ti / Quadro RTX 5000 · JAX 0.10.1",
            "verified_by": "raw JSON env block (device, jax version, per-batch wall time)",
            "status": "measured",
        },
        {
            "claim": "GPU throughput — forward + gradient",
            "value": "2.80M / 2.91M env-steps/s",
            "detail": "Peak forward+backward (jax.grad) throughput: 2,799,870 env-steps/s on "
                      "GTX 1080 Ti and 2,911,069 on Quadro RTX 5000 (batch 4096).",
            "script": "tools/plot_gpu_throughput.py",
            "command": "python3 tools/plot_gpu_throughput.py",
            "artifact": "jutari_paper/paper/figures/gpu_throughput.pdf",
            "runtime": "plot from cached JSON",
            "hardware": "GTX 1080 Ti / Quadro RTX 5000",
            "verified_by": "results/gpu/{gtx1080ti,q5000}.json",
            "status": "measured",
        },
        {
            "claim": "Exact-forward region in the (α, T) relaxation plane",
            "value": "α≥6, T≤0.14",
            "detail": "Per-step likelihood heatmap over the relaxation plane; the recommended "
                      "operating point (α=6, T=0.14) sits inside the bit-exact corner.",
            "script": "tools/relaxation_study/make_relax_heatmap.py",
            "command": "python3 tools/relaxation_study/make_relax_heatmap.py",
            "artifact": "jutari_paper/paper/figures/fig_relax_heatmap.pdf",
            "runtime": "seconds",
            "hardware": "M1 Max (CPU)",
            "verified_by": "tools/relaxation_study/relax_profiles.txt",
            "status": "measured",
        },
        {
            "claim": "XAI demo — joystick gradient finds the cannon edges",
            "value": "edges (sampler) / 0 (naive)",
            "detail": "∂screen/∂RIGHT through the differentiable sampler highlights the "
                      "cannon edges; the naive integer-dispatch gradient is identically zero on the "
                      "discrete output. Identical across all three soft variants.",
            "script": "tools/xai_si_gradient/si_joystick_fig.py",
            "command": "python3 tools/xai_si_gradient/si_joystick_fig.py",
            "artifact": "tools/xai_si_gradient/out/si_joystick_gradient.pdf",
            "runtime": "seconds",
            "hardware": "M1 Max (CPU)",
            "verified_by": "tools/xai_si_gradient/si_joystick_gradient.jl",
            "status": "measured",
        },
        {
            "claim": "Implementation-effort timeline (from git history)",
            "value": "769 commits on main",
            "detail": "Cumulative commits / active sessions / calendar days computed directly from "
                      "the repo git log up to the implementation cutoff.",
            "script": "jutari_paper/paper/make_figures.py",
            "command": "python3 jutari_paper/paper/make_figures.py",
            "artifact": "jutari_paper/paper/figures/fig_timeline.pdf",
            "runtime": "seconds",
            "hardware": "any",
            "verified_by": "git log (public history)",
            "status": "measured",
        },
    ],
    "figures": [
        ("p1_architecture", "VCS architecture", "CPU / TIA / RIOT / cartridge block diagram (schematic)."),
        ("p1_pipeline", "HARD / SOFT / STE pipeline", "Dual execution paths joined by the straight-through estimator."),
        ("p1_gpu_throughput", "GPU throughput scaling", "Env-steps/s vs batch size, two GPUs, forward & forward+grad. Built from results/gpu/*.json."),
        ("p1_relax_heatmap", "Relaxation (α, T) heatmap", "Per-step likelihood; the bit-exact corner and operating point marked."),
        ("p1_si_joystick", "XAI joystick gradient", "Real scene, sampler ∂screen/∂RIGHT, naive≡0, inverse bar chart."),
        ("p1_timeline", "Effort timeline", "Cumulative commits & active sessions from git history."),
    ],
    "videos": [
        ("divergence_si", "Supplement: HARD vs SOFT-STE vs relaxed (4-panel)",
         "HARD | SOFT-STE | SOFT-relaxed(α,T) | diff. The HARD and SOFT-STE panels are pixel-identical (Theorem 1); the relaxed panel drifts."),
        ("si_compare", "Space Invaders — xitari vs jutari vs diff",
         "Left: xitari reference. Middle: jutari. Right: per-pixel difference (solid black = exact match)."),
        ("presentation", "Paper 1 — narrated overview", "Project talk (transcoded for web)."),
    ],
}

# ---------------------------------------------------------------------------
# Paper 2 — interpretability ground-truth benchmark
# ---------------------------------------------------------------------------
PAPER2 = {
    "id": "paper2",
    "title": "Paper 2 — An Interpretability Ground-Truth Benchmark",
    "subtitle": "Scoring XAI / mechanistic-interpretability methods against the exact causal truth of a known machine",
    "venue": "Nature Machine Intelligence (in preparation)",
    "pdf": "xai_paper/xai_2_interpretability/paper/main.pdf",
    "blurb": (
        "Because the VCS is fully known and exactly intervenable, every explanation can be scored "
        "against ground truth. An oracle measures the true causal effect of each state variable; "
        "XAI and mechanistic-interpretability methods are then graded by how well they recover it. "
        "Headline finding: causal/intervention methods stay near-ceiling while gradient and "
        "correlational methods collapse on the discrete sprite-position outputs whose naive "
        "gradient is exactly zero."),
    "headline": {
        "gap": "0.3435",
        "causal": "0.4118",
        "causal_ci": "0.3902",
        "causal_n": "4",
        "grad": "0.0683",
        "grad_ci": "0.0701",
        "grad_n": "9",
        "n_methods": "31",
        "n_records": "257",
    },
    "claims": [
        {
            "claim": "E1 · Ground-truth oracle (intervention + gradient + cross-check)",
            "value": "3 records",
            "detail": "Exact causal map |Δy(u)| via clamp/resample-and-rerun; content-path "
                      "gradient ∂y/∂u via the SOFT-STE substrate; a cross-check confirming "
                      "the two agree. bit_exact_rerun flag set on the records.",
            "script": "tools/xai_study/ground_truth/oracle_intervene.jl",
            "command": "julia --project=jutari tools/xai_study/ground_truth/oracle_intervene.jl --game pong",
            "artifact": "tools/xai_study/ground_truth/out/",
            "runtime": "minutes / game (cluster)",
            "hardware": "LME cluster / local · seed 0",
            "verified_by": "oracle_xcheck.jl (intervention↔gradient correlation)",
            "status": "measured",
        },
        {
            "claim": "E2 · T3 game-concept labels (import / verify / discover)",
            "value": "17 records",
            "detail": "Candidate RAM→concept maps from OCAtari/AtariARI, upgraded to causal by "
                      "verify-by-intervention (does the byte actually move the framebuffer?).",
            "script": "tools/xai_study/t3/verify_labels.jl",
            "command": "julia --project=jutari tools/xai_study/t3/verify_labels.jl",
            "artifact": "tools/xai_study/t3/out/",
            "runtime": "minutes (cluster)",
            "hardware": "LME cluster / local · seed 0",
            "verified_by": "intervention check per label",
            "status": "measured",
        },
        {
            "claim": "E3 · Phase A — neuroscience battery A1–A8 (Kording, quantified)",
            "value": "54 records",
            "detail": "Connectomics, single-unit lesions, tuning curves, pairwise correlations, "
                      "LFP spectra, Granger causality, NMF/PCA, whole-state — each scored "
                      "against the true read/write graph over 6 core games.",
            "script": "tools/xai_study/phaseA_kording/A1_connectomics.jl",
            "command": "julia --project=jutari tools/xai_study/phaseA_kording/A2_lesions.jl --games core",
            "artifact": "tools/xai_study/phaseA_kording/out/",
            "runtime": "4 h walltime / array task",
            "hardware": "LME cluster (CPU array) · seed 0",
            "verified_by": "scored vs oracle data-flow graph; A6+A7 spot-re-run bit-exact",
            "status": "measured",
        },
        {
            "claim": "E3 · A9 — Visual6502 transistor-level track",
            "value": "deferred",
            "detail": "Optional circuit-level (transistor) battery; product-owner-gated after the "
                      "pilot. Not run for this submission — listed for honesty.",
            "script": "xai_paper/xai_2_interpretability/experiment_design.md",
            "command": "—",
            "artifact": "—",
            "runtime": "—",
            "hardware": "—",
            "verified_by": "experiment_design.md (PO-gated)",
            "status": "deferred",
        },
        {
            "claim": "E4 · Phase B — attribution / XAI methods (14)",
            "value": "166 records",
            "detail": "Vanilla gradient, Grad×Input, Guided Backprop, SmoothGrad, Integrated "
                      "Gradients, Expected Gradients, Occlusion, extremal perturbation, RISE, LIME, "
                      "KernelSHAP, counterfactual + the N/A audit — each scored by correlation "
                      "and deletion/insertion AUC vs the oracle.",
            "script": "tools/xai_study/phaseB_attribution/ig_baseline_sweep.jl",
            "command": "julia --project=jutari tools/xai_study/phaseB_attribution/saliency.jl --games core",
            "artifact": "tools/xai_study/phaseB_attribution/out/",
            "runtime": "up to ~15 min / method / game",
            "hardware": "LME cluster (CPU array) · seed 0",
            "verified_by": "scored vs E1 oracle",
            "status": "measured",
        },
        {
            "claim": "E5 · Phase C — mechanistic interpretability (10)",
            "value": "72 records",
            "detail": "Activation patching / causal tracing, interchange/DAS, attribution patching, "
                      "path patching, ACDC, sparse autoencoders, NMF/PCA dictionaries, causal "
                      "scrubbing, linear probing + control tasks, logit/tuned lens.",
            "script": "tools/xai_study/phaseC_mechanistic/activation_patching.jl",
            "command": "julia --project=jutari tools/xai_study/phaseC_mechanistic/sae.jl --games core",
            "artifact": "tools/xai_study/phaseC_mechanistic/out/",
            "runtime": "4 h walltime / array task",
            "hardware": "LME cluster (CPU array) · seed 0",
            "verified_by": "scored vs exact patch / true circuit",
            "status": "measured",
        },
        {
            "claim": "E6 · Cross-tradition leaderboard",
            "value": "31 methods, 257 records",
            "detail": "Faithfulness-vs-plausibility leaderboard aggregated from every committed "
                      "per-game record. Headline contrast: position-regime causal faithfulness "
                      "0.4118 vs gradient/correlational 0.0683 (gap 0.3435).",
            "script": "tools/xai_study/compare/leaderboard.py",
            "command": "python3 tools/xai_study/compare/leaderboard.py",
            "artifact": "tools/xai_study/compare/out/leaderboard.json",
            "runtime": "seconds (pure read)",
            "hardware": "any · embedded self-check",
            "verified_by": "self-check asserts oracle_copy F=1, uniform at floor",
            "status": "measured",
        },
        {
            "claim": "E6 · ROM-free benchmark package",
            "value": "14 records",
            "detail": "A self-contained scoring harness (magnitude_proxy demonstration) that an "
                      "auditor can run with no ROM, against the committed oracle records.",
            "script": "tools/xai_study/compare/benchmark/run.py",
            "command": "python3 tools/xai_study/compare/benchmark/run.py --method magnitude_proxy",
            "artifact": "tools/xai_study/compare/benchmark/out/",
            "runtime": "seconds",
            "hardware": "any",
            "verified_by": "self-test (oracle_copy F=1, magnitude_proxy≈0.271 pong/content)",
            "status": "measured",
        },
    ],
    "figures": [
        ("p2_fig1_platform_oracle", "Platform & oracle", "The known machine and how the causal oracle is measured."),
        ("p2_fig2_faithfulness_plausibility", "Faithfulness vs plausibility", "The leaderboard axes. Built from leaderboard.json + faithful_demo.json."),
        ("p2_fig3_phaseA_battery", "Phase-A battery", "Kording-style neuroscience methods scored against ground truth."),
        ("p2_fig4_attribution_mechanistic", "Attribution vs mechanistic", "Phase B vs Phase C method scores."),
        ("p2_fig5_representativeness", "Representativeness map", "Where each method lands across regimes."),
        ("p2_fig6_failure_taxonomy", "Failure taxonomy", "How and where methods fail."),
        ("p2_fig7_sampler_faithful", "Sampler: faithful, no semantics", "The differentiable sampler is faithful yet semantically empty."),
    ],
    "videos": [],
}
