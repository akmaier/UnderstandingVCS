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
    # (title, desc, link_label, href). href: external URL, an internal page/anchor,
    # or a repo path (rendered as a GitHub blob link).
    "pillars": [
        ("External oracle", "xitari C++ — 64/64 RAM byte-exact + 64/64 screen pixel-exact",
         "google-deepmind/xitari ↗", "https://github.com/google-deepmind/xitari"),
        ("Dual-port cross-check", "PXC2: the JAX port and the Julia port diverge from xitari "
         "<i>identically</i>, in lock-step — a shared root cause, not two coincidences",
         "Read the PXC2 code tour →", "conformance.html#pxc2"),
        ("Conformance harness", "PXC1 / PXC-S / PXC4 replay xitari traces and diff RAM, screen "
         "and a full 6502 functional test", "Read the harness code tour →", "conformance.html"),
        ("Open dev log", "769 commits on <code>main</code> with the prompt→code chain; a "
         "7,525-line running bug-fix log", "bug_fix_log.md", "bug_fix_log.md"),
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
            "value": "byte-identical",
            "detail": "The executed straight-through SOFT path is byte-identical to HARD: with "
                      "relaxation off (the default), <code>soft_rom_peek</code> / "
                      "<code>soft_ram_peek</code> equal the original one-hot dot product over "
                      "5,000 random peeks, and toggling a relaxed run on and back off leaves RAM "
                      "and the rendered frame unchanged (no state leak). The 4-panel video only "
                      "<i>illustrates</i> HARD ≡ SOFT-STE while a relaxed (α,T) path drifts — the "
                      "proof is the regression check, not the video.",
            "script": "tools/relaxation_study/verify_soft_ste.jl",
            "command": "cd jutari && julia --project=. ../tools/relaxation_study/verify_soft_ste.jl",
            "artifact": "tools/relaxation_study/video_out/divergence_si.mp4",
            "inputs": [
                ("tools/relaxation_study/dump_divergence_frames.jl",
                 "renders HARD / SOFT-STE / relaxed scanlines to raw frame streams"),
                ("tools/relaxation_study/make_divergence_video.py",
                 "encodes those frames into the 4-panel mp4"),
            ],
            "note": "The mp4 is an <b>illustration</b>, generated by the two input scripts. "
                    "The bit-exactness is proved by the script in this row "
                    "(<code>verify_soft_ste.jl</code>), not by the video.",
            "runtime": "seconds (check)",
            "hardware": "M1 Max (CPU)",
            "verified_by": "verify_soft_ste.jl (RAM + frame byte-identical) + the 64/64 screen sweep",
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
            "inputs": [
                ("tools/bench_jaxtari_gpu.py",
                 "runs the batched soft-mode benchmark on the GPU → results/gpu/*.json"),
                ("results/gpu/gtx1080ti.json", "raw per-batch wall times (GTX 1080 Ti)"),
                ("results/gpu/q5000.json", "raw per-batch wall times (Quadro RTX 5000)"),
            ],
            "note": "<code>plot_gpu_throughput.py</code> only reads the committed JSON and draws "
                    "the curve — every plotted number traces to a measured wall time in "
                    "<code>results/gpu/*.json</code>.",
            "runtime": "plot from cached JSON",
            "hardware": "GTX 1080 Ti / Quadro RTX 5000",
            "verified_by": "results/gpu/{gtx1080ti,q5000}.json",
            "status": "measured",
        },
        {
            "claim": "Exact-forward region in the (α, T) relaxation plane",
            "value": "α≥6, T≤0.14",
            "detail": "The heatmap is a per-step likelihood model, "
                      "P<sub>step</sub>(α,T) = p_read(T)<sup>ρ</sup> · p_branch(α)<sup>f_b</sup>. "
                      "Its inputs are <b>measured</b> by running the real soft simulator "
                      "(<code>soft_step</code>) on the Space Invaders ROM for 3,000 steps: ρ (mean "
                      "instruction length), f_b (branch fraction), the actual branch-offset set and "
                      "the fetched-address histogram. <code>dump_profiles.jl</code> writes those "
                      "profiles; <code>make_relax_heatmap.py</code> renders the outer combination. "
                      "The operating point α=6, T=0.14 is independently bit-exact-verified by "
                      "verify_soft_ste.jl.",
            "script": "tools/relaxation_study/dump_profiles.jl",
            "command": "cd jutari && julia --project=. ../tools/relaxation_study/dump_profiles.jl",
            "artifact": "tools/relaxation_study/relax_profiles.txt",
            "inputs": [
                ("tools/relaxation_study/make_relax_heatmap.py",
                 "reads relax_profiles.txt → renders fig_relax_heatmap.pdf"),
                ("jutari_paper/paper/figures/fig_relax_heatmap.pdf", "the rendered figure"),
            ],
            "note": "The heatmap is a <b>model</b>, not a per-cell brute-force scan: the closed-form "
                    "<code>p_read</code>/<code>p_branch</code> are evaluated over statistics "
                    "<i>measured</i> from a real 3,000-step soft run on the SI ROM "
                    "(<code>dump_profiles.jl</code>). The operating point α=6, T=0.14 is separately "
                    "bit-exact-verified by <code>verify_soft_ste.jl</code>.",
            "runtime": "seconds",
            "hardware": "M1 Max (CPU)",
            "verified_by": "relax_profiles.txt (measured); plotted by make_relax_heatmap.py",
            "status": "measured",
        },
        {
            "claim": "XAI demo — joystick gradient recovers “push RIGHT”",
            "value": "±35.7 L/R · 0 up/down",
            "detail": "The inverse ∂(move-right)/∂joystick is computed by <b>Zygote autodiff</b> "
                      "through the paper's bilinear sampler: −35.73 for left, +35.73 for right, 0 "
                      "for up/down — identical across all three soft variants (Theorem 1), while "
                      "the naive integer-index path gives 0 in every direction. The forward "
                      "∂screen/∂RIGHT is a finite-difference directional derivative through the "
                      "sampler and lights up the cannon edges. The values live in the committed "
                      "<code>ji_grad.txt</code>; <code>si_joystick_fig.py</code> only plots them — "
                      "it computes nothing.",
            "script": "tools/xai_si_gradient/si_joystick_gradient.jl",
            "command": "cd jutari && julia --project=. ../tools/xai_si_gradient/si_joystick_gradient.jl",
            "artifact": "tools/xai_si_gradient/out/ji_grad.txt",
            "note": "The gradients are computed here, not in the figure script. "
                    "Input is the real <code>space_invaders.bin</code> ROM (not redistributed; "
                    "obtained via AutoROM), stepped to the 35 s scene for the cannon footprint. "
                    "<code>Zygote.gradient</code> differentiates the sampler objective; the forward "
                    "saliency is a central finite difference on the joystick. Outputs land in "
                    "<code>out/ji_grad.txt</code> (committed) and <code>out/ji_*.raw</code> "
                    "(regenerable).",
            "runtime": "seconds",
            "hardware": "M1 Max (CPU)",
            "verified_by": "ji_grad.txt identical across the 3 soft variants; figure si_joystick_gradient.pdf",
            "status": "measured",
        },
        {
            "claim": "XAI joystick figure (plot of the computed gradients)",
            "value": "2×2 panel",
            "detail": "Reads the committed gradient fields/values from "
                      "<code>tools/xai_si_gradient/out/</code> and draws the 2×2 figure (scene, "
                      "sampler saliency, naive≡0, inverse bar chart). A plotting step only — no "
                      "computation. Listed separately so the figure script is not mistaken for the "
                      "source of the numbers.",
            "script": "tools/xai_si_gradient/si_joystick_fig.py",
            "command": "python3 tools/xai_si_gradient/si_joystick_fig.py",
            "artifact": "tools/xai_si_gradient/out/si_joystick_gradient.pdf",
            "inputs": [
                ("tools/xai_si_gradient/si_joystick_gradient.jl",
                 "computes the gradient fields/values → out/ji_grad.txt + out/ji_*.raw"),
                ("tools/xai_si_gradient/out/ji_grad.txt",
                 "the committed inverse-gradient values plotted in panel (d)"),
            ],
            "note": "Pure plotting — no data is computed here. The bar <i>heights</i> in panel (d) "
                    "are read from <code>ji_grad.txt</code>; the constant "
                    "<code>wbar = 0.26</code> (<code>si_joystick_fig.py:73</code>) is only the "
                    "matplotlib <b>bar width</b> / side-by-side offset (3 bars × 0.26 ≈ 0.78 of the "
                    "unit spacing), not a data value.",
            "runtime": "seconds",
            "hardware": "M1 Max (CPU)",
            "verified_by": "inputs from si_joystick_gradient.jl (ji_grad.txt, ji_*.raw)",
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
        ("presentation", "Paper 1 — narrated overview", "Project talk (transcoded for web)."),
    ],
    # Conformance gallery: every clip is xitari | jutari | per-pixel-difference.
    # The difference panel is black on all of them (only the "DIFFERENCE" header
    # label is lit) — verified across the full clip for all 64 games before
    # featuring. (game id, display title)
    "gallery": [
        ("space_invaders", "Space Invaders"),
        ("pong", "Pong"),
        ("breakout", "Breakout"),
        ("ms_pacman", "Ms. Pac-Man"),
        ("qbert", "Q*bert"),
        ("seaquest", "Seaquest"),
        ("enduro", "Enduro"),
        ("pitfall", "Pitfall!"),
        ("montezuma_revenge", "Montezuma's Revenge"),
        ("riverraid", "River Raid"),
        ("beam_rider", "Beam Rider"),
        ("kangaroo", "Kangaroo"),
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
            "note": "Every method, its reference and its implementation script are catalogued in "
                    "the <a href=\"methods.html#phaseB\">method &amp; execution tour</a>.",
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
            "note": "Every method, its reference and its implementation script are catalogued in "
                    "the <a href=\"methods.html#phaseC\">method &amp; execution tour</a>.",
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

# ---------------------------------------------------------------------------
# Paper 2 — per-method explanatory sub-pages (catalogue rows link to these).
# Each method gets m_<key>.html with an explanation + a figure generated from
# its committed record (.json/.npz) paired with a rendered game screenshot.
# (key, phase, title, ref, script, game, record-basename, what-it-does)
# ---------------------------------------------------------------------------
P2_METHODS = [
    # ---- Phase A — Jonas & Kording battery -------------------------------
    dict(key="A1_connectomics", phase="A", title="A1 · Connectomics / data-flow graph",
         ref="cf. Jonas & Kording 2017", script="tools/xai_study/phaseA_kording/A1_connectomics.jl",
         game="pong", record="A1_pong",
         what="Reconstructs the inter-cell data-flow graph by perturbing each RAM cell and "
              "recording which other cells change — the program's 'connectome'. Scored by graph "
              "F1 against the true read/write graph from the disassembly."),
    dict(key="A2_lesions", phase="A", title="A2 · Single-unit lesions",
         ref="cf. Jonas & Kording 2017", script="tools/xai_study/phaseA_kording/A2_lesions.jl",
         game="pong", record="A2_lesions_pong",
         what="Lesions each RAM cell in turn and measures the behavioural change, building a "
              "per-unit importance map. Scored by rank-correlation with each cell's true causal role."),
    dict(key="A3_tuning", phase="A", title="A3 · Tuning curves",
         ref="cf. Jonas & Kording 2017", script="tools/xai_study/phaseA_kording/A3_tuning.jl",
         game="pong", record="A3_tuning_pong",
         what="Builds tuning curves of each cell to luminance and to a game variable, then flags "
              "'spuriously tuned' cells whose tuning does not match their true causal role."),
    dict(key="A4_correlations", phase="A", title="A4 · Pairwise correlations",
         ref="cf. Jonas & Kording 2017", script="tools/xai_study/phaseA_kording/A4_correlations.jl",
         game="pong", record="A4_pong",
         what="Measures the pairwise and global correlation structure across cells and compares "
              "it to the true coupling — weak-pairwise / strong-global, as in the neuroscience original."),
    dict(key="A5_lfp", phase="A", title="A5 · Local field potentials",
         ref="cf. Jonas & Kording 2017", script="tools/xai_study/phaseA_kording/A5_lfp.jl",
         game="pong", record="A5_pong",
         what="Treats pooled regional activity as a local field potential and asks how much of its "
              "power spectrum is just the known clocks (frame/scanline) — i.e. epiphenomenal."),
    dict(key="A6_granger", phase="A", title="A6 · Granger causality",
         ref="cf. Jonas & Kording 2017", script="tools/xai_study/phaseA_kording/A6_granger.jl",
         game="pong", record="A6_pong",
         what="Infers Granger-causal edges between CPU / TIA / RIOT activity and scores its "
              "false-edge and missed-edge rate against the true data-flow."),
    dict(key="A7_dimred", phase="A", title="A7 · Dimensionality reduction (NMF/PCA)",
         ref="cf. Jonas & Kording 2017", script="tools/xai_study/phaseA_kording/A7_dimred.jl",
         game="pong", record="A7_pong",
         what="Runs NMF/PCA on the state tensor and matches the latent components to known "
              "signals (clock, read/write, vsync)."),
    dict(key="A8_wholestate", phase="A", title="A8 · Whole-state recording",
         ref="cf. Jonas & Kording 2017", script="tools/xai_study/phaseA_kording/A8_wholestate.jl",
         game="pong", record="A8_pong",
         what="Records the full RAM + register state over time as a descriptive baseline — the raw "
              "material every other method works from."),
    # ---- Phase B — attribution / XAI -------------------------------------
    dict(key="saliency", phase="B", title="Vanilla gradient (saliency)",
         ref="Simonyan et al. 2014", script="tools/xai_study/phaseB_attribution/saliency.jl",
         game="pong", record="saliency_pong_content",
         what="The input-space gradient of the output with respect to each cause. Scored by "
              "correlation with the oracle's true causal map and by deletion/insertion AUC."),
    dict(key="gradxinput", phase="B", title="Grad×Input / DeepLIFT",
         ref="Shrikumar et al. 2017", script="tools/xai_study/phaseB_attribution/gradxinput.jl",
         game="pong", record="gradxinput_deeplift_pong_content",
         what="Gradient multiplied by input — a DeepLIFT-style attribution with a completeness "
              "property. Scored against the oracle."),
    dict(key="guided_backprop", phase="B", title="Guided Backprop",
         ref="Springenberg et al. 2015", script="tools/xai_study/phaseB_attribution/guided_backprop.jl",
         game="pong", record="guided_backprop_pong_ball_pixel",
         what="Backpropagation that suppresses negative signals, sharpening the saliency map. "
              "Includes the Adebayo et al. 2018 sanity check; scored against the oracle."),
    dict(key="smoothgrad", phase="B", title="SmoothGrad",
         ref="Smilkov et al. 2017", script="tools/xai_study/phaseB_attribution/smoothgrad.jl",
         game="pong", record="smoothgrad_pong_ball_pixel",
         what="Averages the gradient over input noise to denoise the saliency map."),
    dict(key="ig_baseline_sweep", phase="B", title="Integrated Gradients",
         ref="Sundararajan et al. 2017", script="tools/xai_study/phaseB_attribution/ig_baseline_sweep.jl",
         game="pong", record="ig_baseline_sweep_pong_ball_pixel",
         what="Integrates the gradient along a path from a baseline to the input (completeness-"
              "satisfying). We sweep the baseline because the attribution depends on it."),
    dict(key="expected_gradients", phase="B", title="Expected Gradients",
         ref="Erion et al. 2021 (NMI)", script="tools/xai_study/phaseB_attribution/expected_gradients.jl",
         game="pong", record="expected_gradients_pong_ball_pixel",
         what="Integrated Gradients averaged over a distribution of baselines, removing the "
              "single-baseline sensitivity."),
    dict(key="occlusion", phase="B", title="Occlusion",
         ref="Zeiler & Fergus 2014", script="tools/xai_study/phaseB_attribution/occlusion.jl",
         game="pong", record="occlusion_pong_content",
         what="Slides an occluder and measures the output change per region — effectively a coarse "
              "intervention, so it tracks the oracle closely."),
    dict(key="perturbation", phase="B", title="Extremal / meaningful perturbation",
         ref="Fong & Vedaldi 2017; Fong et al. 2019", script="tools/xai_study/phaseB_attribution/perturbation.jl",
         game="pong", record="perturbation_pong_content",
         what="Learns the minimal mask that most changes the output; scored by IoU with the true "
              "causal set."),
    dict(key="rise", phase="B", title="RISE",
         ref="Petsiuk et al. 2018", script="tools/xai_study/phaseB_attribution/rise.jl",
         game="pong", record="rise_pong_ball_pixel",
         what="Averages the output over many random masks (N=500) to estimate per-region importance."),
    dict(key="lime", phase="B", title="LIME",
         ref="Ribeiro et al. 2016", script="tools/xai_study/phaseB_attribution/lime.jl",
         game="pong", record="lime_pong_ball_pixel",
         what="Fits a local linear surrogate around the input and reads off the weights; scored for "
              "correlation and stability."),
    dict(key="kernelshap", phase="B", title="KernelSHAP / Shapley",
         ref="Lundberg & Lee 2017", script="tools/xai_study/phaseB_attribution/kernelshap.jl",
         game="pong", record="kernelshap_pong_ball_pixel",
         what="Estimates Shapley values via weighted least squares; scored against the true causal "
              "contributions and for convergence vs compute."),
    dict(key="counterfactual", phase="B", title="On-distribution counterfactual",
         ref="cf. Olson 2021; Atrey 2020", script="tools/xai_study/phaseB_attribution/counterfactual.jl",
         game="pong", record="counterfactual_pong_ball_pixel",
         what="Finds the minimal valid on-distribution edit that flips the output — set-state-and-"
              "re-render removes the off-manifold objection."),
    dict(key="na_audit", phase="NA", title="N/A audit: Grad-CAM, attention rollout, VIPER",
         ref="Selvaraju 2017; Abnar & Zuidema 2020; Bastani 2018",
         script="tools/xai_study/phaseB_attribution/na_audit.jl",
         game="pong", record="na_audit_pong",
         what="Grad-CAM, attention rollout and VIPER need convolutional/attention layers or a "
              "learned policy that the VCS does not have. Recorded honestly as 'does not apply' "
              "rather than forced."),
    # ---- Phase C — mechanistic interpretability --------------------------
    dict(key="activation_patching", phase="C", title="Activation patching / causal mediation",
         ref="Vig et al. 2020; ROME, Meng et al. 2022", script="tools/xai_study/phaseC_mechanistic/activation_patching.jl",
         game="pong", record="activation_patching_pong",
         what="Patches a recorded state value from one run into another and measures the causal "
              "effect, comparing it to the exact patch effect from the oracle."),
    dict(key="das", phase="C", title="Interchange interventions / DAS",
         ref="Geiger et al. 2021, 2023", script="tools/xai_study/phaseC_mechanistic/das.jl",
         game="pong", record="das_pong",
         what="Distributed Alignment Search aligns a learned subspace to a causal variable and "
              "measures interchange accuracy against the true variable."),
    dict(key="attribution_patching", phase="C", title="Attribution / edge patching",
         ref="Nanda 2023; Syed et al. 2023", script="tools/xai_study/phaseC_mechanistic/attribution_patching.jl",
         game="pong", record="attribution_patching_pong",
         what="A gradient approximation to activation patching; scored by its approximation error "
              "vs true patching and by edge precision/recall."),
    dict(key="path_patching", phase="C", title="Path patching / IOI circuit",
         ref="Wang et al. 2022; Goldowsky-Dill et al. 2023", script="tools/xai_study/phaseC_mechanistic/path_patching.jl",
         game="pong", record="path_patching_pong",
         what="Patches along specific paths to recover the circuit; scored by circuit "
              "precision/recall against the true routine."),
    dict(key="acdc", phase="C", title="ACDC — automatic circuit discovery",
         ref="Conmy et al. 2023", script="tools/xai_study/phaseC_mechanistic/acdc.jl",
         game="pong", record="acdc_pong",
         what="Prunes edges to find a minimal circuit; scored by edge precision/recall and "
              "scrubbing-preserved performance against the true data-flow."),
    dict(key="sae", phase="C", title="Sparse autoencoders",
         ref="Cunningham et al. 2023; Bricken et al. 2023; Templeton et al. 2024",
         script="tools/xai_study/phaseC_mechanistic/sae.jl", game="pong", record="sae_pong",
         what="Trains a sparse autoencoder on the state and matches its features to known "
              "variables; scored by match rate, causal use and monosemanticity."),
    dict(key="dictionaries", phase="C", title="NMF/PCA dictionaries",
         ref="—", script="tools/xai_study/phaseC_mechanistic/dictionaries.jl", game="pong", record="dictionaries_pong",
         what="NMF/PCA dictionary learning over the state; matched-component fraction against the "
              "known variables."),
    dict(key="causal_scrubbing", phase="C", title="Causal scrubbing",
         ref="Chan et al. 2022", script="tools/xai_study/phaseC_mechanistic/causal_scrubbing.jl",
         game="pong", record="causal_scrubbing_pong",
         what="Resamples activations consistent with a hypothesised circuit and checks behaviour "
              "is preserved — a hypothesis pass/fail against the true routine."),
    dict(key="linear_probing", phase="C", title="Linear probing + control tasks",
         ref="Alain & Bengio 2017; Hewitt & Liang 2019", script="tools/xai_study/phaseC_mechanistic/linear_probing.jl",
         game="pong", record="linear_probing_pong",
         what="Trains linear probes for concepts and subtracts a control-task baseline "
              "(selectivity) to expose the present-vs-used gap — information can be decodable "
              "without being causally used."),
    dict(key="logit_lens", phase="C", title="Logit / tuned lens",
         ref="nostalgebraist 2020; Belrose et al. 2023", script="tools/xai_study/phaseC_mechanistic/logit_lens.jl",
         game="pong", record="logit_lens_pong",
         what="Reads intermediate state through the output decoder to see what is represented at "
              "each stage; scored by readout fidelity against the true intermediate value."),
]
