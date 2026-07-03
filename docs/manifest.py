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
    # Qualitative prose ONLY. Every numeric slot is a {placeholder} filled at build
    # time from docs/site_data.json (see build_pages.py). No P2 faithfulness number
    # is hardcoded here — the store is the single source of truth.
    "blurb": (
        "Because the VCS is fully known and exactly intervenable, every explanation can be scored "
        "against ground truth. An oracle measures the true causal effect of each state variable; "
        "XAI and mechanistic-interpretability methods are then graded by how well they recover it. "
        "Headline finding: across all regimes, causal/intervention methods stay well above gradient "
        "and correlational methods — a robust faithfulness gap of {all_regime_gap} "
        "(causal/intervention {causal_all} vs gradient/correlational {grad_all}). On the discrete "
        "sprite-position outputs the naive gradient is exactly zero; the emulator's bilinear sampler "
        "restores a non-zero position gradient, but its faithfulness stays low, so "
        "gradient/correlational methods still collapse there while causal/intervention methods do "
        "not. Bootstrapped over the {scored_games} scored games, the position gap is now significant "
        "(mean {position_gap}, 95% CI [{position_ci_lo}, {position_ci_hi}], excludes zero). "
        "<br><br>The testbed was <b>redesigned</b> to score every method on the same shared "
        "random-action gameplay states — see the "
        "<a href=\"https://github.com/akmaier/UnderstandingVCS/blob/main/xai_paper/xai_2_interpretability/experiment_redesign.md\">"
        "experiment redesign note</a>."),
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
                      "against the true read/write graph over the 42 scored games.",
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
            # {placeholder}s filled from site_data.json at build time (no hardcoded numbers).
            "value": "{n_methods} methods, {n_records} records",
            "detail": "Faithfulness-vs-plausibility leaderboard aggregated from every committed "
                      "per-game record over the {scored_games} scored games. Headline contrast "
                      "(all regimes): causal/intervention faithfulness {causal_all} vs "
                      "gradient/correlational {grad_all} — a robust gap of {all_regime_gap}. The "
                      "position-regime gap is now significant: bootstrapped over the {scored_games} "
                      "games it is {position_gap} (95% CI [{position_ci_lo}, {position_ci_hi}], "
                      "excludes zero; it crossed zero at six games).",
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
            "verified_by": "self-test (oracle_copy F=1; magnitude_proxy reproduces its committed "
                           "pong/content faithfulness within tolerance)",
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
         game="pong", record="guided_backprop_pong_content",
         what="Backpropagation that suppresses negative signals, sharpening the saliency map. "
              "Includes the Adebayo et al. 2018 sanity check; scored against the oracle."),
    dict(key="smoothgrad", phase="B", title="SmoothGrad",
         ref="Smilkov et al. 2017", script="tools/xai_study/phaseB_attribution/smoothgrad.jl",
         game="pong", record="smoothgrad_pong_content",
         what="Averages the gradient over input noise to denoise the saliency map."),
    dict(key="ig_baseline_sweep", phase="B", title="Integrated Gradients",
         ref="Sundararajan et al. 2017", script="tools/xai_study/phaseB_attribution/ig_baseline_sweep.jl",
         game="pong", record="ig_baseline_sweep_pong_content",
         what="Integrates the gradient along a path from a baseline to the input (completeness-"
              "satisfying). We sweep the baseline because the attribution depends on it."),
    dict(key="expected_gradients", phase="B", title="Expected Gradients",
         ref="Erion et al. 2021 (NMI)", script="tools/xai_study/phaseB_attribution/expected_gradients.jl",
         game="pong", record="expected_gradients_pong_content",
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
         game="pong", record="rise_pong_content",
         what="Averages the output over many random masks (N=500) to estimate per-region importance."),
    dict(key="lime", phase="B", title="LIME",
         ref="Ribeiro et al. 2016", script="tools/xai_study/phaseB_attribution/lime.jl",
         game="pong", record="lime_pong_content",
         what="Fits a local linear surrogate around the input and reads off the weights; scored for "
              "correlation and stability."),
    dict(key="kernelshap", phase="B", title="KernelSHAP / Shapley",
         ref="Lundberg & Lee 2017", script="tools/xai_study/phaseB_attribution/kernelshap.jl",
         game="pong", record="kernelshap_pong_content",
         what="Estimates Shapley values via weighted least squares; scored against the true causal "
              "contributions and for convergence vs compute."),
    dict(key="counterfactual", phase="B", title="On-distribution counterfactual",
         ref="cf. Olson 2021; Atrey 2020", script="tools/xai_study/phaseB_attribution/counterfactual.jl",
         game="pong", record="counterfactual_pong_content",
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

# How each method's headline number is computed and compared to the oracle.
# Keyed by method key; rendered as the "How it's scored" section of m_<key>.html.
# Fallback 'How it's scored' one-liners, keyed by method key. QUALITATIVE prose
# only: {F}/{F_content}/{F_position}/{M} are placeholders filled at build time from
# docs/site_data.json (build_pages.py) — no P2 faithfulness number is hardcoded here.
# P2_METHOD_HOWSCORED is the primary source; this is used only if a key is missing there.
P2_METHOD_SCORED = {
 "A1_connectomics": "F1 of the recovered data-flow graph against the true read/write graph from the disassembly (an edge = perturbing one RAM cell changes another). {F} — single-shot perturbation recovers none of the true edges.",
 "A2_lesions": "Spearman correlation between each cell's lesion importance (screen change when the cell is frozen) and its true causal role from the oracle. {F} — lesioning ranks the cells exactly as the oracle does.",
 "A3_tuning": "Spurious-tuning rate: the fraction of cells that are strongly tuned to a game variable yet are NOT among the oracle's causal cells. {F} — half of the strongly-tuned cells are non-causal, so tuning curves are misleading here.",
 "A4_correlations": "Spearman correlation between the measured cell–cell correlation structure and the oracle's true coupling matrix. {F} — correlation partly tracks coupling but conflates it with shared drivers.",
 "A5_lfp": "Fraction of the pooled-activity power spectrum explained by the known clocks (frame / scanline). {F} — most 'LFP' structure is epiphenomenal timing, not computation.",
 "A6_granger": "False-edge rate of the Granger-inferred causal graph against the true data-flow. {F} — every inferred edge is spurious (Granger sees correlation through shared clocks).",
 "A7_dimred": "Fraction of NMF components that match a known signal/variable (vs the oracle's importance). {F} — recovers most known factors, adds some mixed ones.",
 "A8_wholestate": "Minimality M of the whole-state dump vs the oracle's causal set — how much smaller the true causal set is than 'record everything'. Aggregate minimality {M} — the raw dump is almost entirely non-minimal. The exact per-axis triad (F / S / M) is read from the leaderboard in the audit box below.",
 "saliency": "Pearson correlation of the saliency map with the oracle's exact causal map |Δy(u)| over the candidate causes, plus deletion/insertion AUC. All-regime audit faithfulness {F} (content {F_content}, position {F_position}). On discrete <b>sprite-position</b> outputs the naive gradient is exactly 0; the emulator's bilinear sampler restores a non-zero position gradient, but its faithfulness stays low — so the aggregate is moderate, not near-ceiling.",
 "gradxinput": "Pearson correlation of the Grad×Input / DeepLIFT attribution with the oracle. All-regime audit faithfulness {F} (content {F_content}, position {F_position}) — multiplying by the input sharpens the gradient toward the true causes on content outputs, but the position map still leans on the sampler and stays low.",
 "guided_backprop": "Pearson correlation with the oracle's causal map. All-regime audit faithfulness {F} (content {F_content}, position {F_position}). It partly tracks the oracle on <b>content</b> (colour) outputs, but on discrete <b>sprite-position</b> outputs the naive gradient is exactly 0 — the sampler could restore a non-zero gradient, yet guided backprop's rectified path leaves it at 0 here — and it fails the Adebayo et al. 2018 sanity check, so its cross-game audit faithfulness stays low.",
 "smoothgrad": "Pearson correlation with the oracle. All-regime audit faithfulness {F} (content {F_content}, position {F_position}). Averaging input noise sharpens saliency on <b>content</b> outputs a little; on discrete <b>position</b> outputs the naive gradient is 0 and the sampler restores only a low-faithfulness gradient, so the aggregate stays moderate.",
 "ig_baseline_sweep": "Pearson correlation with the oracle at the headline baseline (baselines are swept). All-regime audit faithfulness {F} (content {F_content}, position {F_position}). Moderate on <b>content</b> outputs; on the sprite-position output the naive gradient is 0, and even with the emulator's sampler restoring a non-zero position gradient the faithfulness stays low — completeness does not rescue it on discrete outputs.",
 "expected_gradients": "Pearson correlation with the oracle. Baseline-averaged IG: on a NOOP reference pool the constant bytes give <code>(x−x0)=0</code>, zeroing the attribution; even on the redesigned gameplay reference pool (with the sampler on for position) it is the <b>lowest-faithfulness</b> attribution method — all-regime audit faithfulness {F} (content {F_content}, position {F_position}) despite high plausibility.",
 "occlusion": "Pearson correlation with the oracle (occlusion is itself a coarse intervention). All-regime audit faithfulness {F} (content {F_content}, position {F_position}) — among the most faithful methods, precisely because it perturbs the real system.",
 "perturbation": "Pearson correlation / mask-IoU with the oracle's true causal set. All-regime audit faithfulness {F} (content {F_content}, position {F_position}) — the learned minimal mask partly overlaps the true causes.",
 "rise": "Pearson correlation with the oracle from random-mask averaging. All-regime audit faithfulness {F} (content {F_content}, position {F_position}) — moderate on <b>content</b> outputs, weaker on <b>position</b> outputs where masking a discrete cause is less informative.",
 "lime": "Pearson correlation of the local linear surrogate's weights with the oracle. All-regime audit faithfulness {F} (content {F_content}, position {F_position}) — fits both content and position structure reasonably.",
 "kernelshap": "Pearson correlation of the Shapley values with the oracle. All-regime audit faithfulness {F} (content {F_content}, position {F_position}) — Shapley recovers much of the true contribution on both content and position outputs, given enough coalitions.",
 "counterfactual": "Pearson correlation / minimality of the minimal valid counterfactual edit against the oracle's minimal causal set. All-regime audit faithfulness {F} (content {F_content}, position {F_position}) — set-state-and-re-render is on-distribution, and the minimal-edit search recovers a fair share of the true minimal causes, more strongly on content than on position.",
 "na_audit": "Count of audited methods that cannot run on the VCS at all (they need conv/attention layers or a learned policy). <b>6</b> — recorded honestly, not forced into a faithfulness score.",
 "activation_patching": "Maximum absolute difference between the recovered patch effect and the EXACT patch effect from the oracle. {F} — activation patching is exact by construction, because it <i>is</i> an intervention.",
 "das": "Interchange accuracy of the aligned subspace against the true variable. {F} — DAS aligns to the causal variable exactly on this known circuit.",
 "attribution_patching": "Pearson correlation between the gradient-approximate effect and the exact patch effect. {F} — a good cheap approximation, with a measurable gap from the truth.",
 "path_patching": "F1 of the recovered path/circuit against the true routine's data-flow. {F} — the single-frame path search recovers none of the true routine at this state.",
 "acdc": "Best F1 of the auto-discovered circuit vs the true data-flow over a threshold sweep. {F} — the pruned graph recovers only a fraction of the true edges here.",
 "sae": "Fraction of SAE features that match a known variable (probe F1) plus a causal-use check. {F} — every feature maps to a known variable on this small state.",
 "dictionaries": "Fraction of NMF/PCA dictionary components matching known variables. {F} — partial; PCA mixes variables while NMF separates more.",
 "causal_scrubbing": "Behaviour preserved when resampling activations consistent with the TRUE circuit (should stay ~1) vs a wrong circuit (should drop). {F} — the true-circuit hypothesis passes the scrub.",
 "linear_probing": "Mean selectivity = probe accuracy minus control-task accuracy, averaged over labelled cells. {F} — concepts are decodable above the control, but some decodable cells are not causally used (the present-vs-used gap).",
 "logit_lens": "Readout fidelity (R²) of the lens-decoded intermediate against the true intermediate value. {F} — the state is linearly readable at the right site, as expected on a transparent machine.",
}

# ---------------------------------------------------------------------------
# Long-form "What it does?" prose, keyed by method key (~150-250 words each).
# Explains the method's idea and mechanism in plain but precise terms, and what
# it is normally used for in interpretability. Rendered as the first prose
# section of m_<key>.html. If a key is missing, build_pages falls back to the
# short one-line `what` field on the method dict.
# ---------------------------------------------------------------------------
P2_METHOD_ABOUT = {
 "A1_connectomics":
   "Connectomics tries to map the wiring of a system by watching how its parts affect each "
   "other. In neuroscience it means reconstructing which neurons connect to which. Here we do "
   "the same for the machine's memory. We change one RAM cell, re-run the program, and record "
   "which other cells change as a result. Each such dependency is drawn as a directed edge from "
   "cause to effect. The full set of edges is the program's data-flow graph, its 'connectome'. "
   "The idea is that the pattern of who-influences-whom should reveal the computation, without "
   "reading the code. This is a classic move in systems neuroscience: infer function from "
   "structure. On the VCS we have the true wiring from the disassembly, so we can check whether "
   "the recovered graph is right. The method is attractive because it needs no labels and no "
   "model of what the program means. It only needs the ability to perturb and observe. In "
   "practice it recovers a graph that looks plausible but contains many edges the real program "
   "does not have, because a single perturbation ripples through shared clocks and buffers.",
 "A2_lesions":
   "A lesion study asks what breaks when a part is removed. It is the oldest tool in "
   "neuroscience: damage one region, see what behaviour is lost, and infer that region's role. "
   "We apply the same idea to memory. We freeze one RAM cell at a time, hold it at a fixed "
   "value, re-run the program, and measure how much the screen changes. A cell whose freezing "
   "wrecks the picture is judged important; a cell whose freezing does nothing is judged "
   "irrelevant. Ranking every cell this way gives a per-unit importance map. The appeal is that "
   "the test is causal: we do not just watch the cell, we intervene on it and see the effect. "
   "This is why the single-unit lesion is the one classical method that scores near the top of "
   "our battery. Its weakness is that it changes one cell at a time. It is therefore blind to "
   "causes that only act together, where two cells matter jointly but neither matters alone. So "
   "it recovers a correct ranking of which units matter, but not the way they combine into the "
   "actual computation.",
 "A3_tuning":
   "A tuning curve measures how strongly a unit responds to some variable. In neuroscience one "
   "plots a neuron's firing rate against, say, the angle of a bar, and a sharp peak is read as "
   "the neuron being 'tuned' to that feature. We build the same curves for RAM cells. For each "
   "cell we plot its value against a game variable and against screen luminance, then measure "
   "how strongly the cell tracks each one. A cell that closely follows a game variable is called "
   "tuned to it. The intuition is that a strongly tuned cell must encode that variable and "
   "therefore drive the behaviour. This is one of the workhorses of systems neuroscience. On the "
   "VCS it is actively misleading. Many cells co-vary with the beam clock and the frame counter, "
   "so apparent tuning is cheap and common. A cell can track a game variable perfectly while "
   "playing no causal role in producing it. The method thus flags cells that look meaningful but "
   "are not, which is exactly the trap we set out to measure. It is the same present-versus-used "
   "problem that also defeats linear probing.",
 "A4_correlations":
   "This method studies the correlation structure across the machine's cells. In neuroscience "
   "the 'spike-word' or pairwise-correlation analysis looks at which units fire together, and "
   "reads coordinated firing as evidence of shared function or coupling. We compute the same "
   "statistics over RAM. For every pair of cells we measure how their values co-vary across the "
   "trajectory, and we also measure the global, population-level correlation. The familiar "
   "finding in neural data is weak pairwise but strong global correlation, and the VCS reproduces "
   "that signature. The idea is that cells which move together are functionally linked, so the "
   "correlation matrix should approximate the true coupling. The method is purely observational: "
   "it never intervenes, it only watches. That is its limit. On a clock-locked machine almost "
   "everything co-varies with almost everything else, driven by shared timing signals rather than "
   "by direct influence. So the correlation structure is real but mostly reflects common drivers, "
   "not causal links. It detects that the machine is active and coordinated, but it does not "
   "separate a true dependency from two cells that merely share a clock.",
 "A5_lfp":
   "The local field potential is a pooled, low-resolution signal. In neuroscience it is the "
   "summed electrical activity of many neurons near an electrode, and its power spectrum, the "
   "strength of each oscillation frequency, is studied as a marker of brain state. We build the "
   "analogue on the VCS by pooling the activity of a region of memory into one aggregate signal "
   "and taking its power spectrum. The idea is that rhythms in this pooled signal reflect the "
   "computation. The problem we test is whether those rhythms are real computation or just the "
   "machine's clocks showing through. The VCS is driven by a frame clock and a scanline clock, "
   "and their fixed rhythms dominate any pooled signal. So much of the spectrum is epiphenomenal: "
   "it is the timing of the hardware, not the work the program does. This is a known danger for "
   "field potentials in neuroscience too, where a strong oscillation can reflect a global "
   "pacemaker rather than local processing. The method is easy to compute and produces "
   "impressive-looking spectra, but on a machine whose clocks we know exactly we can show how "
   "little of that structure is actually computational.",
 "A6_granger":
   "Granger causality infers direction from timing. The rule is simple: if the past of signal A "
   "helps predict the future of signal B beyond B's own past, then A is said to Granger-cause B. "
   "It is widely used in neuroscience and economics to draw directed influence graphs from "
   "recorded time series, without any intervention. We apply it to the VCS by treating the CPU, "
   "TIA, and RIOT activity as time series and asking which components predict which. The result "
   "is a directed graph of inferred influence that we compare to the true data-flow. The appeal "
   "is that it turns passive observation into a causal-looking claim. The deep flaw, on this "
   "machine, is that it equates 'happens earlier' with 'is the cause'. The VCS is clock-locked, "
   "so almost every signal is predictable from almost every other signal one cycle earlier, "
   "whether or not there is a real dependency. Granger causality therefore infers edges "
   "everywhere. Neither a longer lag nor a stricter threshold repairs this, because the shared "
   "clock creates predictability that is not causation. It is the cleanest demonstration in our "
   "battery that precedence is not cause, and that only an intervention can tell them apart.",
 "A7_dimred":
   "Dimensionality reduction summarises high-dimensional activity by a few components. "
   "Principal component analysis (PCA) finds the directions of largest variance; non-negative "
   "matrix factorisation (NMF) finds additive, parts-based factors. Both are standard in "
   "neuroscience for turning a large population recording into a handful of interpretable "
   "'latent factors'. We run them on the VCS state tensor, the values of all cells over time, "
   "and read off the recovered components. We then match each component to a known signal: the "
   "frame or scanline clock, a read/write pattern, the vsync flag, or a game variable. The "
   "intuition is that the strongest components should correspond to the machine's real internal "
   "signals, so the factorisation recovers the structure without any labels. The method is "
   "unsupervised and cheap, which is why it is popular. On the VCS it does moderately well: it "
   "recovers most of the known factors, and NMF's non-negativity fits the additive register "
   "basis a little better than PCA. But a large fraction of components stay unmatched or mix "
   "several true signals into one, so the decomposition is suggestive rather than exact. It finds "
   "structure that is genuinely present, without telling us how the program uses it.",
 "A8_wholestate":
   "Whole-state recording is the simplest possible method: write down everything. At each step "
   "we record every bit of RAM and every register, the complete internal state over time. This "
   "is the raw material that every other method in the study works from. In neuroscience it is "
   "the dream of recording every neuron at once, and it is often treated as the thing that would "
   "finally let us understand the system. The intuition is that if we have the full state, we "
   "have hidden nothing, so we must have the explanation. We include it as a deliberate baseline "
   "to test that intuition. By construction the complete record is perfectly faithful and "
   "perfectly sufficient: it contains every cause, so nothing is missing, and it can predict any "
   "intervention because it holds the whole machine. The point of including it is to show what "
   "such a record still lacks. A dump of all 128 RAM cells is not an account of the computation. "
   "It names everything, so it names nothing in particular, and the true cause of any given "
   "output is a tiny handful of those cells. Completeness is not understanding, and this method "
   "makes that concrete.",
 "saliency":
   "Vanilla gradient saliency is the oldest attribution method for neural networks. It computes "
   "the gradient of the output with respect to each input, and reads a large gradient as 'this "
   "input mattered'. The intuition is a first-order sensitivity: if nudging an input would move "
   "the output a lot, that input is important. The result is a heat-map over the inputs, which "
   "for images highlights the pixels the network is most sensitive to. We apply the same idea to "
   "the VCS. The 'inputs' are the candidate causes, RAM cells and registers, and we take the "
   "gradient of a chosen output, a pixel, a score, or a game event, with respect to each one. On "
   "outputs that flow smoothly into the picture, a colour or a register value, the gradient is "
   "meaningful and points at the true cause. The hard case is a sprite's position. The VCS sets a "
   "sprite's screen position by the timing of a strobe write, which is a discrete step, so the "
   "naive gradient there is exactly zero: a tiny change in the position byte moves the sprite by "
   "no fraction of a pixel. Saliency is fast, model-agnostic in appearance, and everywhere in "
   "the literature, which makes it the natural baseline to test against a known answer.",
 "gradxinput":
   "Grad times Input, in the DeepLIFT family, sharpens plain saliency by multiplying the "
   "gradient at each input by the input's own value. The product answers a slightly different "
   "question than the bare gradient: not just how sensitive the output is to an input, but how "
   "much that input, at its actual value, contributes to the output. DeepLIFT frames this as a "
   "difference from a reference, and the multiplication gives the method a completeness-like "
   "property, so the attributions sum to the change in the output. For images this tends to "
   "produce cleaner, less diffuse maps than raw saliency. We apply it to the VCS by multiplying "
   "each cause's gradient by its recorded value and correlating the result with the true causal "
   "map. On outputs that flow smoothly into the picture, the input factor concentrates the "
   "attribution onto the genuine causal byte and improves on plain saliency. On a sprite's "
   "position it inherits the same wall as every gradient method: the position is a discrete step "
   "set by strobe timing, so the underlying gradient is zero and multiplying by the input cannot "
   "create signal where there is none. The method is popular because it keeps the speed of a "
   "gradient while adding a contribution interpretation, so it is a fair, stronger member of the "
   "gradient family to test.",
 "guided_backprop":
   "Guided backpropagation is a variant of saliency that changes how the signal flows backward "
   "through the network. At each rectifier it keeps only the positive contributions and zeroes "
   "the negative ones, on both the forward activation and the backward gradient. The effect is a "
   "much sharper, cleaner-looking saliency map, which is why the method became popular for "
   "visualising what a network 'sees'. The intuition is that suppressing the negative evidence "
   "isolates the features that positively drive the output. We apply the same rectified-backward "
   "rule to the VCS and correlate the resulting map with the true causal map. A well-known "
   "concern, raised by Adebayo and colleagues, is that guided backprop can act more like an edge "
   "detector than a true explanation, producing similar maps even when the model is randomised. "
   "We run exactly that sanity check on the machine itself. On smooth content outputs the method "
   "partly tracks the true cause. On a sprite's position it hits the gradient wall: the position "
   "is a discrete strobe-timed step, so the underlying gradient is zero, and the rectified "
   "backward path leaves it at zero here rather than restoring any signal. Combined with its "
   "failure of the randomisation check, this makes guided backprop a revealing case: it looks "
   "sharp and convincing yet does not depend on what the program actually computes.",
 "smoothgrad":
   "SmoothGrad is a denoising wrapper around gradient saliency. Raw gradients are visually "
   "noisy, so SmoothGrad adds a small amount of random noise to the input many times, computes "
   "the gradient for each noisy copy, and averages the results. The averaging cancels the "
   "high-frequency noise and leaves a cleaner map, on the intuition that the true signal is "
   "stable under small perturbations while the noise is not. It is a simple, widely used way to "
   "make any gradient method look better. We apply it to the VCS by perturbing each cause, "
   "averaging the gradients, and correlating with the true causal map. On smooth content outputs "
   "the averaging gives a modest improvement over plain saliency. The important negative result "
   "is on a sprite's position. There the naive gradient is exactly zero because the position is a "
   "discrete step set by strobe timing, and averaging many zeros is still zero: adding input "
   "noise cannot manufacture a gradient that the hard machine does not provide. Only a "
   "differentiable surrogate can restore one, and even then its faithfulness stays low. SmoothGrad "
   "is a good test of whether the popular 'just denoise the gradient' fix rescues attribution on "
   "the discrete game logic. It does not.",
 "ig_baseline_sweep":
   "Integrated Gradients attributes the output to inputs by integrating the gradient along a "
   "straight path from a chosen baseline to the actual input. Instead of the gradient at a single "
   "point, which can be misleading if the output has saturated, it accumulates the gradient over "
   "the whole path. This gives the method its completeness property: the attributions add up "
   "exactly to the difference between the output at the input and the output at the baseline. The "
   "attribution depends on which baseline is chosen, an all-black image, a mean image, and so on, "
   "so we sweep several baselines and report the behaviour. We apply it to the VCS by integrating "
   "each cause's gradient from a baseline state to the live state and correlating with the true "
   "causal map. On smooth content outputs it behaves like a stronger saliency and concentrates "
   "mass on the true causal byte. On a sprite's position the completeness property does not save "
   "it: the position is a discrete strobe-timed step, so the naive gradient along the whole path "
   "is zero, and the integral of zero is zero. The baseline choice moves the magnitude of the "
   "attribution but not its correlation with the truth. Integrated Gradients is an axiomatic, "
   "much-cited method, which makes it an important reference point for what completeness can and "
   "cannot fix on discrete outputs.",
 "expected_gradients":
   "Expected Gradients removes the single-baseline choice from Integrated Gradients by averaging "
   "over many baselines. Instead of one reference, it draws references from a distribution of "
   "real states and averages the integrated-gradient attribution across them. This makes the "
   "result stable: it no longer depends on one arbitrary baseline, and the attributions are "
   "provably consistent across reference draws. The method was introduced to fix the perennial "
   "worry that Integrated Gradients can be gamed by picking a convenient baseline. We apply it to "
   "the VCS by averaging integrated gradients over a pool of recorded machine states. The result "
   "is very stable but not more accurate. Each integrated-gradient term contains the factor "
   "input-minus-baseline. When the reference pool holds bytes that are constant across states, "
   "that factor is zero and the attribution collapses. Even on a varied gameplay reference pool, "
   "and with the sampler restoring a position gradient, the method carries almost no true causal "
   "signal, and on a sprite's position it still vanishes for the usual reason. It is the clearest "
   "case in the study of a method that is more careful and more expensive, yet buys stability "
   "without buying faithfulness. That trade-off is exactly what makes it worth measuring here.",
 "occlusion":
   "Occlusion is the most direct attribution method. It hides part of the input, re-runs the "
   "system, and measures how much the output changes. A region whose removal changes the output a "
   "lot is judged important; a region whose removal does nothing is judged irrelevant. Sliding "
   "the occluder over the whole input produces an importance map. For images this means graying "
   "out a patch and watching the class score drop. The reason occlusion is powerful is that it is "
   "really a coarse intervention: it does not model sensitivity, it actually changes the input "
   "and observes the real response. We apply it to the VCS by setting each candidate cause, a RAM "
   "cell or register, to an occluded value, re-running the bit-exact program, and recording the "
   "change in the output. Because this is a genuine do-operation on the real machine, it tracks "
   "the intervention oracle closely, and it works even on a sprite's position where every "
   "gradient method fails. Its only weakness relative to the exact oracle is that it perturbs at "
   "the granularity of the occluder and can miss fine or joint effects. Occlusion is therefore "
   "the bridge between the gradient family and the truly causal methods, and it is among the most "
   "faithful attribution methods precisely because it perturbs the real system.",
 "perturbation":
   "Extremal, or meaningful, perturbation learns the smallest mask that most changes the output. "
   "Rather than sliding a fixed occluder, it optimises a soft mask over the inputs, searching for "
   "the smallest region whose removal most disrupts the output, subject to a bound on the mask's "
   "area. The learned mask is read as the explanation: the compact set of inputs the output "
   "really depends on. For images this yields a tight, optimised heat-map instead of a "
   "brute-force sweep. The method was designed to answer a specific criticism, that hand-placed "
   "occluders are arbitrary and can push the input off the data manifold. By optimising a "
   "bounded, smooth mask of real occlusion re-runs, it stays closer to valid interventions. We "
   "apply it to the VCS by optimising an area-limited mask over the candidate causes, each mask "
   "entry a genuine occlude-and-re-run, and comparing the learned mask to the true minimal cause "
   "set. Because every step is a real intervention, it works on a sprite's position where "
   "gradients fail, and it partly overlaps the true causes. Its faithfulness is moderate: the "
   "optimisation finds a compact set that is on the right track but does not exactly match the "
   "oracle's minimal set. It is a strong, on-manifold member of the intervention family and a "
   "fair test of whether optimising the mask beats simply sliding one.",
 "rise":
   "RISE explains a black box by random masking. It generates many random masks, applies each to "
   "the input, records the output for each masked version, and then forms an importance map as "
   "the output-weighted average of the masks. Inputs that tend to be present when the output is "
   "high receive high importance. The method needs no gradients and no access to the model's "
   "internals; it only needs to query the output, which is why it is popular for truly opaque "
   "systems. The intuition is a Monte-Carlo estimate of each input's marginal effect. We apply it "
   "to the VCS by drawing hundreds of random masks over the candidate causes, re-running the "
   "program for each, and averaging. Because it perturbs and re-runs the real machine, it works "
   "on a sprite's position where gradients are dead, unlike the gradient family. Its accuracy "
   "depends on the number of masks and on how informative random masking is for a given output. "
   "On smooth content outputs it does reasonably well. On a discrete position output, masking a "
   "single discrete cause is less informative, so the estimate is noisier and its faithfulness "
   "drops. RISE is a clean example of a perturbation method that is genuinely causal in its "
   "mechanism but pays for using random rather than targeted interventions.",
 "lime":
   "LIME explains one prediction by fitting a simple model nearby. Around the input of interest "
   "it generates many perturbed versions, records the output for each, and then fits a sparse "
   "linear surrogate to that local data. The weights of the surrogate are read as the "
   "explanation: which inputs, locally, push the output up or down. The idea is that even a "
   "complicated system is roughly linear in a small neighbourhood, so a linear fit there is both "
   "faithful and easy to read. LIME is model-agnostic and one of the most cited attribution "
   "methods. We apply it to the VCS by perturbing the candidate causes around the live state, "
   "re-running the program to get outputs, and fitting a local linear model whose weights we "
   "correlate with the true causal map. Because the perturbations are real re-runs, LIME works on "
   "a sprite's position where gradients fail. It fits both content and position structure "
   "reasonably well, so it lands among the more faithful attribution methods. Its main caveats "
   "are the usual ones: the explanation depends on how the neighbourhood is sampled and on the "
   "surrogate's fit quality, so it can be unstable across runs. On the VCS we can check its "
   "surrogate against the exact answer, which is a test the method never gets on a real network.",
 "kernelshap":
   "KernelSHAP estimates Shapley values, the game-theoretic fair share of each input. The "
   "Shapley value asks how much each input contributes to the output on average, over every "
   "possible order in which the inputs could be added. It is the unique attribution that "
   "satisfies a set of fairness axioms, which is why it is widely trusted. Computing it exactly "
   "is exponential, so KernelSHAP approximates it by sampling many coalitions, subsets of inputs "
   "that are present, evaluating the output for each, and solving a weighted least-squares "
   "problem whose solution approximates the Shapley values. We apply it to the VCS by sampling "
   "coalitions of candidate causes, re-running the program for each, and correlating the "
   "estimated Shapley values with the true causal contributions. Because every coalition is a "
   "real re-run of the machine, the method works on a sprite's position where gradients fail. "
   "Given enough coalitions it recovers much of the true contribution on both content and "
   "position outputs, so it is among the more faithful attribution methods, and it reports its "
   "own completeness as a check. Its cost is the number of coalitions needed to converge. On the "
   "VCS we can watch that convergence against the exact answer, which turns a theoretical fairness "
   "guarantee into a measured faithfulness number.",
 "counterfactual":
   "An on-distribution counterfactual asks the smallest realistic change that flips the output. "
   "Instead of occluding with an artificial value, it edits the input toward a genuine "
   "alternative the system could actually produce, and finds the smallest such edit that changes "
   "the output. The explanation is the edit itself: 'change these cells to these plausible "
   "values and the outcome changes'. The point of the on-distribution constraint is to answer the "
   "objection that occlusion and clamping can set an input to a value the running system would "
   "never create, making the explanation off-manifold. We apply it to the VCS by substituting "
   "candidate cells toward a real alternative state, the RAM of another frame of the same ROM, "
   "and re-running the bit-exact program. Because the substituted values are ones the machine "
   "genuinely produced, the edit stays on the data manifold by construction, and because we "
   "re-render the whole machine it is a valid intervention. We compare the minimal edit to the "
   "oracle's minimal cause set. The method works on a sprite's position where gradients fail. Its "
   "faithfulness here is limited: on some frames no on-distribution content byte varies between "
   "frames, so the search finds nothing and the map is flat, and even when it does edit, the "
   "single-edit search only partly recovers the true minimal set.",
 "na_audit":
   "This entry is an honesty audit, not a method that runs. Three well-known techniques, "
   "Grad-CAM, attention rollout, and VIPER, cannot be applied to the VCS at all, and we record "
   "that fact rather than force a number. Grad-CAM weights the final convolutional feature maps "
   "of a network by the gradient of the class score, so it needs a convolutional layer that the "
   "VCS does not have. Attention rollout multiplies the attention matrices across a transformer's "
   "layers, so it needs attention weights that the VCS does not have. VIPER distils a learned "
   "policy into a decision tree, so it needs a trained policy, and the VCS runs a fixed ROM, not "
   "a learned agent. The VCS is a 6502 CPU with a TIA video chip and a RIOT, executing "
   "hand-written machine code. Across all 42 scored games the count of each required substrate, "
   "convolutional layer, attention matrix, learned policy, is exactly zero. Re-targeting "
   "Grad-CAM's pooling onto the raw register state would simply be the content-path gradient "
   "under a different name, so it would add nothing. We therefore mark these methods as 'does not "
   "apply', a structural absence that is the same in every game. Recording this openly is part of "
   "the audit: it shows that some popular tools have no purchase on a real artifact that lacks the "
   "structure they assume.",
 "activation_patching":
   "Activation patching, also called causal mediation, is a core mechanistic-interpretability "
   "method. It runs the system once, records an internal value, and then, on a second run, "
   "overwrites, or 'patches', that internal value with the one recorded from a different input. "
   "The change in the output measures how much that internal value causally mediates the "
   "behaviour. In language-model work this is how researchers localise where a fact or a "
   "computation lives inside the network. The idea is genuinely causal: it does not correlate, it "
   "substitutes a value and observes the effect. We apply it to the VCS by patching a recorded "
   "state value, a RAM cell or register, from one run into another and measuring the causal "
   "effect on the output. On this machine the operation is special. Clamping a cell to a donor "
   "value and re-running the real ROM is literally the same operation the intervention oracle "
   "performs. So activation patching cannot disagree with the oracle: it recovers the exact "
   "causal-effect table by construction. This makes it the clearest demonstration that where a "
   "method's operation is itself a real intervention on a fully observable machine, faithfulness "
   "is automatic. The catch, taken up elsewhere, is that recovering the exact effect table is "
   "still not the same as recovering what the patched variables mean.",
 "das":
   "Distributed Alignment Search, or DAS, looks for a high-level concept inside the "
   "low-level state. It learns a rotation of the internal representation so that a chosen "
   "direction, or subspace, lines up with a supplied causal variable. It then tests the "
   "alignment by an interchange intervention: swap that subspace between two runs and check "
   "whether the output changes exactly as the concept would predict. The method is a leading "
   "tool for arguing that a network represents an abstract variable in a specific, possibly "
   "distributed, part of its activations. We apply it to the VCS by aligning a learned subspace "
   "of the state to a known game variable and measuring interchange accuracy against the true "
   "variable. On this transparent machine the alignment is exact: swapping the aligned subspace "
   "reproduces the concept's effect perfectly, while a mis-aligned control does not. There is an "
   "important caveat that the VCS makes explicit. The concept DAS aligns to is one we supplied "
   "from an external label, not one the method discovered. So a perfect interchange accuracy "
   "confirms an alignment to a given meaning; it does not show the method found the meaning on "
   "its own. DAS is thus a strong test of whether a supplied concept is causally real in the "
   "state, and a clean example of the difference between verifying a label and discovering one.",
 "attribution_patching":
   "Attribution patching is a fast, gradient-based approximation to activation patching. Exact "
   "patching requires a separate re-run for every internal value tested, which is expensive on a "
   "large network. Attribution patching instead uses a first-order, linear estimate: it "
   "multiplies the gradient at each site by the change that patching would apply, to predict the "
   "patch's effect in a single backward pass. This lets researchers screen thousands of sites at "
   "once, at industrial scale, and only re-run the exact patch on the promising ones. We apply it "
   "to the VCS by forming the gradient-linear estimate of each site's patch effect and comparing "
   "it to the exact effect the oracle measures. The comparison is unusually clean here because we "
   "have the exact answer for every site. The linear surrogate keeps the edges precise: it is "
   "usually right about which dependencies exist, because a real edge tends to have a non-zero "
   "gradient. What it gets wrong is the size of an edge, since the true relationship is not "
   "linear, so its numerical agreement with the exact patch is only partial. Attribution "
   "patching is therefore a good cheap screen with a measurable error, and the VCS lets us "
   "quantify exactly how much accuracy the approximation trades away for its speed.",
 "path_patching":
   "Path patching recovers a circuit by patching along specific routes. Rather than measuring "
   "whether one internal value matters overall, it patches a value only where it flows along a "
   "chosen path to the output, so it can separate a direct effect from an indirect one that "
   "passes through other components. This is the technique behind the well-known "
   "indirect-object-identification circuit in GPT-2, where it isolated exactly which heads send "
   "signals to which. We apply it to the VCS by patching along directed edges of the candidate "
   "data-flow and comparing the recovered edge set to the true routine from the disassembly. The "
   "idea is powerful because it targets the wiring, not just the nodes. On the VCS its limit is "
   "recall in a single frame. When we run it as a one-step directed do-operation, it recovers "
   "only the paths that carry a surviving signal in that single step. Many true dependencies are "
   "re-derived within the frame through other cells, so a one-step path does not carry them and "
   "the method misses them. On some games where the direct circuit is genuinely empty it matches "
   "perfectly; on others it recovers little. Path patching is thus precise about what it finds "
   "but silent about dependencies that flow through intermediate cells, a recall limit we can "
   "measure exactly against the known routine.",
 "acdc":
   "ACDC automates circuit discovery. Instead of a human proposing a circuit and patching it by "
   "hand, ACDC starts from the full computational graph and prunes edges one at a time. It "
   "removes an edge by resample-ablation, replacing the signal on that edge with a resampled "
   "value, and keeps the edge only if removing it hurts performance beyond a threshold. Sweeping "
   "the threshold traces out a family of candidate circuits. The output is a minimal subgraph "
   "that, in principle, carries the computation, found automatically. This is one of the main "
   "tools for scaling mechanistic interpretability beyond hand analysis. We apply it to the VCS "
   "by pruning the candidate data-flow graph and comparing the discovered circuit to the true "
   "routine over a threshold sweep. The VCS gives ACDC an unusually fair test, because we have "
   "the exact wiring and can also run a positive control that prunes under exhaustive exact "
   "interventions. Against that control ACDC's shortfall is a real property of single-shot "
   "discovery, not a broken scorer. It recovers a fraction of the true edges: its precision is "
   "reasonable but its recall is limited, because dependencies re-derived through other cells "
   "survive the resample and are pruned away. On one game it recovers a perfect, minimal circuit "
   "that nonetheless fails to regenerate the behaviour, which is one of the paper's key "
   "separations between a correct graph and a working computation.",
 "sae":
   "A sparse autoencoder tries to break a tangled state into clean, single-meaning features. It "
   "is a small neural network trained to reconstruct the recorded state through a wide but "
   "sparsely active hidden layer, so that only a few hidden units fire for any given input. The "
   "hope is that each hidden unit becomes 'monosemantic', standing for one human-meaningful "
   "concept, which would untangle the superposition thought to hide features inside neural "
   "activations. Sparse autoencoders are a central, fast-growing tool in mechanistic "
   "interpretability. We train one on the recorded RAM trajectory of the VCS and match its "
   "learned features to known, verified game-variable cells. On this small state the "
   "reconstruction is nearly perfect and every game-variable cell the SAE is asked to match gets "
   "a matching feature. By the metric the field usually uses to claim a feature 'is' a concept, "
   "the SAE looks like a complete success. The VCS lets us apply a stronger test. We ablate the "
   "matched feature, re-run the real ROM, and ask whether the computation it is supposed to name "
   "actually changes. Often it does not: the feature is named but causally inert. The sparse "
   "autoencoder is therefore our sharpest example of a method that scores well on the standard "
   "matching metric while failing the causal test that only a known machine can provide.",
 "dictionaries":
   "Dictionary learning with NMF or PCA is a simpler cousin of the sparse autoencoder. It "
   "decomposes the recorded state into a small dictionary of components: PCA finds orthogonal "
   "directions of largest variance, NMF finds additive, non-negative parts. Each component is "
   "then matched to a known variable, on the hope that the decomposition recovers the machine's "
   "true internal factors without supervision. This is a classic, cheap way to look for "
   "structure in high-dimensional activity, and it predates the neural sparse-autoencoder "
   "approach. We run it on the VCS state and measure the fraction of dictionary components that "
   "match a known, verified variable. It does partially well: NMF's non-negativity separates the "
   "additive register basis better than PCA, which tends to mix several variables into one "
   "component. So NMF recovers cleaner named components while PCA blurs them. The deeper limit is "
   "the same one that defeats the sparse autoencoder. A component can match a variable by "
   "correlation while playing little causal role, so a good matched fraction does not guarantee "
   "the component is causally used. The VCS lets us check both the match and the causal use, and "
   "the two often disagree. Dictionary learning is thus a useful, interpretable baseline whose "
   "named components must still be tested by intervention before they can be trusted.",
 "causal_scrubbing":
   "Causal scrubbing is a strict test of a circuit hypothesis. Given a claim about which parts of "
   "a system carry a computation, it resamples, or 'scrubs', everything the hypothesis says is "
   "irrelevant, replacing those signals with values from other inputs, and checks whether the "
   "behaviour still holds. If the hypothesis is correct, scrubbing the irrelevant parts should "
   "not hurt performance; if it is wrong, performance should drop. It is one of the most "
   "demanding validation methods in mechanistic interpretability, because it tests a whole "
   "hypothesis at once rather than one edge. We apply it to the VCS by resampling activations "
   "outside a hypothesised circuit and measuring whether the program's behaviour is preserved. On "
   "this machine the true data-flow hypothesis, which we know from the disassembly, passes "
   "cleanly: scrubbing everything outside it preserves the behaviour on every game, while a "
   "deliberately wrong hypothesis fails, and the two are told apart on every game. This makes "
   "causal scrubbing exact here, because the resample-and-re-run operation is a genuine "
   "intervention on a fully observable machine. As with the other exact methods, the caveat is "
   "that confirming the correct wiring is not the same as recovering what the circuit means or "
   "how its variables combine into the computed function.",
 "linear_probing":
   "Linear probing asks whether a concept can be read out of the state by a simple classifier. A "
   "linear probe is trained to predict a concept, say the ball's position, from the internal "
   "state; if it succeeds, the concept is said to be 'encoded' there. The well-known danger is "
   "that a probe can succeed just because the concept happens to be linearly recoverable, not "
   "because the system uses it. Hewitt and Liang's control-task fix addresses this: train a "
   "second probe to predict a random label, and report selectivity, the probe's real accuracy "
   "minus its accuracy on the random control. High selectivity means the state genuinely encodes "
   "the concept rather than the probe memorising it. We apply this on the VCS, training probes "
   "for game concepts and subtracting the control-task baseline. The VCS then adds a test no "
   "neural network allows: we check, by intervention, whether the program actually uses each "
   "decodable cell. It often does not. A cell can be decodable, and pass the control-task check, "
   "yet be provably inert when we clamp it and re-run, because the value is present in the state "
   "but never causally used within the window. Linear probing is thus our clearest illustration "
   "of the present-versus-used gap: decodability shows information is there, but only the "
   "intervention oracle shows whether the program relies on it.",
 "logit_lens":
   "The logit lens reads a system's intermediate state through its own output decoder. In a "
   "transformer it takes the hidden state at some middle layer and applies the final "
   "output projection to it, as if that intermediate state were the last layer, to see what the "
   "model would 'say' at that point. This reveals how the prediction forms across depth, and the "
   "tuned-lens variant learns a small correction so the readout is more faithful at each stage. "
   "It is a standard, cheap tool for tracing how information develops inside a network. We build "
   "the analogue on the VCS by reading an intermediate value through the machine's own decoding, "
   "and comparing the decoded value to the true intermediate value at that site. On this "
   "transparent machine the readout is exact: the state is linearly readable at the right site, "
   "so the lens reproduces the true intermediate value on every game. This is expected, because "
   "the VCS keeps its variables in plain, addressable memory rather than in a distorted "
   "distributed code. The logit lens is therefore a clean positive result on the machine, and it "
   "sits with the exact causal methods. As with those, reading the right value at the right site "
   "confirms that the value is represented and readable, but it does not by itself explain how "
   "that value is computed or what role it plays.",
}

# ---------------------------------------------------------------------------
# Long-form "How it's scored" prose, keyed by method key (~150-250 words each).
# Explains the exact grading procedure against the intervention oracle: the
# primary metric, what F / S / M mean for THIS method, and (gradient family)
# why the discrete sprite-position output is the hard case. Rendered as the
# second prose section of m_<key>.html. If a key is missing, build_pages falls
# back to the short P2_METHOD_SCORED blurb.
# ---------------------------------------------------------------------------
P2_METHOD_HOWSCORED = {
 "A1_connectomics":
   "The score is a data-flow-graph F1 against the true read/write graph from the disassembly. "
   "An edge means that perturbing one RAM cell changes another. We build the recovered graph by "
   "single-shot perturbation and compare its edges to the true edges, taking the harmonic mean of "
   "precision, the fraction of recovered edges that are real, and recall, the fraction of real "
   "edges recovered. Faithfulness for this method is that edge F1: a graph-valued explanation is "
   "graded as a graph, not as a heat-map. The comparison is always against the intervention "
   "oracle's true wiring, never against another method. The hard part is that a single "
   "perturbation ripples through shared clocks and buffers, so the recovered graph contains many "
   "edges the real program does not have. Precision suffers, and the F1 stays low. There is no "
   "separate position hard-case here, because the method never claims a sprite position; it "
   "claims a wiring diagram. The audit box below reports the actual F1, aggregated across the "
   "scored games, so no single game dominates. Read the recovered-versus-true graph as the error: "
   "every edge in one panel but not the other is a mistake the metric counts.",
 "A2_lesions":
   "The score is a rank correlation between each cell's lesion importance and its true causal "
   "role. Lesion importance is how much the screen changes when the cell is frozen; the true "
   "role comes from the intervention oracle. We rank the cells by each and take the Spearman "
   "correlation of the two rankings, so the metric asks whether lesioning orders the cells the "
   "same way the oracle does. Faithfulness for this method is that rank correlation, and it is "
   "high, because lesioning is itself a real intervention on the machine. The grading is always "
   "against the oracle, not against any other method. The important limit the score does not "
   "penalise on its own is interactions. A single-unit lesion changes one cell at a time, so it "
   "is blind to causes that only act together. On some games the per-unit ranking misses a large "
   "fraction of the joint effects the oracle records, where a pair of cells swings the behaviour "
   "far beyond the sum of the two single lesions. So a near-perfect ranking still hides the way "
   "the cells combine. There is no position hard-case for this method; its output is a ranking, "
   "not a sprite coordinate. The audit box reports the measured correlation across the core "
   "games.",
 "A3_tuning":
   "The score is a spurious-tuning rate: the fraction of cells that are strongly tuned to a game "
   "variable yet are not among the oracle's causal cells. We first find the cells whose value "
   "tracks a game variable strongly, then ask the intervention oracle which of those cells "
   "actually cause the output. A tuned cell that is not causal is a false positive, and the "
   "spurious-tuning rate counts them. Faithfulness is one minus that rate, so a method that only "
   "flags true causes scores high and a method that flags many non-causal cells scores low. On "
   "the VCS the rate is high, because many cells co-vary with the beam clock, so tuning is cheap "
   "and often unrelated to causation. This is the same present-versus-used trap that also "
   "defeats linear probing: a cell can carry a variable without the program using it. The grading "
   "is always against the oracle. There is no sprite-position hard-case here, because the output "
   "is a tuning judgement, not a coordinate. Because per-game correlations for this family can go "
   "negative, the aggregate clips each per-game value at zero before averaging, so a method with "
   "no real signal lands near zero rather than at a spurious floor. The audit box reports the "
   "aggregated faithfulness.",
 "A4_correlations":
   "The score is a rank correlation between the measured cell-to-cell correlation structure and "
   "the oracle's true coupling. We compute how each pair of cells co-varies across the "
   "trajectory, assemble that into a coupling matrix, and compare it, by Spearman correlation, to "
   "the true coupling the oracle certifies. Faithfulness for this method is that correlation. It "
   "is only partial, because the method never intervenes; it only watches. On a clock-locked "
   "machine two cells can be strongly correlated because they share a driver, not because one "
   "influences the other. So the correlation matrix conflates true coupling with common drivers, "
   "and the score reflects that confusion. Sufficiency, whether the recovered structure can "
   "predict held-out interventions, is near zero for this method, because a correlation matrix "
   "carries no mechanism to re-generate behaviour. The grading is always against the oracle, not "
   "against another method. There is no sprite-position hard-case here; the output is a coupling "
   "matrix. The audit box reports the measured faithfulness and, where the paper defines it, the "
   "sufficiency, each aggregated across the scored games so no single game dominates the number.",
 "A5_lfp":
   "The score is the fraction of the pooled-activity power spectrum that the known clocks "
   "explain. We pool a region of memory into one signal, take its power spectrum, and measure how "
   "much of that spectrum is accounted for by the frame clock and the scanline clock, whose "
   "frequencies we know exactly. Faithfulness is one minus that clock-explained fraction, because "
   "the part of the spectrum that is just the hardware's timing is epiphenomenal, not "
   "computation. A high clock-explained fraction means the impressive-looking spectrum is mostly "
   "the machine's pacemaker, so the method's faithfulness is low. The grading is always against "
   "the intervention oracle, which certifies what is truly computational. There is no "
   "sprite-position hard-case here; the output is a spectrum, not a coordinate. This method is a "
   "calibration baseline: it shows how a standard neuroscience signal can look rich while "
   "carrying little of the real computation. The audit box reports the measured faithfulness "
   "across the scored games, so the number reflects the whole scored set rather than one game where "
   "the clocks happen to dominate more or less.",
 "A6_granger":
   "The score is the false-edge rate of the Granger-inferred graph against the true data-flow. We "
   "let Granger causality infer directed edges from the timing of CPU, TIA, and RIOT activity, "
   "then compare each inferred edge to the true wiring the oracle certifies. An inferred edge "
   "that is not real is a false edge, and the false-edge rate counts them; faithfulness is one "
   "minus that rate. On the VCS the rate is at or near one, because the machine is clock-locked "
   "and almost every signal is predictable from almost every other one cycle earlier. So Granger "
   "causality infers edges everywhere, and nearly all of them are spurious. Neither a longer lag "
   "nor a stricter threshold repairs this, because the shared clock, not a real dependency, "
   "creates the predictability. The grading is always against the intervention oracle, which is "
   "the only thing that separates precedence from causation here. There is no sprite-position "
   "hard-case for this method; its output is a directed graph. The audit box reports the measured "
   "faithfulness across the scored games, and it is among the lowest in the battery, which is the "
   "point: timing-based causal inference fails on a machine whose timing is shared.",
 "A7_dimred":
   "The score is the fraction of NMF or PCA components that match a known signal or variable. We "
   "decompose the state tensor into components, then match each to the machine's known internal "
   "signals, the frame or scanline clock, a read/write pattern, vsync, or a game variable, and "
   "count the matched fraction against what the oracle marks as important. Faithfulness for this "
   "method is that matched-component fraction. It is moderate: the decomposition recovers most of "
   "the known factors, and NMF's non-negativity fits the additive register basis a little better "
   "than PCA, but a substantial fraction of components stay unmatched or mix several signals into "
   "one. A secondary reconstruction-error composite gives NMF a small edge, but on the primary "
   "matched-fraction metric the two are close. The grading is always against the oracle, not "
   "against another method. There is no sprite-position hard-case here; the output is a set of "
   "components. The audit box reports the measured matched fraction across the scored games, so the "
   "number is an average over the scored set rather than the best single game.",
 "A8_wholestate":
   "The score that matters for this method is minimality, not faithfulness. Recording the whole "
   "state is perfectly faithful, because it contains every cause, and perfectly sufficient, "
   "because it holds the whole machine and can predict any intervention. So F and S are both at "
   "the ceiling by construction. Minimality is defined as the size of the true minimal cause set "
   "divided by the size of the set the explanation names. The whole-state dump names all 128 RAM "
   "cells, while the true cause of any output is a handful of them, so the ratio is tiny and the "
   "minimality is very low. That is the whole point of including this baseline: a complete record "
   "is faithful and sufficient yet says nothing in particular, because it withholds nothing. The "
   "grading is always against the intervention oracle, which supplies the true minimal set. There "
   "is no sprite-position hard-case here; the method makes no positional claim. The audit box "
   "reports the exact per-axis triad, F, S, and M, read straight from the leaderboard, so you can "
   "see faithfulness and sufficiency at the ceiling beside a minimality near the floor.",
 "saliency":
   "The score is the Pearson correlation of the saliency map with the oracle's exact causal map, "
   "the set of true effect sizes over the candidate causes, together with deletion and insertion "
   "area-under-curve as quality measures. Faithfulness for this method is that raw correlation, "
   "reported without any rescaling, so a map that carries no causal signal lands at zero rather "
   "than at a spurious floor. The grading is always against the intervention oracle, never "
   "against another method. The decisive hard case is a sprite's position. The VCS sets a "
   "sprite's screen position by the timing of a strobe write, a discrete step, so the naive "
   "gradient there is provably exactly zero and its correlation with the oracle is zero. To make "
   "the audit as strict as possible we also turn on the differentiable bilinear sampler, which "
   "interpolates that discrete boundary and restores a non-zero position gradient. Even then the "
   "restored gradient is faithful on only a minority of games, so the position score stays low. "
   "On smooth content outputs, by contrast, the gradient points at the true causal byte and "
   "scores high. The audit box below reports the all-regime faithfulness, which averages the "
   "strong content result and the weak position result across the scored games.",
 "gradxinput":
   "The score is the Pearson correlation of the Grad-times-Input, DeepLIFT-style attribution with "
   "the oracle's exact causal map, reported raw. Faithfulness for this method is that "
   "correlation, and the grading is always against the intervention oracle rather than another "
   "method. Multiplying the gradient by the input concentrates the attribution onto the genuine "
   "causal byte, so on smooth content outputs it improves on plain saliency and scores well. The "
   "hard case is the same as for the whole gradient family. A sprite's position is a discrete "
   "step set by strobe timing, so the naive gradient there is provably zero, and multiplying a "
   "zero gradient by the input is still zero. The audit turns on the differentiable sampler to "
   "give the method a fair chance on position, but the restored gradient is faithful on only a "
   "minority of games, so the position score stays low while the content score stays high. The "
   "completeness property, that attributions sum to the output change, does not rescue the "
   "position regime. The audit box reports the all-regime faithfulness, which averages the strong "
   "content result and the weak position result across the 42 scored games, so the single number "
   "sits between the two.",
 "guided_backprop":
   "The score is the Pearson correlation of the guided-backprop map with the oracle's exact "
   "causal map, reported raw, and the grading is always against the intervention oracle. On "
   "smooth content outputs the rectified backward pass partly tracks the true cause. The hard "
   "case is a sprite's position: it is a discrete strobe-timed step, so the naive gradient is "
   "provably zero, and here the method's rectified path leaves it at zero rather than restoring "
   "any signal. On top of that, guided backprop fails the Adebayo program-randomisation sanity "
   "check on the machine itself. We scramble the ROM so the program boots to a completely "
   "different state, and the content-path map stays essentially unchanged, because its Jacobian "
   "is a constant one-hot pointing at the read index, independent of what the program computes. A "
   "faithful explanation must change when the model changes; this one does not. So even where it "
   "looks sharp, it is model-invariant, which the intervention oracle exposes at once by "
   "distinguishing the two programs. Between the zero position gradient and the failed sanity "
   "check, its cross-game faithfulness stays low. The audit box reports the measured all-regime "
   "faithfulness across the scored games.",
 "smoothgrad":
   "The score is the Pearson correlation of the SmoothGrad map with the oracle's exact causal "
   "map, reported raw, and the grading is always against the intervention oracle. Averaging the "
   "gradient over input noise denoises the map, so on smooth content outputs it gives a small "
   "improvement over plain saliency. The hard case is a sprite's position. The position is a "
   "discrete step set by strobe timing, so the naive gradient there is provably exactly zero, and "
   "averaging many zeros is still zero: adding input noise cannot create a gradient the hard "
   "machine does not provide. To be strict, the audit turns on the differentiable sampler, which "
   "restores a non-zero position gradient, but that restored gradient is faithful on only a "
   "minority of games, so the position score stays low. The content score stays high. So the "
   "all-regime faithfulness sits in the middle, as an average of the strong content result and "
   "the weak position result across the 42 scored games. The audit box reports that measured "
   "number. SmoothGrad is a clean test of whether the popular denoise-the-gradient fix rescues "
   "attribution on discrete game logic, and the score shows that it does not.",
 "ig_baseline_sweep":
   "The score is the Pearson correlation of the Integrated-Gradients attribution with the "
   "oracle's exact causal map, reported raw, at the headline baseline, with the baseline swept to "
   "show the dependence. Faithfulness for this method is that correlation, graded always against "
   "the intervention oracle. On smooth content outputs it behaves like a stronger saliency and "
   "scores well, and sweeping the baseline moves the magnitude of the attribution but not its "
   "correlation with the truth, because the constant one-hot Jacobian keeps the mass on the same "
   "byte. The hard case is a sprite's position. The position is a discrete strobe-timed step, so "
   "the naive gradient along the whole integration path is zero, and the integral of zero is "
   "zero; completeness does not rescue it. The audit turns on the differentiable sampler to give "
   "the method a fair chance on position, but the restored gradient is faithful on only a "
   "minority of games, so the position score stays low. The audit box reports the all-regime "
   "faithfulness, which averages the strong content result and the weak position result across "
   "the scored games. This makes Integrated Gradients a key reference for what the completeness "
   "axiom can and cannot fix on discrete outputs.",
 "expected_gradients":
   "The score is the Pearson correlation of the Expected-Gradients attribution with the oracle's "
   "exact causal map, reported raw, and the grading is always against the intervention oracle. "
   "Expected Gradients averages Integrated Gradients over a distribution of baselines, which "
   "makes the attribution provably stable across reference draws, but stability is not accuracy. "
   "Each term carries the factor input-minus-baseline, so when the reference pool holds bytes "
   "that are constant across states, that factor is zero and the attribution collapses. Even on "
   "the redesigned gameplay reference pool, and with the differentiable sampler turned on for the "
   "position regime, the method carries almost no true causal signal, and it is the "
   "lowest-faithfulness attribution method in the study. On a sprite's position it still vanishes "
   "for the usual reason: the naive gradient is zero on a discrete strobe-timed step and "
   "averaging baselines does not change that. Its plausibility proxy is high, which places it "
   "squarely in the danger zone of high plausibility with low faithfulness. The audit box reports "
   "the measured all-regime faithfulness across the 42 scored games, so the low number is an "
   "average over the whole scored set, not one unlucky game.",
 "occlusion":
   "The score is the Pearson correlation of the occlusion map with the oracle's exact causal map, "
   "reported raw, and the grading is always against the intervention oracle. Occlusion is itself "
   "a coarse intervention: we set each candidate cause to an occluded value, re-run the bit-exact "
   "program, and record the change in the output. Because that is a genuine do-operation on the "
   "real machine, its map tracks the oracle closely, and it is among the most faithful "
   "attribution methods. Crucially, there is no sprite-position collapse here. Where the gradient "
   "family scores zero on a discrete position output, occlusion still works, because perturbing "
   "and re-running does not depend on a derivative. So its position-regime faithfulness stays "
   "well above the gradient methods', and its all-regime faithfulness is the highest among the "
   "attribution methods. Its only shortfall relative to the exact oracle is granularity: it "
   "perturbs at the size of the occluder and can miss fine or joint effects that the oracle "
   "captures exactly. The audit box reports the measured all-regime faithfulness across the 42 "
   "scored games. Occlusion is the clearest attribution-side evidence that a method is faithful "
   "exactly when its mechanism is a valid intervention on the real system.",
 "perturbation":
   "The score is the Pearson correlation of the learned mask with the oracle's exact causal map, "
   "together with the mask's overlap, measured as intersection-over-union, against the true "
   "minimal cause set. The grading is always against the intervention oracle. Every entry of the "
   "mask is a real occlude-and-re-run, so the method is a valid intervention and works even on a "
   "sprite's position, where the gradient family scores zero. That is why there is no "
   "position-collapse here. Its faithfulness is moderate rather than high: the area-bounded "
   "optimisation finds a compact set that partly overlaps the true causes but does not exactly "
   "match the oracle's minimal set, so both the correlation and the mask overlap are partial. "
   "Minimality is central to this method, because it explicitly seeks the smallest disruptive "
   "mask, so the score rewards a mask that is both on-target and small. The grading never "
   "compares the mask to another method's mask, only to the oracle's minimal set. The audit box "
   "reports the measured all-regime faithfulness across the 42 scored games, so the number "
   "reflects the whole scored set rather than the single game where the optimised mask happens to "
   "line up best with the true causes.",
 "rise":
   "The score is the Pearson correlation of the RISE importance map with the oracle's exact "
   "causal map, reported raw, and the grading is always against the intervention oracle. RISE "
   "forms its map by averaging the output over hundreds of random masks, each a real re-run of "
   "the machine, so it is a genuine perturbation method and works on a sprite's position where "
   "the gradient family scores zero. That is why it does not collapse on the position regime. Its "
   "faithfulness is moderate: on smooth content outputs the random-mask average recovers the true "
   "causes reasonably well, but on a discrete position output masking a single discrete cause is "
   "less informative, so the estimate is noisier and the position score drops below the content "
   "score. The accuracy also depends on how many masks are drawn, since RISE is a Monte-Carlo "
   "estimate. The grading is always against the oracle, never against another method. The audit "
   "box reports the measured all-regime faithfulness across the 42 scored games, which averages "
   "the stronger content result and the weaker position result. RISE shows that a truly "
   "perturbation-based method can beat the gradient family on discrete outputs while still paying "
   "for using random rather than targeted interventions.",
 "lime":
   "The score is the Pearson correlation of the local linear surrogate's weights with the "
   "oracle's exact causal map, reported raw, and the grading is always against the intervention "
   "oracle. LIME fits a sparse linear model to real re-runs of the machine around the live state, "
   "so its perturbations are valid interventions and it works on a sprite's position where the "
   "gradient family scores zero. That is why there is no position collapse here. It fits both "
   "content and position structure reasonably well, so it lands among the more faithful "
   "attribution methods. The surrogate also reports its own fit quality, the local R-squared, "
   "which acts as an internal check on how linear the neighbourhood really is. The main "
   "caveats the score reflects are the usual ones for LIME: the explanation depends on how the "
   "neighbourhood is sampled and on the surrogate's fit, so it can vary across runs, and we "
   "report that stability alongside the correlation. The grading never compares LIME to another "
   "method, only to the oracle. The audit box reports the measured all-regime faithfulness across "
   "the 42 scored games, so the number reflects the whole scored set. On the VCS we can test the "
   "surrogate against the exact answer, which a real network never allows.",
 "kernelshap":
   "The score is the Pearson correlation of the estimated Shapley values with the oracle's exact "
   "causal map, reported raw, and the grading is always against the intervention oracle. "
   "KernelSHAP samples coalitions of candidate causes, re-runs the machine for each, and solves a "
   "weighted least-squares problem to approximate each cause's Shapley value, so every coalition "
   "is a real intervention and the method works on a sprite's position where the gradient family "
   "scores zero. That is why it does not collapse on the position regime. Given enough coalitions "
   "it recovers much of the true contribution on both content and position outputs, so it is "
   "among the more faithful attribution methods. It also reports its own completeness, that the "
   "Shapley values sum to the difference between the intact and fully masked outputs, which is an "
   "internal consistency check. The cost the score implicitly reflects is convergence: too few "
   "coalitions leave the estimate noisy. The grading is always against the oracle, never against "
   "another method. The audit box reports the measured all-regime faithfulness across the 42 "
   "scored games. On the VCS the theoretical fairness of the Shapley value becomes a measured "
   "faithfulness number, tested against an exact answer the method never sees on a real network.",
 "counterfactual":
   "The score is the Pearson correlation of the minimal counterfactual edit with the oracle's "
   "exact causal map, together with the edit's minimality against the oracle's minimal cause set, "
   "and the grading is always against the intervention oracle. The method substitutes candidate "
   "cells toward a real alternative state, another frame of the same ROM, and re-runs the "
   "bit-exact program, so the edit is on-distribution and a valid intervention, and it works on a "
   "sprite's position where the gradient family scores zero. That is why there is no position "
   "collapse here in principle. Its faithfulness is nonetheless only moderate. On some frames no "
   "on-distribution content byte varies between frames, so the search finds no valid edit and the "
   "map is flat; we mark those cells invalid rather than score them as important. Even when a "
   "valid edit exists, the single-edit search only partly recovers the true minimal set, so the "
   "correlation stays modest. Minimality is central here, because the method explicitly seeks the "
   "smallest flipping edit. The grading is always against the oracle. The audit box reports the "
   "measured all-regime faithfulness across the 42 scored games, so the moderate number reflects both "
   "the no-op frames and the partial recovery on the frames where the search does run.",
 "na_audit":
   "This entry is not scored on the faithfulness leaderboard, and that is deliberate. Grad-CAM, "
   "attention rollout, and VIPER need a convolutional layer, an attention matrix, or a learned "
   "policy that the VCS does not have. Across all 42 scored games the count of each required "
   "substrate is exactly zero, against many genuine candidate causes per game for the methods "
   "that do apply. A faithfulness number would be meaningless, because there is no map to compare "
   "to the oracle. So we record a measured structural fact, applies equals false, rather than "
   "force a score. We also note that re-targeting Grad-CAM's pooling onto the raw register state "
   "would only be the content-path gradient under another name, so it would add no new "
   "information. The grading standard elsewhere in the audit is always correlation, F1, or "
   "effect-agreement against the intervention oracle; here none of those is defined, because the "
   "method cannot run. The audit box below therefore marks this row as excluded from the "
   "leaderboard rather than showing an F, S, and M triad. Recording the absence openly is itself "
   "a finding: several popular tools have no purchase on a real artifact that lacks the neural "
   "structure they assume.",
 "activation_patching":
   "The score is the maximum absolute difference between the recovered patch effect and the exact "
   "patch effect from the oracle, taken across every site and every core game. The grading is "
   "always against the intervention oracle. This method is exact by construction, so the "
   "difference is zero. Clamping a cell to a donor value and re-running the real ROM is literally "
   "the same operation the intervention oracle performs, so the recovered effect table cannot "
   "disagree with the oracle, and site precision and recall are both perfect. Faithfulness for "
   "this method is therefore at the ceiling, level with the oracle-as-method positive control. "
   "There is no sprite-position hard-case here: because the method intervenes on the real machine "
   "rather than taking a derivative, it works on position outputs exactly as it works on content "
   "outputs. The important qualification, which the score does not by itself capture, is that "
   "recovering the exact effect table is not the same as recovering what the patched variables "
   "mean or how they combine into the computed function. The audit box reports the measured "
   "triad. This method is the clearest demonstration that when a method's operation is itself a "
   "real intervention on a fully observable machine, faithfulness is automatic.",
 "das":
   "The score is the interchange accuracy of the aligned subspace against the true variable. We "
   "align a learned subspace of the state to a supplied concept, then swap that subspace between "
   "two runs and check whether the output changes exactly as the concept predicts; the fraction "
   "of correct interchanges is the accuracy. The grading is against the intervention oracle, and "
   "on this transparent machine the alignment is exact, so interchange accuracy is at the "
   "ceiling, while a mis-aligned control scores zero. Faithfulness for this method is that "
   "interchange accuracy. There is no sprite-position hard-case here, because DAS operates by "
   "intervention rather than by a gradient. The essential caveat the audit makes explicit is that "
   "the concept DAS aligns to is one we supplied from an external label, not one the method "
   "discovered. So a perfect interchange accuracy is guaranteed in advance: it confirms an "
   "alignment to a given meaning, and does not show the method found the meaning on its own. The "
   "grading is always against the oracle, never against another method. The audit box reports the "
   "measured triad. DAS is thus a clean example of the difference between verifying a supplied "
   "concept and discovering one, which only a machine with known ground truth can separate.",
 "attribution_patching":
   "The score is the Pearson correlation between the gradient-approximate patch effect and the "
   "exact patch effect from the oracle. The grading is always against the intervention oracle. "
   "Attribution patching forms a first-order, linear estimate of each site's patch effect in a "
   "single backward pass, so we can compare that estimate directly to the exact effect the oracle "
   "measures at the same site. Faithfulness for this method is that correlation, and it is good "
   "but not perfect, because the true relationship is not linear. The linear surrogate keeps the "
   "edges precise: it is usually right about which dependencies exist, since a real edge tends to "
   "have a non-zero gradient, so edge precision and recall stay high. What it gets wrong is the "
   "size of an edge, which is why the numerical agreement with the exact patch is only partial. "
   "There is no sprite-position collapse framed as such here, because the method is scored on "
   "internal patch effects rather than on a rendered position; but its gradient origin is exactly "
   "why it approximates rather than matches. The grading is always against the oracle. The audit "
   "box reports the measured triad, so the correlation is an average across the scored games, "
   "quantifying how much accuracy the fast approximation trades away for its speed.",
 "path_patching":
   "The score is the F1 of the recovered path or circuit against the true routine's data-flow. We "
   "patch along directed edges of the candidate data-flow and compare the recovered edge set to "
   "the true routine from the disassembly, taking the harmonic mean of edge precision and recall. "
   "The grading is always against the intervention oracle's true wiring. Faithfulness for this "
   "method is that circuit F1. Run as a single-frame, one-step directed do-operation, the method "
   "recovers only the paths that carry a surviving signal in that single step. Many true "
   "dependencies are re-derived within the frame through other cells, so a one-step path does not "
   "carry them and recall suffers, which is the main limit the score reflects. On some games the "
   "direct circuit is genuinely empty and the method matches it perfectly; on others it recovers "
   "little. There is no sprite-position hard-case here, because the output is a circuit, not a "
   "coordinate. The grading is always against the oracle, never against another method. The audit "
   "box reports the measured triad across the scored games, so the F1 is an average over the scored "
   "set. Path patching is precise about the paths it finds but silent about dependencies that "
   "flow through intermediate cells.",
 "acdc":
   "The score is the best F1 of the auto-discovered circuit against the true data-flow over a "
   "threshold sweep. ACDC prunes edges by resample-ablation, keeping an edge only if removing it "
   "hurts performance beyond the current threshold, and sweeping the threshold traces a family of "
   "candidate circuits from which we take the best F1 against the true routine. The grading is "
   "always against the intervention oracle's true wiring. Faithfulness for this method is that "
   "best F1, and it is partial: precision is reasonable but recall is limited, because "
   "dependencies re-derived through other cells survive the resample and get pruned away. A "
   "positive control runs the identical pruning under exhaustive exact interventions and reaches "
   "a perfect F1 on every game, so the shortfall is a real measurement of single-shot discovery, "
   "not a broken scorer. There is no sprite-position hard-case here; the output is a circuit. One "
   "of the paper's key separations comes from this method: on one game ACDC recovers a perfect, "
   "minimal circuit that nonetheless fails to regenerate the behaviour under held-out "
   "interventions, so sufficiency drops well below faithfulness. The grading is always against "
   "the oracle. The audit box reports the measured triad, so you can see faithfulness beside a "
   "lower sufficiency on the same discovered graph.",
 "sae":
   "The score is the fraction of sparse-autoencoder features that match a known, verified "
   "game-variable cell, together with a causal-use check by ablation. We first measure the "
   "matched fraction, how many of the concept cells the SAE was asked to represent get a matching "
   "feature, and on this small state that fraction is high. That is the metric the field usually "
   "uses to claim a feature is a concept. The VCS then applies a stronger test the audit treats "
   "as the real faithfulness: we ablate the matched feature, re-run the real ROM, and ask whether "
   "the computation it is supposed to name actually changes. Often it does not, so the causal "
   "faithfulness is low even though the matched fraction is high, and the sufficiency can even go "
   "negative. This split is the point: a feature can be named yet causally inert. There is no "
   "sprite-position hard-case here, because the method is scored on features and their ablation, "
   "not on a rendered position. Its plausibility proxy is high, which places it in the danger "
   "zone of high plausibility with low faithfulness. The grading is always against the "
   "intervention oracle, never against the matching metric alone. The audit box reports the "
   "measured triad, so you can read the low causal faithfulness directly.",
 "dictionaries":
   "The score is the fraction of NMF or PCA dictionary components that match a known, verified "
   "variable. We decompose the state into a small dictionary, match each component to a known "
   "variable, and count the matched fraction. The grading is always against the intervention "
   "oracle. Faithfulness for this method is that matched fraction, and it is partial: NMF's "
   "non-negativity separates the additive register basis better than PCA, so NMF yields cleaner "
   "named components while PCA blurs several variables into one component. The deeper limit, which "
   "the matched fraction alone does not expose, is causal use. A component can match a variable by "
   "correlation while playing little causal role, so a good matched fraction sits beside "
   "near-zero causal use, exactly as it does for the sparse autoencoder. There is no "
   "sprite-position hard-case here; the output is a set of components. The grading is always "
   "against the oracle, never against another method. The audit box reports the measured triad "
   "across the scored games, so the matched fraction is an average over the scored set. Dictionary "
   "learning is a useful interpretable baseline, but its named components must be tested by "
   "intervention before they can be trusted, and the score reflects that gap.",
 "causal_scrubbing":
   "The score is the behaviour preserved when resampling activations consistent with a "
   "hypothesised circuit. We scrub, that is resample, everything the hypothesis marks as "
   "irrelevant and measure whether the program's behaviour still holds. The grading is always "
   "against the intervention oracle. For the true data-flow hypothesis, which we know from the "
   "disassembly, scrubbing preserves the behaviour on every core game, so the preserved "
   "performance is at the ceiling; a deliberately wrong hypothesis fails, and the two are told "
   "apart on every game. Faithfulness for this method is that preserved-performance score, and it "
   "is exact here, because the resample-and-re-run operation is a genuine intervention on a fully "
   "observable machine. There is no sprite-position hard-case here, because the method works by "
   "intervention rather than by a gradient. The grading is a pass/fail of a whole hypothesis "
   "against the oracle, not a comparison to another method. The important qualification, which "
   "the score does not by itself capture, is that confirming the correct wiring is not the same "
   "as recovering what the circuit means or how its variables combine. The audit box reports the "
   "measured triad, with faithfulness at the ceiling on the correct hypothesis.",
 "linear_probing":
   "The score is the mean selectivity: the probe's accuracy minus the accuracy of a control-task "
   "probe trained on a random label, averaged over the labelled cells. The control task follows "
   "Hewitt and Liang and guards against a probe succeeding merely because a concept is linearly "
   "recoverable. The grading is always against the intervention oracle. Faithfulness for this "
   "method is that selectivity, and it is modest, but the deeper finding comes from crossing the "
   "probe against the oracle. Concepts are decodable above the control, so the information is "
   "present in the state, yet the oracle flags cells that are decodable but never causally used "
   "within the window. That is the present-versus-used gap: a value can be read out without the "
   "program relying on it. There is no sprite-position hard-case here; the output is a decodability "
   "judgement. The grading is always against the oracle, which is strictly stronger than the "
   "probe, because the probe asks whether a value can be read while the oracle asks whether "
   "changing it changes the output. The audit box reports the measured triad. Linear probing "
   "earns a high plausibility proxy on a low faithfulness, which places it in the danger zone and "
   "makes it the clearest illustration of decodable-but-unused information.",
 "logit_lens":
   "The score is the readout fidelity of the lens-decoded intermediate against the true "
   "intermediate value, reported as an R-squared. We read an intermediate value through the "
   "machine's own decoding and compare it to the true value at that site. The grading is always "
   "against the intervention oracle. On this transparent machine the readout is exact: the state "
   "is linearly readable at the right site, so the lens reproduces the true intermediate value on "
   "every core game and the fidelity is at the ceiling. Faithfulness for this method is that "
   "readout fidelity. There is no sprite-position hard-case here, because the method reads an "
   "internal value rather than a rendered position, and it does not rely on a gradient. The "
   "result is expected, because the VCS keeps its variables in plain, addressable memory rather "
   "than in a distorted distributed code. The important qualification, which the score does not by "
   "itself capture, is that reading the right value at the right site confirms the value is "
   "represented and readable, but does not explain how the value is computed or what role it "
   "plays. The grading is always against the oracle. The audit box reports the measured triad, "
   "with fidelity at the ceiling.",
}

# Map each method key -> its row name in the actual audit (leaderboard.json).
# na_audit is intentionally excluded from the leaderboard (no applicable causes).
P2_LEADER = {
    "A1_connectomics": "A1_connectomics", "A2_lesions": "A2_lesions", "A3_tuning": "A3_tuning",
    "A4_correlations": "A4_spike_word", "A5_lfp": "A5_local_field_potentials",
    "A6_granger": "A6_granger", "A7_dimred": "A7_dim_reduction", "A8_wholestate": "A8_wholestate",
    "saliency": "vanilla_saliency", "gradxinput": "gradxinput_deeplift",
    "guided_backprop": "guided_backprop", "smoothgrad": "smoothgrad",
    "ig_baseline_sweep": "integrated_gradients", "expected_gradients": "expected_gradients",
    "occlusion": "occlusion", "perturbation": "extremal_perturbation", "rise": "rise",
    "lime": "lime", "kernelshap": "kernelshap", "counterfactual": "on_distribution_counterfactual",
    "activation_patching": "activation_patching", "das": "interchange_interventions_das",
    "attribution_patching": "attribution_patching", "path_patching": "path_patching",
    "acdc": "ACDC", "sae": "sparse_autoencoder", "dictionaries": "nmf_pca_dictionaries",
    "causal_scrubbing": "causal_scrubbing", "linear_probing": "linear_probing_control_tasks",
    "logit_lens": "logit_tuned_lens",
}
