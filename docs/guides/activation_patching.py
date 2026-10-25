# Beginner's guide — activation patching / causal mediation.
# Rendered by build_pages.build_guide_section(); schema in docs/guides/__init__.py.

GUIDE = dict(
    key="activation_patching",
    level="Mechanistic family · the core causal tool",

    wiki=[
        ("Mechanistic interpretability", "https://en.wikipedia.org/wiki/Mechanistic_interpretability"),
        ("Mediation (statistics)", "https://en.wikipedia.org/wiki/Mediation_(statistics)"),
        ("Causal inference", "https://en.wikipedia.org/wiki/Causal_inference"),
        ("Transformer (deep learning)", "https://en.wikipedia.org/wiki/Transformer_(deep_learning)"),
        ("Ablation (artificial intelligence)", "https://en.wikipedia.org/wiki/Ablation_(artificial_intelligence)"),
    ],
    papers=[
        ("Vig et al. (2020). <i>Causal Mediation Analysis for Interpreting Neural NLP: The Case of "
         "Gender Bias.</i> NeurIPS.",
         "https://arxiv.org/abs/2004.12265"),
        ("Meng, Bau, Andonian &amp; Belinkov (2022). <i>Locating and Editing Factual Associations in "
         "GPT</i> (ROME, causal tracing). NeurIPS.",
         "https://arxiv.org/abs/2202.05262"),
        ("Zhang &amp; Nanda (2024). <i>Towards Best Practices of Activation Patching in Language "
         "Models: Metrics and Methods.</i> ICLR.",
         "https://arxiv.org/abs/2309.16042"),
        ("Hase, Bansal, Kim &amp; Ghandeharioun (2023). <i>Does Localization Inform Editing?</i> "
         "NeurIPS. (A critique: where patching localises a fact is not where editing works best.)",
         "https://arxiv.org/abs/2301.04213"),
    ],

    idea=r"""
<p>Gradient methods ask &ldquo;what if I nudged this a tiny bit?&rdquo;. Activation patching asks a
bolder, <b>causal</b> question: <i>what if this internal value had been the one from a different
input?</i> You run the network twice. The <b>clean run</b> uses the input you care about (&ldquo;The
Eiffel Tower is in&rdquo; &rarr; &ldquo;Paris&rdquo;). The <b>corrupted run</b> uses a slightly
different input that gives a different answer. Then you run a third time, and at one chosen place
inside the network you <b>overwrite</b> (&ldquo;patch&rdquo;) the internal value with the one recorded
in the other run. If the answer flips, that place <b>carries</b> the information that decides the
answer.</p>
<p>An internal value of a network is called an <b>activation</b>: the output of one neuron, one layer
or one attention head. Statisticians call a variable that sits on the causal path from cause to effect
a <b>mediator</b>, so the method is also called <b>causal mediation analysis</b>. It is the workhorse of
<b>mechanistic interpretability</b>, the attempt to reverse-engineer the algorithm a network runs.
Unlike a correlation, a patch is an <b>intervention</b>: it sets a value and watches what happens. On
the VCS we can do exactly the same thing to the machine's memory, and it turns out to be identical to
our ground-truth oracle.</p>""",

    nn=r"""
<p>Let \(x\) be the clean input and \(\tilde x\) the corrupted one. Let \(a_\ell(x)\) be the activation
of component \(\ell\) (a neuron, an attention head, or a position in the <b>residual stream</b>: the
running vector that every transformer layer reads from and adds to). Write
\(y(x \mid a_\ell := v)\) for the output when we run on \(x\) but force \(a_\ell\) to the value
\(v\). The <b>patch effect</b> of component \(\ell\) is</p>
\[ \Delta_\ell \;=\; y\big(x \,\big|\, a_\ell := a_\ell(\tilde x)\big) \;-\; y(x). \]
<p>This direction (corrupted value into the clean run) is called <b>noising</b>; the reverse
(clean value into the corrupted run) is <b>denoising</b>. In ROME&rsquo;s &ldquo;causal tracing&rdquo;
the corruption is Gaussian noise on the subject tokens, and the patched-back clean state shows
where a fact is recalled. Each \(\Delta_\ell\) costs one extra forward pass.</p>
<p><b>Toy example.</b> Take a two-step network \(h = 2x_1\), \(g = x_2 + 1\), \(y = h + 3g\). Clean
input \(x = (1, 0)\) gives \(h = 2\), \(g = 1\), \(y = 5\). Corrupted input \(\tilde x = (4, 0)\)
gives \(h = 8\), \(g = 1\), \(y = 11\). Patch \(h\): run on \(x\) but set \(h := 8\). Then
\(y = 8 + 3 = 11\), so \(\Delta_h = 6\): \(h\) mediates the whole difference. Patch \(g\): its
corrupted value is also 1, so \(\Delta_g = 0\). The patch tells us <i>which component carries the
change between these two inputs</i>, not which components matter in general. A different
\(\tilde x\) (say, changing \(x_2\)) would light up \(g\) instead.</p>""",

    recipe=[
        "Choose a clean input and a corrupted input that differ in the one thing you study, and a scalar metric (a logit difference, a probability).",
        "Run both and cache every activation you might patch.",
        "For each component \\(\\ell\\): re-run the clean input, overwrite \\(a_\\ell\\) with its cached corrupted value, let the rest of the network compute normally.",
        "Record \\(\\Delta_\\ell\\), the change in the metric, often normalised by the full clean-vs-corrupted gap.",
        "Plot \\(\\Delta_\\ell\\) over layers and positions: the bright cells are the candidate mediators.",
    ],

    atari_map=[
        ("The network and its forward pass", "The ROM program running on the bit-exact emulated VCS "
         "(<code>jutari</code>). One &ldquo;layer&rdquo; is one video frame of 6507 CPU work."),
        ("The residual stream at layer \\(\\ell\\)", "The whole machine state at a frame boundary: the 128 "
         "bytes of RAM plus TIA video registers. Every frame reads it and writes it back."),
        ("A component (neuron / head)", "One <b>state site</b>: a single RAM byte that the game uses as a "
         "variable (a candidate concept cell such as a ball coordinate), or one TIA register "
         "(the background colour <code>COLUBK</code>)."),
        ("Clean run / corrupted run", "Clean: the seeded random-action gameplay stream, checkpointed at "
         "frame \\(f^*{=}90\\). Donor: the same stream but with LEFT or RIGHT pressed at the last "
         "frame before \\(f^*\\)."),
        ("The logit / metric", "Outputs read 15 frames later: the values of the two most causally active "
         "candidate bytes, and the number of screen pixels that differ from the clean frame."),
        ("Ground truth", "The intervention oracle, which pokes a value and re-runs the real ROM. On the "
         "VCS that <i>is</i> a patch, so activation patching and the truth coincide."),
    ],

    atari=r"""
<p>The runner sits on the <b>shared testbed</b> that every method uses: a seeded stream of random
joystick actions, a checkpoint at frame \(f^*\), and a horizon of 15 frames. Two fresh boots are first
asserted to be byte-identical in RAM and screen, so any later difference is caused by our patch and
nothing else.</p>
<ol>
<li><b>Outputs.</b> The runner picks the two candidate RAM bytes whose own value moves most under the
oracle&rsquo;s interventions, and adds a whole-screen output: the count of pixels that differ from the
clean final frame.</li>
<li><b>Donor patches.</b> Two donor runs share the gameplay prefix and press LEFT or RIGHT just before
\(f^*\). For every candidate byte the donor&rsquo;s value is written into a copy of the clean
checkpoint. The record notes whether the donor actually differed there; equal values are honest
no-op patches.</li>
<li><b>Directed patches.</b> Because a one-frame donor leaves many bytes unchanged, every candidate
byte is also patched to <code>base+17</code> (the same value the oracle uses), plus one TIA patch
that sets the background colour to white.</li>
<li><b>Recovered vs exact.</b> Each patch is re-run on the clean checkpoint (the method). The same
write is then replayed from an <i>independent</i> fresh boot (the oracle). The two effect tables
are compared cell by cell. They are equal, because both are the same single-site write followed by
the same deterministic program.</li>
<li><b>Edge precision/recall.</b> A (patch, output) pair &ldquo;fires&rdquo; if its effect is non-zero.
The recovered firing pattern is scored against the exact one. An honest extra audit lists
<b>transient</b> sites: bytes the program overwrites on the next frame, so their patch never reaches
the outputs.</li>
</ol>
<p>The difference from LLM practice is worth stating plainly. In a transformer, patching and
&ldquo;the truth&rdquo; are different things, and we argue about which metric to trust. On the VCS,
patching a byte and re-running is the oracle&rsquo;s own definition of a cause. So this method is the
<b>reference</b> the other Phase&nbsp;C methods approximate.</p>""",

    code=[
        dict(path="tools/xai_study/phaseC_mechanistic/activation_patching.jl", find="function run_patch(",
             role="xai", desc="The patch itself: copy the checkpoint, write one RAM byte or TIA register, step the tail, snapshot."),
        dict(path="tools/xai_study/phaseC_mechanistic/activation_patching.jl", find="dl = continue_from(boot_replay(vcat(pre, ACT_LEFT)",
             role="xai", desc="The donor (&ldquo;corrupted&rdquo;) runs: same gameplay prefix, LEFT or RIGHT at the last step."),
        dict(path="tools/xai_study/phaseC_mechanistic/activation_patching.jl", find="rec_snap = run_patch(clean_ckpt",
             role="xai", desc="The recovered effect: patch into the clean checkpoint and re-run."),
        dict(path="tools/xai_study/phaseC_mechanistic/activation_patching.jl", find="ex_snap = run_patch(fresh_ckpt",
             role="xai", desc="The exact effect via an independent fresh boot, so equality is not a shared-code artefact."),
        dict(path="tools/xai_study/phaseC_mechanistic/activation_patching.jl", find="rec_fire  = abs.(recovered)",
             role="xai", desc="Site precision/recall of the recovered firing pattern against the exact data-flow."),
        dict(path="tools/xai_study/common/jutari_oracle.jl", find="function intervene_ram!(",
             role="substrate", desc="The write: one byte of RIOT RAM in the live emulator."),
        dict(path="tools/xai_study/common/jutari_oracle.jl", find="function intervene_tia!(",
             role="substrate", desc="The same write for a TIA video register (the background-colour patch)."),
        dict(path="tools/xai_study/common/shared_testbed_impl.jl", find="function build_shared_testbed(",
             role="substrate", desc="The shared gameplay state: seeded actions, checkpoint at \\(f^*\\), causes, cause-density gate."),
        dict(path="jutari/src/env/StellaEnvironment.jl", find="function env_step!(",
             role="substrate", desc="One bit-exact video frame of the real machine; every re-run is a loop over this."),
        dict(path="tools/xai_study/ground_truth/oracle_intervene.jl", find="function run_intervention(",
             role="substrate", desc="The oracle: apply a cause to a copy of the checkpoint and run the horizon. Same operation as a patch."),
    ],

    reading=r"""
<p>The headline number is the largest gap between the recovered and the exact effect table. Expect it
to be zero on every game: this is a <b>positive result by construction</b>, not a discovery. The more
informative parts are the firing pattern and the transient list. Many bytes show <b>no effect</b>
because the game recomputes them every frame. Others move a score byte but not the screen, or the
reverse. Read the page as a map of <i>which variables mediate which outputs over 15 frames</i>, not
as a test the method could fail.</p>""",

    pitfalls=[
        "<b>The answer depends on the corruption.</b> A patch measures what carries the difference between <i>these two</i> runs. A donor that does not change a byte can never reveal that byte, which is why we add directed patches.",
        "<b>Off-distribution values.</b> A directed value like <code>base+17</code>, or Gaussian noise in ROME, may put the system in a state it never visits naturally. The effect is real, but it may not describe normal behaviour.",
        "<b>Effect is not meaning.</b> A perfect effect table says which byte matters, not what it represents or how bytes combine. Hase et al. also found that where patching localises a fact is a poor guide to where editing it works.",
        "<b>Metric choice changes the picture.</b> Zhang &amp; Nanda show that logit difference, probability and noising vs denoising can rank components differently in LLMs.",
        "<b>Cost.</b> One full re-run per site and value. Fine for 128 bytes, expensive for billions of activations; this is what attribution patching tries to fix.",
    ],

    try_it="julia --project=jutari tools/xai_study/phaseC_mechanistic/activation_patching.jl --games core",
    try_note="Writes <code>activation_patching_&lt;game&gt;.json/.npz</code> and a core summary into "
             "<code>tools/xai_study/phaseC_mechanistic/out/</code>. Each patch is replayed twice (clean "
             "checkpoint and fresh boot), so expect minutes per game on one CPU core.",

    next=["attribution_patching", "path_patching", "das", "causal_scrubbing"],
)
