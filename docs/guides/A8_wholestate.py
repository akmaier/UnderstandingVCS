# Beginner's guide — A8 whole-state recording.
# Rendered by build_pages.build_guide_section(); schema in docs/guides/__init__.py.

GUIDE = dict(
    key="A8_wholestate",
    level="Neuroscience battery · the record-everything baseline",

    wiki=[
        ("BRAIN Initiative", "https://en.wikipedia.org/wiki/BRAIN_Initiative"),
        ("Calcium imaging", "https://en.wikipedia.org/wiki/Calcium_imaging"),
        ("Occam's razor", "https://en.wikipedia.org/wiki/Occam%27s_razor"),
        ("Explainable artificial intelligence", "https://en.wikipedia.org/wiki/Explainable_artificial_intelligence"),
    ],
    papers=[
        ("Alivisatos et al. (2012). <i>The Brain Activity Map Project and the Challenge of Functional "
         "Connectomics.</i> Neuron 74(6): 970&ndash;974.",
         "https://doi.org/10.1016/j.neuron.2012.06.006"),
        ("Jonas &amp; Kording (2017). <i>Could a Neuroscientist Understand a Microprocessor?</i> "
         "PLoS Computational Biology 13(1): e1005268.",
         "https://doi.org/10.1371/journal.pcbi.1005268"),
        ("Lipton (2016). <i>The Mythos of Model Interpretability.</i> ICML Workshop on Human "
         "Interpretability in Machine Learning.",
         "https://arxiv.org/abs/1606.03490"),
    ],

    idea=r"""
<p>Every method in this battery looks at part of the system: one unit, one pair, one pooled signal.
The obvious objection is &ldquo;then record <i>everything</i>&rdquo;. Neuroscience has pursued that dream
for decades. Large projects aim to record every neuron of a brain at once, and techniques such as
calcium imaging already capture tens of thousands of cells. The hope behind it is that once nothing is
hidden, understanding will follow.</p>
<p>A8 tests that hope on a machine where the dream is already true. The Atari's complete state is tiny,
and we can copy all of it at every frame. Such a record is a perfect <i>description</i>: it misses
nothing, and from it the machine can be rebuilt exactly. But is it an <i>explanation</i>? An explanation
should also be <b>small</b>: it should point at the few things that matter for the output you asked
about. A8 is the deliberate baseline that scores perfectly on completeness and at the floor on
smallness. It makes the gap between having the data and understanding the system measurable.</p>""",

    nn=r"""
<p><b>Neuroscience origin.</b> Whole-brain or whole-population recording: the Brain Activity Map /
BRAIN Initiative vision (Alivisatos et al. 2012) of measuring every spike of every neuron. Jonas &amp;
Kording had the complete state of their simulated chip at every step, with every transistor known,
and showed that standard analyses of it still failed to reveal how the chip ran its programs.</p>
<p><b>Deep-learning analogue.</b> <b>Dumping every activation</b>: caching all hidden states of every
layer for an input, the way interpretability libraries &ldquo;run with cache&rdquo;. It is always the
first step, never the explanation. To grade it, we use the study's three axes. With \(U^*\) the true
minimal set of causes of an output and \(\hat U\) the set an explanation names:</p>
\[ F = \frac{|U^*\cap \hat U|}{|U^*|}\ \text{(nothing causal left out)}, \qquad
   S = \text{does }\hat U\text{ reproduce the output?}, \qquad
   M = \frac{|U^*|}{|\hat U|} . \]
<p>(The study defines F per method; for a set-valued explanation like this one it is the recall
above.) A full dump names every unit, so \(\hat U\) is everything. It contains every cause (F = 1), it
reproduces the output (S = 1), and M is as small as it can get.</p>
<p><b>Toy example.</b> A network has 1000 hidden units. For one prediction, careful ablations show that
only 3 units matter. The activation dump lists all 1000 numbers: F = 1, S = 1, and
\(M = 3/1000 = 0.003\). A good explanation names those 3 units: F = 1, S = 1, M = 1. Same faithfulness,
opposite usefulness. M is the axis that tells them apart.</p>""",

    recipe=[
        r"Record the complete internal state at every time step (every unit, every register).",
        r"Present the full record as the &ldquo;explanation&rdquo;: its named set \(\hat U\) is the whole state.",
        r"Check faithfulness: does \(\hat U\) contain every true cause? (Trivially yes.)",
        r"Check sufficiency: restore the recorded state and confirm it reproduces the output exactly.",
        r"Find the true minimal cause set \(U^*\) by intervention and compute \(M=|U^*|/|\hat U|\).",
        r"Run the control: the oracle's own minimal set must score F = S = M = 1.",
    ],

    atari_map=[
        ("Recording every neuron / caching every activation",
         "Copying the full machine state each frame: the 128 bytes of <b>RAM</b> (the program's "
         "variables) and the 64 <b>TIA</b> registers (the video/sound chip's settings), 192 bytes in all."),
        ("The output to explain",
         r"The rendered screen 15 frames after the intervention frame \(f^*\)."),
        (r"The true minimal cause set \(U^*\) (unknown in a brain)",
         "Candidate RAM bytes whose poking (to 0, base+17 or base+37) changes at least one screen pixel "
         "when the real ROM is re-run: found by the <b>intervention oracle</b>."),
        (r"The named set \(\hat U\)",
         "All 128 RAM bytes. The TIA registers are recorded too, but the minimality denominator "
         "counts RAM only."),
        ("Sufficiency of a recording",
         "Writing the recorded RAM back into a copy of the machine and checking that RAM and screen "
         "come out byte-identical."),
    ],

    atari=r"""
<p>The runner <code>A8_wholestate.jl</code> is short, because the method is trivial. What it measures
is not.</p>
<ul>
<li><b>Recording.</b> From the shared gameplay state (seeded random actions, checkpoint at frame
\(f^*=90\), 15-frame horizon), <code>record_whole_state</code> steps the real ROM with
<code>env_step!</code> for 150 frames, continuing with NOOP after the gameplay window. It stacks
<code>get_ram</code> and the TIA register file into a \(T\times192\) tape. Per-byte range and variance
are kept as the descriptive map.</li>
<li><b>Ground truth.</b> <code>oracle_causal_importance</code> copies the checkpoint, writes each
candidate byte to 0, base+17 and base+37, runs the 15-frame horizon and counts changed screen pixels.
A byte with any effect is causal. These bytes form \(U^*\).</li>
<li><b>F and S.</b> F is 1 by construction, because the dump contains every byte.
<code>verify_wholestate_sufficiency</code> writes the recorded 128 RAM bytes into a fresh copy of the
checkpoint and compares RAM and screen with the recording. As the code itself notes, this is an identity
restore. It confirms the recorder is lossless; it is not a hard test.</li>
<li><b>M, the headline.</b> \(M = |U^*|/128\). The runner also reports the over-recording factor
\(128/|U^*|\): how many times larger the dump is than the true cause set.</li>
<li><b>Positive control.</b> <code>verify_minimal_sufficiency</code> restores only the causal bytes (again
over the same checkpoint) and checks the frame is still reproduced. The oracle's own set then scores M = 1, proving the harness rewards
small explanations and that the dump's low M is a real measurement.</li>
</ul>
<p><b>Honest caveats.</b> \(U^*\) is searched only among the game's candidate bytes, with three poke values
and one horizon. Bytes outside that list are never tested, so \(|U^*|\) may be undercounted, and M with
it. The verdict does not change: the dump is far larger than any plausible cause set.</p>""",

    code=[
        dict(path="tools/xai_study/phaseA_kording/A8_wholestate.jl", find="function record_whole_state(",
             role="xai", desc="The method: copy RAM and TIA registers into the tape at every frame."),
        dict(path="tools/xai_study/phaseA_kording/A8_wholestate.jl", find="function oracle_causal_importance(",
             role="xai", desc="Ground truth: poke each candidate byte three ways, re-run, count changed pixels."),
        dict(path="tools/xai_study/phaseA_kording/A8_wholestate.jl", find="function verify_wholestate_sufficiency(",
             role="xai", desc="S check: restore the full recorded RAM and compare RAM and screen."),
        dict(path="tools/xai_study/phaseA_kording/A8_wholestate.jl", find="function verify_minimal_sufficiency(",
             role="xai", desc="Positive control: the oracle's minimal set alone reproduces the frame."),
        dict(path="tools/xai_study/phaseA_kording/A8_wholestate.jl", find="M = RAM_SIZE == 0 ? 0.0 : n_causal / RAM_SIZE",
             role="xai", desc="The headline: minimality = causal bytes / 128."),
        dict(path="tools/xai_study/phaseA_kording/A8_wholestate.jl", find="function run_game(",
             role="xai", desc="One game end to end: shared state &rarr; oracle &rarr; recording &rarr; triad &rarr; control."),
        dict(path="jutari/src/env/StellaEnvironment.jl", find="get_ram(env::StellaEnvironment)",
             role="substrate", desc="The 128-byte RAM, read directly: the whole working state of the program."),
        dict(path="jutari/src/env/StellaEnvironment.jl", find="get_screen(env::StellaEnvironment) =",
             role="substrate", desc="The rendered frame: the output whose pixel changes define causality here."),
        dict(path="jutari/src/env/StellaEnvironment.jl", find="function env_step!(env::StellaEnvironment",
             role="substrate", desc="One video frame of the real, bit-exact machine."),
        dict(path="tools/xai_study/common/shared_testbed_impl.jl", find="function build_shared_testbed(",
             role="substrate", desc="The shared gameplay state and cause-density gate every Phase-A runner starts from."),
    ],

    reading=r"""
<p>The audit box shows the three axes side by side. F and S sit at the ceiling. M sits near the floor.
That contrast is the whole result. The figure shows the activity (variance) of every RAM byte over the
recording, with the candidate bytes marked: many bytes change over time, far more than the handful the
oracle marks as causal for the output. Read A8 as the yardstick for
the other methods. Any method that claims to explain should name fewer things than this dump while
keeping F and S high. A method that cannot beat the dump on M has described, not explained.</p>""",

    pitfalls=[
        "<b>Completeness is not understanding.</b> A record that contains every cause also contains every non-cause, with no way to tell them apart.",
        "<b>Faithfulness alone is a weak score.</b> Any explanation can reach F = 1 by naming everything. Always report a size or minimality measure next to it.",
        "<b>The sufficiency check here is easy.</b> Restoring a state into the machine it came from is an identity operation; it shows the recorder loses nothing, not that the state is explanatory.",
        "<b>The true cause set is only as good as the search.</b> Candidate bytes, three poke values and one horizon bound \\(U^*\\). A wider search might find more causes, which would raise M somewhat.",
    ],

    try_it="julia --project=jutari tools/xai_study/phaseA_kording/A8_wholestate.jl --games core",
    try_note="Writes <code>A8_&lt;game&gt;.json/.npz</code> per game and the combined "
             "<code>A8_wholestate.json/.npz</code> into <code>tools/xai_study/phaseA_kording/out/</code>. "
             "<code>--selftest</code> checks the contract on the first game without writing. A few minutes "
             "per game; the cost is the three re-runs per candidate byte.",

    next=["A2_lesions", "A1_connectomics", "causal_scrubbing", "acdc"],
)
