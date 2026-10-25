# Beginner's guide — occlusion.
# Rendered by build_pages.build_guide_section(); schema in docs/guides/__init__.py.

GUIDE = dict(
    key="occlusion",
    level="Perturbation family · the starting point",

    wiki=[
        ("Explainable artificial intelligence", "https://en.wikipedia.org/wiki/Explainable_artificial_intelligence"),
        ("Sensitivity analysis", "https://en.wikipedia.org/wiki/Sensitivity_analysis"),
        ("Ablation (artificial intelligence)", "https://en.wikipedia.org/wiki/Ablation_(artificial_intelligence)"),
        ("Convolutional neural network", "https://en.wikipedia.org/wiki/Convolutional_neural_network"),
    ],
    papers=[
        ("Zeiler &amp; Fergus (2014). <i>Visualizing and Understanding Convolutional Networks.</i> ECCV.",
         "https://arxiv.org/abs/1311.2901"),
        ("Hooker, Erhan, Kindermans &amp; Kim (2019). <i>A Benchmark for Interpretability Methods in Deep "
         "Neural Networks</i> (ROAR: removed pixels push the input off the training distribution). NeurIPS.",
         "https://arxiv.org/abs/1806.10758"),
    ],

    idea=r"""
<p>A network looks at a photo and says &ldquo;dog: 92&nbsp;%&rdquo;. Instead of asking calculus
questions, you can simply <b>cover part of the photo with a grey square</b> and run the network
again. If the dog score drops to 30&nbsp;%, the covered patch mattered. If it stays at 92&nbsp;%,
it did not. Slide the square across the whole photo, write down the drop at every position, and you
get a heat-map. That is <b>occlusion</b>: hide one part, re-run, measure the change.</p>
<p>Occlusion is a <b>black-box</b> method. It never opens the network. It needs no gradients and no
weights, only the ability to feed in an input and read the output. That makes it slow (one full
forward pass per patch) but very honest: it measures what the model <i>actually does</i> when a part
is removed, not a local approximation of it. It is the simplest member of the
<b>perturbation family</b> (RISE, LIME, KernelSHAP, extremal perturbation), which all work by
changing the input and re-running the model.</p>""",

    nn=r"""
<p>Write the network as \(y = f(x)\) with input \(x = (x_1,\dots,x_n)\). Pick a <b>baseline</b>
value \(b\) that stands for &ldquo;absent&rdquo; (grey, black, zero, the dataset mean). Let
\(x_{[i \leftarrow b]}\) be the input with part \(i\) replaced by the baseline. Occlusion gives part
\(i\) the score</p>
\[ O_i(x) \;=\; \big|\, f(x) - f(x_{[i \leftarrow b]}) \,\big| . \]
<p>For images, &ldquo;part \(i\)&rdquo; is a square patch, and the square slides with some stride.
The patch size and the baseline are both <b>choices</b>; the map changes when you change them. Cost:
one forward pass for the intact input plus one per part.</p>
<p><b>Toy example 1 (it works).</b> Let \(f(x_1,x_2,x_3) = 4x_1 + x_3\) at \(x = (1,1,1)\), baseline
0. Then \(f(x) = 5\). Occluding \(x_1\) gives 1 (score 4), occluding \(x_2\) gives 5 (score 0),
occluding \(x_3\) gives 4 (score 1). The map \((4,0,1)\) is exactly right.</p>
<p><b>Toy example 2 (redundancy).</b> Now let \(f = \max(x_1, x_2) + x_3\) at \(x=(1,1,1)\), so
\(f(x)=2\). Occlude \(x_1\): \(\max(0,1)+1 = 2\), score 0. Occlude \(x_2\): also score 0.
Occlude \(x_3\): score 1. Occlusion says \(x_1\) and \(x_2\) are irrelevant. Yet occluding <i>both</i>
gives \(f = 1\): together they matter as much as \(x_3\). Each one is a <b>backup</b> for the other,
and a one-at-a-time probe cannot see a backup. Remember this blind spot; KernelSHAP fixes it.</p>""",

    recipe=[
        r"Choose the output \(y\) to explain and a baseline value that means &ldquo;absent&rdquo;.",
        r"Run the model once on the intact input and record \(y(x)\).",
        r"For each part \(i\): copy the input, set part \(i\) to the baseline, run the model, record \(y(x_{[i\leftarrow b]})\).",
        r"Score part \(i\) by \(|y(x) - y(x_{[i\leftarrow b]})|\).",
        r"Display the scores as a heat-map (or, here, as one bar per candidate cause).",
    ],

    atari_map=[
        (r"The trained network \(f\)", "The cartridge program running on the emulated <b>Atari 2600 VCS</b> "
         "(6507 CPU + TIA video chip + RIOT), bit-exact in <code>jutari</code>."),
        (r"The input parts \(x_i\)", "The candidate <b>causes</b> at the intervention frame \\(f^*\\): "
         "RAM bytes the game uses (score, positions, &hellip;), one TIA colour register, and the joystick input."),
        (r"Grey patch / baseline \(b\)", "Write <b>0</b> into the RAM byte or TIA register "
         "(<code>occlude!</code>). For the joystick cause, &ldquo;absent&rdquo; means &ldquo;no extra "
         "action&rdquo;: the recorded action stream is left unchanged."),
        ("One forward pass", "Restore a <code>deepcopy</code> of the checkpoint, poke the value, and run "
         "the real ROM for the 15-frame horizon with <code>env_step!</code>."),
        (r"The output \(y\)", "A <b>content</b> output (one RAM byte at \\(f^*{+}15\\)) or a "
         "<b>position</b> output (how many screen pixels differ from the un-intervened frame)."),
        ("&ldquo;Ground-truth importance&rdquo;", "The intervention oracle: the same re-run machinery, "
         "but each RAM byte is probed twice (set to base+17, and set to 0), plus a TIA colour and a joystick do."),
    ],

    atari=r"""
<p>On the VCS, occlusion is almost embarrassingly direct. The game is stopped at frame \(f^*\) (after
90 frames of seeded random play) and the machine is saved as a <b>checkpoint</b>. Then, for each
candidate cause \(u\):</p>
<ol>
<li><code>deepcopy</code> the checkpoint (a full copy of CPU, RAM, TIA and RIOT state);</li>
<li><code>occlude!</code> the cause: <code>intervene_ram!(env, idx, 0)</code> for a RAM byte,
<code>intervene_tia!(env, reg, 0)</code> for a TIA register, nothing for the joystick;</li>
<li>replay the same recorded 15 actions on the real ROM with <code>env_step!</code>;</li>
<li>read \(y\) and store \(|y(\text{occluded}) - y(\text{intact})|\).</li>
</ol>
<p>That is one real-emulator re-run per cause, plus one intact run. No surrogate, no gradient, no
learned model. Because it only reads the output after a real run, it works just as well on the
<b>position</b> output, where the VCS places sprites by timing and every gradient is exactly zero.</p>
<p><b>How is this different from the oracle?</b> Occlusion <i>is</i> a coarse oracle. The oracle's
cause list contains, for every candidate RAM byte, two interventions: <code>ram[i]:set</code>
(base value + 17) and <code>ram[i]:occlude</code> (value 0). For the occlude-mode entries, occlusion
runs <b>exactly the same experiment as the oracle</b> and gets exactly the same number. The
disagreements come from the rest: occlusion gives the <code>set</code> entry of a byte the same
score as its <code>occlude</code> twin (it only knows one direction, &ldquo;to zero&rdquo;); it sets
the background-colour register to 0 where the oracle sets it to white; and it gives the joystick
cause zero, because &ldquo;absent&rdquo; leaves the action stream untouched while the oracle forces
a RIGHT press. So we expect a high but not perfect agreement, and that is what the runner checks
(it asserts a <i>positive</i> correlation on the core games, never a perfect one).</p>
<p>The scoring then ranks causes by their occlusion score and runs deletion/insertion curves on the
real machine: occlude the top-1, top-2, &hellip; causes together and watch \(y\) move.</p>""",

    code=[
        dict(path="tools/xai_study/phaseB_attribution/occlusion.jl", find="function occlusion_attr(",
             role="xai", desc="The method: one intact run, then per cause <code>deepcopy</code> &rarr; <code>occlude!</code> &rarr; 15 &times; <code>env_step!</code> &rarr; |&Delta;y|."),
        dict(path="tools/xai_study/phaseB_attribution/occlusion.jl", find="function compute_one(",
             role="xai", desc="One experiment: oracle |&Delta;y| &rarr; occlusion map &rarr; Pearson/Spearman, precision@k &rarr; deletion/insertion on the real VCS &rarr; positive control."),
        dict(path="tools/xai_study/phaseB_attribution/occlusion.jl", find="function compute_game(",
             role="xai", desc="Builds the shared gameplay state and scores both the content and the position output."),
        dict(path="tools/xai_study/phaseB_attribution/saliency.jl", find="function occlude!(env, c::Cause)",
             role="xai", desc="The occluder: RAM byte or TIA register &rarr; 0; the joystick cause leaves the action stream unchanged."),
        dict(path="tools/xai_study/common/jutari_oracle.jl", find="function intervene_ram!(",
             role="substrate", desc="The actual poke into RIOT RAM of the emulated machine."),
        dict(path="tools/xai_study/common/jutari_oracle.jl", find="function intervene_tia!(",
             role="substrate", desc="The poke into the TIA register file (used for the colour-register cause)."),
        dict(path="jutari/src/env/StellaEnvironment.jl", find="function env_step!(",
             role="substrate", desc="One video frame of the bit-exact machine; every occlusion re-run calls it 15 times."),
        dict(path="tools/xai_study/ground_truth/oracle_intervene.jl", find="function build_pong_causes(",
             role="substrate", desc="The cause list: each candidate byte as a <code>set</code> (+17) and an <code>occlude</code> (&rarr;0) cause, a TIA colour and a joystick do."),
        dict(path="tools/xai_study/ground_truth/oracle_intervene.jl", find="function run_intervention(",
             role="substrate", desc="The oracle's own re-run: compare with <code>occlusion_attr</code> &mdash; same machinery, richer set of interventions."),
        dict(path="tools/xai_study/common/shared_testbed_impl.jl", find="function build_shared_testbed(",
             role="substrate", desc="The shared state every method explains: seeded random play, checkpoint at \\(f^*\\), the position output."),
    ],

    reading=r"""
<p>Each cause has two bars: the occlusion score and the oracle's true effect. Look at the pairs of
<code>:set</code> / <code>:occlude</code> causes of the same byte. Occlusion always gives the twins
the same height, while the oracle may not. The <code>:occlude</code> bars should match the oracle
exactly. The joystick bar is zero for occlusion by construction. Expect the tallest bars to line up
with the oracle on both the content and the position output. This is the clearest contrast with the
gradient family, whose position-output maps are flat.</p>""",

    pitfalls=[
        "<b>Single masking misses backups.</b> If two causes can each do the job alone (toy example 2), occluding either one changes nothing. Occlusion then calls both irrelevant. Joint effects need joint masking (RISE, KernelSHAP) or a search (extremal perturbation).",
        "<b>The baseline is a choice.</b> &ldquo;Absent&rdquo; = 0 is arbitrary. A byte whose current value is already 0 cannot be occluded at all and always scores 0, however important it is.",
        "<b>Off-distribution inputs.</b> On images a grey square is something the network never saw during training (Hooker et al.). On the VCS a zeroed byte is a <i>valid</i> machine state the program will run, but possibly one the game never reaches in normal play.",
        "<b>Granularity.</b> The map is only as fine as the parts you hide: patch size on images, the candidate cause list here. A cause that is not on the list gets no score at all.",
    ],

    try_it="julia --project=jutari tools/xai_study/phaseB_attribution/occlusion.jl --games core",
    try_note="Writes <code>occlusion_&lt;game&gt;_{content,position}.json/.npz</code> into "
             "<code>tools/xai_study/phaseB_attribution/out/</code>. Cost per output: one intact run plus "
             "one re-run per cause, plus about twice that again for the deletion/insertion curves. Pass "
             "<code>--game pong</code> to run a single game.",

    next=["kernelshap", "rise", "perturbation", "saliency"],
)
