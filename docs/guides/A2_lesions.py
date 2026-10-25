# Beginner's guide — A2 single-unit lesions (ablation).
# Rendered by build_pages.build_guide_section(); schema in docs/guides/__init__.py.

GUIDE = dict(
    key="A2_lesions",
    level="Phase A · neuroscience battery · knock one unit out",

    wiki=[
        ("Lesion", "https://en.wikipedia.org/wiki/Lesion"),
        ("Phineas Gage", "https://en.wikipedia.org/wiki/Phineas_Gage"),
        ("Ablation (artificial intelligence)", "https://en.wikipedia.org/wiki/Ablation_(artificial_intelligence)"),
        ("Dropout (neural networks)", "https://en.wikipedia.org/wiki/Dropout_(neural_networks)"),
        ("Interaction (statistics)", "https://en.wikipedia.org/wiki/Interaction_(statistics)"),
    ],
    papers=[
        ("Jonas &amp; Kording (2017). <i>Could a Neuroscientist Understand a Microprocessor?</i> "
         "PLOS Computational Biology 13(1): e1005268.",
         "https://doi.org/10.1371/journal.pcbi.1005268"),
        ("Morcos, Barrett, Rabinowitz &amp; Botvinick (2018). <i>On the importance of single "
         "directions for generalization.</i> ICLR.",
         "https://arxiv.org/abs/1803.06959"),
        ("Meyes, Lu, de Puiseau &amp; Meisen (2019). <i>Ablation Studies in Artificial Neural "
         "Networks.</i> arXiv.",
         "https://arxiv.org/abs/1901.08644"),
    ],

    idea=r"""
<p>The oldest question in brain science is: <i>what stops working if this part is gone?</i> In
1848 an iron rod destroyed part of Phineas Gage's frontal lobe. He survived, but his personality
changed. Doctors concluded that this region had something to do with self-control. That is a
<b>lesion study</b>: remove or silence one part, watch which behaviour is lost, and credit the part
with that behaviour.</p>
<p>The appeal is that a lesion is an <b>intervention</b>, not an observation. You do not just
watch a unit fire together with a behaviour; you take the unit away and see whether the behaviour
goes too. That is a causal test. Its weakness is that it tests one unit at a time. If two units do
a job together, or back each other up, removing one may show nothing, or something misleading.
On the Atari we can measure both sides: how well single lesions rank the units, and how much of the
joint behaviour they cannot see.</p>""",

    nn=r"""
<p><b>Neuroscience origin.</b> Lesions come from accidents (Gage), strokes and surgery, and in
animals from deliberate damage, cooling or drugs. Modern tools such as optogenetics can switch a
cell type off for a few milliseconds. Jonas &amp; Kording &ldquo;lesioned&rdquo; the transistors of a
simulated 6502 chip one by one (the VCS CPU, the 6507, is a 6502 variant) and found some that broke only one game. They
warned that this does not mean the transistor is a &ldquo;Donkey Kong transistor&rdquo;.</p>
<p><b>Neural-network analogue.</b> An <b>ablation study</b>: set one unit's activation to zero (or
to its mean) and measure the drop in accuracy or in a logit. <i>Dropout</i> does random ablations
during training. Written as an intervention on unit \(u\) with a set of clamp values \(V\):</p>
\[ I(u) \;=\; \max_{v \in V}\; \big|\,y\big(\mathrm{do}(h_u := v)\big) - y\,\big| . \]
<p>Rank the units by \(I(u)\) and you have an <b>importance map</b>. To test it, compare its ranking
with a trusted ranking by <b>Spearman's \(\rho\)</b>, the correlation of the two rank lists
(\(+1\) = same order).</p>
<p><b>Toy example.</b> Let \(y = a\cdot b + c\) at \(a=b=c=1\), so \(y = 2\). Lesion \(a\) to 0:
\(y = 1\), \(I(a) = 1\). Likewise \(I(b) = 1\) and \(I(c) = 1\). The map says all three matter
equally. Now lesion \(a\) <i>and</i> \(b\): \(y = 1\), a drop of 1, not \(1+1 = 2\). The
<b>interaction term</b></p>
\[ \Delta_{\text{int}}(a,b) = E(a,b) - \big(E(a)+E(b)\big) = 1 - 2 = -1 \]
<p>is non-zero: \(a\) and \(b\) work as a team (a product), \(c\) works alone (a sum). No single-unit
map can tell those two kinds of units apart.</p>""",

    recipe=[
        r"Freeze the system in one state and record a baseline run of the outputs you care about.",
        r"For each unit \(u\): restore the state, clamp \(u\) to a lesion value (zero, or a very different value), re-run, record the outputs.",
        r"Turn the change into one number per unit: the lesion importance \(I(u)\). Take the largest effect over the lesion values tried.",
        r"Rank the units by \(I(u)\). This is the importance map.",
        r"Score the map against a trusted reference ranking with Spearman's \(\rho\).",
        r"Lesion selected <i>pairs</i> too and compare \(E(i,j)\) with \(E(i)+E(j)\) to expose interactions the map hides.",
    ],

    atari_map=[
        ("The neuron", "One <b>candidate RAM byte</b>: one of the 128 bytes of VCS working memory "
         "that published annotations (OCAtari / AtariARI) link to a game concept such as ball&nbsp;x or score."),
        ("The lesion", "<code>intervene_ram!</code> at frame \\(f^*\\): overwrite the byte with "
         "\\(0\\) (&ldquo;silence&rdquo;) or with base&nbsp;XOR&nbsp;255 (every bit flipped, a maximally different value)."),
        ("The behaviour", "After 15 frames of re-run: the absolute change of every candidate byte "
         "<b>plus</b> the number of screen pixels that differ from the un-lesioned run."),
        ("The neurologist's ground truth", "The <b>oracle role</b> of each byte: the same "
         "measurement with a larger set of interventions \\(\\{0,\\ +17,\\ +37,\\ \\text{XOR }255\\}\\), "
         "each a real re-run of the real ROM (the cartridge program)."),
        ("Patients with two lesions", "Pairs of the most important bytes, both flipped at once, "
         "to measure the interaction term."),
    ],

    atari=r"""
<p>The runner <code>A2_lesions.jl</code> works on the same frozen game state as every method in the
study: the cartridge is booted (60 no-op plus 4 reset frames), then 90 frames of seeded random
joystick input are played. That frame is \(f^*\); the horizon is the next 15 frames. Two fresh
replays must agree byte for byte before anything is scored.</p>
<ul>
<li><b>Lesion map (method).</b> For each candidate byte the runner deep-copies the frozen machine,
writes \(0\) or base&nbsp;XOR&nbsp;255 into it, and re-runs the 15 frames with the recorded joystick
inputs. The footprint is the sum over candidate bytes of \(|\text{value} - \text{baseline}|\) plus the
count of changed screen pixels. The importance is the larger footprint of the two clamps. It also
records whether the machine still produced a valid frame; on the VCS a poked byte never crashes the
CPU, so that flag is almost always true.</li>
<li><b>Oracle role (truth).</b> The same footprint, maximised over four interventions: \(0\),
base\(+17\), base\(+37\) and XOR&nbsp;255. The headline score is Spearman's \(\rho\) between the two
rankings. A positive control ranks the oracle against itself and must give \(\rho = 1\).</li>
<li><b>Interactions.</b> The runner takes the top-\(K\) bytes by lesion importance (\(K\) chosen so
that at most 60 pairs are tested), flips both bytes of each pair, and computes
\(\Delta_{\text{int}} = E(i,j) - (E(i)+E(j))\). A pair counts as a missed interaction if
\(|\Delta_{\text{int}}|\) exceeds 5&nbsp;% of the typical single effect.</li>
<li><b>Specificity check.</b> A byte is &ldquo;specific&rdquo; to the lesion map if at least 80&nbsp;%
of its footprint lands on one readout. The runner counts how many of those the oracle calls generic
(touching three or more readouts).</li>
<li><b>Triad.</b> F is \(\rho\). S re-ranks against a held-out value (base\(+37\)) the lesion never
used. M compares how many bytes the map names with how many truly move the output.</li>
</ul>
<p><b>How this differs from a brain lesion.</b> Our &ldquo;lesion&rdquo; is a one-time write at
\(f^*\), not a clamp held for the whole horizon. The game program is free to overwrite the byte on
the next frame, much as a brain can compensate. Also note that the lesion's two clamps are a
<i>subset</i> of the oracle's four interventions, measured with the same footprint. A high rank
correlation is therefore expected by design; the interaction count is the informative part.</p>""",

    code=[
        dict(path="tools/xai_study/phaseA_kording/A2_lesions.jl",
             find="function lesion_continue(",
             role="xai", desc="The lesion primitive: deep-copy, write one or two bytes, step the horizon, snapshot."),
        dict(path="tools/xai_study/phaseA_kording/A2_lesions.jl",
             find="LESION_DOSET(base) =",
             role="xai", desc="The two lesion values {0, base XOR 0xFF}, next to the oracle's four and the held-out one."),
        dict(path="tools/xai_study/phaseA_kording/A2_lesions.jl",
             find="function footprint(",
             role="xai", desc="The behavioural readout: per-byte absolute change plus count of changed screen pixels."),
        dict(path="tools/xai_study/phaseA_kording/A2_lesions.jl",
             find="function lesion_map(",
             role="xai", desc="Per-unit importance map under a given do-set (used for the method, the oracle and the held-out probe)."),
        dict(path="tools/xai_study/phaseA_kording/A2_lesions.jl",
             find="function interaction_misses(",
             role="xai", desc="Pair lesions among the top units: the interaction term the single-unit map cannot see."),
        dict(path="tools/xai_study/phaseA_kording/A2_lesions.jl",
             find="function run_game(",
             role="xai", desc="One game: shared state, bit-exact check, lesion vs oracle maps, Spearman, interactions, triad."),
        dict(path="tools/xai_study/common/shared_testbed_impl.jl",
             find="function build_shared_testbed(",
             role="substrate", desc="The shared seeded-gameplay state and its oracle cause-density gate."),
        dict(path="tools/xai_study/ground_truth/oracle_intervene.jl",
             find="function run_intervention(",
             role="substrate", desc="The reference oracle: poke a cause, re-run the ROM, snapshot (drives the gate)."),
        dict(path="tools/xai_study/common/jutari_oracle.jl",
             find="function intervene_ram!(",
             role="substrate", desc="Write one byte into the console's RAM (the A2 runner carries an identical local copy)."),
        dict(path="jutari/src/env/StellaEnvironment.jl", find="function env_step!(",
             role="substrate", desc="One video frame of the bit-exact emulated VCS."),
    ],

    reading=r"""
<p>The figure shows the game frame, a heat-map of where on the screen the lesions changed pixels,
and a bar chart with two bars per candidate byte: the oracle's true causal role and the lesion
importance, both scaled so the largest is 1. A faithful map has bars of matching height. In Pong
they match exactly: the score bytes and the ball bytes are tall, the paddle bytes are
zero. That is why this is the one classical method near the top of the battery. What the bars do
<i>not</i> show is how the bytes combine. Look in the record's interaction block for the pairs whose
joint lesion is far from the sum of the singles.</p>""",

    pitfalls=[
        "<b>Interactions are invisible.</b> Redundant units (either one suffices) look unimportant alone; cooperating units (both needed) are each over-credited. Only pair or group lesions reveal this.",
        "<b>The lesion value matters.</b> Zero is not &ldquo;off&rdquo; for a RAM byte; it is just another number, and may be a perfectly normal game value. That is why two very different clamp values are tried.",
        "<b>Importance is not function.</b> Knowing a byte matters says nothing about <i>what</i> it does. Jonas &amp; Kording's game-specific transistors were generic circuitry.",
        "<b>Compensation.</b> A one-time poke can be undone by the program on the next frame, just as networks and brains re-route around a damaged unit.",
        "<b>Agreement partly by construction.</b> Here the lesion and the oracle share the same machinery, so a high score says the cheap probe ranks units well, not that the explanation is complete.",
    ],

    try_it="julia --project=jutari tools/xai_study/phaseA_kording/A2_lesions.jl --games core",
    try_note="Writes <code>A2_lesions_&lt;game&gt;.{json,npz}</code> per game and a combined "
             "<code>A2_lesions.{json,npz}</code> into <code>tools/xai_study/phaseA_kording/out/</code>. "
             "<code>--max-pairs K</code> caps the interaction budget (default 60); "
             "<code>--shard i --nshards n</code> runs one game per cluster task.",

    next=["occlusion", "activation_patching", "A1_connectomics", "A3_tuning"],
)
