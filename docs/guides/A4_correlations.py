# Beginner's guide — A4 pairwise correlations ("spike words").
# Rendered by build_pages.build_guide_section(); schema in docs/guides/__init__.py.

GUIDE = dict(
    key="A4_correlations",
    level="Phase A · neuroscience battery · who fires together?",

    wiki=[
        ("Pearson correlation coefficient", "https://en.wikipedia.org/wiki/Pearson_correlation_coefficient"),
        ("Correlation does not imply causation", "https://en.wikipedia.org/wiki/Correlation_does_not_imply_causation"),
        ("Confounding", "https://en.wikipedia.org/wiki/Confounding"),
        ("Single-linkage clustering", "https://en.wikipedia.org/wiki/Single-linkage_clustering"),
        ("Principal component analysis", "https://en.wikipedia.org/wiki/Principal_component_analysis"),
    ],
    papers=[
        ("Schneidman, Berry, Segev &amp; Bialek (2006). <i>Weak pairwise correlations imply strongly "
         "correlated network states in a neural population.</i> Nature 440: 1007&ndash;1012.",
         "https://doi.org/10.1038/nature04701"),
        ("Tka&ccaron;ik et al. (2014). <i>Searching for Collective Behavior in a Large Network of "
         "Sensory Neurons.</i> PLOS Computational Biology 10(1): e1003408.",
         "https://doi.org/10.1371/journal.pcbi.1003408"),
        ("Li, Yosinski, Clune, Lipson &amp; Hopcroft (2015). <i>Convergent Learning: Do different "
         "neural networks learn the same representations?</i> (unit-to-unit correlation in deep nets).",
         "https://arxiv.org/abs/1511.07543"),
        ("Jonas &amp; Kording (2017). <i>Could a Neuroscientist Understand a Microprocessor?</i> "
         "PLOS Computational Biology 13(1): e1005268.",
         "https://doi.org/10.1371/journal.pcbi.1005268"),
    ],

    idea=r"""
<p>Record many neurons at once and you get a table: one row per moment in time, one column per
neuron. A natural first question is <i>which neurons go up and down together?</i> Measure that for
every pair and you get a <b>correlation matrix</b>. Groups of units that move together are
candidates for working together. In neuroscience the patterns of joint firing are called
<b>spike words</b>, and their statistics have been studied in detail.</p>
<p>A famous finding is &ldquo;<b>weak pairwise, strong global</b>&rdquo;. Any two neurons are only
weakly correlated, yet the whole population sits in a few coordinated states far more often than
chance. The tempting reading is that the correlation structure reveals the coupling, the true
influence of one unit on another. But correlation is a passive measurement. Two units can move
together because one drives the other, or because both are driven by a third thing. On the VCS
almost everything is driven by the same clock, and the oracle can tell the two cases apart.</p>""",

    nn=r"""
<p><b>Neuroscience origin.</b> Schneidman et al. recorded about 40 retinal neurons and found
pairwise correlations that were small, yet a model built only from them predicted the frequent
population-wide patterns. Tka&ccaron;ik et al. extended this to larger populations. Jonas &amp;
Kording ran the same statistics on transistors of a 6502 chip and found the same
&ldquo;weak pairwise, strong global&rdquo; picture, in a system whose real wiring is known and looks
nothing like a brain.</p>
<p><b>Neural-network analogue.</b> Correlating unit activations over a dataset is common: to find
redundant units, to match units across two trained networks (Li et al.), or to compare whole layers
(SVCCA, CKA). For units \(i,j\) with activations \(x_i(t)\) over inputs \(t\):</p>
\[ C_{ij} = \rho\big(x_i, x_j\big), \qquad
   \mathrm{PR} = \frac{\big(\sum_k \lambda_k\big)^2}{\sum_k \lambda_k^2}, \]
<p>where \(\lambda_k\) are the eigenvalues of \(C\). The <b>participation ratio</b> PR counts the
effective number of independent directions. A PR far below the number of units means a few global
modes dominate (the same idea as keeping the top principal components).</p>
<p><b>Toy example 1 (common driver).</b> Let a clock run \(t = 1,\dots,5\). Unit \(x = t\) and
unit \(y = 2t + 1\). Neither reads the other. Still \(\rho(x,y) = 1\). If you overwrite \(x\), \(y\)
does not change: the true coupling is \(0\). A correlation of 1, a coupling of 0.</p>
<p><b>Toy example 2 (weak pairwise, strong global).</b> Three units, every pair correlated
\(0.3\). The eigenvalues of \(C\) are \(1.6, 0.7, 0.7\), so \(\mathrm{PR} = 3^2 / (1.6^2 + 0.7^2 + 0.7^2)
= 9 / 3.54 \approx 2.5\). With 300 such units, the top eigenvalue is \(1 + 299 \cdot 0.3 \approx 91\),
the other 299 are \(0.7\), and PR is about 11, far below 300: many weak pairs add up to one strong
collective mode.</p>""",

    recipe=[
        r"Record the activity of all units over a long trajectory: a matrix of time &times; units.",
        r"Compute the Pearson correlation \(C_{ij}\) for every pair of units.",
        r"Group units whose \(|C_{ij}|\) passes a threshold into clusters (here: single linkage, \(\tau = 0.7\)).",
        r"Summarise the global structure: median \(|C_{ij}|\), the share of variance in the top eigenvalues, the participation ratio.",
        r"Obtain an independent measure of true coupling (here: the intervention oracle) for every pair.",
        r"Score: rank-correlate \(|C_{ij}|\) with the true coupling over all pairs, and compare clusters with the known variable groups.",
    ],

    atari_map=[
        ("The neuron", "One <b>candidate RAM byte</b> of the VCS's 128-byte working memory, "
         "named by published annotations (OCAtari / AtariARI); for the population statistics, "
         "every RAM byte that varies."),
        ("The multi-electrode recording", "A RAM <b>tape</b>: all 128 bytes logged after every "
         "frame (one 1/60&nbsp;s picture) while the bit-exact emulator plays the shared input "
         "stream, 150 frames for most games."),
        ("The spike word", "The joint pattern of byte values at one frame; the pairwise statistic "
         "is the Pearson correlation of two bytes' time series."),
        ("The true coupling", "The <b>oracle</b>: overwrite byte \\(i\\) at frame \\(f^*\\) with "
         "\\(0\\) or base\\(+17\\), re-run the real ROM 15 frames, measure \\(|\\Delta|\\) of byte "
         "\\(j\\). Symmetrised as \\(\\max(i\\to j,\\ j\\to i)\\)."),
        ("Known cell types", "<b>Concept families</b> from the labels: all <code>enemy_*</code> "
         "bytes form one true group, all <code>player_*</code> bytes another, and so on."),
    ],

    atari=r"""
<p>The runner <code>A4_correlations.jl</code> has an observational half (the method) and an
interventional half (the truth). Both use the shared game state: boot the cartridge (the ROM), play
90 frames of seeded random joystick input to reach \(f^*\), keep the next 15 frames as the horizon,
and confirm that two fresh replays agree byte for byte.</p>
<ol>
<li><b>Recording.</b> <code>a4_record_ram</code> steps the emulator along the same input stream
(no-ops after the gameplay frames) and stores the 128 RAM bytes after every frame. The length is
set per game (150 frames for most, longer for Q*bert and Ms. Pac-Man).</li>
<li><b>Pairwise correlations.</b> <code>pairwise_corr</code> computes the Pearson matrix over the
candidate bytes. A byte that never changes has zero variance and is given correlation 0 with
everything.</li>
<li><b>Clusters versus true groups.</b> Bytes with \(|C_{ij}| \ge 0.7\) are linked, and connected
components become clusters. These are compared with the concept families using the Rand index,
cluster purity and an <i>over-grouping ratio</i>: the share of within-cluster pairs that belong to
different true groups.</li>
<li><b>Global structure.</b> Over the candidate bytes and over the whole population of varying RAM
bytes, the runner reports mean and median \(|C_{ij}|\), the top-1 and top-3 eigenvalue shares and
the participation ratio. The &ldquo;weak pairwise, strong global&rdquo; flag is set when the
population median \(|C_{ij}|\) is below 0.5 and the top three eigenvalues carry at least 60&nbsp;%
of the total.</li>
<li><b>Oracle coupling.</b> For every candidate byte \(i\), the machine is copied, byte \(i\) is
set to \(0\) or base\(+17\), and the real ROM runs 15 frames. The coupling \(i\to j\) is the larger
\(|\Delta|\) of byte \(j\). A third value, base\(+37\), is kept back as a held-out test.</li>
<li><b>Scores.</b> F is the Spearman rank correlation between \(|C_{ij}|\) and the oracle coupling
over all off-diagonal pairs. S is the Pearson correlation with the held-out coupling. M compares how
many pairs the method calls coupled with how many truly are. A positive control scores the oracle
against itself.</li>
</ol>
<p><b>How this differs from the neural original.</b> Our units are bytes with up to 256 values, not
binary spikes, so we use Pearson correlation rather than a maximum-entropy model of spike words.
And the &ldquo;true coupling&rdquo; is a measured total effect over 15 frames, which includes
indirect paths, not a list of direct synapses.</p>""",

    code=[
        dict(path="tools/xai_study/phaseA_kording/A4_correlations.jl",
             find="function a4_record_ram(",
             role="xai", desc="The recording: a (frames &times; 128) RAM tape from the bit-exact emulator."),
        dict(path="tools/xai_study/phaseA_kording/A4_correlations.jl",
             find="function pairwise_corr(",
             role="xai", desc="The method: Pearson correlation matrix over the candidate bytes."),
        dict(path="tools/xai_study/phaseA_kording/A4_correlations.jl",
             find="function score_grouping(",
             role="xai", desc="Clusters vs concept families: Rand index, purity, over-grouping ratio."),
        dict(path="tools/xai_study/phaseA_kording/A4_correlations.jl",
             find="function global_structure(",
             role="xai", desc="Weak-pairwise / strong-global: eigenvalue shares and participation ratio, candidates and full population."),
        dict(path="tools/xai_study/phaseA_kording/A4_correlations.jl",
             find="function oracle_coupling(",
             role="xai", desc="The ground truth: do(0) / do(+17) per byte, 15-frame re-run, |&Delta;| of every other byte; +37 held out."),
        dict(path="tools/xai_study/phaseA_kording/A4_correlations.jl",
             find="function score_triad(",
             role="xai", desc="F = Spearman(|corr|, coupling) over pairs; S on the held-out value; M named vs true pairs."),
        dict(path="tools/xai_study/common/shared_testbed_impl.jl",
             find="function build_shared_testbed(",
             role="substrate", desc="The shared seeded-gameplay state and its oracle cause-density gate."),
        dict(path="tools/xai_study/common/triad_sm.jl",
             find="function minimality_score(",
             role="substrate", desc="The shared minimality score M = |U*| / |U&#770;| used by every method."),
        dict(path="jutari/src/env/StellaEnvironment.jl", find="function env_step!(",
             role="substrate", desc="One video frame of the bit-exact emulated VCS (called once per recorded frame)."),
        dict(path="jutari/src/env/StellaEnvironment.jl", find="get_ram(env::StellaEnvironment)",
             role="substrate", desc="Read the 128 RAM bytes: one row of the recording."),
    ],

    reading=r"""
<p>The figure shows two square matrices over the same candidate bytes. Left: the oracle's true
coupling; a bright square means overwriting one byte changes the other within 15 frames. Right: the
measured correlation. The diagonal of the right matrix is always 1 and is not scored. If the two
matrices lit up the same off-diagonal squares, correlation would be a faithful map of coupling. In
Pong most candidate bytes are zero in both, and both light up the same small group of ball and
paddle bytes, so the ranking is partly right. But inside that group the two disagree: correlation
gives every pair a similar, high value, because the bytes all move with the same ball, while the
oracle says only one pair is strongly coupled, two weakly, and the player paddle not at all. Correlation found the group,
not the wiring inside it. That is the partial score in the audit box.</p>""",

    pitfalls=[
        "<b>Common drivers.</b> Two units driven by the same clock or the same input correlate strongly without any influence between them. On a clock-locked machine this is the rule, not the exception.",
        "<b>No direction.</b> A correlation is symmetric. It cannot say whether \\(i\\) drives \\(j\\), \\(j\\) drives \\(i\\), or neither.",
        "<b>Silent but important units.</b> A byte that is constant during the recording has zero correlation with everything, however much it matters.",
        "<b>Collective structure is not mechanism.</b> A low participation ratio says activity is low-dimensional. Jonas &amp; Kording found the same signature in a chip, so it says little about how the system computes.",
        "<b>Window dependence.</b> Correlations are computed over one stretch of play. A different stretch, or a longer one, can give a different matrix.",
    ],

    try_it="julia --project=jutari tools/xai_study/phaseA_kording/A4_correlations.jl --games core",
    try_note="Writes <code>A4_&lt;game&gt;.{json,npz}</code> per game into "
             "<code>tools/xai_study/phaseA_kording/out/</code>. <code>--tau-corr 0.7</code> sets "
             "the clustering threshold, <code>--traj-frames N</code> the recording length, "
             "<code>--selftest</code> runs the checks on the first game without writing.",

    next=["A6_granger", "A7_dimred", "A3_tuning", "A1_connectomics"],
)
