# Beginner's guide — A7 dimensionality reduction (PCA / NMF).
# Rendered by build_pages.build_guide_section(); schema in docs/guides/__init__.py.

GUIDE = dict(
    key="A7_dimred",
    level="Neuroscience battery · latent factors of population activity",

    wiki=[
        ("Dimensionality reduction", "https://en.wikipedia.org/wiki/Dimensionality_reduction"),
        ("Principal component analysis", "https://en.wikipedia.org/wiki/Principal_component_analysis"),
        ("Non-negative matrix factorization", "https://en.wikipedia.org/wiki/Non-negative_matrix_factorization"),
        ("Singular value decomposition", "https://en.wikipedia.org/wiki/Singular_value_decomposition"),
    ],
    papers=[
        ("Lee &amp; Seung (1999). <i>Learning the parts of objects by non-negative matrix "
         "factorization.</i> Nature 401: 788&ndash;791.",
         "https://doi.org/10.1038/44565"),
        ("Cunningham &amp; Yu (2014). <i>Dimensionality reduction for large-scale neural "
         "recordings.</i> Nature Neuroscience 17: 1500&ndash;1509.",
         "https://doi.org/10.1038/nn.3776"),
        ("Jonas &amp; Kording (2017). <i>Could a Neuroscientist Understand a Microprocessor?</i> "
         "PLoS Computational Biology 13(1): e1005268.",
         "https://doi.org/10.1371/journal.pcbi.1005268"),
    ],

    idea=r"""
<p>Modern recordings are huge: hundreds of neurons, or thousands of hidden units, over thousands of
time steps. Nobody can stare at that table. <b>Dimensionality reduction</b> compresses it. It looks
for a few <b>components</b>: patterns of joint activity that, added together, rebuild most of the
table. Each component has a <b>loading</b> (how strongly each unit takes part) and an
<b>activation</b> over time (how strongly the pattern is &ldquo;on&rdquo; at each step). The hope is that
these few <b>latent factors</b> are the system's real internal variables: &ldquo;arm direction&rdquo;,
&ldquo;choice&rdquo;, &ldquo;ball position&rdquo;.</p>
<p>Two classic tools are used. <b>PCA</b> (principal component analysis) finds the directions along which the
data vary most. <b>NMF</b> (non-negative matrix factorisation) instead rebuilds the data as a sum of
non-negative &ldquo;parts&rdquo;, which often look more like separate objects. Both are
unsupervised: they need no labels. That is their charm and their risk. They find what is
<i>large</i> in the data, and large is not the same as <i>important</i>. On the Atari we can check
which components match real signals, and whether the bytes they highlight are the ones that
actually cause what appears on screen.</p>""",

    nn=r"""
<p><b>Neuroscience origin.</b> PCA, factor analysis and NMF are staples of population analysis
(Cunningham &amp; Yu 2014): reduce a recording to a few latent trajectories, then interpret them. Jonas
&amp; Kording ran NMF on the transistor activity of a simulated 6502 and found components that tracked the
clock and bus activity, not the program's logic.</p>
<p><b>Deep-learning analogue.</b> <b>Dimensionality reduction of activations</b>: PCA of a layer's
activations over a dataset, NMF of CNN feature maps into &ldquo;concept&rdquo; parts, or the top
singular directions of a residual stream. Arrange the recording as a matrix \(X\) of size \(T\times n\)
(time steps &times; units).</p>
<p><b>PCA</b> centres each column (subtracts its mean, giving \(X_c\)) and takes the singular value
decomposition \(X_c = U S V^\top\). Column \(j\) of \(V\) is component \(j\)'s loading over the \(n\)
units; \(X_c V\) gives the activations; the share of variance explained is \(s_j^2/\sum_k s_k^2\).</p>
<p><b>NMF</b> requires \(X\ge 0\) and finds \(W\ge 0\) (\(T\times r\), activations) and \(H\ge 0\)
(\(r\times n\), loadings) that minimise \(\lVert X - WH\rVert_F\). The Lee&ndash;Seung multiplicative
updates keep everything non-negative:</p>
\[ H \leftarrow H \odot \frac{W^\top X}{W^\top W H}, \qquad
   W \leftarrow W \odot \frac{X H^\top}{W H H^\top} . \]
<p><b>Toy example.</b> Record two bytes over four frames. Byte A is a frame counter:
\((0, 64, 128, 192)\). Byte B is the score: \((0, 0, 1, 1)\). Byte A has variance 5120; byte B has
variance 0.25. PCA's first component points almost entirely at A and explains more than 99.99&nbsp;% of
the variance. The score barely registers. Now suppose the score is what the screen shows and the counter
only drives a blink. The top component has picked the <i>loudest</i> byte, not the <i>causal</i> one.
Variance is a statement about the data, not about cause.</p>""",

    recipe=[
        r"Record the population as a \(T\times n\) matrix \(X\) (here: 128 RAM bytes over \(T\) frames).",
        r"PCA: centre the columns, take the SVD, keep the variance fraction of each component.",
        r"NMF: pick a rank \(r\), run multiplicative updates from a seeded random start, and order components by energy.",
        r"For each leading component, ask whether it <i>matches</i> a known variable (its loading sits on one variable) or a known signal (its activation tracks it).",
        r"Turn the loadings into a per-unit &ldquo;recovery&rdquo; score: how strongly the leading components load on each unit.",
        r"On the VCS, compare that score with the oracle's true causal importance of each unit.",
    ],

    atari_map=[
        ("Population recording / layer activations",
         "The \\(T\\times128\\) RAM tape: every byte of the VCS's working memory, once per frame. "
         "Bytes are 0&ndash;255, so the tape is non-negative and NMF applies directly."),
        ("Latent factors",
         "The top-5 PCA components and the top-5 of a rank-8 NMF, each with a loading over the "
         "128 bytes and an activation over frames."),
        ("Known task variables",
         "The game's <b>candidate bytes</b>, grouped into variable families (score, ball, enemy, "
         "&hellip;), plus three structural signals: a frame <b>clock</b> ramp, the population "
         "<b>write activity</b> (bytes changed per frame), and a periodic <b>frame-phase</b> wave."),
        ("&ldquo;This component is the choice variable&rdquo;",
         "A <b>matched</b> component: at least half of its candidate-byte loading lies on one family "
         "(and candidates carry a real share of the loading), or its activation correlates "
         "\\(|r|\\ge0.5\\) with one structural signal."),
        ("Importance of a unit (unknown in a brain)",
         "The <b>intervention oracle</b>: set the byte to 0 or base+17 at frame \\(f^*\\), re-run the "
         "real ROM for the horizon, count changed screen pixels."),
    ],

    atari=r"""
<p>The runner <code>A7_dimred.jl</code> does two things: a descriptive match of components to
known signals, and an oracle-scored test of whether the components point at causal bytes.</p>
<ol>
<li><b>The recording.</b> From the shared gameplay state (seeded random actions, checkpoint at frame
\(f^*=90\), 15-frame horizon), <code>a7_record_ram</code> steps the real ROM with
<code>env_step!</code> and stores the RAM each frame. The trajectory is 150 frames for most games,
continued with NOOP after the gameplay window.</li>
<li><b>Decompositions.</b> <code>run_pca</code> centres the raw byte values and takes an SVD.
<code>run_nmf</code> runs 300 Lee&ndash;Seung updates at rank 8 from a seeded random start, then sorts
components by energy (the size of each rank-one part \(W_{:,k}H_{k,:}\)). No package is added; both are
written out in the file.</li>
<li><b>Matching.</b> <code>known_signals</code> builds the clock ramp, the write-activity count, and a
sinusoid at the dominant autocorrelation period of that count. <code>match_components</code> then
labels each of the top-5 components as a game variable, a known signal, or unmatched (purity threshold
0.5). The headline record value is the <b>NMF matched-component fraction</b>.</li>
<li><b>Causal check.</b> <code>recovery_scores</code> gives each candidate byte a score: its squared
loading on the top-5 components, weighted by variance (PCA) or energy (NMF).
<code>oracle_importance</code> pokes each candidate byte to 0 and to base+17, re-runs the horizon and
counts changed pixels. A third value, base+37, is held out. <code>score_triad</code> gives F (Spearman
rank correlation of recovery vs importance), S (Pearson against the held-out effect) and M (the true
cause set's size over the named set's size).</li>
</ol>
<p><b>Read the match metric carefully.</b> A component that tracks the frame clock counts as
&ldquo;matched&rdquo;. That is honest bookkeeping (the component <i>is</i> a real signal), but it is
the epiphenomenal signal A5 warns about. The triad is what asks whether the components point at
causes. Note also that the &ldquo;frame-phase&rdquo; wave is a stand-in: per-frame sampling cannot see the real
VSYNC pulse, so it is a periodic proxy derived from the data.</p>""",

    code=[
        dict(path="tools/xai_study/phaseA_kording/A7_dimred.jl", find="function run_pca(",
             role="xai", desc="PCA by SVD of the centred RAM tape: loadings and variance fractions."),
        dict(path="tools/xai_study/phaseA_kording/A7_dimred.jl", find="function _nmf(",
             role="xai", desc="Lee&ndash;Seung multiplicative-update NMF, seeded and dependency-free."),
        dict(path="tools/xai_study/phaseA_kording/A7_dimred.jl", find="function known_signals(",
             role="xai", desc="The structural signals a component may match: clock ramp, write activity, frame-phase wave."),
        dict(path="tools/xai_study/phaseA_kording/A7_dimred.jl", find="function match_components(",
             role="xai", desc="Labels each leading component as a game variable, a known signal, or unmatched."),
        dict(path="tools/xai_study/phaseA_kording/A7_dimred.jl", find="function recovery_scores(",
             role="xai", desc="Per-byte recovery score: variance- or energy-weighted squared loading on the top components."),
        dict(path="tools/xai_study/phaseA_kording/A7_dimred.jl", find="function oracle_importance(",
             role="xai", desc="Ground truth per byte: poke it, re-run the ROM, count changed pixels."),
        dict(path="tools/xai_study/phaseA_kording/A7_dimred.jl", find="function score_triad(",
             role="xai", desc="F (Spearman), S (held-out Pearson) and M (minimality) against the oracle."),
        dict(path="tools/xai_study/common/triad_sm.jl", find="function minimality_score(",
             role="substrate", desc="The shared minimality scorer \\(M=|U^*|/|\\hat U|\\) used across the study."),
        dict(path="tools/xai_study/common/jutari_oracle.jl", find="function intervene_ram!(env::StellaEnvironment",
             role="substrate", desc="The intervention: write one value into one RAM byte."),
        dict(path="jutari/src/env/StellaEnvironment.jl", find="function env_step!(env::StellaEnvironment",
             role="substrate", desc="One video frame of the real, bit-exact machine."),
    ],

    reading=r"""
<p>The figure shows two bars per candidate byte (Pong): the oracle's true causal importance and the
byte's NMF recovery score, both scaled so the largest is 1. A faithful decomposition would give bars
of matching height. Expect them not to match. The leading components favour the bytes that vary most,
such as a paddle position the oracle calls inert, while bytes that matter but barely change, such as
the scores, get almost no recovery. The match metric in the record tells a kinder story: a good share
of the leading components match a game variable or a known signal, for PCA and NMF alike. The
structure is real; it is just not the same thing as the program's causal story.</p>""",

    pitfalls=[
        "<b>Variance is not importance.</b> A wide-ranging counter dominates PCA while a one-bit flag that decides the game barely moves it (the toy example). Raw byte values are not rescaled here, so wide-range bytes weigh more.",
        "<b>Components mix variables.</b> Nothing forces one component per true variable. PCA's orthogonality constraint and NMF's additivity both produce blends when variables co-vary.",
        "<b>A match is not an explanation.</b> Matching the clock is a correct label for a causally inert signal. Matching a game variable says the variable varies a lot, not that the program uses it here.",
        "<b>Arbitrary choices.</b> Rank, number of components kept, purity threshold and NMF seed all change the answer. NMF also has no unique solution.",
        "<b>Correlational.</b> Like every recording-only method, it never intervenes. Only the oracle comparison tells you whether a component points at causes.",
    ],

    try_it="julia --project=jutari tools/xai_study/phaseA_kording/A7_dimred.jl --games core",
    try_note="Writes <code>A7_&lt;game&gt;.json/.npz</code> (variance curves, loadings, matches, recovery "
             "scores, oracle importance, PCA and NMF triads) plus <code>A7_dimred_all.json</code> into "
             "<code>tools/xai_study/phaseA_kording/out/</code>. Flags <code>--nmf-rank</code>, "
             "<code>--topk</code> and <code>--purity</code> let you test the sensitivity. Minutes per game.",

    next=["A5_lfp", "A3_tuning", "sae", "linear_probing"],
)
