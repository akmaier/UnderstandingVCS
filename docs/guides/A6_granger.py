# Beginner's guide — A6 Granger causality.
# Rendered by build_pages.build_guide_section(); schema in docs/guides/__init__.py.

GUIDE = dict(
    key="A6_granger",
    level="Neuroscience battery · causal discovery from timing alone",

    wiki=[
        ("Granger causality", "https://en.wikipedia.org/wiki/Granger_causality"),
        ("Vector autoregression", "https://en.wikipedia.org/wiki/Vector_autoregression"),
        ("Confounding", "https://en.wikipedia.org/wiki/Confounding"),
        ("Correlation does not imply causation",
         "https://en.wikipedia.org/wiki/Correlation_does_not_imply_causation"),
    ],
    papers=[
        ("Granger (1969). <i>Investigating Causal Relations by Econometric Models and Cross-spectral "
         "Methods.</i> Econometrica 37(3): 424&ndash;438.",
         "https://doi.org/10.2307/1912791"),
        ("Seth, Barrett &amp; Barnett (2015). <i>Granger Causality Analysis in Neuroscience and "
         "Neuroimaging.</i> Journal of Neuroscience 35(8): 3293&ndash;3297.",
         "https://doi.org/10.1523/JNEUROSCI.4399-14.2015"),
        ("Stokes &amp; Purdon (2017). <i>A study of problems encountered in Granger causality analysis "
         "from a neuroscience perspective.</i> PNAS 114(34): E7063&ndash;E7072.",
         "https://doi.org/10.1073/pnas.1704663114"),
        ("Jonas &amp; Kording (2017). <i>Could a Neuroscientist Understand a Microprocessor?</i> "
         "PLoS Computational Biology 13(1): e1005268.",
         "https://doi.org/10.1371/journal.pcbi.1005268"),
    ],

    idea=r"""
<p>You record two signals over time, say the activity of brain areas A and B. You cannot cut
wires or stimulate anything. Can you still say which one drives the other? Clive Granger's idea
(1969) is a clever proxy: <i>a cause comes before its effect, and it carries information about the
effect</i>. So try to predict B's next value from B's own past. Then add A's past. If the prediction
gets clearly better, A is said to <b>Granger-cause</b> B. Run this for every ordered pair and you get
a directed graph of &ldquo;who influences whom&rdquo;, built purely from passive recordings.</p>
<p>The word &ldquo;causality&rdquo; is generous. What the test really measures is <b>temporal
precedence plus predictive information</b>. That is not the same as a causal wire. If both A and B are
driven by a hidden third signal C, and A sees C a moment earlier than B, then A's past predicts B
perfectly, with no wire from A to B at all. This hidden common driver is a <b>confounder</b>. A
clock-driven machine like the Atari is full of them. A6 measures how far the Granger graph sits from
the true wiring, which on the VCS we can establish by direct intervention.</p>""",

    nn=r"""
<p><b>Neuroscience origin.</b> Granger causality is a workhorse for &ldquo;directed functional
connectivity&rdquo; in EEG, MEG, LFP and fMRI data (Seth et al. 2015). Jonas &amp; Kording applied it
to a simulated 6502 chip and found a graph that bore little resemblance to the real circuit.</p>
<p><b>Deep-learning analogue.</b> In XAI it is <b>temporal-precedence causal discovery on activation
time series</b>: record hidden units of a recurrent network, the residual stream across tokens, or the
latent state of an RL agent over steps, then infer which units &ldquo;drive&rdquo; which.</p>
<p><b>The test.</b> For an ordered pair \(x\to y\) and a lag \(p\), fit two linear autoregressions by
least squares:</p>
\[ \text{restricted:}\; y_t = a_0 + \sum_{l=1}^{p} a_l\,y_{t-l} + e_t, \qquad
   \text{unrestricted:}\; y_t = a_0 + \sum_{l=1}^{p} a_l\,y_{t-l} + \sum_{l=1}^{p} b_l\,x_{t-l} + e'_t . \]
<p>With residual sums of squares \(\mathrm{RSS}_r \ge \mathrm{RSS}_u\) over \(n\) usable steps, the
Granger statistic and its F-test are</p>
\[ G(x\to y) = \ln\frac{\mathrm{RSS}_r}{\mathrm{RSS}_u}, \qquad
   F = \frac{(\mathrm{RSS}_r-\mathrm{RSS}_u)/p}{\mathrm{RSS}_u/(n-2p-1)} . \]
<p>An edge \(x\to y\) is declared when the F-test's p-value falls below a significance level
\(\alpha\) (usually 0.05).</p>
<p><b>Toy example.</b> A hidden driver produces random numbers \(d = (3,1,4,1,5,9,2,6)\). Unit \(x\)
copies it one step late, unit \(y\) two steps late: \(x_t = d_{t-1}\), \(y_t = d_{t-2}\). So
\(x = (\cdot,3,1,4,1,5,9,2)\) and \(y = (\cdot,\cdot,3,1,4,1,5,9)\). Because \(d\) is random, \(y\)'s own
past says little about its next value: \(\mathrm{RSS}_r\) is large. But \(y_t = x_{t-1}\) exactly, so
adding \(x\)'s past drives \(\mathrm{RSS}_u\) to zero. \(G\) is huge, the p-value is tiny, and the
test reports a strong edge \(x\to y\). There is no wire between them: both just read \(d\). Cutting
\(x\) would leave \(y\) unchanged. Only that intervention reveals the truth.</p>""",

    recipe=[
        r"Record the time series you want to connect (unit activations, region activity, RAM bytes).",
        r"Drop constant series: with zero variance the test is undefined.",
        r"For every ordered pair \(x\to y\), fit the restricted and unrestricted autoregressions at lag \(p\).",
        r"Compute \(G=\ln(\mathrm{RSS}_r/\mathrm{RSS}_u)\) and the F-test p-value.",
        r"Declare an edge \(x\to y\) if \(p\le\alpha\) (optionally also \(G\ge\tau\)).",
        r"Read the resulting directed graph as the system's influence structure, and here score it against the intervention-defined true graph.",
    ],

    atari_map=[
        ("Recorded brain areas / hidden units",
         "Two scopes: (a) the <b>candidate RAM bytes</b>, bytes the game is known to use for a score, "
         "a position, a timer, &hellip;; (b) three <b>subsystems</b>: the CPU's working RAM, the "
         "<b>TIA</b> (the video/sound chip, 64 registers) and the <b>RIOT</b> (timer and I/O chip)."),
        ("An activity time series",
         "Cell scope: the byte's value at every frame. Subsystem scope: how many bytes of that "
         "subsystem changed since the previous frame (its &ldquo;firing rate&rdquo;)."),
        ("Directed functional connectivity",
         r"The Granger graph: an edge \(i\to j\) if byte \(i\)'s past significantly improves the "
         r"prediction of byte \(j\) (lag 2, \(\alpha = 0.05\))."),
        ("The true anatomical wiring (unknown in a brain)",
         r"The <b>intervention oracle's</b> data-flow graph: edge \(i\to j\) if setting byte \(i\) to "
         r"a new value at frame \(f^*\) and re-running the real ROM changes byte \(j\) 15 frames later."),
        ("A shared pacemaker / confounder",
         "The frame loop. Every byte is updated in the same fixed order every frame, so many bytes "
         "move in lock-step without any data flowing between them."),
    ],

    atari=r"""
<p>The runner <code>A6_granger.jl</code> builds the method's graph and the true graph on the same
nodes, then compares them.</p>
<ol>
<li><b>The recording.</b> From the shared gameplay state (seeded random actions, checkpoint at frame
\(f^*=90\)) it records a trajectory of 150 frames for most games, continuing with NOOP after the
gameplay window. <code>a6_record_tape</code> calls <code>env_step!</code> each frame and stores the
128 RAM bytes plus the 64 TIA registers.</li>
<li><b>Granger, cell scope.</b> <code>run_cell_granger</code> tests every ordered pair of candidate
bytes with <code>granger</code>: lag \(p=2\), \(\alpha=0.05\), magnitude floor \(\tau=0\) (so
significance alone decides, the textbook rule). The least-squares fits and the F-distribution
p-value are written out by hand, so no statistics package is needed.</li>
<li><b>Granger, subsystem scope.</b> <code>subsystem_activity</code> turns the tape into three
change-count series. The CPU series counts changes in RAM bytes 0x00&ndash;0x5F, the RIOT series in
0x60&ndash;0x7F, the TIA series in its register file. They are scored against a fixed truth: the CPU
writes the TIA, and the CPU reads the RIOT.</li>
<li><b>The true graph.</b> <code>true_dataflow_graph</code> takes the checkpoint at \(f^*\) and, for each
candidate byte \(i\), pokes three values (0, base+17, base+37) with <code>intervene_ram!</code>. It
re-runs the 15-frame horizon and marks \(i\to j\) for every candidate byte \(j\) that ends up
different. A second graph from base+37 alone serves as a held-out probe.</li>
<li><b>Scores.</b> <code>edge_rates</code> gives the headline <b>false-edge rate</b> (share of Granger
edges that are not real), the missed-edge rate, and a <b>direction disagreement</b>: where both graphs
have one arrow between a pair, how often they point opposite ways. <code>score_triad</code> adds F (edge
F1 against the true graph), S (agreement with the held-out graph) and M (true edges divided by inferred
edges). A positive control scores the true graph against itself and must give F1 = 1.</li>
</ol>
<p><b>Differences from the textbook use.</b> The true graph is a <i>15-frame reachability</i> graph: an
edge means byte \(j\) changed by any route, direct or indirect. It is measured at one state, while
Granger is fitted on the whole trajectory. Both choices favour Granger, not penalise it. The bytes are
also integers that jump in steps, not the smooth, noisy signals the linear model assumes.</p>""",

    code=[
        dict(path="tools/xai_study/phaseA_kording/A6_granger.jl", find="function granger(x::",
             role="xai", desc="The bivariate Granger test: \\(G=\\ln(\\mathrm{RSS}_r/\\mathrm{RSS}_u)\\), F statistic and p-value."),
        dict(path="tools/xai_study/phaseA_kording/A6_granger.jl", find="function _granger_rss(",
             role="xai", desc="Builds the lagged design matrices for the restricted and unrestricted fits."),
        dict(path="tools/xai_study/phaseA_kording/A6_granger.jl", find="function run_cell_granger(",
             role="xai", desc="All ordered pairs of candidate bytes &rarr; the inferred directed graph."),
        dict(path="tools/xai_study/phaseA_kording/A6_granger.jl", find="function subsystem_activity(",
             role="xai", desc="CPU / TIA / RIOT &ldquo;firing rates&rdquo;: bytes changed per frame."),
        dict(path="tools/xai_study/phaseA_kording/A6_granger.jl", find="function true_dataflow_graph(",
             role="xai", desc="The ground-truth graph: poke each byte, re-run the ROM, see which bytes move."),
        dict(path="tools/xai_study/phaseA_kording/A6_granger.jl", find="function edge_rates(",
             role="xai", desc="False-edge rate, missed-edge rate and direction disagreement."),
        dict(path="tools/xai_study/phaseA_kording/A6_granger.jl", find="function compute_game(",
             role="xai", desc="One game end to end: recording &rarr; both Granger scopes &rarr; true graph &rarr; triad &rarr; control."),
        dict(path="tools/xai_study/common/jutari_oracle.jl", find="function intervene_ram!(env::StellaEnvironment",
             role="substrate", desc="The intervention: write one value into one RAM byte of a copied machine."),
        dict(path="jutari/src/env/StellaEnvironment.jl", find="function env_step!(env::StellaEnvironment",
             role="substrate", desc="One video frame of the real, bit-exact machine (recording and re-runs)."),
        dict(path="tools/xai_study/ground_truth/oracle_intervene.jl", find="function run_intervention",
             role="substrate", desc="The shared oracle that gates the gameplay state used by every Phase-A runner."),
    ],

    reading=r"""
<p>The figure sets the Granger graph beside the true data-flow graph over the same candidate bytes
(Pong). Here the two hardly overlap: the true graph has whole rows of edges (the score and ball bytes
reach many cells within 15 frames), while Granger asserts almost nothing, and what it does assert is
not in the truth. In games with many moving bytes (Ms. Pac-Man, Q*bert) the Granger graph turns
<b>dense</b> instead, with many bytes appearing to drive many others. Either way most Granger edges
have no counterpart in the truth, which is why the false-edge rate sits near its ceiling and the
faithfulness is among the lowest in the battery. The subsystem scope in the record tells the same
story at coarse scale. Changing the lag or the threshold moves edges around but does not fix the
picture, because the shared frame loop creates the predictability.</p>""",

    pitfalls=[
        "<b>Precedence is not causation.</b> A common driver seen at different delays produces strong, significant, directed edges between units that never interact (the toy example).",
        "<b>Lock-step systems break the test.</b> When everything is updated by one clock, almost every series predicts almost every other one. Deep networks run by a fixed schedule over time steps or layers share the problem.",
        "<b>Model mismatch.</b> The test assumes linear, stationary, noisy series. Integer counters that wrap at 256, flags and step jumps violate all three. Stokes &amp; Purdon show that filtering and model order distort results even on well-behaved data.",
        "<b>Multiple testing.</b> With \\(n\\) units there are \\(n(n-1)\\) tests. At \\(\\alpha=0.05\\) some edges appear by chance alone, before any confounding.",
        "<b>The fix is an intervention, not a better statistic.</b> Only cutting or setting a unit separates &ldquo;earlier&rdquo; from &ldquo;cause&rdquo;. See lesions and activation patching.",
    ],

    try_it="julia --project=jutari tools/xai_study/phaseA_kording/A6_granger.jl --games core",
    try_note="Writes <code>A6_&lt;game&gt;.json/.npz</code> (Granger statistics, p-values, inferred and "
             "true adjacency matrices, edge rates, triad) into "
             "<code>tools/xai_study/phaseA_kording/out/</code>. Try <code>--lag 4</code> or "
             "<code>--alpha 0.01</code> to see that the false edges persist. Minutes per game; the "
             "true graph costs three ROM re-runs per candidate byte.",

    next=["A1_connectomics", "A4_correlations", "A2_lesions", "activation_patching"],
)
