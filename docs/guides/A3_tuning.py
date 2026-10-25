# Beginner's guide — A3 tuning curves (single-unit selectivity).
# Rendered by build_pages.build_guide_section(); schema in docs/guides/__init__.py.

GUIDE = dict(
    key="A3_tuning",
    level="Phase A · neuroscience battery · what does this unit respond to?",

    wiki=[
        ("Neuronal tuning", "https://en.wikipedia.org/wiki/Neuronal_tuning"),
        ("Receptive field", "https://en.wikipedia.org/wiki/Receptive_field"),
        ("Grandmother cell", "https://en.wikipedia.org/wiki/Grandmother_cell"),
        ("Pearson correlation coefficient", "https://en.wikipedia.org/wiki/Pearson_correlation_coefficient"),
        ("Correlation does not imply causation", "https://en.wikipedia.org/wiki/Correlation_does_not_imply_causation"),
    ],
    papers=[
        ("Hubel &amp; Wiesel (1962). <i>Receptive fields, binocular interaction and functional "
         "architecture in the cat's visual cortex.</i> Journal of Physiology 160: 106&ndash;154.",
         "https://pmc.ncbi.nlm.nih.gov/articles/PMC1359523/"),
        ("Bau, Zhou, Khosla, Oliva &amp; Torralba (2017). <i>Network Dissection: Quantifying "
         "Interpretability of Deep Visual Representations.</i> CVPR.",
         "https://arxiv.org/abs/1704.05796"),
        ("Leavitt &amp; Morcos (2020). <i>Selectivity considered harmful: evaluating the causal "
         "impact of class selectivity in DNNs.</i> ICLR 2021.",
         "https://arxiv.org/abs/2003.01262"),
        ("Jonas &amp; Kording (2017). <i>Could a Neuroscientist Understand a Microprocessor?</i> "
         "PLOS Computational Biology 13(1): e1005268.",
         "https://doi.org/10.1371/journal.pcbi.1005268"),
    ],

    idea=r"""
<p>Put an electrode next to one neuron in a cat's visual cortex. Show the cat a bar of light at
different angles. At one angle the neuron fires a lot; at others it is quiet. Plot firing rate
against angle and you get a bump: the neuron's <b>tuning curve</b>. A sharp bump makes the neuron
<b>selective</b>, or &ldquo;tuned&rdquo;, to that angle. Hubel and Wiesel won a Nobel prize for such
curves. The tempting conclusion is that the neuron <i>encodes</i> the angle and that the brain
<i>uses</i> it to see.</p>
<p>Deep learning does the same thing under other names: finding the &ldquo;cat neuron&rdquo;, the
&ldquo;edge detector&rdquo; in layer&nbsp;1, or a unit that lights up for one class. The method is
purely <b>observational</b>. It watches a unit and a variable change together; it never asks whether
the unit is needed. On the VCS we can check that second question with the oracle. We find that many
strongly tuned bytes play no causal role at all. Being correlated with a variable and being used to
produce behaviour are different things.</p>""",

    nn=r"""
<p><b>Neuroscience origin.</b> A tuning curve is the mean response \(r_i(s)\) of unit \(i\) as a
function of a stimulus variable \(s\) (bar angle, sound pitch, head direction). Its sharpness is
often summarised by a <i>selectivity index</i>. The extreme form is the &ldquo;grandmother
cell&rdquo; that fires only for one person.</p>
<p><b>Neural-network analogue.</b> <i>Network dissection</i> (Bau et al.) measures how well the
activation map of a unit overlaps a labelled concept (&ldquo;dog&rdquo;, &ldquo;wood&rdquo;).
<i>Class-selectivity</i> indices compare a unit's mean activation on its favourite class with the
others. A simple, general version is the absolute <b>Pearson correlation</b> between a unit's
activation over many inputs and a variable \(g\):</p>
\[ s_i \;=\; \max_{g}\; \big|\rho\big(x_i(t),\, g(t)\big)\big|,\qquad
   \rho(x,g) = \frac{\operatorname{cov}(x,g)}{\sigma_x\,\sigma_g}. \]
<p>A unit is <b>strongly tuned</b> if \(s_i \ge \tau\). Leavitt &amp; Morcos then ablated highly
selective units and found that removing them often did <i>not</i> hurt, and sometimes helped:
selectivity is a poor guide to importance.</p>
<p><b>Toy example.</b> A game runs for 10 frames, \(t = 1,\dots,10\). The ball moves right at a
steady speed: \(\text{ball\_x}(t) = 20 + 2t\). A separate byte is a frame counter:
\(\text{timer}(t) = t\). The timer is never read when the ball is drawn. Still,
\(\rho(\text{timer}, \text{ball\_x}) = 1\) exactly, because both are straight lines in \(t\). The
tuning method calls the timer perfectly tuned to ball position. If you overwrite the timer, the
ball does not move differently. In the language of statistics, the shared driver \(t\) is a
<i>confounder</i>: it makes two unrelated things move together.</p>""",

    recipe=[
        r"Record a long trajectory: the value of every unit and every candidate variable at every time step.",
        r"For each unit and each variable, compute a tuning statistic (here: \(|\rho|\) over time).",
        r"Take the unit's selectivity \(s_i\) as its best tuning over the variables, leaving out the unit itself.",
        r"Flag units with \(s_i \ge \tau\) as strongly tuned (\(\tau = 0.7\) here).",
        r"Ask an independent causal test which units actually matter (here: the intervention oracle).",
        r"Report the <b>spurious-tuning rate</b>: the share of strongly tuned units that are not causal.",
    ],

    atari_map=[
        ("The neuron", "One <b>candidate RAM byte</b> of the VCS's 128-byte working memory, taken "
         "from published RAM annotations (OCAtari / AtariARI) for that game."),
        ("The recording", "A 150-frame <b>trajectory</b>: RAM and screen logged after every frame "
         "while the bit-exact emulator plays the shared seeded random input (then no-ops)."),
        ("The stimulus variable", "Two families: <b>game variables</b> (the other labelled bytes' "
         "own time series, e.g. ball&nbsp;x, plus a raw score byte) and <b>luminance</b> (mean "
         "palette value of the screen) as a low-level &ldquo;hardware&rdquo; feature."),
        ("The tuning curve", "Absolute Pearson correlation of the byte's value with each variable "
         "over the 150 frames; the frame clock is kept as a confound control."),
        ("&ldquo;Is it used?&rdquo;", "The <b>oracle</b>: overwrite the byte at frame \\(f^*\\) with "
         "\\(0\\), base\\(+17\\) or base\\(+37\\), re-run the real ROM 15 frames, count changed "
         "screen pixels. Any change means causal."),
    ],

    atari=r"""
<p>The runner <code>A3_tuning.jl</code> combines one observational part (the method) and one
interventional part (the truth) for each game.</p>
<ol>
<li><b>Shared state.</b> The cartridge (the ROM) is booted with the game's own settings, then
played for 90 frames of seeded random joystick input; that frame is \(f^*\). Two fresh replays must
match byte for byte. An oracle gate checks that enough causes move the screen.</li>
<li><b>Recording.</b> <code>record_parity_trajectory</code> steps the emulator frame by frame for
150 frames along the same input stream (no-op after the 105 gameplay frames) and stores the 128 RAM
bytes and the full screen after every frame. This is the &ldquo;electrode recording&rdquo;.</li>
<li><b>Tuning.</b> For each candidate byte the runner correlates its time series with (a) every
<i>other</i> candidate byte's time series and a raw score byte, (b) mean screen luminance, and
(c) the frame index. The byte's own column is excluded, since a byte trivially correlates with
itself. Selectivity is the maximum over (a).</li>
<li><b>Oracle.</b> For each candidate byte, three interventions at \(f^*\), each followed by a real
15-frame re-run. The causal importance is the largest number of screen pixels that differ from the
unperturbed run. A byte is causal if that number is above zero.</li>
<li><b>Scores.</b> The spurious-tuning rate (strongly tuned but not causal), for game variables and
for luminance. The triad: F is the Spearman rank correlation between selectivity and oracle
importance; S correlates selectivity with the effect of the held-out base\(+37\) intervention alone;
M compares how many bytes the method names with how many truly matter. A positive control scores
the oracle against itself.</li>
</ol>
<p><b>Why the trap is so easy to fall into on the VCS.</b> Everything in a game is locked to the
60&nbsp;Hz frame clock. Timers, animation counters and positions of moving objects all change in
step with it, so over a short window many bytes look like straight lines in time and correlate
strongly with each other. Some of those bytes are also never used for what reaches the screen within
15 frames. The game variables themselves are other RAM bytes, not hand-measured quantities, so a
&ldquo;tuned&rdquo; pair may simply be two outputs of the same routine.</p>""",

    code=[
        dict(path="tools/xai_study/phaseA_kording/A3_tuning.jl",
             find="function record_parity_trajectory(",
             role="xai", desc="The recording: step the ROM frame by frame and stack RAM plus screen into a (T &times; n) tape."),
        dict(path="tools/xai_study/phaseA_kording/A3_tuning.jl",
             find="function game_variable_signals(",
             role="xai", desc="The stimulus variables: each labelled byte's time series plus a raw score byte."),
        dict(path="tools/xai_study/phaseA_kording/A3_tuning.jl",
             find="function luminance_and_clock(",
             role="xai", desc="The hardware variable (mean screen luminance) and the frame-clock confound."),
        dict(path="tools/xai_study/phaseA_kording/A3_tuning.jl",
             find="function run_a3_tuning(",
             role="xai", desc="The method: |Pearson| tuning, self-column excluded, threshold &tau; = 0.7, spurious-tuning rates."),
        dict(path="tools/xai_study/phaseA_kording/A3_tuning.jl",
             find="function oracle_causal_importance(",
             role="xai", desc="The ground truth: do(0 / +17 / +37) per byte, 15-frame re-run, max changed pixels."),
        dict(path="tools/xai_study/phaseA_kording/A3_tuning.jl",
             find="function score_triad(",
             role="xai", desc="F (Spearman vs oracle), S (held-out +37 effect), M (named vs true movers)."),
        dict(path="tools/xai_study/phaseA_kording/A3_tuning.jl",
             find="function compute_game(",
             role="xai", desc="One game end to end: shared state, oracle, recording, tuning, triad."),
        dict(path="tools/xai_study/common/shared_testbed_impl.jl",
             find="function build_shared_testbed(",
             role="substrate", desc="The shared seeded-gameplay state and its oracle cause-density gate."),
        dict(path="jutari/src/env/StellaEnvironment.jl", find="function env_step!(",
             role="substrate", desc="One video frame of the bit-exact emulated VCS (called once per recorded frame)."),
        dict(path="jutari/src/env/StellaEnvironment.jl", find="get_ram(env::StellaEnvironment)",
             role="substrate", desc="Read the 128 RAM bytes out of the console: the &ldquo;electrode&rdquo;."),
    ],

    reading=r"""
<p>The figure is a scatter plot with one dot per candidate byte. The x-axis is the oracle's true
causal importance, the y-axis the byte's tuning strength to a game variable. Red dots are
<b>spurious</b>: strongly tuned (high on the y-axis) yet at zero on the x-axis. Green dots are
genuine. If tuning were a good guide, the dots would rise from bottom-left to top-right. In Pong the
top-left corner is occupied: bytes that track a game variable almost perfectly but do not change
the screen when overwritten. Also look at the bottom-right: causal bytes, such as a score byte that
does not change during the window, show no tuning at all. Tuning misses them as well.</p>""",

    pitfalls=[
        "<b>Present is not used.</b> A unit can carry information about a variable without the system ever reading it. This is the same trap that defeats linear probing.",
        "<b>Shared drivers.</b> Anything that changes steadily with time (a clock, a scrolling position) correlates with everything else that does. Always include a clock or time control.",
        "<b>Constant causes are invisible.</b> A byte that is crucial but does not change during the recording has zero variance, so zero correlation, and is never called tuned.",
        "<b>Threshold and window.</b> The rate depends on \\(\\tau\\) and on which 150 frames are recorded. A different phase of play can tune different bytes.",
        "<b>Linear only.</b> Pearson's \\(\\rho\\) sees straight-line relations. A byte that encodes a variable through a table or a bit mask can look untuned.",
    ],

    try_it="julia --project=jutari tools/xai_study/phaseA_kording/A3_tuning.jl --games core",
    try_note="Writes <code>A3_tuning_&lt;game&gt;.{json,npz}</code> per game and the index "
             "<code>A3_tuning_all.json</code> into <code>tools/xai_study/phaseA_kording/out/</code>. "
             "<code>--tau 0.7</code> sets the tuning threshold, <code>--traj-frames 150</code> the "
             "recording length, <code>--selftest</code> runs the checks without writing.",

    next=["linear_probing", "A4_correlations", "A2_lesions", "A7_dimred"],
)
