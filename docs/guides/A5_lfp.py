# Beginner's guide — A5 local field potentials (pooled-activity spectra).
# Rendered by build_pages.build_guide_section(); schema in docs/guides/__init__.py.

GUIDE = dict(
    key="A5_lfp",
    level="Neuroscience battery · pooled signals and their rhythms",

    wiki=[
        ("Local field potential", "https://en.wikipedia.org/wiki/Local_field_potential"),
        ("Spectral density", "https://en.wikipedia.org/wiki/Spectral_density"),
        ("Discrete Fourier transform", "https://en.wikipedia.org/wiki/Discrete_Fourier_transform"),
        ("Window function", "https://en.wikipedia.org/wiki/Window_function"),
        ("Epiphenomenon", "https://en.wikipedia.org/wiki/Epiphenomenon"),
    ],
    papers=[
        ("Jonas &amp; Kording (2017). <i>Could a Neuroscientist Understand a Microprocessor?</i> "
         "PLoS Computational Biology 13(1): e1005268.",
         "https://doi.org/10.1371/journal.pcbi.1005268"),
        ("Buzs&aacute;ki, Anastassiou &amp; Koch (2012). <i>The origin of extracellular fields and "
         "currents &mdash; EEG, ECoG, LFP and spikes.</i> Nature Reviews Neuroscience 13: 407&ndash;420.",
         "https://doi.org/10.1038/nrn3241"),
    ],

    idea=r"""
<p>Put a microphone in a busy restaurant. You cannot follow any single conversation, but you
still hear something: a loud hum that swells and fades. Neuroscientists do the same with the
brain. An electrode in brain tissue records the <b>local field potential</b> (LFP): the summed
electrical activity of thousands of nearby neurons. Nobody sees a single cell. What people study
instead is the <b>rhythm</b> of this hum: how strong the slow waves are, how strong the fast ones.
That breakdown of a signal into rhythms is called its <b>power spectrum</b>, and LFP bands such as
&ldquo;theta&rdquo; or &ldquo;gamma&rdquo; are read as windows onto what the region computes.</p>
<p>The worry is simple. A strong rhythm might be the computation. Or it might be a
<b>pacemaker</b>: a global clock that ticks everywhere and makes every part of the system hum at
the same beat, whether or not that part does useful work. Activity that is caused by the clock but
carries no computation is called <b>epiphenomenal</b>. In a brain you cannot easily tell the two
apart. On the Atari you can, because we know every clock of the machine exactly. A5 measures how
much of the &ldquo;LFP&rdquo; spectrum is just those clocks.</p>""",

    nn=r"""
<p><b>Neuroscience origin.</b> The LFP is a <i>mesoscale</i> signal: coarser than a single neuron,
finer than an EEG over the whole skull. It is the analysis you can run when single-cell access is
impossible. Jonas &amp; Kording (2017) recorded the analogous &ldquo;LFP&rdquo; of a simulated
6502 chip and found rich rhythms that said nothing about the program it ran.</p>
<p><b>Deep-learning analogue.</b> In a network, the LFP corresponds to <b>pooled or aggregate
activation statistics</b>: the mean activation of a layer, the norm of a block's output, or the
average over a group of units, tracked over time steps (tokens, video frames, RL steps). Write the
activation of unit \(i\) at time \(t\) as \(a_i(t)\). Standardise each unit (subtract its mean,
divide by its standard deviation: a <b>z-score</b> \(z_i\)) and average over a region \(R\):</p>
\[ L(t) \;=\; \frac{1}{|R|}\sum_{i\in R} z_i(t), \qquad
   P(f_k) \;=\; \Big|\sum_{t=0}^{T-1} L(t)\,w(t)\,e^{-2\pi i k t/T}\Big|^2 . \]
<p>\(L(t)\) is the pooled signal, \(w(t)\) a smooth <b>window</b> that tapers the ends, and
\(P(f_k)\) the power at frequency \(f_k = k/T\) cycles per step (the <b>discrete Fourier
transform</b>). A peak in \(P\) at period \(p = 1/f\) means the pooled activity repeats every
\(p\) steps.</p>
<p><b>Toy example.</b> Take two units over \(T=8\) steps. Unit 1 is a counter that flips every
step: \(z_1 = (+1,-1,+1,-1,+1,-1,+1,-1)\). Unit 2 holds a real variable that jumps once:
\(z_2 = (-1,-1,-1,-1,+1,+1,+1,+1)\). Their pool is \(L = (0,-1,0,-1,1,0,1,0)\). Its spectrum has a
peak at period 2 (from unit 1) and a slow component at period 8 (from unit 2). If we
<i>know</i> unit 1 is a clock, we can fit sin/cos waves of period 2 to \(L\) and ask what fraction
of \(L\)'s energy they explain. Here it is exactly half (energy 2 of 4): half of this &ldquo;LFP&rdquo; is pure clock,
however striking the period-2 peak looks.</p>""",

    recipe=[
        r"Record the activity of every unit over \(T\) time steps (here: all 128 RAM bytes, once per video frame).",
        r"Choose regions: one global pool of every unit that varies, plus smaller pools of related units.",
        r"Z-score each varying unit and average per time step: this is the region's pooled signal \(L(t)\).",
        r"Remove the mean and the linear trend, multiply by a Hann window, and take the DFT power spectrum.",
        r"Report the dominant periods. Then build a basis of the <i>known clocks</i> (sin/cos at their periods and harmonics).",
        r"Project \(L\) onto that clock basis: the fraction of its energy captured is the <b>clock-explained</b>, epiphenomenal fraction.",
    ],

    atari_map=[
        ("Neurons in a brain region / units in a layer",
         "The 128 bytes of the VCS's <b>RAM</b> (random-access memory: the program's working "
         "variables, e.g. ball position, score, timers), read once per <b>frame</b> (one full "
         "TV picture, 1/60 s)."),
        ("The LFP: pooled, z-scored activity",
         "The mean z-scored value of a set of RAM bytes per frame: the <b>global</b> pool (every "
         "byte that changes) and one <b>regional</b> pool per game-variable family (all score "
         "bytes, all enemy bytes, &hellip;)."),
        ("Theta / gamma bands",
         "Peaks in the per-frame power spectrum, reported as periods in frames (a period-2 "
         "blink, a period-8 animation cycle, &hellip;)."),
        ("A global pacemaker",
         "The machine's <b>known clocks</b>: the frame cadence itself, plus every RAM byte that is "
         "an exact free-running counter (it ticks by a fixed step, or repeats with a fixed short "
         "period). Games use these as animation and blink timers."),
        ("&ldquo;Is this rhythm computation?&rdquo;",
         "Answered exactly: the fraction of the pooled signal's energy that lies in the span of "
         "the known-clock waves. That part is causally inert timing, not game logic."),
    ],

    atari=r"""
<p>The runner <code>A5_lfp.jl</code> follows the recipe on the real, bit-exact emulator. Concretely:</p>
<ol>
<li><b>The trajectory.</b> It builds the study's shared gameplay state: a seeded stream of random
joystick actions (90 frames of play, then a 15-frame horizon). It drives the game with that stream
and keeps going with NOOP (no button pressed) until the trajectory is long enough for a spectrum
(256 frames for most games). Two fresh runs from power-on must give byte-identical RAM and screen,
or the runner refuses to analyse. Then <code>a5_record_ram</code> calls <code>env_step!</code> and
<code>get_ram</code> once per frame to fill a \(T\times128\) tape.</li>
<li><b>Finding the clocks without labels.</b> <code>detect_clock_cells</code> scans each RAM byte.
A byte is a <i>ramp</i> if its frame-to-frame difference (mod 256) is one constant non-zero step:
a frame counter. It is <i>periodic</i> if it repeats exactly with some period between 2 and 64
frames: a blink or animation timer. No game knowledge is used.</li>
<li><b>The clock basis.</b> <code>clock_basis</code> stacks sin/cos waves at two slow
frame-cadence periods (\(T\) and \(T/2\), always present) plus the first three harmonics of every
detected clock period. Each wave goes through the same detrend and window as the data. A QR
factorisation makes the basis orthonormal.</li>
<li><b>Pooling and spectra.</b> <code>pool_region</code> builds the global pool and one pool per
variable family from the candidate-cause list. <code>dft_power</code> computes a plain
\(O(T^2)\) DFT (no FFT package is added to the emulator environment). The headline number is the
<b>global pool's clock-explained fraction</b>, from <code>variance_explained_by</code>.</li>
<li><b>Controls.</b> The self-test projects a pure clock wave (must score above 0.9) and a
mid-band sinusoid placed between frequency bins (must score below 0.5). So a high clock fraction is a
measurement, not a basis that explains everything.</li>
</ol>
<p><b>Where we differ from the neuroscience original.</b> We sample once per frame. The VCS also
has a <b>scanline</b> clock (262 TV lines per frame) and a CPU clock, but those tick far faster than
our sampling rate and cannot appear in a per-frame spectrum. The clocks A5 actually removes are the
frame cadence and the frame-driven counter bytes. A5 is also <b>descriptive</b>. It never
intervenes; the oracle is used only to build and gate the shared gameplay state. Its score asks
&ldquo;how much of this signal is known-inert timing?&rdquo;, not &ldquo;does it find the causes of an
output?&rdquo;.</p>""",

    code=[
        dict(path="tools/xai_study/phaseA_kording/A5_lfp.jl", find="function compute_game(",
             role="xai", desc="One game end to end: shared gameplay stream &rarr; bit-exact check &rarr; RAM tape &rarr; clocks &rarr; regional spectra."),
        dict(path="tools/xai_study/phaseA_kording/A5_lfp.jl", find="function a5_record_ram(",
             role="xai", desc="The recording: step the ROM one frame at a time and copy the 128 RAM bytes into the tape."),
        dict(path="tools/xai_study/phaseA_kording/A5_lfp.jl", find="function pool_region(",
             role="xai", desc="The LFP itself: z-score every varying byte of a region and average them per frame."),
        dict(path="tools/xai_study/phaseA_kording/A5_lfp.jl", find="function dft_power(",
             role="xai", desc="A dependency-free one-sided DFT power spectrum (cycles per frame)."),
        dict(path="tools/xai_study/phaseA_kording/A5_lfp.jl", find="function detect_clock_cells(",
             role="xai", desc="Label-free detection of the known clocks: constant-stride ramps and fixed-period timers."),
        dict(path="tools/xai_study/phaseA_kording/A5_lfp.jl", find="function clock_basis(",
             role="xai", desc="Frame-cadence waves plus harmonics of every clock period, detrended, windowed and orthonormalised."),
        dict(path="tools/xai_study/phaseA_kording/A5_lfp.jl", find="function variance_explained_by(",
             role="xai", desc="The score: the share of a pooled signal's energy that lies in the clock subspace."),
        dict(path="tools/xai_study/common/shared_testbed_impl.jl", find="function build_shared_testbed(",
             role="substrate", desc="The shared gameplay state: seeded random actions, checkpoint at frame 90, oracle cause-density gate."),
        dict(path="jutari/src/env/StellaEnvironment.jl", find="function env_step!(env::StellaEnvironment",
             role="substrate", desc="One video frame of the real, bit-exact machine."),
        dict(path="jutari/src/env/StellaEnvironment.jl", find="get_ram(env::StellaEnvironment)",
             role="substrate", desc="Direct read of the 128-byte RAM: the &ldquo;electrode&rdquo; that records every unit."),
    ],

    reading=r"""
<p>The figure shows the global pool's power spectrum in cycles per frame. Expect most of the power
at the lowest frequencies, slow sweeps over the whole recording (periods of \(T\) or \(T/2\)
frames), and in some games sharp peaks at short integer periods: these look like the &ldquo;brain
rhythms&rdquo; of the game. Then compare them with the clock basis. The slow sweeps are the frame
cadence, the short peaks sit on a detected timer period or its harmonics, and in most games the
clock-explained fraction is large. The rich-looking spectrum is mostly the machine's timing.
Regional pools for real game variables (score, enemies) are often just as clock-dominated, or flat
when their bytes never change; either way the spectrum cannot tell you <i>what</i> they compute or
what causes them.
The live faithfulness number is in the audit box below.</p>""",

    pitfalls=[
        "<b>Rhythm is not computation.</b> A strong, clean peak is often the easiest thing to see and the least informative: a pacemaker shows up everywhere it is wired.",
        "<b>Sampling hides fast clocks.</b> A per-frame recording cannot see the scanline or CPU clocks. They do not vanish; anything faster than half the sampling rate folds into lower frequencies or is averaged away. The same holds for an LFP filtered at a few hundred hertz.",
        "<b>Pooling destroys identity.</b> Averaging many units cancels opposite signals and lets one loud unit dominate. A flat pooled signal does not mean the units are idle.",
        "<b>The clock basis is a choice.</b> A larger basis explains more of any signal. That is why the runner checks that an off-grid sinusoid stays below 0.5; any LFP analysis should run a similar control.",
        "<b>Descriptive only.</b> Nothing here is an intervention. The spectrum says how activity varies, never which byte drives which output.",
    ],

    try_it="julia --project=jutari tools/xai_study/phaseA_kording/A5_lfp.jl --games core",
    try_note="Writes <code>A5_&lt;game&gt;.json/.npz</code> (pooled signal, spectrum, clock cells, "
             "per-region clock fractions) into <code>tools/xai_study/phaseA_kording/out/</code>. "
             "Add <code>--selftest</code> to run the controls on the first game without writing. "
             "A few minutes per game on one CPU core, most of it building the shared gameplay state.",

    next=["A4_correlations", "A6_granger", "A7_dimred", "A2_lesions"],
)
