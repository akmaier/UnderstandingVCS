# Beginner's guide — logit lens / tuned lens.
# Rendered by build_pages.build_guide_section(); schema in docs/guides/__init__.py.

GUIDE = dict(
    key="logit_lens",
    level="Mechanistic family · reading intermediate states",

    wiki=[
        ("Transformer (deep learning)", "https://en.wikipedia.org/wiki/Transformer_(deep_learning)"),
        ("Residual neural network", "https://en.wikipedia.org/wiki/Residual_neural_network"),
        ("Softmax function", "https://en.wikipedia.org/wiki/Softmax_function"),
        ("Logit", "https://en.wikipedia.org/wiki/Logit"),
        ("Coefficient of determination", "https://en.wikipedia.org/wiki/Coefficient_of_determination"),
    ],
    papers=[
        ("nostalgebraist (2020). <i>interpreting GPT: the logit lens.</i> LessWrong.",
         "https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens"),
        ("Belrose, Furman, Smith, Halawi, Ostrovsky, McKinney, Biderman &amp; Steinhardt (2023). "
         "<i>Eliciting Latent Predictions from Transformers with the Tuned Lens.</i> "
         "(follow-up and critique: the raw logit lens is biased and brittle)",
         "https://arxiv.org/abs/2303.08112"),
    ],

    idea=r"""
<p>A transformer language model is a stack of layers. Each layer reads a shared vector, the
<b>residual stream</b>, and adds its contribution back to it. Only at the very end does an
<b>unembedding</b> matrix turn that vector into one score per word in the vocabulary. Those scores are
the <b>logits</b>; a <b>softmax</b> turns them into probabilities for the next word.</p>
<p>The <b>logit lens</b> asks a cheeky question: what if we applied that final unembedding to the
residual stream <b>halfway up</b> the stack, as if the model stopped there? Because every layer writes
into the same stream, this often gives sensible words. You can watch the prediction form layer by layer:
vague at first, then sharpening into the final answer. The <b>tuned lens</b> improves on it. It learns a
small linear correction for each layer, so that earlier layers, whose &ldquo;language&rdquo; drifts a
little from the last layer's, are translated before decoding. Both are cheap, popular ways to trace how
information develops through depth.</p>""",

    nn=r"""
<p>Let \(h_\ell\) be the residual stream after layer \(\ell\), \(W_U\) the unembedding and
\(\mathrm{LN}\) the final layer normalisation. Then</p>
\[ \text{logits}_L = W_U\,\mathrm{LN}(h_L), \qquad
   \text{logit lens: } W_U\,\mathrm{LN}(h_\ell), \qquad
   \text{tuned lens: } W_U\,\mathrm{LN}(A_\ell h_\ell + b_\ell). \]
<p>The tuned-lens translator \((A_\ell, b_\ell)\) is trained so its output distribution matches the
model's final one (by minimising a KL divergence). No new information is created: the lens only reads
what is already linearly there.</p>
<p><b>Toy example.</b> A vocabulary of two words, &ldquo;Paris&rdquo; and &ldquo;Rome&rdquo;, and
\(W_U = I\) (identity). The residual stream after layers 1, 2 and 3 is \((0.2, 0.1)\),
\((1.5, 0.3)\) and \((3.0, 0.2)\). The softmax probability of &ldquo;Paris&rdquo; is
\(1/(1+e^{-(z_1-z_2)})\): 0.52 after layer 1, 0.77 after layer 2, 0.94 after layer 3. The logit lens
shows the answer &ldquo;Paris&rdquo; forming gradually. If layer 1 stored the answer in a rotated
basis, the raw lens would read nonsense there, and a learned \(A_1\) would undo the rotation: that is the
tuned lens.</p>""",

    recipe=[
        "Run the model and save the residual stream \\(h_\\ell\\) at every layer.",
        "Logit lens: apply the final norm and unembedding to each \\(h_\\ell\\) and read the decoded prediction.",
        "Tuned lens: for each layer, fit an affine map \\(A_\\ell h_\\ell + b_\\ell\\) on training data so the decoded output matches the final output.",
        "Evaluate on held-out data: how well does each layer's readout match the final answer, and from which layer on?",
        "Check what the readout relies on: are its weights on components that actually drive the output?",
    ],

    atari_map=[
        ("Layers \\(\\ell\\)", "The <b>frames</b> of the 15-frame horizon after \\(f^*\\): step 0 (the checkpoint) to step 15."),
        ("Residual stream \\(h_\\ell\\)", "The full 128-byte RAM at that step. RAM is the one state every frame reads and writes, like the residual stream."),
        ("Final logit", "The value of a target variable's RAM cell <b>at the end</b> of the horizon (one per candidate cell; 8 for Pong)."),
        ("Unembedding \\(W_U\\) (logit lens)", "The identity readout: read the target's own cell at step \\(t\\). There is no learned matrix on the VCS."),
        ("Tuned-lens translator \\(A_\\ell, b_\\ell\\)", "A ridge-regression readout from all 128 bytes at step \\(t\\) to the final value."),
        ("True intermediate value (unknown in an LLM)", "The actual byte at that step on the bit-exact run, plus the oracle's mask of which cells causally drive each final value."),
    ],

    atari=r"""
<p>Our runner <code>logit_lens.jl</code> builds both lenses on the VCS and scores them against values
we can measure directly.</p>
<ul>
<li><b>A set of runs to fit on.</b> A lens needs many examples. From the shared gameplay checkpoint at
\(f^*\) we record 48 runs of 16 steps each: the clean run, a source run whose last prefix action is
RIGHT, and 46 runs that each start with one candidate cell shifted by 17, 34, 64, 128 or 200. Every run is
a genuine bit-exact replay of the ROM, so every row is a real machine state.</li>
<li><b>Logit lens.</b> For each target variable and step \(t\), read its own cell. Against the true
intermediate value this has fidelity (\(R^2\)) of exactly 1, because the readout <i>is</i> the cell. It
is a <b>positive control</b> that the readout site is right, not a discovery. Against the
<i>final</i> value it shows how early the cell already holds its end-of-horizon value.</li>
<li><b>Tuned lens.</b> For each variable and step, fit a ridge regression (\(\lambda = 10^{-3}\)) from
all 128 bytes to the final value on 70&nbsp;% of the runs, and score \(R^2\) on the other 30&nbsp;%.
The earliest step with \(R^2 \ge 0.9\) is when the answer is &ldquo;already there&rdquo;.</li>
<li><b>Faithful or spurious?</b> The oracle sets each candidate cell to its value + 17 at \(f^*\),
replays, and marks which final values change. At the tuned lens's earliest decoding step we measure what
share of its weight magnitude sits on those truly causal cells. A high share means the lens reads the
mechanism. A low share means it found a correlate, for example a counter that happens to predict the
answer.</li>
</ul>
<p><b>Where we differ from the original.</b> A transformer has one shared unembedding; on the VCS each
variable lives in its own address, so the &ldquo;unembedding&rdquo; is just &ldquo;look at that
address&rdquo;. Our layers are time steps, not stacked modules. And the training runs are mostly
deliberate pokes, not a natural data distribution. Readout fidelity reaching the ceiling is
therefore <b>expected</b>: the machine keeps its variables in plain, addressable memory. The more
informative numbers are the tuned-lens curves and the faithful-weight share, which is the record's
scored faithfulness.</p>""",

    code=[
        dict(path="tools/xai_study/phaseC_mechanistic/logit_lens.jl", find="function build_trajset(base_ckpt::StellaEnvironment",
             role="xai", desc="Builds the 48 real runs (clean, source, and single-cell pokes) the lenses are fit on."),
        dict(path="tools/xai_study/phaseC_mechanistic/logit_lens.jl", find="function record_ram_tape(base_ckpt::StellaEnvironment",
             role="xai", desc="Records the &ldquo;residual stream over layers&rdquo;: the RAM at steps 0 to 15."),
        dict(path="tools/xai_study/phaseC_mechanistic/logit_lens.jl", find="function ridge_fit(X::Matrix{Float64}",
             role="xai", desc="The tuned-lens translator: a closed-form ridge regression per variable and step."),
        dict(path="tools/xai_study/phaseC_mechanistic/logit_lens.jl", find="function r2(y::Vector{Float64}",
             role="xai", desc="Readout fidelity: the coefficient of determination \\(R^2\\)."),
        dict(path="tools/xai_study/phaseC_mechanistic/logit_lens.jl", find="function build_causal_mask(base_ckpt::StellaEnvironment",
             role="xai", desc="The oracle mask: which cells, poked by +17, change each variable's final value."),
        dict(path="tools/xai_study/phaseC_mechanistic/logit_lens.jl", find="function _run_game_body(",
             role="xai", desc="Per variable and step: logit lens, tuned lens, earliest decoding step, faithful-weight share."),
        dict(path="tools/xai_study/common/shared_testbed_impl.jl", find="function build_shared_testbed(",
             role="substrate", desc="The shared gameplay checkpoint at \\(f^*\\) and the 15-frame action tail."),
        dict(path="tools/xai_study/common/jutari_oracle.jl", find="function intervene_ram!(env::StellaEnvironment",
             role="substrate", desc="The single-byte RAM write behind every poked run and the causal mask."),
        dict(path="jutari/src/env/StellaEnvironment.jl", find="function env_step!(env::StellaEnvironment",
             role="substrate", desc="One bit-exact frame: one &ldquo;layer&rdquo; of the VCS lens."),
        dict(path="jutari/src/env/StellaEnvironment.jl", find="get_ram(env::StellaEnvironment)",
             role="substrate", desc="The 128-byte RAM read at each step: the residual-stream analogue."),
    ],

    reading=r"""
<p>The figure plots readout fidelity against the true intermediate value, step by step, and it sits at
the ceiling on every game. Read that as a sanity check: the variable is stored at the site we read, in plain form.
The richer story is in the per-step curves. The logit lens reaches the final value only once the cell
stops changing. The tuned lens can sometimes predict it earlier by combining other bytes. Then check the
faithful-weight share: an early prediction built on causal cells is the mechanism showing through; one
built on non-causal cells is a correlation, the same present-versus-used trap as in probing.</p>""",

    pitfalls=[
        "<b>Readable is not explained.</b> Reading the right value at the right place confirms it is represented. It says nothing about how it is computed or what role it plays later.",
        "<b>Representation drift.</b> In a transformer, early layers may use a different basis than the last one, so the raw logit lens reads nonsense there. That is why Belrose et al. propose the tuned lens.",
        "<b>A tuned lens can learn too much.</b> A flexible translator fit per layer can find correlates that the model never uses. Check which inputs carry its weight, as we do with the oracle mask.",
        "<b>Off-distribution fitting.</b> Our training runs are mostly deliberate single-cell pokes. Readouts fit on them may not transfer to ordinary gameplay states.",
    ],

    try_it="julia --project=jutari tools/xai_study/phaseC_mechanistic/logit_lens.jl --games core",
    try_note="Writes <code>logit_lens_&lt;game&gt;.{json,npz}</code> and <code>logit_lens_core_summary.json</code> "
             "into <code>tools/xai_study/phaseC_mechanistic/out/</code>. <code>--n-traj K</code> changes the "
             "number of runs the lenses are fit on (default 48).",

    next=["linear_probing", "activation_patching", "das", "A8_wholestate"],
)
