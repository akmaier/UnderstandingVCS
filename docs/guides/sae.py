# Beginner's guide — sparse autoencoders (SAE).
# Rendered by build_pages.build_guide_section(); schema in docs/guides/__init__.py.

GUIDE = dict(
    key="sae",
    level="Mechanistic family · features from superposition",

    wiki=[
        ("Autoencoder", "https://en.wikipedia.org/wiki/Autoencoder"),
        ("Sparse dictionary learning", "https://en.wikipedia.org/wiki/Sparse_dictionary_learning"),
        ("Lasso (statistics)", "https://en.wikipedia.org/wiki/Lasso_(statistics)"),
        ("Representation learning", "https://en.wikipedia.org/wiki/Representation_learning"),
        ("Mechanistic interpretability", "https://en.wikipedia.org/wiki/Mechanistic_interpretability"),
    ],
    papers=[
        ("Cunningham, Ewart, Riggs, Huben &amp; Sharkey (2023). <i>Sparse Autoencoders Find Highly "
         "Interpretable Features in Language Models.</i> ICLR 2024.",
         "https://arxiv.org/abs/2309.08600"),
        ("Bricken et al. (2023). <i>Towards Monosemanticity: Decomposing Language Models With Dictionary "
         "Learning.</i> Transformer Circuits Thread.",
         "https://transformer-circuits.pub/2023/monosemantic-features"),
        ("Templeton et al. (2024). <i>Scaling Monosemanticity: Extracting Interpretable Features from "
         "Claude 3 Sonnet.</i> Transformer Circuits Thread.",
         "https://transformer-circuits.pub/2024/scaling-monosemanticity/"),
        ("Elhage et al. (2022). <i>Toy Models of Superposition.</i> (the theory SAEs are built on)",
         "https://arxiv.org/abs/2209.10652"),
        ("Heap et al. (2025). <i>Automated Interpretability Metrics Do Not Distinguish Trained and "
         "Random Transformers.</i> (critique: SAE features look interpretable even on random networks)",
         "https://arxiv.org/abs/2501.17727"),
    ],

    idea=r"""
<p>Look inside a language model and pick one neuron. You will often find it fires for many unrelated
things: Python code, the French word for &ldquo;cat&rdquo;, and closing brackets. Such a neuron is
<b>polysemantic</b> (many meanings). One explanation is <b>superposition</b>: the model wants to store
more concepts (&ldquo;features&rdquo;) than it has neurons, so it packs them in as overlapping
directions in activation space. That works because most features are <b>sparse</b>: in any one input,
only a few of them are active, so the overlaps rarely collide.</p>
<p>A <b>sparse autoencoder</b> (SAE) tries to undo the packing. It is a small, separate network trained
to rebuild the model's activation vector through a hidden layer with many units, under a penalty that
keeps <b>only a few units active at a time</b>. If it works, each hidden unit stands for one concept: it
is <b>monosemantic</b>. These units are then called the model's &ldquo;features&rdquo;, given names, and
used to explain and steer the model. SAEs are among the most widely used tools in interpretability
today.</p>""",

    nn=r"""
<p>Take an activation vector \(x \in \mathbb{R}^d\) (e.g. the residual stream of a transformer at one
token). The SAE encodes and decodes it:</p>
\[ z = \mathrm{ReLU}(W_e x + b_e), \qquad \hat x = W_d z + b_d, \qquad
   \mathcal{L} = \|x - \hat x\|^2 + \lambda \|z\|_1 . \]
<p>\(z\) is the <b>code</b> (one number per feature). ReLU sets negative values to zero. The first loss
term asks for a good reconstruction. The second, an <b>L1 penalty</b> (the sum of absolute values,
known from the lasso), pushes most entries of \(z\) to exactly zero. \(\lambda\) trades the two off.
Usually the hidden layer is <b>wider</b> than \(d\) (&ldquo;overcomplete&rdquo;), so it has room for
more features than neurons. Column \(k\) of \(W_d\) is feature \(k\)'s <b>direction</b> in activation
space.</p>
<p><b>Toy example.</b> A model stores 3 features in a 2-D space, at 120&deg; apart:
\(u_1=(1,0)\), \(u_2=(-0.5,0.87)\), \(u_3=(-0.5,-0.87)\). An input has only feature 2 on:
\(x=u_2=(-0.5,0.87)\). The two raw neurons read \(-0.5\) and \(0.87\): both respond, and each
would also respond to the other features. Now an SAE with three hidden units, encoder rows
\(u_1,u_2,u_3\) and bias \(-0.4\): the pre-activations are \((-0.5, 1.0, -0.5) - 0.4 =
(-0.9, 0.6, -0.9)\), and ReLU gives \(z=(0, 0.6, 0)\). Only unit 2 fires. Three features, cleanly
separated, from a 2-D space.</p>""",

    recipe=[
        "Record many activation vectors from the model (one layer, many inputs).",
        "Normalise them, then train the SAE on the reconstruction loss plus \\(\\lambda\\cdot\\) L1 of the code.",
        "Check reconstruction quality (fraction of variance explained) and sparsity (average number of active units, \\(L_0\\)).",
        "Interpret each unit: list the inputs where it fires most, or correlate it with known concepts.",
        "Test causally: set a feature to zero (or clamp it high), decode, put the result back into the model, and see whether the behaviour changes as the feature's name predicts.",
    ],

    atari_map=[
        ("Activation vector \\(x\\)", "The 128 bytes of RIOT RAM at one video frame. A recorded run gives one row per frame."),
        ("Layer being decomposed", "The whole RAM state. The VCS has no layers; RAM is the single place where game state lives between frames."),
        ("Feature (SAE hidden unit)", "A learned direction over RAM bytes. We hope each one tracks one <b>known game variable</b>, e.g. ball x or the player score."),
        ("Human label for a feature", "The verified concept&harr;cell map (<code>candidates_&lt;game&gt;.json</code>). For Pong: 8 cells, including scores, paddle and ball positions."),
        ("Feature steering / ablation", "Zero the feature in the code, decode to bytes, <b>write all 128 bytes into the real machine</b>, replay 15 frames, count screen pixels that change."),
        ("Ground truth (never available for an LLM)", "The intervention oracle: how much the screen breaks when each cell itself is set to 0 or base+17 and the ROM re-runs."),
    ],

    atari=r"""
<p>Our runner <code>sae.jl</code> is a complete SAE pipeline written by hand in Julia (no deep-learning
library), applied to the VCS state instead of a transformer.</p>
<ul>
<li><b>Data.</b> We record the RAM after every frame along the shared gameplay stream (90 seeded
random-action frames plus the 15-frame horizon, continued with NOOP; 150 frames for Pong). Each byte is
z-scored (mean 0, standard deviation 1). Bytes that never change are set to zero. Every fifth frame is
held out for testing.</li>
<li><b>Model and sweep.</b> \(z=\mathrm{ReLU}(W_e x+b_e)\), \(\hat x = W_d z + b_d\), loss = mean
squared error + \(\lambda\cdot\)mean\(|z|\). Full-batch gradient descent, 4000 epochs, learning rate
0.5, with the gradients written out explicitly. We sweep the width \(\in\{16,32,64\}\) and
\(\lambda\in\{0.01,0.05,0.2\}\) and keep the model with the best <i>held-out</i> reconstruction. Note:
these widths are smaller than 128, so the SAE is not overcomplete in raw bytes. But most of the 128 bytes
are constant over a short run, so the number of bytes that actually vary is far smaller.</li>
<li><b>Matching.</b> For each known cell that varies, we find the feature whose activation has the
largest absolute Pearson correlation with the cell's value over the run. The cell is
&ldquo;matched&rdquo; if that correlation is at least 0.7. The <b>match rate</b> is the share of
varying known cells that get a match. This is the kind of evidence papers use to say &ldquo;this feature
<i>is</i> concept X&rdquo;.</li>
<li><b>Causal use.</b> This is the test a language model cannot give us. For each matched cell that the
oracle says is causal, we encode the RAM at \(f^*\), set its best feature to zero, decode, undo the
z-scoring, round to bytes, and write the full 128-byte vector into a copy of the machine. Then we play
the 15-frame tail. The effect is the number of screen pixels that differ from the same write
<i>without</i> the ablation, so reconstruction error cancels out. We also count <b>decode leakage</b>
(how many bytes of the ablated decode differ from the real RAM) and the share of the effect that the target cell alone
explains.</li>
<li><b>Controls.</b> One ablated write is replayed through an independent fresh boot and must match
the checkpoint path exactly. The oracle's own importance map, scored as if it were the method, must get
a perfect rank correlation.</li>
</ul>
<p>The triad: <b>F</b> is the Spearman rank correlation between each cell's SAE &ldquo;recovery
score&rdquo; (best feature correlation &times; decoder weight mass on that cell) and the oracle's
importance. <b>S</b> predicts a held-out lesion (base+37) from the same score. <b>M</b> asks whether the
cells the SAE highlights are the cells that matter.</p>""",

    code=[
        dict(path="tools/xai_study/phaseC_mechanistic/sae.jl", find="function SAE(d_in::Integer, d_hidden::Integer, l1::Real",
             role="xai", desc="The model: encoder, decoder, L1 weight, random initialisation."),
        dict(path="tools/xai_study/phaseC_mechanistic/sae.jl", find="function loss(s::SAE, X::AbstractMatrix)",
             role="xai", desc="The objective: mean squared reconstruction error + \\(\\lambda\\cdot\\)mean\\(|z|\\)."),
        dict(path="tools/xai_study/phaseC_mechanistic/sae.jl", find="function fit!(s::SAE, X::AbstractMatrix",
             role="xai", desc="Training: full-batch gradient descent with hand-written gradients through ReLU and L1."),
        dict(path="tools/xai_study/phaseC_mechanistic/sae.jl", find="function feature_variable_matching(",
             role="xai", desc="Naming features: best |Pearson r| between a feature and each known RAM cell, threshold 0.7."),
        dict(path="tools/xai_study/phaseC_mechanistic/sae.jl", find="function causal_use(game, checkpoint",
             role="xai", desc="The causal test: zero a feature, decode, write all 128 bytes, replay, count changed pixels."),
        dict(path="tools/xai_study/phaseC_mechanistic/sae.jl", find="function sae_record_ram(",
             role="xai", desc="Records the per-frame 128-byte RAM tape the SAE is trained on."),
        dict(path="tools/xai_study/phaseC_mechanistic/sae.jl", find="function compute_game(game::AbstractString",
             role="xai", desc="One game: record, sweep width &times; \\(\\lambda\\), match, oracle importance, causal use, triad."),
        dict(path="tools/xai_study/common/jutari_oracle.jl", find="function intervene_ram!(env::StellaEnvironment",
             role="substrate", desc="Writes one byte into the emulated RIOT RAM; the decoded state is written with 128 of these."),
        dict(path="jutari/src/env/StellaEnvironment.jl", find="get_ram(env::StellaEnvironment)",
             role="substrate", desc="Exposes the live 128-byte RAM: the &ldquo;activation vector&rdquo; the SAE sees."),
        dict(path="jutari/src/env/StellaEnvironment.jl", find="function env_step!(env::StellaEnvironment",
             role="substrate", desc="One bit-exact video frame, used for recording and for every causal re-run."),
    ],

    reading=r"""
<p>Expect two very different pictures side by side. The reconstruction is close to perfect and the
match rate is high: almost every known cell has a feature that tracks it. By the usual standard of the
field, the SAE has &ldquo;found the variables&rdquo;. Then look at the causal-use bars. Ablating a
matched feature often moves the screen much less than the oracle says that cell matters, or moves it
through other bytes the decoder also touched. The lesson is the gap between <b>named</b> and
<b>used</b>: a feature can track a variable perfectly and still not be the handle by which the program's
computation runs.</p>""",

    pitfalls=[
        "<b>Correlation is not mechanism.</b> A feature that correlates with ball x may also correlate with a frame counter or the paddle. High correlation names a feature; it does not prove the model uses that direction.",
        "<b>Decode leakage.</b> Zeroing one feature changes the decoded value of many bytes at once. The observed effect then mixes several variables, which breaks the &ldquo;one feature, one concept&rdquo; story.",
        "<b>Hyper-parameters shape the dictionary.</b> Width and \\(\\lambda\\) decide how features split or merge. A different sweep can give different &ldquo;features&rdquo; on the same data.",
        "<b>Interpretable-looking is not evidence.</b> Heap et al. found that standard automated interpretability scores for SAE features look similar on trained and randomly initialised transformers. On the VCS we score against the oracle instead.",
        "<b>Small state, easy case.</b> The VCS keeps variables in separate bytes, so there is little superposition to undo. A high match rate here is expected and says little about how SAEs fare on dense LLM activations.",
    ],

    try_it="julia --project=jutari tools/xai_study/phaseC_mechanistic/sae.jl --games core",
    try_note="Writes <code>sae_&lt;game&gt;.{json,npz}</code> and <code>sae_core_summary.json</code> into "
             "<code>tools/xai_study/phaseC_mechanistic/out/</code>. The 9-point sweep dominates the run time; "
             "use <code>--d-hidden-sweep 32 --l1-sweep 0.05</code> for a quick single model.",

    next=["dictionaries", "linear_probing", "das", "A7_dimred"],
)
