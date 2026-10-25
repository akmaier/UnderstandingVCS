# Beginner's guide — LIME (local interpretable model-agnostic explanations).
# Rendered by build_pages.build_guide_section(); schema in docs/guides/__init__.py.

GUIDE = dict(
    key="lime",
    level="Perturbation family · fit a local surrogate",

    wiki=[
        ("Surrogate model", "https://en.wikipedia.org/wiki/Surrogate_model"),
        ("Weighted least squares", "https://en.wikipedia.org/wiki/Weighted_least_squares"),
        ("Ridge regression", "https://en.wikipedia.org/wiki/Ridge_regression"),
        ("Coefficient of determination (R²)", "https://en.wikipedia.org/wiki/Coefficient_of_determination"),
        ("Cosine similarity", "https://en.wikipedia.org/wiki/Cosine_similarity"),
    ],
    papers=[
        ("Ribeiro, Singh &amp; Guestrin (2016). <i>&ldquo;Why Should I Trust You?&rdquo;: Explaining the "
         "Predictions of Any Classifier.</i> KDD.",
         "https://arxiv.org/abs/1602.04938"),
        ("Alvarez-Melis &amp; Jaakkola (2018). <i>On the Robustness of Interpretability Methods</i> "
         "(LIME explanations can change a lot between nearby inputs and runs). ICML WHI workshop.",
         "https://arxiv.org/abs/1806.08049"),
        ("Slack, Hilgard, Jia, Singh &amp; Lakkaraju (2020). <i>Fooling LIME and SHAP: Adversarial Attacks "
         "on Post hoc Explanation Methods.</i> AIES.",
         "https://arxiv.org/abs/1911.02508"),
    ],

    idea=r"""
<p>A deep network is a very curvy function. But zoom in far enough on any smooth curve and it looks
like a straight line. <b>LIME</b> uses this idea. To explain one prediction, it builds a <b>simple
model</b> (a straight-line, i.e. linear, model) that imitates the network <i>only near this one
input</i>. Such a stand-in model is called a <b>surrogate</b>. The weights of the surrogate are the
explanation: a large positive weight means &ldquo;this part pushes the output up, around here&rdquo;.</p>
<p>How do you get data to fit the surrogate? You make it. Switch random parts of the input off (for
an image: grey out random <i>superpixels</i>, patches of similar colour), run the network on each
version, and record the outputs. Versions that are close to the original get a high
<b>weight</b> in the fit, far-away versions a low one. Then fit the line. LIME is
<b>model-agnostic</b>: it treats the model as a black box, so it works for networks, random
forests, or, as here, a game console.</p>""",

    nn=r"""
<p>Describe each perturbed version by a binary vector \(z \in \{0,1\}^d\): \(z_j = 1\) if part
\(j\) is present, \(0\) if it was switched off. The original input is \(z = \mathbf 1\) (all
present). Sample \(z_1,\dots,z_N\), run the model on each, \(y_k = f(h(z_k))\), where \(h\) maps
\(z\) back to a real input. Weight each sample by its closeness to the original,
\(\pi_k = \exp(-D(z_k,\mathbf 1)^2/\sigma^2)\), with distance \(D\) and <b>kernel width</b>
\(\sigma\). Then solve the <b>weighted least-squares</b> problem</p>
\[ (w, b) \;=\; \arg\min_{w,b}\; \sum_{k=1}^{N} \pi_k\,\big(y_k - w\cdot z_k - b\big)^2 . \]
<p>The coefficients \(w_j\) are the attribution. The library version also keeps only the top few
features (a <i>sparse</i> fit) and adds a small ridge penalty. The fit's \(R^2\) (the share of the
output variation the line explains) tells you how linear the neighbourhood really is.</p>
<p><b>Toy example.</b> Two parts, \(f(z) = 3z_1 + 2z_1z_2\). The four possible samples are
\(f(1,1)=5\), \(f(1,0)=3\), \(f(0,1)=0\), \(f(0,0)=0\). Give the original weight 1, the two
one-off samples weight 0.5, the all-off sample 0.25. Solving the \(3\times3\) normal equations gives
\(w_1 = 13/3 \approx 4.33\), \(w_2 = 4/3 \approx 1.33\), \(b = -8/9\). Compare with the truth at the
original: switching off \(z_1\) costs 5, switching off \(z_2\) costs 2. With equal weights the fit
would give \(w_1 = 4\), \(w_2 = 1\), \(b = -0.5\). The locality weights pull the line toward the
original, so \(w_2\) moves from 1 toward the local value 2. No line can fit the interaction term
\(2z_1z_2\) exactly; the surrogate spreads it over both coefficients, and \(R^2 < 1\) shows it.</p>""",

    recipe=[
        r"Choose the output \(y\), the interpretable parts, a number of samples \(N\), a keep-probability and a kernel width \(\sigma\).",
        r"Draw \(N\) random present/absent vectors \(z_k\) (include the original \(z=\mathbf 1\)).",
        r"Turn each \(z_k\) into a real input (absent parts set to the baseline), run the model, record \(y_k\).",
        r"Weight each sample by its closeness to the original, \(\pi_k = \exp(-D(z_k,\mathbf 1)^2/\sigma^2)\).",
        r"Fit \(y \approx w\cdot z + b\) by weighted least squares; report \(w\) and the fit's \(R^2\).",
        r"Repeat with other random seeds to see how stable \(w\) is.",
    ],

    atari_map=[
        (r"The black-box model \(f\)", "The cartridge program on the bit-exact emulated <b>Atari 2600 VCS</b> (<code>jutari</code>)."),
        ("Superpixels (interpretable parts)", "The <b>candidate RAM cells</b>, de-duplicated (8 to 23 per core game). TIA and joystick causes are not in the interpretable space."),
        (r"\(h(z)\): switch parts off", "<code>deepcopy</code> the checkpoint and <code>intervene_ram!(env, j, 0)</code> for every absent cell; then 15 frames of the real ROM."),
        (r"Samples \(N\), kernel \(\pi\)", "\\(N = 300\\), keep-prob 0.5, the first sample is the all-present original; cosine-distance kernel with \\(\\sigma = 0.75\\sqrt{d}\\)."),
        ("Surrogate \\(g(z)=w\\cdot z+b\\)", "Weighted least squares with a tiny ridge (\\(10^{-6}\\)) on the coefficients; all cells kept (no sparsity step)."),
        ("&ldquo;Ground-truth importance&rdquo;", "The oracle's single-cause |&Delta;y| per cause, from real re-runs."),
    ],

    atari=r"""
<p><b>Sampling.</b> <code>lime_sample</code> draws \(N = 300\) present/absent vectors over the
candidate RAM cells. Row 1 is forced to the original (all present); every other cell is present with
probability 0.5. For each row, <code>masked_read_y</code> copies the checkpoint, zeroes the absent
cells with <code>intervene_ram!</code>, plays the recorded 15 actions with <code>env_step!</code>,
and reads \(y\). These 300 real-emulator runs are the whole cost of one fit.</p>
<p><b>Weights.</b> <code>lime_weights</code> uses the cosine distance between \(z\) and
\(\mathbf 1\) and the kernel \(\exp(-D^2/\sigma^2)\) with the LIME library's default width
\(\sigma = 0.75\sqrt{d}\) (\(d\) = number of cells). Cosine distance here lies between 0 and 1, while
\(\sigma\) is above 2 for 8 or more cells, so all weights stay between roughly 0.8 and 1. In practice
the fit is close to an <b>unweighted</b> least-squares fit over a broad neighbourhood (about half the
cells zeroed per sample), not a tight local one.</p>
<p><b>Fit.</b> <code>fit_weighted_linear</code> solves the ridge-regularised normal equations and
returns the coefficients, the intercept and the weighted \(R^2\). Each RAM cause gets \(|w_j|\) of its
cell; TIA and joystick causes get 0. The sign is dropped because the oracle column is a magnitude.</p>
<p><b>Stability.</b> LIME depends on its random samples, so the runner refits from 5 independent
seeds (5 &times; 300 re-runs per output) and records the spread of the oracle correlation, of
\(R^2\), and the average cosine similarity between the coefficient vectors.</p>
<p><b>How is this different from the oracle?</b> The oracle asks one exact question per cause:
&ldquo;change only this, at the real state, what happens?&rdquo; LIME asks many joint questions
(half the cells zeroed at once) and then <i>compresses</i> the answers into one line. If the game
logic were additive, the line's coefficients would equal the zeroing effects. Where it is not
(thresholds, AND-like interactions, sprite timing), the line averages the effect over damaged
contexts, and the \(R^2\) tells you how much it missed.</p>""",

    code=[
        dict(path="tools/xai_study/phaseB_attribution/lime.jl", find="function lime_sample(",
             role="xai", desc="Draws the present/absent design matrix (row 1 = the original) and re-runs the ROM once per row."),
        dict(path="tools/xai_study/phaseB_attribution/lime.jl", find="function masked_read_y(",
             role="xai", desc="One perturbed run: <code>deepcopy</code>, zero the absent cells, 15 &times; <code>env_step!</code>, read y."),
        dict(path="tools/xai_study/phaseB_attribution/lime.jl", find="function lime_weights(",
             role="xai", desc="The locality kernel \\(\\exp(-D^2/\\sigma^2)\\) with cosine distance to the original."),
        dict(path="tools/xai_study/phaseB_attribution/lime.jl", find="function fit_weighted_linear(",
             role="xai", desc="The surrogate: ridge-regularised weighted least squares, plus the weighted \\(R^2\\)."),
        dict(path="tools/xai_study/phaseB_attribution/lime.jl", find="_resolve_kernel_width(kw, n_cells) =",
             role="xai", desc="Default kernel width \\(0.75\\sqrt{d}\\), the LIME library's choice."),
        dict(path="tools/xai_study/phaseB_attribution/lime.jl", find="for s in 1:(stability_seeds - 1)",
             role="xai", desc="The stability sweep: refits from fresh seeds and compares them."),
        dict(path="tools/xai_study/common/jutari_oracle.jl", find="function intervene_ram!(",
             role="substrate", desc="The poke that switches a cell &ldquo;off&rdquo; in the emulated RAM."),
        dict(path="jutari/src/env/StellaEnvironment.jl", find="function env_step!(",
             role="substrate", desc="One frame of the real machine; each sample costs 15 of them."),
        dict(path="tools/xai_study/ground_truth/oracle_intervene.jl", find="function run_intervention(",
             role="substrate", desc="The oracle's single-cause re-run, the ground truth for the surrogate's weights."),
        dict(path="tools/xai_study/common/shared_testbed_impl.jl", find="function build_shared_testbed(",
             role="substrate", desc="The shared gameplay checkpoint and the position output."),
    ],

    reading=r"""
<p>Bars are \(|w_j|\) per cause next to the oracle's true effect; TIA and joystick causes are zero
for LIME by construction, and both causes of a byte share a height. Read the record's surrogate
\(R^2\) before trusting the bars: a low value is LIME admitting that a line is a poor model of the
game near this state. Also read the stability block: if refits with other seeds disagree, the
ranking is partly sampling luck. Expect LIME to track the main structure on both content and
position outputs reasonably well.</p>""",

    pitfalls=[
        "<b>The neighbourhood is a choice.</b> Kernel width, keep-probability and the &ldquo;off&rdquo; value all change the answer. With our defaults the weights barely vary, so &ldquo;local&rdquo; really means &ldquo;about half the cells zeroed&rdquo;.",
        "<b>Interactions get smeared.</b> A linear surrogate cannot represent AND-like effects; it splits them across coefficients (toy example). Check \\(R^2\\).",
        "<b>Instability.</b> Different random samples give different coefficients (Alvarez-Melis &amp; Jaakkola). That is why we refit with several seeds and report the spread.",
        "<b>Explanations can be gamed.</b> Slack et al. built models that detect LIME's off-distribution samples and behave differently on them. The VCS cannot do that, but it does run the program on states the game would never reach.",
    ],

    try_it="julia --project=jutari tools/xai_study/phaseB_attribution/lime.jl --games core",
    try_note="Writes <code>lime_&lt;game&gt;_{content,ball_pixel}.json/.npz</code> into "
             "<code>tools/xai_study/phaseB_attribution/out/</code>, with the surrogate \\(R^2\\) and the "
             "stability block. Knobs: <code>--n-samples</code>, <code>--kernel-width</code>, "
             "<code>--stability-seeds</code>.",

    next=["kernelshap", "rise", "occlusion", "linear_probing"],
)
