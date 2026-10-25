# Beginner's guide — the N/A audit (when an XAI method does not apply).
# Rendered by build_pages.build_guide_section(); schema in docs/guides/__init__.py.

GUIDE = dict(
    key="na_audit",
    level="Applicability audit · what a method needs before it can run",

    wiki=[
        ("Convolutional neural network", "https://en.wikipedia.org/wiki/Convolutional_neural_network"),
        ("Attention (machine learning)", "https://en.wikipedia.org/wiki/Attention_(machine_learning)"),
        ("Transformer (deep learning)", "https://en.wikipedia.org/wiki/Transformer_(deep_learning)"),
        ("Reinforcement learning", "https://en.wikipedia.org/wiki/Reinforcement_learning"),
        ("Decision tree learning", "https://en.wikipedia.org/wiki/Decision_tree_learning"),
    ],
    papers=[
        ("Selvaraju et al. (2017). <i>Grad-CAM: Visual Explanations from Deep Networks via "
         "Gradient-based Localization.</i> ICCV.",
         "https://arxiv.org/abs/1610.02391"),
        ("Abnar &amp; Zuidema (2020). <i>Quantifying Attention Flow in Transformers.</i> ACL.",
         "https://arxiv.org/abs/2005.00928"),
        ("Bastani, Pu &amp; Solar-Lezama (2018). <i>Verifiable Reinforcement Learning via Policy "
         "Extraction</i> (VIPER). NeurIPS.",
         "https://arxiv.org/abs/1805.08328"),
        ("Jain &amp; Wallace (2019). <i>Attention is not Explanation.</i> NAACL.",
         "https://arxiv.org/abs/1902.10186"),
    ],

    idea=r"""
<p>Every XAI method is a recipe, and every recipe has ingredients. Saliency needs a differentiable
function. Occlusion needs a way to blank out an input. Some popular methods need much more specific
ingredients: a <b>convolutional layer</b> whose feature maps can be weighted, <b>attention
matrices</b> that can be multiplied together, or a <b>trained agent</b> whose behaviour can be
copied. If the system you study lacks the ingredient, the recipe cannot even start. That is what
<b>applicability</b> means. A method applies to a system only if every object its formula
refers to actually exists there, and every assumption behind its interpretation holds.</p>
<p>The Atari 2600 VCS is a hand-written program on a 1977 games console. It has a CPU, a video chip and
a timer chip. It has no neurons, no layers, no attention and no learned agent. So some famous methods
simply have nothing to bind to. This page is an <b>audit</b> of those methods. We do not invent a stand-in
and report a score for it. We record &ldquo;does not apply&rdquo;, name the missing ingredient, and explain
why every stand-in we could build would be a different method under a borrowed name. An empty cell
in a results table is more honest than a number that measures nothing.</p>""",

    nn=r"""
<p>Here are the three flagship methods of the audit, and exactly what each one needs on a network.</p>
<p><b>Grad-CAM</b> (Selvaraju et al. 2017). Take the last convolutional layer: \(K\) feature maps
\(A^k\), each a small grid of activations. Average the gradient of the class score \(y^c\) over each map
to get one weight per channel, then form a ReLU-ed weighted sum:</p>
\[ \alpha_k^c = \frac{1}{Z}\sum_{i,j}\frac{\partial y^c}{\partial A^k_{ij}}, \qquad
   L^c = \mathrm{ReLU}\Big(\sum_k \alpha_k^c A^k\Big) . \]
<p>Ingredients: <i>learned channels</i> \(k\), a <i>spatial grid</i> \((i,j)\) aligned with the image, and a
<i>class score</i>. CAM, Grad-CAM++ and feature visualisation / TCAV (concept vectors in a learned
activation space) need the same kind of learned, spatially arranged channels.</p>
<p><b>Attention rollout</b> (Abnar &amp; Zuidema 2020). A transformer layer \(l\) has an attention matrix
\(A^{(l)}\): how much each token reads from each other token. Mix in the residual connection and multiply
through the layers:</p>
\[ \tilde A^{(l)} = \tfrac12 A^{(l)} + \tfrac12 I, \qquad R = \tilde A^{(L)}\cdots\tilde A^{(1)} . \]
<p>Ingredients: <i>tokens</i> and <i>softmax attention weights</i> in every layer.</p>
<p><b>VIPER</b> (Bastani et al. 2018). Take a trained reinforcement-learning policy \(\pi(a\mid s)\) with
its Q-function (how good each action is in each state). Train a decision tree to imitate it, weighting
states by how much a wrong action would cost. Ingredients: a <i>learned policy</i> and its <i>Q-values</i>.</p>
<p><b>Toy example: what a forced re-target produces.</b> On a tiny CNN with two \(2\times2\) maps
\(A^1=\begin{pmatrix}1&0\\0&0\end{pmatrix}\), \(A^2=\begin{pmatrix}0&2\\0&2\end{pmatrix}\) and channel weights
\(\alpha_1=0.5\), \(\alpha_2=-0.25\), Grad-CAM gives
\(L=\mathrm{ReLU}\begin{pmatrix}0.5&-0.5\\0&-0.5\end{pmatrix}=\begin{pmatrix}0.5&0\\0&0\end{pmatrix}\): one hot
cell. Now try it on RAM. Suppose we declare the 128 bytes one &ldquo;map&rdquo; with one channel. Then
\(\alpha\) is the mean gradient, and \(L_i=\mathrm{ReLU}(\alpha\,\mathrm{ram}_i)\) is just a rescaled copy of the
RAM. It says nothing about which byte matters. Suppose instead each byte is its own \(1\times1\) channel.
Then \(\alpha_k\,\mathrm{ram}_k = \frac{\partial y}{\partial \mathrm{ram}_k}\,\mathrm{ram}_k\), which is
Gradient&times;Input, a method we already score. Either way the &ldquo;Grad-CAM&rdquo; label adds nothing.</p>""",

    recipe=[
        r"Write down the method's formula and list every object it refers to (feature maps, attention matrices, a policy, learned directions).",
        r"Check each object against the system. Does it exist? Is it learned, so that the method's interpretation makes sense?",
        r"If one is missing, look for a faithful substitute. Would the method's defining steps still mean what they mean on a network?",
        r"If the only substitutes collapse into another method (or into the ground truth itself), record <b>does not apply</b> and the missing ingredient.",
        r"Contrast with methods that <i>do</i> apply: confirm they have real causes to work on, so the absence is structural, not a coverage gap.",
        r"Keep the n/a row out of the faithfulness leaderboard rather than giving it a zero or an invented score.",
    ],

    atari_map=[
        ("Convolutional feature maps \\(A^k\\) (Grad-CAM, Grad-CAM++, CAM)",
         "None. The VCS has no learned intermediate layers. The TIA (video chip) draws pixels directly "
         "from register values set by hand-written code, line by line."),
        ("Attention matrices \\(A^{(l)}\\) (attention rollout)",
         "None. Which byte influences which is fixed by the program's instructions. That true data flow "
         "is what the intervention oracle and path patching measure; calling it &ldquo;attention&rdquo; "
         "would relabel ground truth as a method."),
        ("A learned policy \\(\\pi(a\\mid s)\\) (VIPER)",
         "None. The study explains the console itself. Joystick inputs come from a seeded random "
         "stream, not from a trained agent."),
        ("Learned channels / concepts (feature visualisation, TCAV)",
         "None. RAM bytes and registers have fixed, known meanings (the score byte <i>is</i> the "
         "score). Testing such meanings is the job of probing with control tasks."),
        ("A ground truth to score against",
         "Present: the intervention oracle and many candidate causes per game. The applicable methods "
         "use them; the n/a methods produce no map to compare."),
    ],

    atari=r"""
<p>The audit is a runner like every other, <code>na_audit.jl</code>, but it runs no method, because there
is nothing to run. Instead it writes a structured, checkable record.</p>
<ul>
<li><b>Six audited methods.</b> <code>NA_METHODS</code> lists Grad-CAM, Grad-CAM++, attention rollout,
VIPER, CAM, and feature visualisation / TCAV. Each entry names the required ingredient, its kind (conv
feature maps, attention, learned policy, learned channels), sets <code>present_in_vcs = false</code>, and
gives two texts. One says why the method does not apply. The other says why a naive re-target onto VCS
state would be a different method.</li>
<li><b>Grounding per game.</b> <code>ground_game</code> loads the game's candidate-cause file: the RAM
bytes the applicable methods attribute over. It records that count next to zero conv maps, zero
attention heads, zero learned policies and zero learned channels. That <b>asymmetry</b> is the finding:
there are real causes to explain, and none of the ingredients these methods need.</li>
<li><b>The contrast.</b> <code>APPLICABLE_CONTRAST</code> names methods that do bind to real causes:
vanilla gradient, Integrated Gradients, occlusion, counterfactuals. They are scored against the same
oracle (<code>run_intervention</code>) elsewhere, and gradient methods flow through
<code>soft_ram_peek</code>. A re-targeted Grad-CAM would collapse onto exactly those gradients.</li>
<li><b>A self-check.</b> <code>selftest</code> asserts that every entry is marked absent with a named
ingredient, that the canonical methods are covered, that every core game has candidate causes, and that
the record round-trips. The headline value is the count of methods that do not apply; the record carries
no F, S or M.</li>
</ul>
<p><b>Why not force a number?</b> Any score needs a map to compare with the oracle. Building one would mean
choosing a stand-in for the missing ingredient, and every stand-in we can build is either uninformative
(the rescaled-input case above), a method already on the leaderboard (Gradient&times;Input), or the ground
truth itself (the data-flow graph). Scoring it under the famous name would credit or blame that method for
something it never did. The absence is also <b>structural</b>. It is the same in every game, because no
cartridge contains a network. That is why the record is game-independent and why it stays off the
leaderboard.</p>""",

    code=[
        dict(path="tools/xai_study/phaseB_attribution/na_audit.jl", find="struct NAMethod",
             role="xai", desc="The audit schema: required ingredient, its kind, presence flag, why it fails, why no re-target."),
        dict(path="tools/xai_study/phaseB_attribution/na_audit.jl", find="const NA_METHODS = NAMethod[",
             role="xai", desc="The six audited methods, each with its missing ingredient."),
        dict(path="tools/xai_study/phaseB_attribution/na_audit.jl", find="\"grad_cam\", \"Grad-CAM\"",
             role="xai", desc="Grad-CAM: needs last-layer conv feature maps; a RAM re-target is just the content-path gradient."),
        dict(path="tools/xai_study/phaseB_attribution/na_audit.jl", find="\"attention_rollout\", \"Attention rollout\"",
             role="xai", desc="Attention rollout: needs per-layer attention matrices; the VCS data flow is ground truth, not attention."),
        dict(path="tools/xai_study/phaseB_attribution/na_audit.jl", find="\"viper\", \"VIPER",
             role="xai", desc="VIPER: needs a learned policy and Q-function; the VCS runs a fixed program."),
        dict(path="tools/xai_study/phaseB_attribution/na_audit.jl", find="function ground_game(",
             role="xai", desc="Per-game grounding: real candidate-cause count against zero required channels."),
        dict(path="tools/xai_study/phaseB_attribution/na_audit.jl", find="const APPLICABLE_CONTRAST",
             role="xai", desc="The methods that do apply and are scored against the oracle."),
        dict(path="tools/xai_study/phaseB_attribution/na_audit.jl", find="function selftest(groundings",
             role="xai", desc="Consistency checks: every entry absent and explained, every core game grounded."),
        dict(path="tools/xai_study/ground_truth/oracle_intervene.jl", find="function run_intervention",
             role="substrate", desc="The oracle the applicable methods are scored against; the n/a methods give it nothing to compare."),
        dict(path="jutari/src/diff/SoftStep.jl", find="function soft_ram_peek",
             role="substrate", desc="The differentiable RAM read: where a re-targeted Grad-CAM would end up as a plain gradient."),
    ],

    reading=r"""
<p>This page has no faithfulness figure, and that is the point. The audit box marks the row as
excluded from the leaderboard and shows how many methods were audited. Read the per-method table as a
list of preconditions: for each method, the ingredient it needs, the fact that the VCS has none of it,
and the reason a substitute would not be the same method. The same verdict holds for every game. Set it
beside the gradient and perturbation pages: those methods need only a function and a way to change its
inputs, and both exist on the VCS.</p>""",

    pitfalls=[
        "<b>Forcing a method produces a different method.</b> Re-targeting Grad-CAM onto raw state yields a rescaled input or Gradient&times;Input. Reporting it as &ldquo;Grad-CAM&rdquo; misattributes the result.",
        "<b>A zero is not the same as n/a.</b> Entering a score of 0 would claim the method ran and failed. Leaving the row out of the leaderboard records that it could not run.",
        "<b>Applicability is about meaning, not just shapes.</b> You can reshape 128 bytes into an 8&times;16 grid, but neighbouring bytes are not neighbouring pixels. Grad-CAM's spatial reading assumes a grid that comes from the image.",
        "<b>Attention is not explanation anyway.</b> Even where attention exists, its weights may not reflect what drives the output (Jain &amp; Wallace 2019). Applicability is a necessary condition, not a guarantee.",
        "<b>The verdict is scoped.</b> These methods do apply to a neural agent <i>playing</i> Atari. The audit concerns the console itself, which is the subject of this study.",
    ],

    try_it="julia --project=jutari tools/xai_study/phaseB_attribution/na_audit.jl",
    try_note="Writes <code>na_audit_&lt;game&gt;.json/.npz</code> for the six core games, "
             "<code>na_audit_combined.json</code> into <code>tools/xai_study/phaseB_attribution/out/</code>, "
             "and the write-up <code>na_audit.md</code>. Runs in seconds: no emulation is needed. "
             "<code>--games labeled</code> grounds every labelled game; <code>--selftest</code> checks without writing.",

    next=["saliency", "gradxinput", "path_patching", "linear_probing"],
)
