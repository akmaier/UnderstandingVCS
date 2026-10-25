# Beginner's guide — on-distribution counterfactual explanations.
# Rendered by build_pages.build_guide_section(); schema in docs/guides/__init__.py.

GUIDE = dict(
    key="counterfactual",
    level="Perturbation family · the smallest change that flips the outcome",

    wiki=[
        ("Counterfactual conditional", "https://en.wikipedia.org/wiki/Counterfactual_conditional"),
        ("Causal model", "https://en.wikipedia.org/wiki/Causal_model"),
        ("Explainable artificial intelligence", "https://en.wikipedia.org/wiki/Explainable_artificial_intelligence"),
        ("Manifold hypothesis", "https://en.wikipedia.org/wiki/Manifold_hypothesis"),
        ("Greedy algorithm", "https://en.wikipedia.org/wiki/Greedy_algorithm"),
    ],
    papers=[
        ("Wachter, Mittelstadt &amp; Russell (2017). <i>Counterfactual Explanations without Opening the "
         "Black Box: Automated Decisions and the GDPR.</i> Harvard J. of Law &amp; Technology.",
         "https://arxiv.org/abs/1711.00399"),
        ("Olson, Khanna, Neal, Li &amp; Wong (2021). <i>Counterfactual State Explanations for "
         "Reinforcement Learning Agents via Generative Deep Learning.</i> Artificial Intelligence.",
         "https://arxiv.org/abs/2101.12446"),
        ("Atrey, Clary &amp; Jensen (2020). <i>Exploratory Not Explanatory: Counterfactual Analysis of "
         "Saliency Maps for Deep Reinforcement Learning.</i> ICLR.",
         "https://arxiv.org/abs/1912.05743"),
        ("Laugel et al. (2019). <i>The Dangers of Post-hoc Interpretability: Unjustified Counterfactual "
         "Explanations.</i> IJCAI. (A critique: many counterfactuals land where no real data lives.)",
         "https://arxiv.org/abs/1907.09294"),
    ],

    idea=r"""
<p>A bank&rsquo;s model rejects your loan. A heat-map of &ldquo;important features&rdquo; does not help
you much. What helps is a sentence like: <i>&ldquo;If your income had been 4&nbsp;000 higher, the loan
would have been approved.&rdquo;</i> That sentence is a <b>counterfactual explanation</b>. It names the
<b>smallest change to the input that flips the output</b>. The word comes from logic: a counterfactual
is a &ldquo;what if things had been different&rdquo; statement. Wachter et al. made it popular in
machine learning because it needs no access to the model&rsquo;s insides and is easy for a person to act
on.</p>
<p>There is a trap. The smallest change in raw numbers can produce an input that could never occur:
an image with a few strange pixels, a person aged 12 with a mortgage. Such an input is
<b>off-distribution</b> (or <b>off-manifold</b>): it lies outside the region where real data lives.
An explanation built on it describes the model&rsquo;s behaviour on nonsense. A good counterfactual is
therefore also <b>plausible</b>: the changed input should look like something the world really
produces. On the VCS we get plausibility for free, by copying values from other frames the game
really visited, and validity for free, by re-running the real program.</p>""",

    nn=r"""
<p>For a model \(f\), an input \(x\) and a desired different output, Wachter et al. solve</p>
\[ x^{\ast} \;=\; \arg\min_{x'} \; \lambda\,\big(f(x') - y_{\text{target}}\big)^2 \;+\; d(x, x'), \]
<p>where \(d\) is a distance (often the \(L_1\) norm, which favours changing few features). The
explanation is the difference \(x^\ast - x\). Later work adds a <b>plausibility</b> term that keeps
\(x'\) near real data, for example a generative model (Olson et al. use one to create counterfactual
Atari frames for RL agents). Two properties are checked: <b>validity</b> (does \(f(x^\ast)\) really
reach the target?) and <b>sparsity / minimality</b> (how few features were changed?).</p>
<p><b>Toy example.</b> A loan model approves when \(f(x) = 2\,\text{income} + \text{savings} - 3\,\text{debt}
\ge 10\). Applicant: income 3, savings 2, debt 1, so \(f = 6 + 2 - 3 = 5\): rejected. Real applicants
in the data have one of these alternative profiles: (income 5, savings 2, debt 1) or (income 3, savings
4, debt 0). Try copying one feature at a time from the first real profile: income 3&rarr;5 gives
\(f = 9\), still rejected. Copying two features from the second profile, savings 2&rarr;4 and debt
1&rarr;0, gives \(f = 6 + 4 - 0 = 10\): approved. So the plausible counterfactual is
&ldquo;two more savings and no debt&rdquo;. A purely numerical optimiser might instead suggest income
3&rarr;5.5, a value no applicant has. Both are valid; only one is grounded in real data.</p>""",

    recipe=[
        "Fix the input \\(x\\), the output \\(y = f(x)\\), and what counts as a flip (a class change, or a move larger than a threshold).",
        "Collect real alternative inputs \\(x'\\) from the data, so every substituted value is plausible.",
        "Greedily build a feature set \\(S\\): at each step copy the feature from \\(x'\\) that moves \\(y\\) furthest, until the move is large enough.",
        "Prune: drop any feature from \\(S\\) whose removal keeps the counterfactual valid.",
        "Report \\(S\\) and the new values as the explanation, and check validity and minimality.",
    ],

    atari_map=[
        ("The model \\(f\\)", "The ROM program on the bit-exact emulated VCS (<code>jutari</code>), run for "
         "15 frames from the checkpoint."),
        ("The input \\(x\\) and its features", "The machine state at frame \\(f^*{=}90\\); the features are "
         "the candidate RAM bytes (the game&rsquo;s known variables)."),
        ("A real alternative \\(x'\\)", "The genuine RAM at another frame of the same seeded gameplay "
         "stream (frames such as 15, 30, 45, &hellip; before \\(f^*\\))."),
        ("The output \\(y\\)", "A <b>content</b> output (one candidate RAM byte that is causally active and "
         "varies across the alternatives) and a <b>position</b> output (how many pixels change in the "
         "shared screen region)."),
        ("&ldquo;Flip&rdquo;", "A move of at least \\(\\tau{=}0.5\\) of the move you get by copying "
         "<i>all</i> candidate bytes from \\(x'\\)."),
        ("Evaluating \\(f(x')\\)", "Write the bytes into a copy of the checkpoint and re-run the real ROM. "
         "No surrogate, no generative model."),
    ],

    atari=r"""
<p>The runner uses the shared gameplay testbed: seeded random actions, a checkpoint at \(f^*\), a
15-frame horizon. Every number below comes from a real re-run of the emulator.</p>
<ol>
<li><b>Alternative pool.</b> The runner replays the same action stream to several earlier frames and
keeps their full RAM. These are states the game truly reached, so any byte copied from them is an
on-distribution value. This answers the off-manifold objection Atrey et al. raised against saliency
explanations for Atari agents.</li>
<li><b>Content byte.</b> The content output is a candidate byte that both reacts to the oracle&rsquo;s
interventions and actually takes different values across the pool. If no such byte exists, the run
falls back to the most causally active byte and records that it does not vary.</li>
<li><b>Choosing \(x'\).</b> For each output, the runner copies all candidate bytes from each pool
frame, re-runs, and picks the frame that moves \(y\) the most: the most different real world.</li>
<li><b>Per-cell map.</b> Copy one byte at a time from \(x'\) and re-run. The size of the change in
\(y\) is that byte&rsquo;s attribution, which is scored against the oracle like every other
attribution method (correlation, precision@3, deletion/insertion curves).</li>
<li><b>Minimal valid edit.</b> Greedy forward selection adds the byte that moves \(y\) furthest until
the move reaches \(\tau\) of the full move; a backward pass removes redundant bytes. The runner reports
validity and minimality: \(|S|\) divided by the number of bytes that move \(y\) on their own.</li>
</ol>
<p>Because the method intervenes rather than differentiates, it works on the <b>position</b> output,
where the gradient methods see an exact zero. The difference from Wachter et al. is that we never
optimise continuous values. We only substitute whole bytes from real frames. That keeps every edit
plausible per byte, but a <i>combination</i> of bytes from another frame inside the current state is
not guaranteed to be a reachable state.</p>""",

    code=[
        dict(path="tools/xai_study/phaseB_attribution/counterfactual.jl", find="function cf_read_y(",
             role="xai", desc="One counterfactual evaluation: copy chosen bytes from \\(x'\\) into the checkpoint, re-run, read \\(y\\)."),
        dict(path="tools/xai_study/phaseB_attribution/counterfactual.jl", find="function single_cell_cf(",
             role="xai", desc="The per-byte counterfactual map scored against the oracle."),
        dict(path="tools/xai_study/phaseB_attribution/counterfactual.jl", find="function greedy_minimal_cf(",
             role="xai", desc="Greedy forward selection plus backward pruning to the minimal valid edit."),
        dict(path="tools/xai_study/phaseB_attribution/counterfactual.jl", find="pool_frames = sort(unique(",
             role="xai", desc="The on-distribution pool: genuine RAM from other frames of the same gameplay stream."),
        dict(path="tools/xai_study/phaseB_attribution/counterfactual.jl", find="function pick_cf_content_idx(",
             role="xai", desc="Picks a content byte that is causally active and really varies across the pool."),
        dict(path="tools/xai_study/phaseB_attribution/counterfactual.jl", find="function compute_one(",
             role="xai", desc="One experiment: oracle &rarr; choose \\(x'\\) &rarr; per-byte map &rarr; scores &rarr; minimal edit &rarr; positive control."),
        dict(path="tools/xai_study/common/jutari_oracle.jl", find="function intervene_ram!(",
             role="substrate", desc="Writes one substituted byte into the live emulator."),
        dict(path="tools/xai_study/common/shared_testbed_impl.jl", find="function shared_gameplay_actions(",
             role="substrate", desc="The seeded action stream that defines both the explained state and the pool frames."),
        dict(path="jutari/src/env/StellaEnvironment.jl", find="function env_step!(",
             role="substrate", desc="One bit-exact frame; each counterfactual is a 15-frame loop over it."),
        dict(path="tools/xai_study/ground_truth/oracle_intervene.jl", find="function run_intervention(",
             role="substrate", desc="The ground truth the map is scored against."),
    ],

    reading=r"""
<p>The bars show the per-byte counterfactual effect next to the oracle&rsquo;s true effect. Expect
moderate agreement, stronger on content than on position. The two do not test the same edit: the
oracle uses fixed pokes (<code>base+17</code>, zero), while the counterfactual copies a value from a
real frame. When a byte has the same value in \(x'\) as in \(x\), the counterfactual cannot see it,
however important it is. On some states no candidate byte varies, the map is flat, and the run marks
the counterfactual invalid rather than inventing one. When an edit is found, it is usually small: one
or two bytes suffice.</p>""",

    pitfalls=[
        "<b>Off-manifold edits.</b> Unconstrained optimisation finds inputs no real system produces (Laugel et al.). Copying values from real frames avoids that per feature, but a mix of features from different frames can still be unreachable.",
        "<b>The answer depends on the alternative.</b> A counterfactual only reveals features that differ between \\(x\\) and \\(x'\\). A different reference frame gives a different explanation.",
        "<b>Many valid counterfactuals.</b> Usually several small edits flip the output. Greedy search returns one of them, not necessarily the smallest, and it can miss sets whose bytes only matter together.",
        "<b>Valid is not causal in the world.</b> In a neural net a valid counterfactual describes the <i>model</i>, not reality. On the VCS model and reality coincide, which is exactly why we can check it.",
    ],

    try_it="julia --project=jutari tools/xai_study/phaseB_attribution/counterfactual.jl --games core",
    try_note="Writes <code>counterfactual_&lt;game&gt;_{content,ball_pixel}.json/.npz</code> and "
             "<code>counterfactual_core_summary.json</code> into "
             "<code>tools/xai_study/phaseB_attribution/out/</code>. The greedy search costs a re-run per "
             "candidate byte per step; expect minutes per game on one CPU core. <code>--tau</code> sets the "
             "validity threshold.",

    next=["occlusion", "perturbation", "activation_patching", "kernelshap"],
)
