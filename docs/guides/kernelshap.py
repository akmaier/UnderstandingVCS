# Beginner's guide — KernelSHAP / Shapley values.
# Rendered by build_pages.build_guide_section(); schema in docs/guides/__init__.py.

GUIDE = dict(
    key="kernelshap",
    level="Perturbation family · fair shares from game theory",

    wiki=[
        ("Shapley value", "https://en.wikipedia.org/wiki/Shapley_value"),
        ("Cooperative game theory", "https://en.wikipedia.org/wiki/Cooperative_game_theory"),
        ("Weighted least squares", "https://en.wikipedia.org/wiki/Weighted_least_squares"),
        ("Lagrange multiplier", "https://en.wikipedia.org/wiki/Lagrange_multiplier"),
        ("Binomial coefficient", "https://en.wikipedia.org/wiki/Binomial_coefficient"),
    ],
    papers=[
        ("Lundberg &amp; Lee (2017). <i>A Unified Approach to Interpreting Model Predictions.</i> NeurIPS.",
         "https://arxiv.org/abs/1705.07874"),
        ("Covert &amp; Lee (2021). <i>Improving KernelSHAP: Practical Shapley Value Estimation via Linear "
         "Regression.</i> AISTATS.",
         "https://arxiv.org/abs/2012.01536"),
        ("Kumar, Venkatasubramanian, Scheidegger &amp; Friedler (2020). <i>Problems with Shapley-value-based "
         "explanations as feature importance measures.</i> ICML.",
         "https://arxiv.org/abs/2002.11097"),
    ],

    idea=r"""
<p>Three friends share a taxi and the fare is 30 euros. How should they split it fairly, if each lives
at a different distance? Game theory has a famous answer: the <b>Shapley value</b>. Imagine the
friends joining the ride one after another, in every possible order. Each time someone joins, note
how much the fare goes up. A person's fair share is the <b>average increase they cause, over all
orders</b>. Lloyd Shapley proved this is the <i>only</i> split that obeys a few fairness rules, for
example: the shares add up to the total, and two people who always contribute the same pay the same.</p>
<p>For a neural network, the &ldquo;players&rdquo; are the input features and the &ldquo;total&rdquo; is
the output minus a baseline output. A feature's Shapley value is its fair share of the prediction.
Unlike occlusion, which removes one feature at a time, Shapley values look at every
<b>coalition</b> (subset) of features, so backups and interactions are shared out correctly. The
catch: \(n\) features have \(2^n\) coalitions. <b>KernelSHAP</b> estimates the values from a random
sample of coalitions by solving one cleverly weighted linear regression.</p>""",

    nn=r"""
<p>Let \(N\) be the set of \(n\) features and \(v(S)\) the model output when only the features in
\(S\) are present (the others replaced by a baseline). The Shapley value of feature \(i\) is</p>
\[ \phi_i \;=\; \sum_{S \subseteq N\setminus\{i\}} \frac{|S|!\,(n-|S|-1)!}{n!}\,\big[v(S\cup\{i\}) - v(S)\big] , \]
<p>the order-averaged marginal contribution. The values always satisfy <b>efficiency</b>:
\(\sum_i \phi_i = v(N) - v(\varnothing)\). Lundberg and Lee showed that the same \(\phi\) solves a
weighted least-squares problem over coalitions \(z\in\{0,1\}^n\):</p>
\[ \min_{\phi}\; \sum_{z} \pi(z)\Big(v(z) - v(\varnothing) - \textstyle\sum_i z_i\phi_i\Big)^2
\quad\text{s.t.}\;\; \sum_i\phi_i = v(N)-v(\varnothing),\qquad
\pi(z) = \frac{n-1}{\binom{n}{|z|}\,|z|\,(n-|z|)} . \]
<p>The <b>Shapley kernel</b> \(\pi\) puts most weight on very small and very large coalitions.
KernelSHAP samples coalitions instead of enumerating all \(2^n\), and solves this regression.</p>
<p><b>Toy example: exact Shapley values of a 3-player game.</b> \(f(x) = 4x_1x_2 + 2x_3\) at
\(x = (1,1,1)\), baseline 0. The coalition values are \(v(\varnothing)=0\), \(v(1)=v(2)=0\),
\(v(3)=2\), \(v(12)=4\), \(v(13)=v(23)=2\), \(v(123)=6\). Walk through the 6 orders and record
what each player adds:</p>
<table>
<tr><th>order</th><th>\(x_1\) adds</th><th>\(x_2\) adds</th><th>\(x_3\) adds</th></tr>
<tr><td>1,2,3</td><td>0</td><td>4</td><td>2</td></tr>
<tr><td>1,3,2</td><td>0</td><td>4</td><td>2</td></tr>
<tr><td>2,1,3</td><td>4</td><td>0</td><td>2</td></tr>
<tr><td>2,3,1</td><td>4</td><td>0</td><td>2</td></tr>
<tr><td>3,1,2</td><td>0</td><td>4</td><td>2</td></tr>
<tr><td>3,2,1</td><td>4</td><td>0</td><td>2</td></tr>
</table>
<p>Averages: \(\phi = (2, 2, 2)\), and \(2+2+2 = 6 = v(123)-v(\varnothing)\). The product term
\(4x_1x_2\) is split evenly between its two players. Occlusion on the same toy gives \((4, 4, 2)\),
which sums to 10: it counts the shared term <i>twice</i>. With \(n = 3\) the kernel gives every
coalition of size 1 or 2 the same weight \(\pi = 1/3\), and the constrained regression over those
6 coalitions returns exactly \((2,2,2)\).</p>""",

    recipe=[
        r"Choose the output \(y\), the features, and the &ldquo;absent&rdquo; baseline (a fixed value or a background dataset).",
        r"Evaluate the two endpoints: all features present, \(v(N)\), and all absent, \(v(\varnothing)\).",
        r"Sample coalitions \(z\): draw a size \(s\) with probability proportional to the kernel mass of that size, then a random subset of that size.",
        r"Run the model once per distinct coalition (absent features set to the baseline), recording \(v(z)\).",
        r"Solve the kernel-weighted least squares for \(\phi\) under the efficiency constraint \(\sum_i\phi_i = v(N)-v(\varnothing)\).",
        r"Check convergence: re-solve with fewer coalitions and see whether \(\phi\) has settled.",
    ],

    atari_map=[
        (r"The model \(f\)", "The cartridge program on the bit-exact emulated <b>Atari 2600 VCS</b> (<code>jutari</code>)."),
        ("The players (features)", "The <b>candidate RAM cells</b>, de-duplicated (8 to 23 per core game). TIA and joystick causes are not players."),
        ("Feature absent / background", "One fixed baseline: the cell is set to <b>0</b> with <code>intervene_ram!</code>. No background dataset is averaged over."),
        (r"Value \(v(z)\) of a coalition", "<code>deepcopy</code> the checkpoint, zero the absent cells, run 15 frames of the real ROM, read \\(y\\). No surrogate model of \\(f\\) is trained."),
        ("Sampling budget", "512 sampled coalitions plus the two endpoints; repeated draws are merged, so each distinct coalition costs one re-run."),
        ("&ldquo;Ground-truth importance&rdquo;", "The oracle's single-cause |&Delta;y| per cause, from real re-runs."),
    ],

    atari=r"""
<p><b>The value function is the real machine.</b> In textbook SHAP, \(v(S)\) is often approximated
(e.g. by averaging over a background dataset). Here <code>coalition_value</code> computes it exactly:
copy the checkpoint, zero every absent candidate cell with <code>intervene_ram!</code>, play the
recorded 15 actions with <code>env_step!</code>, read the output. The only approximations left are
the <b>coalition sampling</b> and the <b>additive form</b> of the explanation.</p>
<ol>
<li><b>Endpoints.</b> \(f_0 = v(\varnothing)\) (all candidate cells zeroed) and \(f_1 = v(N)\)
(intact) are each one real run.</li>
<li><b>Sampling.</b> <code>sample_coalitions</code> draws 512 coalitions: a size \(s\in\{1,\dots,C-1\}\)
with probability \(\propto (C-1)/(s(C-s))\), then a uniform random subset of that size. Identical
draws are merged by <code>dedup_coalitions</code>; their kernel weights are summed.</li>
<li><b>Solve.</b> <code>kernelshap_solve</code> solves the weighted least squares (tiny ridge
\(10^{-8}\)) and projects the answer onto \(\sum_i\phi_i = f_1 - f_0\) with a
<b>Lagrange multiplier</b>. The runner records the remaining completeness residual as a check.</li>
<li><b>To causes.</b> Each RAM cause gets \(|\phi|\) of its cell (the signed \(\phi\) are kept in the
record); TIA and joystick causes get 0.</li>
<li><b>Convergence.</b> The solve is repeated on the first \(N/8\), \(N/4\), \(N/2\) draws of the same
pool, so the record shows faithfulness versus compute at no extra re-runs.</li>
</ol>
<p><b>An implementation detail worth knowing.</b> The coalitions are <i>drawn</i> in proportion to the
kernel and then <i>also weighted</i> by the kernel \(\pi(s)\) in the solve. Textbook KernelSHAP does
one or the other, not both. Our estimate therefore leans even more on near-empty and near-full
coalitions (&ldquo;only this cell kept&rdquo;, &ldquo;only this cell zeroed&rdquo;) than exact Shapley
values would. Efficiency still holds exactly, by the constraint.</p>
<p><b>How is this different from the oracle?</b> The oracle measures each cause <i>alone</i> at the
real state, in two directions (+17 and &rarr;0). Shapley values average a cell's effect over many
coalitions of other cells being zeroed, and split shared effects fairly. So when two cells form an
AND-like pair, the oracle (like occlusion) gives each one the full effect, while SHAP gives each half.
Agreement in <i>ranking</i> is what we score.</p>""",

    code=[
        dict(path="tools/xai_study/phaseB_attribution/kernelshap.jl", find="function coalition_value(",
             role="xai", desc="The value function \\(v(z)\\): <code>deepcopy</code>, zero the absent cells, 15 &times; <code>env_step!</code>, read y."),
        dict(path="tools/xai_study/phaseB_attribution/kernelshap.jl", find="function shapley_kernel_weight(",
             role="xai", desc="The Shapley kernel \\(\\pi(s) = (C-1)/(\\binom{C}{s}s(C-s))\\); endpoints handled by the constraint."),
        dict(path="tools/xai_study/phaseB_attribution/kernelshap.jl", find="function sample_coalitions(",
             role="xai", desc="Draws coalition sizes from the kernel's size distribution, then a uniform subset."),
        dict(path="tools/xai_study/phaseB_attribution/kernelshap.jl", find="function dedup_coalitions(",
             role="xai", desc="Merges repeated draws (one re-run each) and sums their kernel weights."),
        dict(path="tools/xai_study/phaseB_attribution/kernelshap.jl", find="function kernelshap_solve(",
             role="xai", desc="Weighted least squares plus the Lagrange projection onto \\(\\sum\\phi = f_1-f_0\\)."),
        dict(path="tools/xai_study/phaseB_attribution/kernelshap.jl", find="function compute_one(",
             role="xai", desc="One experiment: oracle, endpoints, 512 coalitions, solve, scoring, convergence sweep."),
        dict(path="tools/xai_study/common/jutari_oracle.jl", find="function intervene_ram!(",
             role="substrate", desc="The poke that makes a cell &ldquo;absent&rdquo; in the emulated RAM."),
        dict(path="jutari/src/env/StellaEnvironment.jl", find="function env_step!(",
             role="substrate", desc="One frame of the real machine; each coalition costs 15 of them."),
        dict(path="tools/xai_study/ground_truth/oracle_intervene.jl", find="function run_intervention(",
             role="substrate", desc="The oracle's single-cause re-run, the ground truth the Shapley values are compared with."),
        dict(path="tools/xai_study/common/shared_testbed_impl.jl", find="function build_shared_testbed(",
             role="substrate", desc="The shared gameplay checkpoint and the position output."),
    ],

    reading=r"""
<p>Bars are \(|\phi|\) per cause next to the oracle's true effect. TIA and joystick causes are zero
for SHAP by construction, and both causes of a byte share a height. Where the oracle shows two cells
with large, similar effects, SHAP may show them at about half height each: that is fair sharing, not
an error. Check the completeness residual (should be near zero) and the convergence block: if
agreement still changes between \(N/2\) and \(N\), more coalitions would help. Expect SHAP to recover
much of the true structure on both content and position outputs.</p>""",

    pitfalls=[
        "<b>&ldquo;Absent&rdquo; must be defined.</b> Shapley values depend completely on the baseline. Zero is one choice; a background dataset of real states would give different values.",
        "<b>Fair share is not causal effect.</b> Kumar et al. show Shapley values can credit features in ways that do not match the question you care about. Splitting an interaction in half is a convention, not a fact about the game.",
        "<b>Sampling error.</b> With many features, a few hundred coalitions cover a tiny fraction of the \\(2^n\\). Covert &amp; Lee analyse the estimator's variance; we report a convergence sweep instead of assuming it has converged.",
        "<b>Our weighting differs from the textbook.</b> Drawing and weighting by the kernel both emphasise extreme coalitions, so our values are a Shapley-like estimate, not the exact Shapley values even at infinite budget.",
        "<b>Off-distribution coalitions.</b> Zeroing many cells creates machine states the game never reaches. The VCS runs them faithfully, but they may say little about normal play.",
    ],

    try_it="julia --project=jutari tools/xai_study/phaseB_attribution/kernelshap.jl --games core",
    try_note="Writes <code>kernelshap_&lt;game&gt;_{content,ball_pixel}.json/.npz</code> into "
             "<code>tools/xai_study/phaseB_attribution/out/</code>, with the signed \\(\\phi\\), the completeness "
             "residual and the convergence sweep. Change the budget with <code>--n-coalitions</code>.",

    next=["lime", "rise", "occlusion", "ig_baseline_sweep"],
)
