# Beginner's guide — extremal / meaningful perturbation.
# Rendered by build_pages.build_guide_section(); schema in docs/guides/__init__.py.

GUIDE = dict(
    key="perturbation",
    level="Perturbation family · optimise the mask",

    wiki=[
        ("Mathematical optimization", "https://en.wikipedia.org/wiki/Mathematical_optimization"),
        ("Greedy algorithm", "https://en.wikipedia.org/wiki/Greedy_algorithm"),
        ("Regularization (mathematics)", "https://en.wikipedia.org/wiki/Regularization_(mathematics)"),
        ("Jaccard index (intersection over union)", "https://en.wikipedia.org/wiki/Jaccard_index"),
    ],
    papers=[
        ("Fong &amp; Vedaldi (2017). <i>Interpretable Explanations of Black Boxes by Meaningful "
         "Perturbation.</i> ICCV.",
         "https://arxiv.org/abs/1704.03296"),
        ("Fong, Patrick &amp; Vedaldi (2019). <i>Understanding Deep Networks via Extremal Perturbations "
         "and Smooth Masks.</i> ICCV.",
         "https://arxiv.org/abs/1910.08485"),
        ("Chang, Creager, Goldenberg &amp; Duvenaud (2019). <i>Explaining Image Classifiers by "
         "Counterfactual Generation</i> (masks with blur/grey create artefacts the classifier reacts to). ICLR.",
         "https://arxiv.org/abs/1807.08024"),
    ],

    idea=r"""
<p>Occlusion slides a fixed grey square and reports a number for every position. That raises two
questions. Why a square? Why that size? <b>Extremal perturbation</b> turns the question around. It
asks: <i>what is the smallest region I must remove to destroy the prediction?</i> The answer is a
<b>mask</b>, a map that says for each input whether it is removed. The mask is found by
<b>optimisation</b>, not by sliding.</p>
<p>The mask is the explanation. A good mask is <b>small</b> (it points at few inputs) and
<b>extremal</b> (removing those inputs changes the output as much as possible). Fong and Vedaldi
called the 2017 version &ldquo;meaningful perturbation&rdquo;, because the removal should look like
a plausible change (a blur, not a random noise pattern). The 2019 version fixes the mask
<b>area</b> in advance, say 10&nbsp;% of the image, and finds the best mask of exactly that size.
This removes a hard-to-tune penalty weight. It is still a black-box method: it only needs to run the
model on masked inputs.</p>""",

    nn=r"""
<p>Let \(m \in [0,1]^n\) be a <b>soft mask</b> (\(m_i = 1\): input \(i\) removed, \(0\): kept) and
\(\Phi(x; m)\) the perturbed input, e.g. \(x\) blended with a blurred copy where \(m\) is high. The
2017 &ldquo;deletion game&rdquo; solves</p>
\[ m^\star = \arg\min_{m}\; \lambda\,\|m\|_1 \;+\; f\big(\Phi(x; m)\big) , \]
<p>trading mask size (the \(\ell_1\) <b>regulariser</b> \(\|m\|_1\), weighted by \(\lambda\))
against how low the class score goes. The 2019 &ldquo;extremal&rdquo; version replaces the penalty
by a hard <b>area constraint</b> \(\|m\|_1 = a\cdot n\). Both are solved by gradient descent on
\(m\), with smoothness tricks so the mask does not become a noisy pattern that fools the network
(an <i>adversarial</i> mask).</p>
<p><b>Toy example.</b> \(f(x) = 4x_1x_2 + x_3\) at \(x = (1,1,1,1)\) (\(x_4\) unused), removal sets
an input to 0, and the budget is 2 inputs. Removing one input at a time changes \(f\) by 4, 4, 1, 0.
A greedy search takes \(x_1\) first (change 4). Next it tries each remaining input <i>together
with</i> \(x_1\): adding \(x_2\) still gives 4 (the product is already dead), adding \(x_3\) gives 5.
So the mask is \(\{x_1, x_3\}\), which destroys the whole output with two inputs. Note what happened
to \(x_2\): it is just as important as \(x_1\), but once \(x_1\) is removed it adds nothing, so it is
<b>not in the mask</b>. A minimal mask names <i>a</i> sufficient set, not <i>all</i> important
inputs.</p>""",

    recipe=[
        r"Choose the output \(y\), a removal operator (blur, grey, zero) and an area budget \(a\) (fraction of inputs that may be removed).",
        r"Start from an empty mask; record the intact output \(y(x)\).",
        r"Search for the mask of at most \(\lceil a\cdot n\rceil\) inputs that maximises \(|y(x) - y(\text{masked } x)|\): gradient descent on a soft mask for a network, greedy forward selection on a black box.",
        r"Stop when the budget is used up, or when adding one more input no longer increases the change.",
        r"Report the mask (the selected set), and score it against the true causal set with intersection-over-union (IoU)."
    ],

    atari_map=[
        (r"The network \(f\)", "The cartridge program on the bit-exact emulated <b>Atari 2600 VCS</b> (<code>jutari</code>)."),
        (r"The mask \(m\) over pixels", "A <b>set</b> of candidate causes (RAM bytes, one TIA colour register, the joystick) to remove together."),
        (r"The perturbation \(\Phi(x;m)\)", "<code>occlude!</code> every cause in the set (RAM/TIA &rarr; 0) on a <code>deepcopy</code> of the checkpoint, then run 15 frames of the real ROM."),
        (r"Gradient descent on a soft mask", "<b>Greedy forward selection</b> on a hard mask: the VCS output is discrete, so a mask gradient would be zero."),
        (r"Area constraint \(a\)", "At most \\(\\lceil 0.25\\cdot C\\rceil\\) of the \\(C\\) causes may be selected (<code>--area 0.25</code>)."),
        ("&ldquo;Is this mask right?&rdquo;", "IoU of the mask with the oracle's top-\\(k\\) true movers, plus the same correlation and deletion/insertion scores as every other method."),
    ],

    atari=r"""
<p><b>Why greedy, not gradients.</b> The original optimises a soft mask by gradient descent. On the
VCS that would fail for the same reason saliency fails on the position output: the machine is a
discrete index computer and \(\partial y/\partial m = 0\) almost everywhere. So the runner optimises
the mask <b>directly on the real emulator</b> with greedy forward selection, the discrete,
budget-bounded cousin of the original optimiser.</p>
<ol>
<li><b>Single-cause pass.</b> For every cause \(i\), <code>masked_abs_dy</code> removes just that
cause and re-runs: \(|y(\text{occlude}\{i\}) - y(\text{intact})|\). This pass is literally occlusion.</li>
<li><b>Greedy growth.</b> Start with \(M = \varnothing\). In each step try every unselected cause
\(i\): <code>deepcopy</code> the checkpoint, <code>occlude!</code> all of \(M\cup\{i\}\), run 15
<code>env_step!</code> frames, read \(y\). Add the cause with the largest change.</li>
<li><b>Stop</b> when \(|M| = \lceil 0.25\,C\rceil\), or when the best addition does not increase the
change. If no single removal moves \(y\) at all, the mask stays empty: an honest null.</li>
<li><b>Per-cause scores.</b> Selected causes get the larger of their entry value and single-cause score, plus a constant lift (the largest
single-cause score), so the mask ranks first and its internal order is kept. Unselected causes keep
their single-cause occlusion score.</li>
</ol>
<p><b>How is this different from the oracle?</b> The oracle probes each cause <i>alone</i>, with two
directions (+17 and &rarr;0). Extremal perturbation probes <i>sets</i> of causes, and only in the
&ldquo;remove&rdquo; direction. It can therefore see joint effects the oracle's per-cause map does not
contain. But it deliberately reports a <i>minimal</i> set, so a cause that is redundant given the
mask (like \(x_2\) in the toy) is left out, even if the oracle ranks it high. For the IoU, causes are
reduced to the cell they actually touch, because <code>ram[i]:set</code> and
<code>ram[i]:occlude</code> both zero the same byte when masked.</p>
<p><b>Cost.</b> At most \(\lceil aC\rceil\cdot C\) greedy re-runs plus \(C\) single-cause re-runs and
one intact run; every one is a full 15-frame run of the real ROM from the checkpoint.</p>""",

    code=[
        dict(path="tools/xai_study/phaseB_attribution/perturbation.jl", find="function extremal_mask(",
             role="xai", desc="The optimiser: single-cause pass, then greedy forward selection under the area budget, with the no-gain stop rule."),
        dict(path="tools/xai_study/phaseB_attribution/perturbation.jl", find="function masked_abs_dy(",
             role="xai", desc="Value of one mask: <code>deepcopy</code>, <code>occlude!</code> every masked cause, run the horizon, |&Delta;y|."),
        dict(path="tools/xai_study/phaseB_attribution/perturbation.jl", find="function extremal_attr_per_cause(",
             role="xai", desc="Maps the mask onto the per-cause axis (selected causes lifted above all others)."),
        dict(path="tools/xai_study/phaseB_attribution/perturbation.jl", find="function mask_iou(",
             role="xai", desc="The headline score: IoU of the mask with the oracle's top-k true movers, on the touched-cell axis."),
        dict(path="tools/xai_study/phaseB_attribution/perturbation.jl", find="max_cells = max(1, ceil(Int, area * C))",
             role="xai", desc="The area budget \\(\\lceil a\\cdot C\\rceil\\); <code>area = 0.25</code> by default."),
        dict(path="tools/xai_study/phaseB_attribution/ig_baseline_sweep.jl", find="function occlude!(",
             role="xai", desc="The removal operator shared with occlusion, RISE, LIME and KernelSHAP."),
        dict(path="tools/xai_study/common/jutari_oracle.jl", find="function intervene_ram!(",
             role="substrate", desc="The poke into the emulated RAM that every mask entry performs."),
        dict(path="jutari/src/env/StellaEnvironment.jl", find="function env_step!(",
             role="substrate", desc="One frame of the real machine; each mask evaluation runs 15 of them."),
        dict(path="tools/xai_study/ground_truth/oracle_intervene.jl", find="function run_intervention(",
             role="substrate", desc="The oracle's single-cause re-run, the ground truth the mask is compared with."),
        dict(path="tools/xai_study/common/shared_testbed_impl.jl", find="function build_shared_testbed(",
             role="substrate", desc="The shared gameplay checkpoint and the position output."),
    ],

    reading=r"""
<p>The figure shows the per-cause scores next to the oracle, and the record lists the selected mask.
The selected causes sit on top because of the lift. Check two things. First, are the mask's causes
among the oracle's tallest bars? That is what the IoU measures. Second, are there tall oracle bars
<i>outside</i> the mask? Usually there are: the mask is a small sufficient set, not the full list of
important causes, so expect a partial overlap rather than a perfect match. The method works on the
position output too, since it never uses a gradient.</p>""",

    pitfalls=[
        "<b>Minimal is not complete.</b> A mask names one small sufficient set. Redundant causes (backups) are left out even when each is important on its own.",
        "<b>Greedy is not optimal.</b> Forward selection can be trapped: two causes that only matter together are never picked if neither helps alone. Ties are broken by list order, so the choice between equally good causes is arbitrary.",
        "<b>Artefact masks.</b> On images, an unconstrained mask can become an adversarial pattern; the smoothness terms and blur perturbation of Fong et al. exist to stop this. On the VCS every mask is a valid machine state, but zeroing several bytes at once can push the game into states it never reaches in play.",
        "<b>The budget shapes the answer.</b> A different area fraction gives a different mask. The 2019 paper reports masks at several areas for this reason; we use one default.",
    ],

    try_it="julia --project=jutari tools/xai_study/phaseB_attribution/perturbation.jl --games core",
    try_note="Writes <code>perturbation_&lt;game&gt;_{content,position}.json/.npz</code> into "
             "<code>tools/xai_study/phaseB_attribution/out/</code>, including the mask, its growth curve and "
             "the IoU. Change the budget with <code>--area 0.1</code>.",

    next=["occlusion", "counterfactual", "rise", "kernelshap"],
)
