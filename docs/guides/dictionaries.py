# Beginner's guide — NMF / PCA dictionaries.
# Rendered by build_pages.build_guide_section(); schema in docs/guides/__init__.py.

GUIDE = dict(
    key="dictionaries",
    level="Mechanistic family · classical dictionary learning",

    wiki=[
        ("Principal component analysis", "https://en.wikipedia.org/wiki/Principal_component_analysis"),
        ("Non-negative matrix factorization", "https://en.wikipedia.org/wiki/Non-negative_matrix_factorization"),
        ("Sparse dictionary learning", "https://en.wikipedia.org/wiki/Sparse_dictionary_learning"),
        ("Singular value decomposition", "https://en.wikipedia.org/wiki/Singular_value_decomposition"),
    ],
    papers=[
        ("Lee &amp; Seung (1999). <i>Learning the parts of objects by non-negative matrix "
         "factorization.</i> Nature 401.",
         "https://doi.org/10.1038/44565"),
        ("Olshausen &amp; Field (1996). <i>Emergence of simple-cell receptive field properties by "
         "learning a sparse code for natural images.</i> Nature 381.",
         "https://doi.org/10.1038/381607a0"),
        ("Locatello et al. (2019). <i>Challenging Common Assumptions in the Unsupervised Learning of "
         "Disentangled Representations.</i> ICML. (critique: unsupervised factors are not identifiable "
         "without extra assumptions)",
         "https://arxiv.org/abs/1811.12359"),
    ],

    idea=r"""
<p>Suppose you record a network's activations for thousands of inputs and stack them into one big
table: one row per input, one column per neuron. A <b>dictionary</b> is a short list of
&ldquo;building-block&rdquo; patterns, called <b>atoms</b> or <b>components</b>, such that every row can
be written as a weighted sum of a few atoms. If the atoms line up with real concepts, you have found the
concepts <b>without any labels</b>. This is <b>dictionary learning</b>, and it is much older than deep
learning: neuroscientists used it in the 1990s to explain how the visual cortex codes images.</p>
<p>Two classical recipes are fast and need no training loop. <b>PCA</b> (principal component analysis)
finds the directions along which the data varies most; they are orthogonal and may have negative
entries. <b>NMF</b> (non-negative matrix factorisation) only allows non-negative atoms and weights, so
everything is built by <i>adding</i> parts, never by cancelling them. That often gives &ldquo;parts&rdquo;
a human can recognise. The sparse autoencoder is the modern, learned, sparse member of the same family.
Running the classical ones first shows how much the extra machinery buys.</p>""",

    nn=r"""
<p>Put the data in a matrix \(X\) with \(T\) rows (samples) and \(n\) columns (neurons). Both methods
approximate it by a product of a thin matrix of <b>activations</b> \(A\) (\(T\times r\)) and a short
matrix of <b>atoms</b> \(H\) (\(r\times n\)), with \(r \ll n\):</p>
\[ X \;\approx\; \mu + A H \quad(\text{PCA, } H H^\top = I), \qquad
   X \;\approx\; A H \quad(\text{NMF, } A \ge 0,\ H \ge 0). \]
<p>PCA takes \(\mu\) = the column means and \(H\) = the top \(r\) right singular vectors of the centred
data. NMF minimises \(\|X - AH\|^2\) with the Lee&ndash;Seung multiplicative updates, which keep every
entry non-negative. An SAE is the same picture with a learned, non-linear, sparse encoder for \(A\).</p>
<p><b>Toy example.</b> Three RAM-like cells \((a, b, c)\). Cells \(a, b\) move together (a ball's x and
a copy of it); cell \(c\) is a score. Four frames: \((4,2,0)\), \((2,1,0)\), \((0,0,5)\),
\((4,2,5)\). NMF with \(r=2\) finds the exact answer \(H_1=(2,1,0)\) (&ldquo;ball&rdquo;),
\(H_2=(0,0,1)\) (&ldquo;score&rdquo;), with weights \((2,0), (1,0), (0,5), (2,5)\). Each atom is one
variable. PCA on the same data gives a first component of roughly \(0.92\cdot\)score \(-\,0.38\cdot\)ball:
both variables vary a lot, so the direction of largest variance <b>mixes</b> them, with a sign. Same data,
same rank; only the constraint differs.</p>""",

    recipe=[
        "Record many activation vectors and stack them into a samples &times; units matrix.",
        "Choose the number of atoms \\(r\\) (try several).",
        "Fit PCA (singular value decomposition of the centred matrix) and NMF (multiplicative updates on the raw, non-negative matrix).",
        "Inspect each atom's loadings: which units does it weigh, and do they belong to one concept?",
        "Count the share of atoms that match a known concept (the matched fraction).",
        "Test causally: remove one atom from the reconstruction, put the state back into the model, and measure the change in output.",
    ],

    atari_map=[
        ("Activation table \\(X\\)", "The RAM tape: one row per video frame, 128 columns (one per RAM byte)."),
        ("Atom / component", "A pattern over the 128 bytes. A clean atom loads on the bytes of <b>one</b> game variable."),
        ("Concept label", "The verified concept&harr;cell map (for Pong: scores, paddle and ball positions), grouped into <b>families</b> such as &ldquo;ball&rdquo; or &ldquo;score&rdquo;."),
        ("Structural signals (no network analogue)", "Known machine-level signals: the frame clock, how many bytes change per frame, and a fitted periodic phase."),
        ("Removing an atom", "Zero its weight, rebuild the 128 bytes, round them, write them into the real machine, replay 15 frames."),
        ("Ground truth", "The bit-exact re-run itself: an atom is causally used only if removing it changes the screen."),
    ],

    atari=r"""
<p>Our runner <code>dictionaries.jl</code> applies PCA, NMF and a small sparse autoencoder to the
<b>same</b> RAM tape and scores all three with the <b>same</b> tests. That makes it a fair three-way
comparison.</p>
<ul>
<li><b>Data.</b> The RAM after each frame along the shared gameplay stream (90 seeded random-action
frames, the 15-frame horizon, then NOOP; 150 frames for Pong). PCA uses the centred bytes. NMF uses the
raw bytes, which are naturally non-negative (0&ndash;255). The SAE uses z-scored bytes.</li>
<li><b>Models.</b> PCA is one singular value decomposition. NMF runs 300 Lee&ndash;Seung updates. The
SAE has as many hidden units as the rank, L1 weight 0.05, and 3000 epochs of gradient descent. All three
are run at ranks 8, 4 and 12; rank 8 is the headline.</li>
<li><b>Matching.</b> An atom counts as matched to a game variable if at least a quarter of its squared
loading mass sits on candidate cells and at least half of that mass sits on <b>one</b> concept family.
Otherwise it can still be matched to a structural signal if its activation over time correlates with the
frame clock, the per-frame change count, or the periodic phase at \(|r|\ge 0.5\). The <b>matched
fraction</b> is matched atoms over all atoms.</li>
<li><b>Causal use.</b> The RAM at \(f^*\) is projected onto the dictionary. For each atom we zero its
weight, decode, round and clamp to bytes, write all 128 bytes into a copy of the machine, and play the
15-frame tail. The effect is the number of screen pixels that differ from the same write <i>without</i>
the ablation, so the lossy reconstruction does not count as an effect. The same ablation is also played
under a different, held-out action tail.</li>
<li><b>Positive control.</b> Zeroing <i>all</i> atoms must break the screen. Otherwise the causal test
would be blind.</li>
</ul>
<p>The triad: <b>F</b> is the Spearman correlation between how strongly each atom matches a variable
and its causal effect (&ldquo;are the named atoms the used ones?&rdquo;). <b>S</b> correlates the effect
with the held-out-tail effect. <b>M</b> is one minus the share of matched atoms whose removal does
nothing: naming a causally dead atom is an over-claim.</p>
<p>Why NMF tends to win on the VCS: RAM is literally a list of separate bytes that the program adds to
and overwrites. That is an additive, non-negative &ldquo;parts&rdquo; structure, which is what NMF
assumes. PCA's orthogonality has no reason to hold, so its components blend variables that happen to vary
together.</p>""",

    code=[
        dict(path="tools/xai_study/phaseC_mechanistic/dictionaries.jl", find="function run_pca(tape::AbstractMatrix, rank::Integer)",
             role="xai", desc="PCA: SVD of the centred RAM tape; rows of \\(V^\\top\\) are the atoms."),
        dict(path="tools/xai_study/phaseC_mechanistic/dictionaries.jl", find="function _nmf(X::AbstractMatrix{<:Real}, r::Integer",
             role="xai", desc="NMF: Lee&ndash;Seung multiplicative updates that keep both factors non-negative."),
        dict(path="tools/xai_study/phaseC_mechanistic/dictionaries.jl", find="function run_sae(tape::AbstractMatrix, hidden::Integer",
             role="xai", desc="The small L1 sparse autoencoder used as the third, learned dictionary."),
        dict(path="tools/xai_study/phaseC_mechanistic/dictionaries.jl", find="function match_components(m::DictModel",
             role="xai", desc="Matched fraction: loading purity on one concept family, or correlation with a structural signal."),
        dict(path="tools/xai_study/phaseC_mechanistic/dictionaries.jl", find="function causal_use(m::DictModel, checkpoint",
             role="xai", desc="Ablate one atom, decode, write the state, replay, count pixels against the reconstruction control."),
        dict(path="tools/xai_study/phaseC_mechanistic/dictionaries.jl", find="function rerun_with_ram(checkpoint, ram",
             role="xai", desc="Writes a full 128-byte state into a copy of the machine and replays the tail."),
        dict(path="tools/xai_study/phaseC_mechanistic/dictionaries.jl", find="function compute_game(game::AbstractString",
             role="xai", desc="One game: record the tape, then PCA / NMF / SAE at each rank, matching, causal use, triad."),
        dict(path="tools/xai_study/common/jutari_oracle.jl", find="function intervene_ram!(env::StellaEnvironment",
             role="substrate", desc="The single-byte write into emulated RIOT RAM that the decoded state is built from."),
        dict(path="jutari/src/env/StellaEnvironment.jl", find="function env_reset!(env::StellaEnvironment",
             role="substrate", desc="Boots the cartridge with the xitari-parity reset before any recording."),
        dict(path="jutari/src/env/StellaEnvironment.jl", find="function env_step!(env::StellaEnvironment",
             role="substrate", desc="One bit-exact video frame: the step behind every recorded row and every re-run."),
    ],

    reading=r"""
<p>The figure compares the matched fraction of PCA, NMF and the SAE. Expect a partial result: NMF
recovers more cleanly named atoms than PCA, because its atoms are additive parts, while PCA folds
several variables into one signed direction. Then look at causal use. Several matched atoms do little
or nothing when removed, while some unmatched ones matter. So a good matched fraction is evidence that a
variable is <b>present</b> in the data, not that the program <b>uses</b> that atom. This is the same
gap the SAE page shows.</p>""",

    pitfalls=[
        "<b>Variance is not importance.</b> PCA ranks directions by how much they vary. A busy frame counter can dominate the first component while a rarely changing but crucial byte is buried.",
        "<b>The rank is a choice.</b> Too few atoms merge variables; too many split one variable over several atoms. We sweep ranks 4, 8 and 12 for this reason.",
        "<b>Unsupervised factors are not unique.</b> Locatello et al. proved that, without extra assumptions, many different factorisations explain the data equally well. A matching atom may be luck of the initialisation.",
        "<b>Rounding to bytes.</b> On the VCS a decoded state must be rounded to integers before it can be written. Small atoms can vanish in the rounding, which reads as &ldquo;no causal effect&rdquo;.",
    ],

    try_it="julia --project=jutari tools/xai_study/phaseC_mechanistic/dictionaries.jl --games core",
    try_note="Writes <code>dictionaries_&lt;game&gt;.{json,npz}</code> and "
             "<code>dictionaries_core_summary.json</code> into <code>tools/xai_study/phaseC_mechanistic/out/</code>. "
             "Use <code>--rank 8</code> for the headline rank only.",

    next=["sae", "A7_dimred", "linear_probing", "das"],
)
