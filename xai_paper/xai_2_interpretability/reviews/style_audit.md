# P2 writing-clarity audit (2026-06-24) — overview

Whole-paper audit vs the author standard (simple, precise, short, non-native-readable, not poetic). 227 findings.

# Style Audit — Clarity Overview

## 1. Verdict

**Widespread and systemic.** 227 flagged spans across 11 sections, dominated by *poetic ornament* (125, 55%) and *non-native-reader friction* (47, 21%); 42 sentences are simply too long. The problem is not isolated typos — it is a consistent house style of aphoristic refrains ("wiring, not the meaning"; "the X, not the Y"; "buys / lands / carries"; halving rhetoric the author already banned) that trades precision for cadence. **Fix the refrains and the long sentences and ~80% of findings resolve.**

## 2. Counts

### By problem type
| Type | Count | Share |
|---|---|---|
| poetic | 125 | 55% |
| hard_for_nonnative | 47 | 21% |
| too_long | 42 | 19% |
| vague | 11 | 5% |
| meaning_inverting | 2 | <1% |
| **Total** | **227** | |

### By section (worst first)
| Section | Total | poetic | nonnative | too_long | vague | inverting |
|---|---|---|---|---|---|---|
| 07_results_compare.tex | 33 | 19 | 8 | 6 | 1 | 0 |
| 08_discussion.tex | 42 | 24 | 9 | 12 | 3 | 1 |
| 06_results_C.tex | 26 | 16 | 6 | 4 | 3 | 0 |
| 03_methods.tex | 24 | 13 | 9 | 2 | 1 | 0 |
| 04_results_A.tex | 23 | 16 | 4 | 1 | 1 | 1 |
| 01_intro.tex | 21 | 11 | 4 | 4 | 2 | 0 |
| 05_results_B.tex | 16 | 11 | 5 | 1 | 0 | 0 |
| 02_related.tex | 9 | 1 | 3 | 5 | 0 | 0 |
| 09_endmatter.tex | 2 | 0 | 0 | 2 | 0 | 0 |

**Worst offenders: `08_discussion`, `07_results_compare`, and `06_results_C`** carry 101 of 227 findings (44%) and most of the high-severity ones. The Discussion alone has 12 over-long sentences. The two `meaning_inverting` bugs (in `04_results_A` and `08_discussion`) are the only correctness-level issues and should be fixed first.

## 3. Top ~15 Worst Offenders (high-severity)

1. **04_results_A** — *"No longer lag or stricter threshold repairs that..."* — **meaning-inverting**: "No longer lag" parses as a time adverb, reversing the intent → **"Neither a longer lag nor a stricter threshold fixes this. Precedence and causation differ here, and only the intervention oracle separates them."**

2. **01_intro** — *"Securing faithfulness, then, is not the smaller half of the problem but the wrong half."* — banned halving ornament, says nothing → **"Securing faithfulness is necessary, but it does not solve the main problem."**

3. **08_discussion** — *"We have not solved it. We have made it a number, and a number is where understanding begins."* — closing aphorism over substance → **"We have not solved this problem. We have defined and measured it, which is the necessary first step."**

4. **06_results_C** — *"...separates into a causal core that is faithful by construction (... at $1.0$; phase mean ... $0.63$ ...) and a recovery layer that is precise-but-partial ... or present-but-inert ..."* — 55 words, two nested asides, coined labels → split into two sentences, one per group (see full list).

5. **08_discussion** — *"...the gap appearing where tuning cannot reach, running in both directions at once, and surviving the sampler-on test that makes the gradient methods faithful on the games where a moving tracked sprite gives the gradient something to flow to."* — 48 words, three-part tail → **"It rests on three facts: the gap appears where tuning cannot reach it; it runs in both directions; and it survives the sampler-on test that makes the gradient methods faithful on games with a moving tracked sprite."**

6. **01_intro** — *"the residual between a faithful attribution and an account of the computation is not philosophy but a number"* — rhetorical antithesis → **"On this machine we can measure the gap between a faithful attribution and an account of the computation as a number."**

7. **07_results_compare** — *"The leaderboard is the way-station; the gap is the destination."* — travel metaphor the author rejects → **"The leaderboard is an intermediate result; the semantic gap is the main result."**

8. **04_results_A** — *"Both see the machine breathe without seeing it compute."* — metaphor over precision → **"Both methods detect the machine's activity but not its computation."**

9. **05_results_B** — *"...the entire gradient family inherits this boundary, and it cuts the leaderboard cleanly in two."* / *"Returning the empty map ... not a partial success but a structural blind spot, exactly where the discrete game logic lives."* — stacked metaphors → **"...so the whole gradient family hits this limit, splitting the leaderboard into two groups."** / **"...is not partial success. It is a structural failure, exactly where the discrete game logic is."**

10. **06_results_C** — *"The result splits cleanly... the causal methods are exact, yet exactness buys the wiring, not the meaning."* — "buys" + the core refrain → **"The result splits cleanly. The causal methods are exact, but exactness recovers only the wiring, not the meaning."**

11. **08_discussion** — *"When the gradient is made honest it names the cause and names no meaning, so the meaning was never in the wiring..."* — refrain chain → **"Even when corrected, the gradient identifies the cause but no meaning. So the meaning was never in the wiring for a method to recover."**

12. **02_related** — *"Read against Marr's levels ... and Lazebnik's 'fix-the-radio' parable ..., the point that drives this paper holds: a faithful description ... is not yet an account of the algorithm ..."* — inverted opener, 40+ words → **"Marr's levels and Lazebnik's parable support this paper's main point. A faithful description of a system's wiring is not yet an account of the algorithm it runs. On a known machine, we measure that gap."**

13. **08_discussion** — *"The reference against which a cause becomes a meaning lives in the system's documented behavior, where no perturbation of the chip can find it."* — boundary refrain as metaphor → **"A cause becomes a meaning only against an external reference: the system's documented behavior. No perturbation of the chip can reach that reference."**

14. **07_results_compare** — *"We have now run three traditions on one machine, and the result that organizes this section is not the ranking ... but the boundary they all share: every method ... returns the system's causal wiring and not its meaning."* — 40 words, antithesis + refrain → **"This section is organized not by the ranking among the methods, but by a boundary they share. Every method, faithful or not, returns the system's causal wiring, not its meaning."**

15. **03_methods** — *"Separating F, which the wiring satisfies, from S and M ... is what lets us name, later, the residual that faithfulness alone leaves behind."* — refrain + ornament → **"F is satisfied by the wiring; S and M must be satisfied by the computation. Separating them lets us measure, later, what faithfulness alone leaves out."**

**Honorable mentions (high-severity, same pattern):** 04_results_A "The map of which units matter is right, the model of how they combine absent" (omitted verb) → *"The ranking of which units matter is correct, but the model of how they combine is missing."*; 06_results_C "Linear probing closes the triptych with the classic trap" → *"Linear probing shows a third version of the same gap, the well-known probing trap."*; 07_results_compare "It buys no understanding." → *"But this does not give understanding."*

## 4. Full List by Section (quote → rewrite)

### 01_intro.tex
- "Interpretability now carries real weight, because neural networks drive cars, read scans, and file loans..." → "Interpretability now matters in practice. Neural networks drive cars, read scans, and approve loans. Before we trust such a decision, we ask why."
- "a saliency map ... may look convincing and shifts with the action, but this does not mean..." → "A saliency map can look convincing and can change with the action. That does not mean the explanation is correct."
- "Jonas and Kording made this vivid by turning the neuroscientist's toolkit on a microprocessor..." → "Jonas and Kording showed this clearly. They applied standard neuroscience methods to a microprocessor in which every transistor is known."
- "a method that finds structure cannot be told apart from one that finds the right structure" → "We cannot distinguish a method that finds some structure from one that finds the correct structure."
- "Faithfulness has stood in for understanding without anyone being able to test the proxy." → "Faithfulness has been used as a substitute for understanding, but no one could test this substitute."
- "It operates over the platform Fig. lays out." → "It operates over the platform shown in Fig.~\ref{fig:platform}."
- "Across the neuroscience, attribution, and mechanistic-interpretability traditions --- 30 ... 31 ... 257 ---" → "We study three traditions: neuroscience, attribution, and mechanistic interpretability. This covers 30 methods plus the oracle, across 31 experiments and 257 per-game records. We score each on two axes: faithfulness against the oracle, and a documented plausibility proxy."
- "carries the higher plausibility (0.9 versus 0.5), looking most convincing where it is least correct." → "still has the higher plausibility score (0.9 versus 0.5). It looks most convincing exactly where it is least correct."
- "we read it as a way-station, not the headline." → "We treat it as an intermediate step, not the main finding."
- "The dividing line, it turns out, is not faithfulness." → "Faithfulness is not what separates good explanations from bad ones."
- "so we put the question in its strongest form and remove the escape" → "so we test the strongest version of the question and close this objection"
- "The rise is honest rather than universal." → "The increase is real but does not occur on every game."
- "it finds the right cause and names no meaning" → "it identifies the correct cause but gives no meaning for it"
- "Securing faithfulness ... is not the smaller half ... but the wrong half." → "Securing faithfulness is necessary, but it does not solve the main problem."
- "Faithfulness is necessary for understanding but radically insufficient" → "...but far from sufficient"
- "every method returns the same 'wiggle X and Y moves' wiring" → "every method returns the same wiring fact: change input X and outputs X and Y move"
- "the residual ... is not philosophy but a number" → "we can measure the gap ... as a number"
- "A concept can also be linearly decodable yet provably unused, since what is present need not be used." → "A concept can be linearly decodable yet provably unused. Being present in the state does not mean being used."
- "The wiring is in the mechanism for any faithful method to recover; the meaning is not..." → "The wiring is in the mechanism, so any faithful method can recover it. The meaning is not, and no method we tested supplied it."
- "The hard part of interpretability is not faithfulness but what remains once faithfulness is secured" → "The hard part of interpretability is not faithfulness. It is recovering meaning, which remains unsolved even when faithfulness is achieved."

### 02_related.tex
- "yet all ask what, inside a complex system, caused this output" → "yet all ask the same question: inside a complex system, what caused this output?"
- "turned this battery on a microprocessor running classic games" → "applied these methods to a microprocessor running classic games"
- "not how far each fell from the mechanism; we supply that number" → "but not how much each analysis deviated from the true mechanism. We supply that number."
- "On a learned policy that doubt can only be argued, the true cause being unknown; we settle it where it is computable" → "On a learned policy the true cause is unknown, so this doubt can only be argued. We resolve it on a system where the true cause is computable."
- "Yet a recovered circuit, the same oracle shows, is a correct data-flow graph, not the computed function." → "Yet the same oracle shows that a recovered circuit is a correct data-flow graph, not the computed function."
- "Yet such checks are only negative: with no reference map none can certify a right method..." → "But such checks are only negative. With no reference map, they cannot certify that a method is correct. So the field optimises a proxy it cannot directly score."
- "Never designed to be interpretable, the VCS yields its causal structure to an exhaustive intervention oracle" → "The VCS was never designed to be interpretable. Even so, an exhaustive intervention oracle can recover its full causal structure."
- "the engineered benchmarks validate faithfulness alone, because there the circuit is the meaning by fiat" → "...validate faithfulness only, because in them the circuit is defined to be the meaning"
- "Hence we can do what they cannot: separate faithfulness from understanding..." → "This lets us separate faithfulness from understanding. We can show that a method which perfectly recovers the true wiring still has not recovered the semantics."
- "Read against Marr's levels ... the point that drives this paper holds..." → "Marr's levels and Lazebnik's parable support this paper's main point. A faithful description of a system's wiring is not yet an account of the algorithm it runs. On a known machine, we measure that gap."

### 03_methods.tex
- "the correctness triad that turns 'do we understand?' into a set of numbers." → "the correctness triad that scores understanding as numbers."
- "yet whose every causal dependency we can read off exactly." → "yet whose every causal dependency we can measure exactly."
- "The inputs we drive ... are the three faces ... so one substrate carries all three batteries." → "A neural interpretability study works with three things: the inputs, the state, and the outputs. Our substrate gives us all three."
- "We honor one boundary throughout." → "We respect one limit throughout."
- "The oracle is the heart of the method: for any output y and any candidate cause u..." → "The oracle is central to the method. For any output y and cause u, it returns the true effect of u on y. We score every method against it."
- "The true causal effect ... is thus the bit-exact difference its surgical edit makes ..." → "The true causal effect of a cause is the difference its edit makes to the output. We measure it by running the altered machine. We assume no world model and no smoothness."
- "The interventions ... are of distinct manifold kinds, and we state each because the kind bounds what a score means." → "The oracle uses several kinds of intervention. We list each one, because the kind limits what a score means."
- "Off-manifold interventions are acceptable ... precisely because the oracle measures a causal effect, not a likelihood..." → "Off-manifold interventions are still valid for oracle scoring. The oracle measures a causal effect, not a likelihood. We ask what the output would be if u were v'. The bit-exact machine answers this exactly for any v', on- or off-manifold."
- "The content-path gradient ... is its companion construction." → "The content-path gradient is the second way to build the oracle. We read it through the differentiable substrate."
- "Yet, a single decisive caveat carries through the whole study." → "One important limit applies to the whole study."
- "an infinitesimal change in u cannot move the sprite by a fractional pixel." → "a tiny change in u cannot move the sprite by part of a pixel."
- "We push the faithfulness audit to its strongest form by removing the one escape the position regime leaves open." → "We make the faithfulness audit as strict as possible. The position regime leaves one objection open, and we close it."
- "The naive zero ... could be read as a fixable artifact, so we fix it..." → "A reader might object that the naive zero is just a fixable artifact. So we fix it: we turn on the Paper-1 bilinear sampler."
- "rather than a cause that an external label later christens." → "rather than a cause that an external label only names later."
- "Faithfulness and semantic recovery are thus orthogonal axes, the one measuring ... the other ..." → "Faithfulness and semantic recovery are two independent axes. Faithfulness asks whether the method names the right cause. Semantic recovery asks whether it names any meaning."
- "The ground truth comes in three tiers, and the tiering is not bookkeeping; it carries the central argument..." → "The ground truth comes in three tiers. The tiers are not just bookkeeping; they support the paper's central argument."
- "T1 and T2 are everything the machine is and need no external label." → "T1 and T2 describe what the machine does internally, and need no external label."
- "Tier 3 (T3) is different in kind, and that difference is the point." → "Tier 3 (T3) is different in kind. This difference matters for our argument."
- "names not computable from the substrate alone, because they are facts about what the program means, not what it does." → "These names cannot be computed from the substrate alone. They are facts about what the program means, which is separate from what it computes."
- "We are deliberate about what this buys us." → "We are careful about what this gives us."
- "A faithful attribution is not yet an understanding, and to make that a measurement we need an operational criterion." → "A faithful attribution is not yet an understanding. To measure understanding, we need an operational criterion."
- "Separating F, which the wiring satisfies, from S and M ... is what lets us name, later, the residual..." → "F is satisfied by the wiring; S and M must be satisfied by the computation. Separating them lets us measure, later, what faithfulness alone leaves out."
- "Its job is to expose the danger zone ... convincing yet wrong." → "The proxy lets us find explanations with high plausibility but low faithfulness: explanations that convince but are wrong."
- "We turn three traditions onto the machine and score each by the triad..." → "We apply three traditions to the machine and score each by the triad of Eq.~\eqref{eq:triad}."
- "Against these numbers the semantic gap is measured." → "We measure the semantic gap relative to these numbers."

### 04_results_A.tex
- "We begin where the field's doubt began." → "We begin with a known concern in the field."
- "They could diagnose the failure but not grade it, because on the 6502 they had no causal ground truth..." → "They could show the failure but not measure it. On the 6502 they had no causal ground truth to score each method against."
- "We supply that number." → "We supply that missing number."
- "turns Kording's qualitative lesson into a measured one." → "makes Kording's qualitative lesson quantitative."
- "The structure these methods report is real; yet structure is not cause, and the gap is what the score measures." → "The structure these methods report is real. But structure is not the same as cause. The score measures the difference."
- "The wiring itself is largely recoverable." → "The dependency structure is largely recoverable."
- "at a faithfulness of 0.41, half spurious." → "at a faithfulness of 0.41: about half its edges are spurious."
- "Lesioning is causal, and the one classical number near the ceiling proves it." → "Lesioning is causal, which is why A2 is the one classical method that scores near the top."
- "Yet a near-perfect rank order hides an interaction blind-spot." → "Yet a near-perfect rank order still misses interactions."
- "The map of which units matter is right, the model of how they combine absent." → "The ranking of which units matter is correct, but the model of how they combine is missing."
- "an effect map is not the algorithm that produces it." → "An effect map does not show the algorithm behind it."
- "Tuning curves report structure that points the wrong way." → "Tuning curves report structure that does not match the true cause."
- "which foreshadows the present-versus-used distinction..." → "This is the same present-versus-used distinction we discuss later for linear probing."
- "Correlation structure and field potentials are real and largely epiphenomenal." → "Correlation structure and field potentials are real, but they are mostly side effects, not causes."
- "Both see the machine breathe without seeing it compute." → "Both methods detect the machine's activity but not its computation."
- "Granger causality mistakes temporal precedence for cause." → "Granger causality treats 'happens earlier' as 'is the cause.'"
- "No longer lag or stricter threshold repairs that; precedence and causation come apart here..." → "Neither a longer lag nor a stricter threshold fixes this. Precedence and causation differ here, and only the intervention oracle separates them."
- "Dimensionality reduction lands in the middle, with NMF and PCA tied." → "Dimensionality reduction scores in the middle, with NMF and PCA tied."
- "A7's 0.60 ... leaves the latent space half interpretable and half opaque." → "A7's 0.60 is the second-highest score, behind only the causal lesion. It recovers about 60 percent of the known components and leaves the rest unexplained."
- "A perfect record is still not an explanation." → "A complete record is still not an explanation."
- "a recording of everything is, at the algorithmic level, an explanation of nothing." → "at the algorithmic level, a recording of every cell explains nothing."
- "Faithfulness and sufficiency without minimality is a log file ... the paper's centre of gravity." → "Faithfulness and sufficiency without minimality give a log file, not an account of the computation. This is the first sign that the three axes are independent. The semantic-gap result makes that point central later."
- "As such, the gap is the method, not the machine." → "So the limitation lies in the analysis methods, not in the machine."
- "That residual is the problem the paper measures." → "This remaining gap is what the paper measures."

### 05_results_B.tex
- "Attribution hands back a map ... the one question a real network never lets us ask, whether the map is true." → "Attribution returns a map of which inputs mattered. We apply these methods to the VCS to ask a question a real network cannot answer: is the map true?"
- "The headline is a split, not a ranking." → "The main result is a split between two output types, not a ranking of methods."
- "the entire gradient family inherits this boundary, and it cuts the leaderboard cleanly in two." → "So the whole gradient family hits this limit, and it splits the leaderboard into two clear groups."
- "On the content path the gradient is faithful and unremarkable." → "On the content path the gradient is faithful, as expected."
- "lands its mass on the true causal byte" → "puts most of its attribution on the true causal byte"
- "Smoothing leaves the score put ... nothing for the methods to disagree about." → "Smoothing does not change the score, and guided backprop reduces to vanilla saliency. On this path all methods give the same answer."
- "The position output is where the family fails, and it fails by vanishing." → "The gradient family fails on the position output. Its attribution goes to zero."
- "Returning the empty map ... not a partial success but a structural blind spot, exactly where the discrete game logic lives." → "Returning an empty map for half of a game's outputs is not partial success. It is a structural failure, exactly where the discrete game logic is."
- "Two finer results sharpen the warning." → "Two further results make this point stronger."
- "the more careful method buys stability, not truth ... 0.034 is the lowest in the study." → "the more careful method gives stability but not accuracy. Its all-regime faithfulness of 0.034 is the lowest in the study."
- "A faithful explanation must change when the model changes, and this one cannot." → "A faithful explanation must change when the model changes. This one does not."
- "The map lands on the right cause yet stays blind to the model that produced it." → "The map identifies the right cause but does not depend on the model that produced it."
- "Replace the gradient with an actual intervention and the position blind spot disappears." → "When we replace the gradient with a real intervention, the position failure disappears."
- "the coarse twin of the oracle." → "a coarse version of the oracle."
- "A perturbation is faithful exactly when it is a valid intervention." → "A perturbation is faithful only when it is a valid intervention."
- "A part of the popular toolkit returns no answer ... nothing on the VCS to attach to." → "Some popular methods return no answer at all, because the VCS has no structure for them to use."
- "That a slice of the field's headline methods is silent on a real, deployed artifact is itself part of the audit." → "Several well-known methods produce no result on a real artifact. This absence is itself a finding of the audit."
- "the leaderboard separates along the axis that matters and crosses the axis the field actually optimizes." → "faithfulness and plausibility move in opposite directions: methods rank high on plausibility but low on faithfulness."
- "the provably empty map looks more convincing than the one that is exactly right." → "the empty map has a higher plausibility score than the exactly correct one."
- "That inversion ... is the danger zone, and on the VCS it is not a worry but a measurement." → "Low faithfulness with high plausibility is the danger zone. On the VCS we can measure it directly, not just warn about it."
- "We claim no more than the faithfulness of these maps." → "We claim only that these maps are faithful, nothing more."
- "faithfulness can be had, and ... the most popular way of seeking it does not deliver it." → "faithfulness is achievable, and the most popular methods do not achieve it."

### 06_results_C.tex
- "Mechanistic interpretability makes the strongest claim ...: not where a system attends, but how it computes." → "Mechanistic interpretability makes the strongest claim in the field. It asks not where a system attends, but how it computes."
- "We therefore turn ten ... methods onto a circuit whose answer is already on the page, and score each against the same oracle..." → "We therefore run ten methods on a circuit whose answer we already know. We score each against the same oracle as in §A: the exact intervention table, the true read/write graph, and the F∧S∧M triad."
- "a circuit whose answer is already on the page" → "a circuit whose answer we already know"
- "the split is the point of the paper: the causal methods are exact, yet exactness buys the wiring, not the meaning." → "The result splits cleanly. The causal methods are exact, but exactness recovers only the wiring, not the meaning."
- "exactness buys the wiring, not the meaning" → "exactness recovers only the wiring, not the meaning"
- "The headline is bare." → "The main result is simple."
- "Clamping a RAM cell ... is, on this machine, literally the oracle's own operation, so it cannot disagree with it." → "On this machine, clamping a RAM cell to a donor value and re-running the real ROM is literally the oracle's own operation. So it cannot disagree with the oracle."
- "faithfulness is not hard to earn on a fully observable machine; it is automatic." → "faithfulness is automatic on a fully observable machine."
- "That is the easy half of the problem, and it is genuinely solved here." → "This is the first part of the problem, and it is solved here."
- "The causal account weakens the moment a method trades exact re-runs for an approximation or a search." → "The causal account weakens as soon as a method replaces exact re-runs with an approximation or a search."
- "because the linearisation errs in magnitude, not in sign of presence" → "because the linearisation is wrong about the size of an edge, not about whether the edge exists"
- "On a machine where it is cheap they leave structure on the table: precise ... silent ... a recall ceiling rather than a precision failure." → "On a machine where the exact intervention is cheap, these methods miss real structure. They are precise about what they find but silent about dependencies re-derived through other cells. The limit is recall, not precision."
- "they leave structure on the table" → "they miss real structure"
- "Here the paper turns." → "This subsection presents the central result."
- "A circuit can be the right graph and still not be the computed function." → "A circuit can have the correct graph and still not reproduce the computed function."
- "One method is exact and the other is graph-perfect, and between them the behaviour still escapes." → "One method is exact and the other recovers a perfect graph. Even so, neither reproduces the behaviour."
- "This is the separation a 'methods underperform' story can never produce, because nothing here underperformed." → "A 'methods underperform' explanation cannot account for this separation, because no method underperformed here."
- "The dictionary methods sharpen the gap from the other side." → "The dictionary methods show the same gap from the opposite direction."
- "By the metric ... the SAE has succeeded outright. By the oracle it has not:" → "By the metric the field uses to claim a feature 'is' a concept, the SAE succeeds. By the oracle, it does not:"
- "The matched feature, removed, does not change the computation it supposedly names." → "When we remove the matched feature, the computation it is supposed to name does not change."
- "the circuit was causal but behaviourally incomplete, the feature is perfectly named but causally inert. The gap runs in both directions at once." → "This separation is the reverse of the first. The circuit was causal but behaviourally incomplete; the feature is perfectly named but causally inert. The two failures point in opposite directions."
- "Linear probing closes the triptych with the classic trap." → "Linear probing shows a third version of the same gap, the well-known probing trap."
- "Yet presence is not use." → "Yet the program does not necessarily use that information."
- "We state the caveat in the same breath:" → "We state the caveat directly:"
- "DAS's alignment accuracy of 1.0 holds 'by construction' because it confirms a supplied variable" → "DAS's alignment accuracy of 1.0 is guaranteed in advance, because it only confirms a variable we supplied"
- "The probe tells us a concept is in the representation; only the intervention tells us the program uses it." → "The probe tells us a concept is present. Only the intervention tells us the program uses it."
- "Phase C is the cleanest demonstration of the paper's claim." → "Phase C is the clearest test of the paper's claim."
- "The mechanistic toolkit ... separates into a causal core that is faithful by construction (...) and a recovery layer that is precise-but-partial (...) or present-but-inert (...)." → "Run against a known real circuit, the mechanistic toolkit splits into two groups. The first is a causal core that is faithful by construction: activation patching, causal scrubbing, DAS, and the logit lens, all at 1.0 (phase mean faithfulness 0.63, the highest of the three phases). The second is a recovery layer that is either precise but partial (attribution and path patching, ACDC) or present but inert (SAEs, dictionaries, probing)."
- "The three separations are adjudicated by the oracle, not by our preference, and they appear exactly where the field's success metrics report victory:" → "The oracle, not our judgement, decides the three separations. They appear exactly where the field's success metrics report success:"
- "Faithfulness, on a machine that grants it for free, is therefore the necessary half of understanding and not the sufficient one." → "On a machine that gives faithfulness for free, faithfulness is necessary for understanding but not sufficient."
- "The residual, the wiring recovered and the meaning withheld, is the semantic gap, which §... places on the shared axes..." → "The residual is the semantic gap: the wiring is recovered, but the meaning is not. §\ref{sec:results_compare} places this gap on the shared axes, and the Discussion treats it as the field's open problem."
- "the meaning withheld" → "the meaning is not recovered"

### 07_results_compare.tex
- "We have now run three traditions ... not the ranking among them but the boundary they all share: every method ... returns the system's causal wiring and not its meaning." → "We have now run three traditions on one machine. This section is organized not by the ranking among them, but by a boundary they share. Every method, faithful or not, returns the system's causal wiring, not its meaning."
- "Then we press past it: we make the unfaithful methods faithful ... leaves the picture unchanged, because faithfulness was never the dividing line." → "Second, we go further. We make the unfaithful methods faithful by turning on the bilinear sampler. Making every method faithful does not change the result, because faithfulness was never the dividing line."
- "The harder half of the problem, the variable's meaning and the routine's algorithm, is a separate object that no method ... recovers, and the distance to it is a number we name the semantic gap." → "The harder part of the problem is the variable's meaning and the routine's algorithm. This is a separate object, and no method on this fully known machine recovers it. We call the distance to it the semantic gap."
- "The leaderboard is the way-station; the gap is the destination." → "The leaderboard is an intermediate result; the semantic gap is the main result."
- "Faithfulness asks whether an explanation names the true causes, and on the VCS we answer it exactly." → "Faithfulness is whether an explanation names the true causes. On the VCS we can measure it exactly."
- "...returns the ground truth no real network affords." → "It returns ground truth that no real network can provide."
- "the familiar story that faithful and convincing are not the same" → "the known result that faithful and convincing are not the same"
- "The leaderboard separates cleanly along the axis that matters, and we read it as an exploratory cross-tradition ranking rather than a single commensurable score: the traditions measure different objects..." → "The leaderboard separates cleanly along the faithfulness axis. We read it as an exploratory cross-tradition ranking, not a single commensurable score, because the traditions measure different objects. We report the ranking with its robustness."
- "On the position regime, the discrete sprite-position outputs whose naive gradient is provably zero ..., the gap holds with the same sign: ..." → "The position regime is the set of discrete sprite-position outputs whose naive gradient is provably zero under the hard forward map. On this regime the gap holds with the same sign: causal and intervention 0.412 versus gradient and correlational 0.068."
- "the gradient family and the classical correlational analyses ... fall toward the floor" → "the gradient family and the classical correlational analyses (tuning curves, Granger causality, spike-word correlations) score near zero"
- "The starkest pair is bit-exact on every game." → "The clearest contrast is between two methods, and it holds bit-exactly on every game."
- "Yet two qualifications belong in the same breath." → "Two qualifications must be stated alongside this result."
- "so the leaderboard's headline contrast is carried in part by methods that approach the oracle operation itself" → "so part of the leaderboard's main contrast comes from methods close to the oracle operation itself"
- "The faithfulness axis would be unremarkable if it agreed with how convincing a method looks, and it does not." → "If the faithfulness axis matched how convincing a method looks, it would tell us nothing new. It does not match."
- "the cloud separates into a benign diagonal and a hazardous corner: the methods the field reaches for first sit near the faithfulness floor while carrying the highest proxy tags" → "the points split into a safe diagonal and a dangerous corner. The methods the field uses first sit near zero faithfulness while having the highest plausibility-proxy values."
- "The inversion is the point: the map that is provably empty ... looks more convincing than the map that is exactly right." → "This inversion is the key finding: the map provably empty on the position regime looks more convincing than the exactly correct map."
- "the ranking it inverts is the field's own habit of trust..." → "the ranking it inverts is what the field tends to trust, so the hazard is real even at the proxy's coarse resolution"
- "The danger zone is a measurement, not a worry." → "The danger zone is a quantitative result, not a speculation."
- "A reader could grant the danger zone and still escape its lesson by blaming the gradient." → "A reader could accept the danger zone and still dismiss it by blaming the gradient."
- "On this reading the danger zone is a technical artifact ... and once repaired ... the worry would dissolve." → "On this reading, the danger zone is a technical artifact a smoother forward pass would repair. Once repaired, the gradient methods would become faithful and the concern would disappear."
- "The repair works, on the games where there is a moving tracked sprite to repair." → "The repair works on the games that have a moving, tracked sprite."
- "Where a sprite's position is genuinely tracked, however, the formerly empty saliency map becomes a faithful one, and the faithfulness escape is closed." → "Where a sprite's position is actually tracked, the previously empty saliency map becomes faithful. This rules out the gradient-blaming objection."
- "It buys no understanding." → "But this does not give understanding."
- "A faithful attribution to ram[54] names a byte the program reads; it does not say that the byte is the paddle..." → "A faithful attribution to ram[54] names a byte the program reads. It does not say that the byte is the paddle, that reading it is tracking, or that the routine computes a bounce."
- "Faithfulness was repairable; meaning was not." → "We could repair faithfulness, but not meaning."
- "This is the keystone for the rest of the section:" → "This result sets up the rest of the section:"
- "Hand the audit its best case, a method that scores 1.0 ..., and ask whether it has given us an understanding..." → "Take the best case for the audit: a method that scores 1.0 against the oracle. We ask whether it gives an understanding of what the program does."
- "The missing object is of a different type, not a worse-fit instance of the same one." → "The missing object is a different kind of thing. It is not just a poorer version of the wiring."
- "The three show the axes pulling apart in both directions at once." → "The three separations show the failures pointing in opposite directions."
- "A correct graph of which cells feed which is not the function those cells compute." → "A correct graph of which cells feed which other cells is not the function those cells compute."
- "'Looks like the ball's x' is not 'is how the program computes with the ball's x.'" → "A feature that looks like the ball's x is not the same as the mechanism the program uses to compute with the ball's x."
- "Where the circuit was correct yet incomplete, the dictionary is named yet wrong; the gap runs the other way on the same axis, which a story of methods scoring too low cannot produce." → "The circuit was correct but incomplete; the dictionary is named but causally wrong. The gap runs in the opposite direction on the same axis. An account of methods simply scoring too low cannot explain this."
- "the oracle is strictly stronger than the probe that walks into it" → "the oracle is strictly stronger than the probe"
- "Probing answers 'is it there,' the oracle answers 'is it used,' and only the second is about how the program works." → "Probing answers whether the concept is present. The oracle answers whether it is used. Only the second tells us how the program works."
- "The claim does not rest on any single low number ... but on the gap appearing where tuning cannot reach --- ... --- and running in opposite directions on the same axis at once." → "The claim does not rest on any single low number; a lone low S could be a tuning artifact. It rests on two things. First, the gap appears where tuning cannot help: the effect table is already exact. Second, the failures run in opposite directions on the same axis."
- "That two-sidedness is what a 'methods are not good enough yet' account cannot generate." → "A 'methods are not good enough yet' account cannot explain this two-sided result."
- "We restate the honesty contract here, where the temptation to overread is greatest." → "We restate the honesty contract here, because this is the easiest place to overread the results."
- "Hence, the faithful methods recover the wiring; the meaning is a separate object, and the distance to it is what we have measured." → "So the faithful methods recover the wiring. The meaning is a separate thing, and we have measured the distance to it."

### 08_discussion.tex
- "We have turned the interpretability toolkit onto a machine whose ground truth is computable, and that let us measure two things at once." → "We applied the interpretability toolkit to a machine whose ground truth is computable. This let us measure two things at once."
- "the leaderboard separates them along the only axis that admits an exact answer" → "the leaderboard separates them by faithfulness, the one axis we can score exactly"
- "A faithful causal account scores 1.0 where the field's default scores 0.0 ... Yet this is the easier half." → "A faithful causal account scores 1.0 where the field's default scores 0.0, on a machine whose answer is known exactly. But faithfulness is the easier of the two measurements."
- "Even a method at the oracle ceiling returns the causal wiring ... and not the semantics, the variable's meaning or the routine's algorithm." → "Even a method at the oracle ceiling returns only the causal wiring: an effect map and a data-flow graph. It does not return the semantics: the meaning of a variable or the algorithm of a routine."
- "The residual is a number, and we name it the semantic gap." → "We call this residual the semantic gap, and it is a number we can measure."
- "Three traditions that rarely share a page, a neuroscience battery, the attribution toolkit, and mechanistic interpretability, occupy a single plane once a common oracle scores them." → "We score three traditions with one common oracle: a neuroscience battery, the attribution toolkit, and mechanistic interpretability. These traditions are rarely compared, but the oracle places them on one scale."
- "that rarely share a page" → "that are rarely studied together"
- "occupy a single plane once a common oracle scores them" → "can be placed on one common scale once a shared oracle scores them"
- "the gradient family collapses to the floor on the discrete position regime" → "the gradient methods score zero on the discrete position regime"
- "We add the missing case, a real, hand-authored, deployed artifact never designed to be interpretable, whose ground truth is independent of the artifact rather than built into it." → "We add the missing case: a real, hand-written, deployed artifact that was never designed to be interpretable. Its ground truth is independent of the artifact, not built into it."
- "Hence the leaderboard calibrates these methods against ground truth on a system the field did not get to plant." → "So the leaderboard calibrates these methods against ground truth on a system the field did not design to be interpretable."
- "Yet the same oracle marks a boundary the leaderboard cannot cross, and that boundary is the subject of this paper." → "The same oracle also reveals a limit: a high leaderboard score cannot show meaning. This limit is the subject of this paper."
- "We saw the boundary three times, from three sides" → "We observed this limit in three cases"
- "The gap runs in both directions on the same axis at once, correct-but-incomplete and named-but-wrong, which is precisely what a 'methods are not good enough yet' story cannot produce." → "The gap appears in two opposite forms: correct-but-incomplete, and named-but-wrong. A simple 'the methods are not good enough yet' story cannot produce both."
- "We pressed the most natural version of that story until it broke." → "We tested the strongest version of that story and it failed."
- "the danger zone could be dismissed as a fixable technical accident. We removed the accident." → "so the failure could be dismissed as a fixable technical artifact. We removed that artifact."
- "their position-regime faithfulness rises off the floor to 0.791 on Pong ... and to 0.529 on Q*bert ..." → "their position-regime faithfulness rises above zero. On Pong it reaches 0.791 (vanilla saliency, grad×input, smoothgrad, expected gradients). On Q*bert it reaches 0.529 (vanilla saliency, grad×input, smoothgrad, integrated gradients)."
- "We engineered the faithfulness deficit away and the meaning did not appear." → "We removed the faithfulness deficit, but the meaning still did not appear."
- "When the gradient is made honest it names the cause and names no meaning, so the meaning was never in the wiring for a method to recover." → "Even when the gradient is corrected, it identifies the cause but no meaning. So the meaning was never present in the wiring for a method to recover."
- "The rise is honest about where it does and does not occur, which is itself part of the evidence." → "The improvement occurs only in specific cases, and that pattern is itself evidence."
- "Where the position byte exists and moves, the sampler makes the gradient faithful; everywhere the faithfulness story permitted, we let it win." → "Where a position byte exists and moves, the sampler makes the gradient faithful. We gave the faithfulness explanation every case it could win."
- "None of it bought a single semantic label" → "None of it produced a single semantic label"
- "This is Lazebnik's lament, that one can characterize a system in exhaustive mechanistic detail and still not understand it." → "Lazebnik made the same point: one can describe a system in complete mechanistic detail and still not understand it."
- "We turn that lament into a measured quantity" → "We turn that observation into a measured quantity"
- "The bar is sharpest when borrowed from software engineering." → "Software engineering gives the clearest version of this requirement."
- "A faithful circuit recovers part of the program; understanding the system means recovering its documentation, what each variable is for and what each routine is meant to do." → "A faithful circuit recovers part of the program. To understand the system you must also recover its documentation: what each variable is for and what each routine should do."
- "The reference against which a cause becomes a meaning lives in the system's documented behavior, where no perturbation of the chip can find it." → "A cause becomes a meaning only against an external reference: the system's documented behavior. No perturbation of the chip can reach that reference."
- "We are deliberate that this paper defines and measures the gap." → "We state clearly that this paper defines and measures the gap."
- "Closing it, recovering the program's documentation and design from the bare machine, is the reverse-engineering bar and the charge of the companion ... paper (Paper 4)." → "Closing the gap means recovering the program's documentation and design from the bare machine. That is the reverse-engineering goal, and the task of the companion design-recovery paper (Paper 4)."
- "it shows that a concept is present in the state, not that it is used" → "it shows that a concept is present in the state, but not that the concept is used"
- "Note that the three traditions are, at bottom, one toolkit under different names, as tuning curves are feature visualization, lesions are ablations, and Granger causality is the correlational ancestor of causal attribution." → "At base, the three traditions are one toolkit under different names. Tuning curves are feature visualization, lesions are ablations, and Granger causality is the predecessor of causal attribution."
- "as tuning curves are feature visualization, lesions are ablations, and Granger causality is the correlational ancestor of causal attribution" → "tuning curves match feature visualization, lesions match ablations, and Granger causality is the correlational predecessor of causal attribution"
- "The obvious objection deserves a hearing." → "We address the obvious objection."
- "When a method fails here, with perfect ground truth ... and no learning to muddy the picture, that failure is strong negative evidence..." → "Here a method has perfect ground truth, full observability, exact interventions, and no learning to confuse the result. If it still fails, that failure is strong evidence against any claim that the same method reliably recovers causal mechanisms in harder, less observable systems."
- "no learning to muddy the picture" → "no learning to complicate the result"
- "Passing here is necessary, not sufficient; failing here is telling." → "Passing here is necessary but not sufficient; failing here is informative."
- "a hand-coded echo of superposition" → "a hand-coded analogue of superposition"
- "so that a property provable here is one suspected there" → "so that a property we can prove on the VCS is one to suspect in neural networks"
- "What the chip does not capture we state without hedging: learned features, scale, and stochasticity are absent, and the learned agent that folds all three back in is Paper 5." → "We state plainly what the chip does not capture: learned features, scale, and stochasticity are absent. The learned agent that adds all three back is Paper 5."
- "What the chip does not capture we state without hedging" → "We state plainly what the chip does not capture"
- "Three recommendations follow, and we state them flatly." → "Three recommendations follow."
- "a map provably empty on the position regime looks more convincing than the map that is exactly right" → "a saliency map provably empty on the position regime can look more convincing than the exactly correct map"
- "We learned this the hard way, by closing the faithfulness side on the gradient methods and watching the meaning stay absent." → "We confirmed this directly: we closed the faithfulness gap for the gradient methods, and the meaning still did not appear."
- "Faithfulness is the half we now know how to measure and, on causal methods, how to win; meaning is the half that remains..." → "Faithfulness is the part we can now measure, and on causal methods we can achieve it. Meaning remains unsolved, because no internal method carries the external reference it needs."
- "how to win" → "how to achieve it"
- "so the claim rests not on any one number but on the gap appearing where tuning cannot reach, running in both directions at once, and surviving the sampler-on test ..." → "So the claim does not rest on any single number. It rests on three facts: the gap appears where tuning cannot reach it; it runs in both directions; and it survives the sampler-on test that makes the gradient methods faithful on games with a moving tracked sprite."
- "a moving tracked sprite gives the gradient something to flow to" → "a moving tracked sprite gives the gradient a continuous signal to follow"
- "We close by reaching past the present result." → "We end with the broader implication."
- "so a saliency map over sprite position is empty by construction, not by accident" → "so a saliency map over sprite position is empty for a structural reason, not by chance"
- "That statement about gradients as a class travels to any system where a discrete decision sits downstream of a continuous one." → "This statement about gradients generalizes to any system where a discrete decision follows a continuous one."
- "where the semantic gap can only widen" → "where the semantic gap is likely to be larger"
- "It is foreseeable that closing it will be the work of the next decade ... turning a faithful account of a learned system into an understanding of what it has learned." → "Closing this gap may take the next decade of interpretability research. The goal is to turn a faithful account of a learned system into an understanding of what it has learned."
- "We have not solved it. We have made it a number, and a number is where understanding begins." → "We have not solved this problem. We have defined and measured it, which is the necessary first step."
- "because finding structure is not the same as explaining it" → "because describing structure is not the same as explaining how it works"

### 09_endmatter.tex
- "Following the practice of the underlying emulator port, we acquire them through the AutoROM retrieval procedure and report a SHA-256 hash table ... so that the exact binaries can be reproduced and verified independently." → "We follow the practice of the underlying emulator port. We acquire the ROMs through the AutoROM procedure and report a SHA-256 hash for each one in the Supplementary Information. This lets others reproduce and verify the exact binaries independently."
- "The differentiable, bit-exact emulator that the benchmark rests on, in its two interchangeable realizations (jutari, Julia; jaxtari, JAX/Python), is already public." → "The benchmark rests on a differentiable, bit-exact emulator. It has two interchangeable versions (jutari in Julia, jaxtari in JAX/Python), and both are already public."

---

**Recommended fix order:** (1) the 2 meaning-inverting spans (04_results_A "No longer lag", 08_discussion "not that it is used") — correctness; (2) the banned halving/refrain phrases ("smaller/wrong/easier/harder half", "wiring not the meaning", "buys X not Y", way-station/destination) — a find-and-replace pass; (3) split the 42 long sentences in `08`, `07`, `06` at the colons and "and" joins. Files most in need: `08_discussion.tex`, `07_results_compare.tex`, `06_results_C.tex`.
