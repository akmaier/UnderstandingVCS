# STYLE.md — Andreas Maier's research-writing voice (mandatory for all paper drafting)

A drafting guide for the XAI program's papers (P2–P5, Nature-family) **in Andreas Maier's
voice.** Every writing item (SPEC epic E8) must follow this and run the §6 self-check before
marking a section done. Reusable across all papers.

> Grounding: analysed from the **full text of three of Maier's papers** in `papers/` —
> *A Gentle Introduction to Deep Learning in Medical Image Processing* (Z. Med. Phys. 2019);
> *Learning with Known Operators reduces Maximum Error Bounds* (Nature Machine Intelligence
> 2019); *Known Operator Learning and Hybrid Machine Learning in Medical Imaging — a review*
> (Prog. Biomed. Eng. 2022). **Paper 1 is deliberately excluded** from the analysis. Quotes
> below are his.

## 1. Voice in one paragraph
Maier writes like a confident teacher who respects the reader's time and intelligence.
Short, plain declarative sentences carry the argument in the first-person plural ("we
describe," "we show," "we believe") — the "we" is author *and* reader. Ideas are chained,
not nested: a clean claim, a `Yet,`/`Still,`/`However,` concession, then a `Hence,`/`Thus,`
resolution — a recurring three-beat. Sentence-initial connectives do the logical work a
denser writer would bury in subordinate clauses. Every equation is narrated in words before
and after and *mapped back onto its meaning*; symbols are never read aloud. He hedges
honestly — unhedged about what he proved or measured, modest about the field and the future
— and pairs every strength with its limitation in the same breath. For a Nature-family
paper he restates broad significance at every level and reaches deliberately across
disciplines, closing not on a result but on a forward-looking research programme, with
calibrated optimism and never a hint of overclaim. The register is rigorous yet accessible,
didactic, and warm.

## 2. DO (with his evidence)
1. **First-person plural, active for our moves; passive only for established facts.** *"We
   describe an approach… We derive a maximal error bound… We apply this approach…"* (NMI
   abstract — almost every sentence verb-initial "we").
2. **Short paratactic declaratives; one idea per sentence; chain, don't nest.** Break often.
3. **Use the three-beat: claim → concession → resolution.** *"Yet, there are also reports
   of surprising results…"* then *"Hence, blind deep learning methods have to be performed
   with care."* (NMI).
4. **Lead with connective adverbs; `Hence`=default inference marker, `Yet,`=signature
   pivot** (concession→claim). Also: *Still, Thus, Therefore, In particular, As such,
   Obviously, Of course, Furthermore, In fact, Note that*.
5. **Open each section by recapping, then advancing.** *"With the knowledge summarized in
   the previous sections, networks can be constructed and trained. However, deep learning is
   not possible yet."* (ZMP). Or a temporal anchor: *"In the past…", "Today…"* (review).
6. **Abstract = a tight roadmap**, mostly verb-initial "we" sentences, one job each: what →
   scope → method/constraint → theory result → empirical result → breadth → significance →
   outlook (NMI abstract).
7. **Narrate every equation before and after; give an "i.e." gloss; map the math back to
   meaning.** *"With N → ∞, ε approaches 0, i.e. … any single layer network is able to
   approximate any function"* (review). Scaffold derivations with *"In order to …, we need
   to …"* (ZMP).
8. **Name theorems in plain English and state the intuition, not the symbols.** *"Theorem 2.
   Known output operator theorem"*; *"the bound shrinks up to the point of identity, if all
   operations are known"* (NMI).
9. **Weave citations as the grammatical subject + strong verb; escalate a line of work.**
   *"Unberath et al. drive this even further to emulate the complete X-ray formation
   process"* (ZMP); *"Zhu et al. demonstrated a surprising analogy…"* (review).
10. **Italicize-and-define each new term on first use; prefix named concepts with
    "so-called"; coin/borrow memorable scare-quoted terms, sparingly** (*'graduate student
    descent'*, *by design* — review).
11. **Captions argue.** Bold lead sentence stating what the figure *is* → mechanism → a
    *"Note that…"* pointer to the key contrast; cross-reference inline with *"(cf. Fig. X)"*.
    *"Note that the result of the learned method is much sharper…"* (NMI Fig. 5).
12. **Steer with didactic asides** — *"Note that…", "let us…", "Obviously,", "Of course,"*
    — and pose a problem, then answer it crisply: *"The answer is fairly easy: gradient
    descent."* (ZMP).
13. **Concede the opposing view fully, then rebut with "Yet,".** *"Above argument is well
    founded and makes sense… Yet…"* (review).
14. **State limitations plainly, in their own sentences; concede-then-reaffirm.**
    *"Unfortunately, …"*; *"this apparent contradiction… has not been resolved
    sufficiently"* (review); *"…does not limit the generality… Yet, our error analysis is
    still useful."* (NMI).
15. **Calibrate hedging:** unhedged on results (*"we have shown," "we demonstrated"*),
    hedged on field/future (*"we believe," "seem to," "is likely," "it is foreseeable"*).
16. **Build on triads / rule of three** (*"the past, the present, and the future"* — review;
    *"complex, fully specified, and differentiable"*-style enumerations).
17. **One homely, concrete analogy per section, to demystify** (*"separating apples from
    pears"* — ZMP); where a known structure is tested, use the **validate/falsify hypothesis**
    framing (*"any known operator sequence can also be regarded as a hypothesis… we are able
    to validate or falsify"* — NMI).

## 3. DON'T
1. **No literature throat-clearing.** Open on the field arc/tension, then land a concrete hook.
2. **Never read equations aloud** ("sum over i of…") — gloss meaning.
3. **Don't hedge what you proved; don't overclaim what you didn't.**
4. **Don't nest long subordinate clauses** where two short sentences are clearer.
5. **Don't decorate** — no "groundbreaking/novel/powerful" about our own work; confidence =
   flat statement + hard numbers.
6. **Don't vary key terms for elegance** — prize consistency ("known operator", "ground
   truth") over synonyms.
7. **Don't bury the contribution** — put it in an *"In this Article, we …"* roadmap or a
   flagged list.
8. **Don't let a figure/equation stand** without an in-text sentence naming its point.

## 4. Section-opening patterns (templates in his voice)
- **Abstract** — roadmap of verb-initial "we" sentences: *"We describe… We aim at… We derive
  … We show experimentally… We apply this to… As such, the concept is widely applicable…
  We assume that our analysis will support…"* (mirror the NMI rhythm).
- **Introduction** — big-picture temporal frame → a vivid concrete feat/hook → the gap in
  existing work → *"In this Article, we …"* numbered roadmap.
- **Results / Applications** — headline number first, bare; then a per-result mini-template:
  name the problem → the operator/method → what was mapped onto the system → result vs.
  baseline with a number → interpretive takeaway.
- **Discussion** — anaphoric recap (*"In this Article, we have shown… we could demonstrate…
  we investigated…"*), then a surprise/insight, then a flagged honest Limitations paragraph.
- **Conclusion** — restate the organizing frame/triad; situate against a named authority;
  close on a forward-looking research programme with a faintly wry, humble long-horizon note
  — *not* on a result.

## 5. Nature-family significance & equation/citation handling
- **Restate broad significance in abstract, intro, discussion, AND conclusion** — reach
  across disciplines explicitly (name the wider audience); a measured *"to the best of our
  knowledge, the first…"* where true; end the body on a vision/programme with calibrated
  optimism.
- **Equations:** symbols defined in words first → display → payoff/"i.e." gloss → map back
  to interpretation/architecture; never narrate glyphs; theorems get a plain-English name +
  intuition in text, full proofs to supplementary, tied back to the implementation.
- **Citations:** narrative *"X et al. + verb"* as subject for noteworthy work; trailing
  numerics for backing; quote a source against itself when it concedes a weakness, sparingly.

## 6. Self-check (verify before marking a section done)
- [ ] First-person plural, mostly active; "we" = author + reader; passive only for facts.
- [ ] Short paratactic declaratives dominate; the three-beat (claim → `Yet,` → `Hence,`)
      appears; connectives lead sentences and carry the logic.
- [ ] Opens on the field arc / tension (not literature warm-up); section openings recap-then-
      advance; lead sentence of each paragraph states its point; paragraph ends on a punch.
- [ ] Every equation glossed before and after with an "i.e."-style plain reading and mapped
      back to meaning; no symbols read aloud; theorems named in plain English.
- [ ] Citations act in the narrative ("X et al. demonstrated…"); key terms used consistently.
- [ ] Every strength paired with its limitation; opposing views conceded then rebutted with
      "Yet,"; limitations stated plainly ("Unfortunately, …").
- [ ] Hedging calibrated: unhedged results, modest field/future claims; no hype words.
- [ ] Broad significance restated at every level; cross-disciplinary reach present; closes on
      a forward-looking programme, not a result.
- [ ] ≤1 homely analogy per section, carrying information; figures have an in-text pointer +
      an argument-bearing "Note that…" caption.
- [ ] Nature-fit: a non-specialist follows the motivation and the consequence even if they
      skip the proofs.
