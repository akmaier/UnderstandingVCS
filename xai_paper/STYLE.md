# STYLE.md — Andreas Maier's research-writing voice (mandatory for all paper drafting)

A drafting and restyling contract for the XAI program's papers (P2–P5). Every writing item
must follow this and run the §7 self-check before a section is marked done. Reusable across
all papers.

> Grounding: derived from voice profiles of **three of Maier's real papers** —
> *Learning with Known Operators reduces Maximum Error Bounds* (Nature Machine Intelligence
> 2019); *A Gentle Introduction to Deep Learning in Medical Image Processing* (Z. Med. Phys.
> 2019); *Known Operator Learning and Hybrid Machine Learning in Medical Imaging — a review*
> (Prog. Biomed. Eng. 2022). **Paper 1 is deliberately excluded.** Quotes below are his.

---

## 1. The single hardest rule — read this first

**Paragraphs open with an ordinary topic sentence and run as flowing prose. Never with a
label.** Maier does NOT use run-in `\paragraph{...}` headings and does NOT use bold or italic
lead-in labels at the start of a paragraph. There is no *"\textbf{The opportunity.}"*, no
*"\paragraph{Setup.}"*, no *"\textit{An honesty contract.}"*, no *"\textbf{Our
contribution.}"* anywhere in his expository, argumentative, methods, results, discussion, or
conclusion prose. Every paragraph begins with a normal declarative sentence that names its
subject in the first few words and then flows.

The ONE narrow exception, which our XAI papers should generally avoid anyway: inside a pure
**catalog/survey subsection** that walks a list of named architectures or concepts, he will
let the *name of the thing being defined* begin the sentence in bold and flow straight into
prose — *"**Autoencoders** use a contracting and an expanding branch…"*, *"**Precision
learning** is a strategy to include known operators…"*. These are term-definition lead-ins,
not rhetorical signpost labels, and they appear only in survey list sections, never in the
intro, derivations, discussion, or conclusion. If you are not literally cataloguing named
methods one per sentence, do not use them.

When in doubt: no label, plain topic sentence, flowing prose.

---

## 2. Voice in one paragraph

Maier writes like a confident teacher who respects the reader's time. Paragraphs open with a
flat declarative claim — a statement of where we are, never a question and never a hook — and
close on a consequence or a careful caveat rather than a flourish. The argument is carried in
the first-person plural ("we describe," "we show," "we believe," "let us consider"), active
voice for the authors' own moves, passive only for established mathematical facts. Sentences
are medium-length with deliberate variation: a long qualified sentence followed by a short
one that lands the point. His signature pivot is sentence-initial **"Yet,"** (concede, then
rebut), and his default inference connector is **"Hence,"**; he leans on *"In particular,"*
*"Note that,"* *"As such,"* *"Still,"* and *"In the following,"* and he does **not** stack
"Moreover/Furthermore/Notably." Equations carry the quantitative load and are glossed in
plain words before and after; numbers in prose are sparse and inline. He hedges mechanisms
honestly (*"seem to," "likely," "probably"*) while staying confident on flagged opinions
(*"In our view," "the authors believe"*), and he pairs strengths with their limitations in
the same breath. The register is rigorous, didactic, and warm.

---

## 3. DO NOT (the bans — enforced by the self-check)

1. **No run-in `\paragraph{...}` headings.** Do not start a paragraph with a `\paragraph{}`
   command. Use `\section`/`\subsection` for structure (§5) and plain topic sentences for
   paragraphs.
2. **No bold or italic lead-in labels on paragraphs.** No *"\textbf{The opportunity.}"*, no
   *"\textbf{Setup.}"*, no *"\textit{An honesty contract.}"*, no any-word-then-period bold
   tag that functions as a mini-heading. Every paragraph opens with a normal sentence.
   (The only allowed bold-first-word case is a literal term-definition lead-in inside a
   catalog/survey subsection — see §1; avoid it unless you are truly cataloguing.)
3. **No "Moreover / Furthermore / Notably" pile-ups.** His additive glue is the plain
   *"Also,"* and *"In particular,"*; his workhorse is the adversative *"Yet,"*. At most one
   "Furthermore" in a long stretch.
4. **No "It is worth noting that…"** and no *"It is important to…"* boilerplate. When he
   flags importance he writes a terse *"Note that…"* or owns it in the first person.
5. **No bulleted prose.** Narrative runs as continuous paragraphs. Lists appear only as
   numbered equations/theorems, or (rarely) a genuine enumerated list of items — never as a
   substitute for flowing argument.
6. **No triadic "First / Second / Third" scaffolding** as paragraph or section structure. A
   single paper-level *"First… Then… Finally…"* roadmap is fine once; do not litter
   paragraphs with it. When enumerating, name the items in running prose instead.
7. **No em-dash overuse.** Default joints are commas, *"i.e./e.g."* parentheticals, and the
   *"Yet,"* pivot. Em-dashes are allowed only for a genuine aside, sparingly.
8. **No hype adjectives or marketing register.** No *groundbreaking / novel / powerful /
   robust / seamless / comprehensive / crucial / vital / delve / leverage* (as filler).
   His evaluative words are concrete: *sharper, blurring, elegant, useful, beneficial,
   astonishing* (used once, not stacked).
9. **No overclaim and no absolute certainty.** Bound the claim (*"may," "we believe,"
   "maximum error bound," "statistically not distinguishable"*). State scope honestly.
10. **No hollow summary closers** (*"In summary, this demonstrates…"*). Close paragraphs and
    sections on a specific consequence, a forward verdict, or a "handle with care" caveat.
11. **No questions or hooks as openers.** Paragraphs and sections open on a flat factual
    claim, not a rhetorical question.
12. **No reading equations aloud** (*"sum over i of…"*). Gloss the meaning instead.
13. **No varying key terms for elegance.** Keep *"known operator," "ground truth,"
    "inductive bias"* consistent; do not reach for synonyms.

---

## 4. DO (with his evidence)

1. **Open every paragraph with a flat declarative topic sentence** that names the subject in
   the first words. *"With the rise of deep learning, researchers became aware that these
   methods… are applicable to much more than these kinds of perceptual tasks."* (NMI)
   *"Computer-aided diagnosis is regarded as one of the most challenging problems in the
   field of medical image processing."* (ZMP)
2. **Close the paragraph on a consequence or a soft caveat**, signposted often by *"Hence,"*.
   *"Hence, blind deep learning methods have to be performed with care to be successful."*
   (NMI) *"…have to be handled with care."* (ZMP)
3. **Use the concede-then-rebut two-beat, pivoting on "Yet,".** *"Yet, there are also reports
   of surprising results in which parts of the image are hallucinated."* (NMI) *"Above
   argument is well founded… Yet…"* (review).
4. **Lead with plain connectives that do logical, not additive, work.** *"Hence,"* (default
   inference), *"Yet,"* (signature pivot), *"Still,"* *"In particular,"* *"As such,"*
   *"Note that,"* *"In the following,"* *"Following the paradigm of,"* *"Obviously,"*.
5. **Carry the argument in active first-person-plural "we" / "let us."** *"we explore," "we
   derive," "we apply," "we are interested in," "let us consider a slightly more complicated
   network structure…"* Reserve passive for the math objects (*"the approximation is bounded
   by," "can be shown that"*).
6. **Lay out the roadmap once, with "we will…", at a section start — not in every paragraph.**
   *"First, we analyse the problem from a theoretical perspective… Then… Finally, we discuss
   our observations…"* (NMI, used once at paper level). *"In the following, we will describe
   the past…"* (review).
7. **Narrate every equation before and after; give an "i.e." gloss; map the math back to
   meaning.** Define symbols in words first, display, then say what it means: *"With N → ∞, ε
   approaches 0, i.e. … any single layer network is able to approximate any function."*
   (review) Scaffold derivations with *"In order to…, we need to…"* (ZMP).
8. **Name theorems in plain English and state the intuition, not the symbols.** *"Known
   output operator theorem"*; *"the bound shrinks up to the point of identity, if all
   operations are known."* (NMI)
9. **Keep results in prose as compact figures-of-merit; cite one decisive number and move
   on.** *"Frangi-Net contains less than 6% of the number of trainable parameters, while
   achieving an area under the curve (AUC) score around 0.960, which is only 1% inferior to
   that of the U-Net."* (NMI) Defer experimental detail explicitly: *"The results and
   experimental details are demonstrated in the Supplementary Information."*
10. **Weave citations as the grammatical subject + a strong verb; chain a line of work.**
    *"Kobler et al demonstrated… This was done by Hammernik et al… Later, Adler and Öktem
    demonstrated…"* (review); the escalation idiom *"X drive this even further."* (ZMP)
11. **Hedge mechanisms; be confident on flagged opinions and on what you measured.** Hedged:
    *"seem to be beneficial," "is likely," "probably," "it is foreseeable that…"*. Confident:
    *"we have shown," "we demonstrated," "In our view," "the authors believe."* State open
    problems plainly: *"Unfortunately, …"; "this apparent contradiction… has not been
    resolved sufficiently."* (review)
12. **Write captions that interpret, not just label.** A bold noun-phrase title fragment, then
    one to three sentences explaining the mechanism and what to conclude, with an in-caption
    pointer and a credit clause where reproduced. *"Schematic of the idea of known operator
    learning."* (NMI title); *"Note that the result of the learned method is much sharper…"*;
    *"(after [27])"* / *"Reproduced from [12]. CC BY 4.0."*
13. **Teach with didactic asides and the colon-punchline.** *"let us…," "Note that…,"
    "Obviously,"* and *"The answer is fairly easy: gradient descent."* (ZMP)
14. **Use at most one homely, concrete analogy per stretch, to demystify** (*"separating
    apples from pears"* — ZMP). Coin or borrow a memorable scare-quoted term sparingly
    (*'graduate student descent,' 'bitter lesson,' 'operator mining'* — review), and anchor a
    point with a famous quote where it earns its place (Box's *"All models are wrong, but
    some are useful"*).
15. **Italicize-and-define a new term on first use; prefix jargon with "so-called."** *"the
    so-called loss function," "the well known XOR problem."* (ZMP)

---

## 5. Section structure (use `\section`/`\subsection`, NOT labelled mini-paragraphs)

- **Use real sectioning commands with short noun-phrase titles**, sentence-case, never full
  sentences and never questions. His titles: *"Known operator learning," "Application
  examples," "Important architectures in deep learning," "Network and operator mining,"
  "The past," "The present," "The future," "Summary."* He goes at most three deep and keeps
  granularity coarse: a handful of sections, each holding several full paragraphs. He does
  **not** chop the text into many tiny labelled fragments.
- **Standard skeletons he uses** (pick the closest fit): journal — *Introduction; Materials
  and methods; Results; Discussion; Conclusion*. Nature-Article — *Known operator learning;
  Application examples; Discussion; Conclusion; Data availability; Code availability*.
  Narrative review — a plain temporal arc *Introduction; The past; The present; The future;
  Summary*.
- **Abstract** — a tight roadmap of mostly verb-initial "we" sentences, one job each: what →
  scope → method/constraint → theory result → empirical result → breadth → outlook.
- **Introduction** — open on the field's state as a flat claim, give a concrete feat or the
  tension, name the gap, then a single *"In this Article, we…"* roadmap. No literature
  throat-clearing, no labelled sub-headings inside it.
- **Results / Applications** — headline number bare, then per result: name the problem → the
  operator/method → what was mapped onto the system → result vs. baseline with one number →
  the interpretive takeaway. All flowing prose.
- **Discussion** — anaphoric recap (*"we have shown… we investigated…"*), then a surprise or
  insight, then a plainly-stated limitations paragraph (*"Unfortunately, …"*) — as ordinary
  prose, not a labelled "Limitations." block.
- **Conclusion / Summary** — restate the organizing frame, situate against a named authority
  if apt, and close on a forward-looking research programme with calibrated, slightly humble
  optimism — not on a result.

---

## 6. Significance and cross-disciplinary reach (Nature-family)

Restate broad significance in the abstract, introduction, discussion, and conclusion, and
reach across disciplines explicitly by naming the wider audience. Use a measured *"to the
best of our knowledge, the first…"* only where true. End the body on a vision or programme
with calibrated optimism, never on a metric. Keep the non-specialist able to follow the
motivation and the consequence even if they skip the proofs.

## 6b. Stakes and vividness (this paper earns it)

This paper identifies and *quantifies* an important problem, the way Jonas and Kording asked
whether a neuroscientist could understand a microprocessor. Write so the reader feels that
weight — this is not a dry methods report. But the vividness must come from the **problem and
the numbers, not from ornament**: no hype, no marketing adjectives (§3 still holds).

- **Open on the stakes.** State the real, slightly provocative problem plainly and early: the
  field trusts explanations it cannot check, because real models offer no ground truth; we
  built the one complex system where the explanation *can* be checked, and the verdict is
  sharp. A flat declarative sentence carrying a big idea is more striking than any flourish.
- **Carry one throughline** the reader remembers — *the wiring, not the meaning* — and let it
  recur naturally at the section level, never as a slogan repeated verbatim.
- **Let the result be the drama.** The strongest sentences are the plainest: a faithful causal
  method scores 1.0 where the field's default scores 0.0, on a machine whose answer is known
  exactly. Anchor every vivid claim to an exact number; concrete imagery (an "empty saliency
  map," a "wiring diagram that is not an account of the computation") over abstraction.
- **One or two grounded rhetorical reaches** are allowed in the introduction and discussion —
  as Maier does (*"may be key to gaining a better understanding of deep networks"*) — each
  tied to evidence and kept calibrated. Energy, not exaggeration.

---

## 7. Self-check (run before marking a section done)

- [ ] **No `\paragraph{}` and no bold/italic lead-in labels.** Every paragraph opens with a
      plain topic sentence in flowing prose. (Bold-first-word only inside a true
      catalog/survey list, and even then prefer plain prose.)
- [ ] Paragraphs open on a flat declarative claim (no questions, no hooks) and close on a
      consequence or a "handle with care" caveat.
- [ ] First-person plural, mostly active; passive only for math/established facts.
- [ ] The concede-then-rebut "Yet," pivot appears; "Hence," carries inference; connectives
      lead sentences and do logical work.
- [ ] No "Moreover/Furthermore/Notably" pile-ups; no "It is worth noting"; no triadic
      First/Second/Third scaffolding; em-dashes rare; no bulleted prose where prose belongs.
- [ ] Every equation glossed before and after with an "i.e."-style plain reading and mapped
      back to meaning; no symbols read aloud; theorems named in plain English.
- [ ] Numbers sparse and inline as figures-of-merit; detail deferred to supplementary;
      citations act in the narrative ("X et al. demonstrated…"); key terms used consistently.
- [ ] Every strength paired with its limitation; opposing views conceded then rebutted with
      "Yet,"; limitations stated plainly, in ordinary prose.
- [ ] Hedging calibrated: unhedged on measured results, modest on field/future; no hype
      adjectives, no marketing register, no overclaim.
- [ ] `\section`/`\subsection` with short noun-phrase titles; coarse granularity; no tiny
      labelled fragments.
- [ ] Captions interpret (title fragment → mechanism → "Note that…" pointer), with an in-text
      reference; broad significance restated across levels; closes on a forward programme.
- [ ] ≤1 homely analogy per stretch, carrying information; a non-specialist follows the
      motivation and the consequence even skipping the proofs.
