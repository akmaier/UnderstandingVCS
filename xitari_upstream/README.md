# xitari upstream contribution — DRAFTS (hold until arXiv)

These are **draft** texts for an upstream bug report + fix to
**[google-deepmind/xitari](https://github.com/google-deepmind/xitari)**
(the Stella fork used by ALE / the DQN line of work).

**Do NOT submit yet.** Per the project decision, these go out **only after our
paper is on arXiv** (so we can cite it and the finding is on record). The
non-determinism is also reported in the paper itself — see
`jutari_paper/paper_plan.md`.

Contents:
- [`ISSUE.md`](ISSUE.md) — the GitHub *issue*: the non-deterministic Superchip-RAM
  initialisation bug, with a minimal reproduction, root-cause analysis, and impact.
- [`PULL_REQUEST.md`](PULL_REQUEST.md) — the GitHub *pull request*: a deterministic
  fix that honours `random_seed` for on-cart (Superchip) RAM.

The concrete one-shot patch we apply locally for our conformance suite is
[`../tools/xitari_conformance_seed.patch`](../tools/xitari_conformance_seed.patch);
the PR proposes the cleaner upstream form (seed before cartridge construction).

**Before submitting (checklist):**
1. Paper is live on arXiv → fill in the arXiv ID / URL placeholders below and in
   `ISSUE.md` / `PULL_REQUEST.md`.
2. Re-verify the reproduction against a fresh upstream `xitari` checkout (the line
   numbers / file paths below are from our working copy and may have drifted).
3. Confirm the maintainers' preferred fix shape (honour-seed vs. seeded-default)
   and trim the PR to match; open the issue first, then the PR referencing it.
