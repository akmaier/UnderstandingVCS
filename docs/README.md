# docs/ — results-audit GitHub Page

A static, dependency-free site that lets a reviewer audit every result in **Paper 1**
(differentiable VCS emulator) and **Paper 2** (interpretability ground-truth benchmark)
down to the script, command, artifact, runtime, hardware, and verifying conformance gate.

Live at **https://akmaier.github.io/UnderstandingVCS/** (GitHub Pages, source = `main` `/docs`).

## Layout
- `manifest.py` — **single source of truth**. The claim ledger, environment, provenance.
  Numbers were read from the committed result files (`results/gpu/*.json`,
  `tools/rom_sweep/results_*.md`, `tools/xai_study/compare/out/leaderboard.json`,
  `tools/xai_study/repro/rom_hash_table.csv`) — not transcribed.
- `build_assets.py` — regenerates `assets/{img,gif,video}/` from the source PDFs and MP4s
  (needs `ffmpeg` + `pdftoppm`).
- `build_pages.py` — renders `manifest.py` into `index.html`, `paper1.html`, `paper2.html`,
  `conformance.html`, `provenance.html`, `environment.html`, `reproduce.html`.
  `conformance.html` is the guided code tour of the PXC1/PXC2/PXC-S/PXC4 harnesses; its
  GitHub blob links carry verified line numbers — re-check them with the snippet at the
  bottom of `build_pages.py`'s history if the harness files move.
- `guides/<key>.py` — the **beginner's guide** for each of the 31 method pages ("Learn the
  method": the method on neural networks, the translation to the VCS, code links into both the
  XAI runner and the jutari substrate, pitfalls, a reproduce command). Schema in
  `guides/__init__.py`. Code links are `(path, find-substring)` pairs resolved to line numbers at
  build time — the build fails if a function moves, so links cannot rot silently.
- `primer.py` — content for `primer.html`, the XAI primer (machine, NN↔VCS dictionary, oracle,
  scoring, reading order, glossary) that every guide links to.
- `check_guide_links.py` — verifies every Wikipedia / DOI / paper link in the guides and primer
  (Wikipedia via the API: exists, not a disambiguation page, redirects reported).
- `assets/` — generated PNG figures, looping GIFs, web MP4s, and the hand-written `css/style.css`.
- `.nojekyll` — serve paths verbatim (no Jekyll processing).

## Rebuild
```bash
python3 docs/build_assets.py   # PDFs/MP4s -> web img/gif/mp4   (only when sources change)
python3 docs/build_pages.py    # manifest.py + guides/ -> docs/*.html  (needs numpy)
python3 docs/check_guide_links.py   # after editing a guide
```

## Scope
Papers 1 and 2 only. Papers 3–5 are deliberately excluded.

## Enabling Pages (one-time, in the GitHub UI)
Settings → Pages → Build and deployment → Source = **Deploy from a branch**,
Branch = **`main`**, folder = **`/docs`**.
