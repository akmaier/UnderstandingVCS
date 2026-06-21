#!/usr/bin/env bash
# Build the arXiv submission package: a single named (non-anonymous) PDF =
# main paper (NO reproducibility checklist) + the supplement as an appendix,
# with a link to the video on GitHub. Produces a SOURCE zip for arXiv upload
# (arXiv compiles it to one PDF) plus a local preview PDF.
#
# NOTE: this script carries the real author names, so it lives OUTSIDE tools/
# (which the anonymous supplement zip bundles). Output zip is not versioned.
#
# Usage:  bash jutari_paper/make_arxiv_submission.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PAPER="$ROOT/jutari_paper/paper"
OUT_ZIP="$ROOT/jutari_paper/arxiv_submission.zip"
OUT_PDF="$ROOT/jutari_paper/arxiv_preview.pdf"
VIDEO_URL="https://github.com/akmaier/UnderstandingVCS/blob/main/jutari_paper/presentation/presentation.mp4"
STAGE="$(mktemp -d)"; trap 'rm -rf "$STAGE"' EXIT
step() { echo; echo "=== $* ==="; }

# --- 1. slice the two documents' bodies (pattern-based, robust to line shifts) ---
step "1/5 extract main + supplement bodies"
awk '/\\maketitle/{p=1;next} /^\{\\small/{p=0} p' "$PAPER/paper.tex"          > "$STAGE/body_main.tex"
awk '/\\maketitle/{p=1;next} /\\bibliography\{references\}/{p=0} p' "$PAPER/supplementary.tex" > "$STAGE/body_supp.tex"
echo "  body_main: $(wc -l < "$STAGE/body_main.tex") lines   body_supp: $(wc -l < "$STAGE/body_supp.tex") lines"

# arXiv-only abstract edit: the code is already public, so link the repo. The
# anonymous AAAI build (paper.tex) is left untouched -- it keeps "released upon
# acceptance" because a repo link would break double-blind.
perl -0777 -i -pe 's{The full code of both ports will be released under the\s+MIT\s+license upon acceptance\.}{The full code of both ports is available under the MIT license at \\url{https://github.com/akmaier/UnderstandingVCS}.}s' "$STAGE/body_main.tex"
grep -q 'is available under the MIT license at' "$STAGE/body_main.tex" \
    && echo "  abstract: code-availability sentence -> public repo link" \
    || { echo "  FATAL: abstract MIT sentence not found/rewritten"; exit 1; }

# --- 2. support files (style, bib, figures) ---
step "2/5 copy style / bib / figures"
cp "$PAPER/aaai2027.sty" "$PAPER/aaai2027.bst" "$PAPER/references.bib" "$STAGE/"
mkdir -p "$STAGE/figures"; cp "$PAPER"/figures/*.pdf "$STAGE/figures/"
echo "  figures: $(ls "$STAGE"/figures | wc -l | tr -d ' ') pdf"

# --- 3. the combined, NAMED master document ([preprint] = authors shown, no AAAI copyright) ---
step "3/5 write arxiv.tex"
cat > "$STAGE/arxiv.tex" <<TEX
\\documentclass[letterpaper]{article}
\\usepackage[preprint]{aaai2027}
\\usepackage[hyphens]{url}
\\usepackage{graphicx}
\\urlstyle{rm}
\\def\\UrlFont{\\rm}
\\usepackage{natbib}
\\usepackage{caption}
\\frenchspacing
\\usepackage{algorithm}
\\usepackage{algorithmic}
\\usepackage{amsmath}
\\usepackage{amssymb}
\\usepackage{amsthm}
\\usepackage{booktabs}
\\pdfinfo{/TemplateVersion (2027.1)}
\\setcounter{secnumdepth}{1}
\\newtheorem{theorem}{Theorem}
\\newtheorem{corollary}{Corollary}
\\newtheorem{assumption}{Assumption}
\\newcommand{\\sg}{\\mathrm{sg}}
\\newcommand{\\softmax}{\\mathrm{softmax}}

\\title{A Differentiable Atari VCS: A Complex, Fully Known\\\\
Ground Truth for Explainable AI}

\\author{
    Andreas Maier\\textsuperscript{\\rm 1},\\quad
    Siming Bayer\\textsuperscript{\\rm 1},\\quad
    Patrick Krauss\\textsuperscript{\\rm 1,2}
}
\\affiliations{
    \\textsuperscript{\\rm 1}Pattern Recognition Lab, Friedrich-Alexander-University Erlangen-Nuremberg, Germany\\\\
    \\textsuperscript{\\rm 2}Mannheim Center for Neuromodulation and Neuroprosthetics, Heidelberg University, Germany
}

\\begin{document}
\\maketitle

\\input{body_main.tex}

\\paragraph{Supplementary video.} A narrated walkthrough of the two ports, the
bit-exact side-by-side comparisons, and the gradient study is available at
\\url{${VIDEO_URL}}.

{\\small
\\bibliography{references}
}

\\clearpage
\\appendix
\\section*{Supplementary Material}
\\input{body_supp.tex}

\\end{document}
TEX
echo "  written"

# --- 4. compile (pdflatex + bibtex) ---
step "4/5 compile arxiv.pdf"
( cd "$STAGE" && latexmk -pdf -interaction=nonstopmode arxiv.tex >build.log 2>&1 ) || {
    echo "  COMPILE FAILED -- tail of log:"; tail -25 "$STAGE/build.log"; exit 1; }
ERR=$(grep -cE '^! |LaTeX Error' "$STAGE/build.log" || true)
MULT=$(grep -ic 'multiply defined' "$STAGE/build.log" || true)
PAGES=$(pdfinfo "$STAGE/arxiv.pdf" 2>/dev/null | awk '/Pages/{print $2}')
echo "  errors=$ERR  multiply-defined=$MULT  pages=$PAGES"

# --- verify named, no checklist, supplement present, video linked ---
TXTF="$STAGE/arxiv.txt"; pdftotext "$STAGE/arxiv.pdf" "$TXTF" 2>/dev/null
chk()   { if grep -q "$1" "$TXTF"; then echo "  OK   present : $2"; else echo "  FAIL missing : $2"; FAILED=1; fi; }
chkno() { if grep -q "$1" "$TXTF"; then echo "  FAIL present : $2"; FAILED=1; else echo "  OK   absent  : $2"; fi; }
FAILED=0
chk   'Andreas Maier'            'author Andreas Maier'
chk   'Patrick Krauss'           'author Patrick Krauss'
chk   'Erlangen'                 'affiliation'
chkno 'Anonymous'               'no "Anonymous"'
chkno 'Reproducibility Checklist' 'no reproducibility checklist'
chk   'Supplementary Material'    'supplement appendix'
chk   'Temperature-limit bound'  'supplement theorem content'
chk   'presentation.mp4'         'video link'
chk   'github.com/akmaier/UnderstandingVCS' 'abstract repo link'
chkno 'upon acceptance'          'no "released upon acceptance"'
[[ "$ERR" == 0 && "$FAILED" == 0 ]] || { echo "VERIFICATION FAILED"; exit 1; }

# --- 5. zip the SOURCE for arXiv (arXiv compiles it); also drop a preview PDF ---
step "5/5 zip arXiv source + preview PDF"
rm -f "$OUT_ZIP"
( cd "$STAGE" && zip -rqX "$OUT_ZIP" \
    arxiv.tex body_main.tex body_supp.tex references.bib arxiv.bbl \
    aaai2027.sty aaai2027.bst figures )
cp "$STAGE/arxiv.pdf" "$OUT_PDF"
echo
echo "wrote $OUT_ZIP"
echo "  size:  $(echo "$(stat -f %z "$OUT_ZIP")/1024" | bc) KB   contents:"
unzip -l "$OUT_ZIP" | awk 'NR>3 && $4{print "    "$4}' | head -40
echo "preview PDF: $OUT_PDF  ($PAGES pages)"
