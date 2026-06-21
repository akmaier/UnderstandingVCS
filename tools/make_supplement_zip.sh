#!/usr/bin/env bash
# Build the anonymized AAAI-27 supplementary-material zip from a clean git export.
#
# Produces  jutari_paper/supplement_anonymous.zip  containing the two ports, the
# conformance/benchmark tooling, the supplement PDF, the narrated video, the GPU
# results, an artifact README, and the ROM hash manifest -- with every author
# identifier scrubbed (double-blind). The working repo is NOT modified: the real
# repo keeps its real attribution (released with the author's name on publication);
# only the EXPORT is anonymized. The script FAILS if any identity string survives,
# so a leaky zip can never be produced.
#
# Usage:  bash tools/make_supplement_zip.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
OUT="$ROOT/jutari_paper/supplement_anonymous.zip"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
step() { echo; echo "=== $* ==="; }

# --- 1. export a CURATED set of tracked files (no .git, no ROMs, no dev docs) ---
step "1/5 export curated tracked files"
git archive --format=tar HEAD -- \
    jutari jaxtari tools \
    results/gpu \
    jutari_paper/paper/supplementary.pdf \
    jutari_paper/presentation/presentation.mp4 \
    LICENSE \
  | tar -x -C "$STAGE"
# drop anything heavy/irrelevant that lives under the included trees, and the
# packaging script itself (meta-tooling, not part of the artifact -- and its own
# grep pattern literally contains identity words, which would trip the leak scan).
rm -f  "$STAGE"/tools/make_supplement_zip.sh 2>/dev/null || true
rm -rf "$STAGE"/tools/comparison_videos "$STAGE"/jaxtari/.venv "$STAGE"/**/__pycache__ 2>/dev/null || true
find "$STAGE" -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true
find "$STAGE" -name '.DS_Store' -delete 2>/dev/null || true
echo "  staged $(find "$STAGE" -type f | wc -l | tr -d ' ') files"

# --- 2. surface the ROM hash manifest at top level (ROMs themselves are NOT shipped) ---
step "2/5 ROM hash manifest"
if [[ -f "$STAGE/tools/rom_sweep/manifest.txt" ]]; then
    cp "$STAGE/tools/rom_sweep/manifest.txt" "$STAGE/ROMS.sha256"
    echo "  copied tools/rom_sweep/manifest.txt -> ROMS.sha256"
else
    echo "  (no manifest found; README points to the supplement's per-game table)"
fi

# --- 3. anonymize the EXPORT in place (text files only) ---
step "3/5 anonymize export (author name / handle / absolute paths)"
# LICENSE copyright holder
perl -0pi -e 's/Copyright \(c\) \d{4} .*/Copyright (c) 2026 Anonymous (under double-blind review)/' "$STAGE/LICENSE" 2>/dev/null || true
# all text files: scrub paths, handle, name (longest path first)
find "$STAGE" -type f \( -name '*.py' -o -name '*.jl' -o -name '*.toml' -o -name '*.sh' \
    -o -name '*.sbatch' -o -name '*.md' -o -name '*.txt' -o -name '*.cfg' -o -name '*.ini' \
    -o -name '*.yml' -o -name '*.yaml' -o -name '*.jsonl' -o -name '*.json' -o -name 'LICENSE' \) -print0 \
  | xargs -0 perl -pi -e '
      s{/Users/maier/Documents/code/UnderstandingVCS}{/path/to/repo}g;
      s{/cluster/maier}{/path/to/scratch}g;
      s{/Users/maier}{/path/to/repo}g;
      s{/home/maier}{/path/to/repo}g;
      s{\bakmaier\b}{anonymous}g;
      s{Andreas\s+Maier}{Anonymous}g;
      s{\bmaier\b}{anonymous}gi;
  '
echo "  scrubbed"

# --- 4. artifact README (anonymous) ---
step "4/5 write ARTIFACT_README.md"
cat > "$STAGE/ARTIFACT_README.md" <<'README'
# Supplementary material — A Differentiable Atari VCS

Anonymized artifact for double-blind review. Contents:

- `jutari/`  — the Julia / Zygote differentiable VCS port.
- `jaxtari/` — the JAX / XLA differentiable VCS port.
- `tools/`   — conformance harnesses (xitari-trace comparison, 64-ROM sweeps),
  throughput benchmarks, and the deterministic action-stream generators.
- `results/gpu/` — GPU throughput measurements behind the supplement's tables.
- `supplementary.pdf` — proofs, relaxation study, per-game conformance table,
  throughput tables (the technical appendix).
- `presentation.mp4` — narrated walkthrough (synthetic voice; anonymous).
- `ROMS.sha256` — SHA-256 of every cartridge used.

## ROMs are NOT included

The Atari 2600 cartridge ROMs are copyrighted and are not redistributed. Obtain
the standard set via ALE / AutoROM; identities are pinned by `ROMS.sha256` (and
the per-game table in `supplementary.pdf`). Place them where the harness expects
(see `tools/rom_sweep/`).

## Running

- jutari:  Julia 1.10+, then `cd jutari && julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.test()'`.
- jaxtari: Python 3.12, then `cd jaxtari && pip install -e . && pytest`.
- Conformance sweeps and benchmarks: see scripts under `tools/`.

Released under the MIT License upon publication.
README
echo "  written"

# --- 5. leak scan (FAIL if any identifier survives), then zip ---
step "5/5 leak scan + zip"
# A real cluster-home leak (/cluster/maier) is caught by \bmaier\b; the repo's own
# tools/cluster/ dir and the generic /cluster/<user> placeholder are not identity.
# Hostnames are caught by \blme[0-9]+\b.
HITS=$(grep -rInE 'andreas|akmaier|\bmaier\b|fau\.de|erlangen|/Users/|/home/[a-z]|\blme[0-9]+\b|Co-Authored' "$STAGE" || true)
if [[ -n "$HITS" ]]; then
    echo "FATAL: identity strings still present in the export:"; echo "$HITS" | head -40
    echo "(zip NOT written)"; exit 1
fi
echo "  leak scan CLEAN (no author/handle/host/path identifiers)"

rm -f "$OUT"
( cd "$STAGE" && zip -rqX "$OUT" . )
echo
echo "wrote $OUT"
echo "  size:  $(echo "$(stat -f %z "$OUT")/1048576" | bc -l | cut -c1-5) MB"
echo "  files: $(unzip -l "$OUT" | tail -1 | awk '{print $2}')"
echo "  top-level:"; unzip -l "$OUT" | awk 'NR>3{print $4}' | sed '/^$/d' | cut -d/ -f1 | sort -u | sed 's/^/    /'
