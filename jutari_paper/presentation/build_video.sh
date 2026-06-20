#!/usr/bin/env bash
# Build the ~5-min anonymous supplementary video from the beamer deck, the
# narration script, the comparison clips, and the relaxation animations.
# See make_video.md. Output: presentation.mp4 (anonymous, metadata-stripped).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

VID="../../tools/comparison_videos/output"   # xitari-vs-port comparison clips
FIG="../paper/figures"                        # relaxation GIFs
W=1920; H=1080; FPS=30
GRID_GAMES="space_invaders seaquest enduro"   # segment 6 grid cycle
GRID_SECS=10                                   # per-game grid length (jaxtari clips are 10 s)
ONE_LARGE="space_invaders_xitari_vs_jutari"    # segment 5 single clip

# black-padded 16:9 normaliser for clips; white-padded for slides
PAD_BLACK="scale=${W}:${H}:force_original_aspect_ratio=decrease,pad=${W}:${H}:(ow-iw)/2:(oh-ih)/2:black,fps=${FPS},format=yuv420p,setsar=1"
PAD_WHITE="scale=${W}:${H}:force_original_aspect_ratio=decrease,pad=${W}:${H}:(ow-iw)/2:(oh-ih)/2:white,fps=${FPS},format=yuv420p,setsar=1"
VENC=(-c:v libx264 -preset medium -crf 20 -pix_fmt yuv420p -r "$FPS")
AENC=(-c:a aac -ar 48000 -ac 2 -b:a 192k)
FF=(ffmpeg -hide_banner -loglevel error -y -nostdin)   # -nostdin: do not eat the while-read manifest

TTS_VENV="${TTS_VENV:-$HOME/venvs/chatterbox}"
if [[ -f "$TTS_VENV/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "$TTS_VENV/bin/activate"
    echo "activated venv: $TTS_VENV ($(which python3))"
else
    echo "FATAL: Chatterbox venv not found at $TTS_VENV"; exit 2
fi

mkdir -p build
# always rebuild the cheap mux/concat outputs; rebuild assets (slides, clips,
# narration) only when REBUILD_ASSETS=1 (default). Set REBUILD_ASSETS=0 to reuse
# the existing build/ artifacts and re-run just the mux + concat (skips ~4-min TTS).
REBUILD_ASSETS="${REBUILD_ASSETS:-1}"
rm -f build/seg-*.mp4 build/concat.txt presentation.mp4
step() { echo; echo "=== $* ==="; }

if [[ "$REBUILD_ASSETS" == 1 ]]; then
rm -f build/page-*.png build/clip-*.mp4 build/grid_*.mp4 build/slide-*.txt build/slide-*.wav

# ---------------------------------------------------------------------------
step "0/8 lint script (Chatterbox: short sentences, no - ; : in spoken text)"
python3 - script.md <<'PY'
import re, sys
bad = 0
for ln in open(sys.argv[1]):
    s = ln.rstrip("\n")
    if not s or s.startswith("#"):
        continue
    for sent in re.split(r'(?<=[.!?])\s+', s.strip()):
        if not sent:
            continue
        n = len(sent.split())
        if n > 20:
            print(f"  WARN long sentence ({n} w): {sent[:70]}..."); bad += 1
        if re.search(r'[;:—]|--', sent):
            print(f"  WARN punctuation trigger: {sent[:70]}..."); bad += 1
print(f"  lint: {bad} warning(s)" + ("" if bad else " -- clean"))
PY

# ---------------------------------------------------------------------------
step "1/8 compile beamer deck"
latexmk -pdf -interaction=nonstopmode presentation.tex >/dev/null 2>&1 || {
    echo "latexmk failed; see presentation.log"; exit 1; }

step "2/8 render beamer pages to PNG"
pdftoppm -png -r 200 presentation.pdf build/page
i=1
for f in build/page-[0-9]*.png; do
    base=$(printf "build/page-%02d.png" "$i")
    [[ "$f" != "$base" ]] && mv "$f" "$base"
    i=$((i+1))
done
echo "  rendered $((i-1)) pages"

# ---------------------------------------------------------------------------
step "3/8 segment 5 clip: one game large"
"${FF[@]}" -i "$VID/${ONE_LARGE}.mp4" -filter_complex "[0:v]${PAD_BLACK}[o]" \
    -map "[o]" -an "${VENC[@]}" build/clip-05.mp4

step "4/8 segment 6 clip: stacked jutari-top / jaxtari-bottom grid"
: > build/grid_concat.txt
for g in $GRID_GAMES; do
    top="$VID/${g}_xitari_vs_jutari.mp4"
    bot="$VID/${g}_xitari_vs_jaxtari.mp4"
    [[ -f "$top" && -f "$bot" ]] || { echo "  skip $g (missing clip)"; continue; }
    "${FF[@]}" -t "$GRID_SECS" -i "$top" -t "$GRID_SECS" -i "$bot" \
        -filter_complex "[0:v][1:v]vstack=inputs=2[v];[v]${PAD_BLACK}[o]" \
        -map "[o]" -an "${VENC[@]}" "build/grid_${g}.mp4"
    echo "file 'grid_${g}.mp4'" >> build/grid_concat.txt
done
"${FF[@]}" -f concat -safe 0 -i build/grid_concat.txt -c copy build/clip-06.mp4

step "5/8 segment 8 clip: relaxation animations (GIFs, looped)"
"${FF[@]}" -ignore_loop 0 -i "$FIG/fig_alpha_anim.gif" -ignore_loop 0 -i "$FIG/fig_temp_anim.gif" \
    -filter_complex "[0:v]scale=-2:400[a];[1:v]scale=-2:400[b];[a][b]hstack=inputs=2[h];[h]${PAD_BLACK}[o]" \
    -map "[o]" -an -t 45 "${VENC[@]}" build/clip-08.mp4

# ---------------------------------------------------------------------------
step "6/8 split script + synthesise narration (Chatterbox default voice)"
python3 split_script.py script.md build
python3 tts_chatterbox.py build "$@"
fi   # end REBUILD_ASSETS

# ---------------------------------------------------------------------------
step "7/8 mux each segment (visual + matched narration)"
: > build/concat.txt
while IFS=$'\t' read -r nn type src; do
    [[ "$nn" =~ ^# ]] && continue
    [[ -z "${nn:-}" ]] && continue
    wav="build/slide-${nn}.wav"
    out="build/seg-${nn}.mp4"
    [[ -f "$wav" ]] || { echo "  MISSING $wav"; exit 1; }
    if [[ "$type" == "slide" ]]; then
        "${FF[@]}" -loop 1 -i "build/${src}.png" -i "$wav" \
            -filter_complex "[0:v]${PAD_WHITE}[v]" -map "[v]" -map 1:a \
            -tune stillimage "${VENC[@]}" "${AENC[@]}" -shortest "$out"
    else
        "${FF[@]}" -stream_loop -1 -i "build/${src}.mp4" -i "$wav" \
            -map 0:v:0 -map 1:a:0 "${VENC[@]}" "${AENC[@]}" -shortest "$out"
    fi
    echo "file 'seg-${nn}.mp4'" >> build/concat.txt
    printf "  seg %s (%s) %.1fs\n" "$nn" "$type" \
        "$(ffprobe -v error -show_entries format=duration -of default=nk=1:nw=1 "$out")"
done < segments.tsv

# ---------------------------------------------------------------------------
step "8/8 concatenate + strip metadata (anonymity)"
"${FF[@]}" -f concat -safe 0 -i build/concat.txt -map_metadata -1 \
    "${VENC[@]}" "${AENC[@]}" -movflags +faststart presentation.mp4

dur=$(ffprobe -v error -show_entries format=duration -of default=nk=1:nw=1 presentation.mp4)
sz=$(stat -f %z presentation.mp4)
echo
printf "wrote presentation.mp4  %.0f s  %.1f MB\n" "$dur" "$(echo "$sz/1048576" | bc -l)"
echo "metadata check (should show no author/title):"
ffprobe -v error -show_entries format_tags -of default=nk=0 presentation.mp4 || true
