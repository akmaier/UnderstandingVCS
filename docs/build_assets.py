#!/usr/bin/env python3
"""Regenerate the web assets for the results-audit site (docs/).

Reads source figures (PDF) and videos (MP4) from the repo and produces
web-ready PNG / GIF / MP4 under docs/assets/. Idempotent: safe to re-run.

Requires: ffmpeg, pdftoppm (poppler) on PATH.

Run from the repo root:  python3 docs/build_assets.py
"""
import os
import shutil
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(REPO, "docs", "assets")
IMG = os.path.join(ASSETS, "img")
GIF = os.path.join(ASSETS, "gif")
VID = os.path.join(ASSETS, "video")
for d in (IMG, GIF, VID):
    os.makedirs(d, exist_ok=True)


def run(cmd):
    print("  $", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=REPO,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def have(tool):
    return shutil.which(tool) is not None


def sizemb(path):
    return os.path.getsize(path) / 1e6 if os.path.exists(path) else 0.0


# --- 1. PDF figures -> PNG (150 dpi) -------------------------------------
# (src_pdf relative to repo, out_png basename)
FIGURES = [
    # Paper 1
    ("jutari_paper/paper/figures/fig_architecture.pdf", "p1_architecture"),
    ("jutari_paper/paper/figures/fig_pipeline.pdf", "p1_pipeline"),
    ("jutari_paper/paper/figures/fig_timeline.pdf", "p1_timeline"),
    ("jutari_paper/paper/figures/fig_relax_heatmap.pdf", "p1_relax_heatmap"),
    ("jutari_paper/paper/figures/fig_temp_heatmap.pdf", "p1_temp_heatmap"),
    ("jutari_paper/paper/figures/fig_alpha_temp.pdf", "p1_alpha_temp"),
    ("jutari_paper/paper/figures/fig_divergence_frame.pdf", "p1_divergence_frame"),
    ("jutari_paper/paper/figures/gpu_throughput.pdf", "p1_gpu_throughput"),
    ("jutari_paper/paper/figures/si_joystick_gradient.pdf", "p1_si_joystick"),
    # Paper 2
    ("xai_paper/xai_2_interpretability/paper/figures/fig1_platform_oracle.pdf", "p2_fig1_platform_oracle"),
    ("xai_paper/xai_2_interpretability/paper/figures/fig2_faithfulness_vs_plausibility.pdf", "p2_fig2_faithfulness_plausibility"),
    ("xai_paper/xai_2_interpretability/paper/figures/fig3_phaseA_battery.pdf", "p2_fig3_phaseA_battery"),
    ("xai_paper/xai_2_interpretability/paper/figures/fig4_attribution_vs_mechanistic.pdf", "p2_fig4_attribution_mechanistic"),
    ("xai_paper/xai_2_interpretability/paper/figures/fig5_representativeness_map.pdf", "p2_fig5_representativeness"),
    ("xai_paper/xai_2_interpretability/paper/figures/fig6_failure_taxonomy.pdf", "p2_fig6_failure_taxonomy"),
    ("xai_paper/xai_2_interpretability/paper/figures/fig7_sampler_faithful_no_semantics.pdf", "p2_fig7_sampler_faithful"),
]


def build_figures():
    print("[figures] PDF -> PNG")
    if not have("pdftoppm"):
        print("  !! pdftoppm not found, skipping figures")
        return
    for src, base in FIGURES:
        s = os.path.join(REPO, src)
        if not os.path.exists(s):
            print("  -- missing:", src)
            continue
        # pdftoppm appends -1 to single-page output with -singlefile suppressed
        run(["pdftoppm", "-png", "-r", "150", "-singlefile", s,
             os.path.join(IMG, base)])
        print("     %s.png  (%.2f MB)" % (base, sizemb(os.path.join(IMG, base + ".png"))))


# --- 2. comparison MP4 -> looping GIF ------------------------------------
# (src_mp4, out_gif, start_s, dur_s, width)
GIFS = [
    ("tools/breakout_video/output/space_invaders_xitari_vs_jutari.mp4",
     "si_compare", 2, 8, 640),
    ("tools/breakout_video/output/enduro_xitari_vs_jutari.mp4",
     "enduro_compare", 1, 8, 640),
    ("tools/breakout_video/output/seaquest_xitari_vs_jutari.mp4",
     "seaquest_compare", 1, 6, 640),
]


def build_gifs():
    print("[gifs] MP4 -> GIF (palette)")
    if not have("ffmpeg"):
        print("  !! ffmpeg not found, skipping gifs")
        return
    for src, base, ss, dur, w in GIFS:
        s = os.path.join(REPO, src)
        if not os.path.exists(s):
            print("  -- missing:", src)
            continue
        pal = os.path.join(GIF, base + "_palette.png")
        out = os.path.join(GIF, base + ".gif")
        vf = "fps=12,scale=%d:-1:flags=lanczos" % w
        run(["ffmpeg", "-y", "-ss", str(ss), "-t", str(dur), "-i", s,
             "-vf", vf + ",palettegen=stats_mode=diff", pal])
        run(["ffmpeg", "-y", "-ss", str(ss), "-t", str(dur), "-i", s, "-i", pal,
             "-lavfi", vf + " [x]; [x][1:v] paletteuse=dither=bayer:bayer_scale=3",
             "-loop", "0", out])
        if os.path.exists(pal):
            os.remove(pal)
        print("     %s.gif  (%.2f MB)" % (base, sizemb(out)))

    # copy pre-rendered animation gifs from paper 1 figures
    for src, base in [
        ("jutari_paper/paper/figures/fig_temp_anim.gif", "p1_temp_anim"),
        ("jutari_paper/paper/figures/fig_alpha_anim.gif", "p1_alpha_anim"),
    ]:
        s = os.path.join(REPO, src)
        if os.path.exists(s):
            shutil.copy(s, os.path.join(GIF, base + ".gif"))
            print("     copied %s.gif  (%.2f MB)" % (base, sizemb(os.path.join(GIF, base + ".gif"))))


# --- 3. videos -> web MP4 ------------------------------------------------
def build_videos():
    print("[videos] MP4 -> web MP4")
    if not have("ffmpeg"):
        print("  !! ffmpeg not found, skipping videos")
        return
    # small comparison clips: copy as-is (already web-sized, h264)
    for src, base in [
        ("tools/relaxation_study/video_out/divergence_si.mp4", "divergence_si"),
        ("tools/breakout_video/output/space_invaders_xitari_vs_jutari.mp4", "si_compare"),
        ("tools/breakout_video/output/enduro_xitari_vs_jutari.mp4", "enduro_compare"),
        ("tools/breakout_video/output/seaquest_xitari_vs_jutari.mp4", "seaquest_compare"),
        ("tools/breakout_video/output/pitfall_xitari_vs_jutari.mp4", "pitfall_compare"),
    ]:
        s = os.path.join(REPO, src)
        out = os.path.join(VID, base + ".mp4")
        if not os.path.exists(s):
            print("  -- missing:", src)
            continue
        # re-encode to faststart (web streaming) + yuv420p for broad support
        run(["ffmpeg", "-y", "-i", s, "-c:v", "libx264", "-pix_fmt", "yuv420p",
             "-crf", "23", "-preset", "slow", "-movflags", "+faststart", "-an", out])
        print("     %s.mp4  (%.2f MB)" % (base, sizemb(out)))

    # presentation talk: transcode 12 MB -> compact web clip
    s = os.path.join(REPO, "jutari_paper/presentation/presentation.mp4")
    out = os.path.join(VID, "presentation.mp4")
    if os.path.exists(s):
        run(["ffmpeg", "-y", "-i", s, "-vf", "scale=1280:-2", "-c:v", "libx264",
             "-pix_fmt", "yuv420p", "-crf", "30", "-preset", "slow",
             "-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart", out])
        print("     presentation.mp4  (%.2f MB)" % sizemb(out))


def main():
    print("== build_assets.py ==  repo:", REPO)
    build_figures()
    build_gifs()
    build_videos()
    print("done.")


if __name__ == "__main__":
    main()
