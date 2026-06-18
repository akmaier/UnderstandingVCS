#!/usr/bin/env python3
"""Figure: real-ROM Space Invaders joystick gradient via the paper's sampler.
2x2: (a) real 35 s scene; (b) d screen / d RIGHT via the sampler (cannon edges,
identical across the 3 soft variants); (c) the NAIVE gradient through the integer
sprite index = 0 (vanishes); (d) inverse d(move-right)/d joystick -> push RIGHT,
up/down vanish. Run with system python3 (numpy, matplotlib, PIL).
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from breakout_video import decode_palette, load_ntsc_palette

OUT = os.path.join(os.path.dirname(__file__), "out")
def raw(n, dt, sh): return np.fromfile(os.path.join(OUT, n), dt).reshape(sh)

H, W = (int(x) for x in open(f"{OUT}/ji_scene.shape").read().split())
scene = raw("ji_scene.raw", np.uint8, (H, W))
R0, R1, C0, C1 = (int(x) for x in open(f"{OUT}/ji_crop.txt").read().split())
ch, cw = R1 - R0 + 1, C1 - C0 + 1
sal = raw("ji_sal_STE.raw", np.float32, (ch, cw))
naive = raw("ji_naive.raw", np.float32, (ch, cw))

# parse inverse gradients
variants, grads = [], {}
cur = None
for ln in open(f"{OUT}/ji_grad.txt"):
    if ln.startswith("VARIANT"):
        cur = ln.split()[1]; variants.append(cur); grads[cur] = {}
    elif "d(move_right)" in ln:
        d = ln.split("d(")[2].split(")")[0]; grads[cur][d] = float(ln.split("=")[1])

pal = load_ntsc_palette()
rgb = decode_palette(scene, pal).astype(np.float32)

# zoom window around the cannon for (b)/(c)
zr0, zr1, zc0, zc1 = 168, 205, 16, 70
def zoom(img): return img[zr0:zr1, zc0:zc1]

# place a crop map into a full-frame field
def embed(crop):
    f = np.full((H, W), np.nan, np.float32); f[R0:R1 + 1, C0:C1 + 1] = crop; return f
sal_f, naive_f = embed(sal), embed(naive)

plt.rcParams.update({"xtick.labelsize": 8, "ytick.labelsize": 8, "axes.labelsize": 9})
fig, ax = plt.subplots(2, 2, figsize=(4.7, 4.1))      # tuned for single-column width
vmax = np.nanmax(np.abs(sal)) + 1e-6

# (a) real scene
ax[0, 0].imshow(rgb.astype(np.uint8), interpolation="none")
ax[0, 0].set_title("(a) real SI, 35 s", fontsize=9)

# (b) sampler saliency over dimmed zoom — alpha-masked so zeros are transparent
def overlay(a, field, title, cmap, vmx):
    import matplotlib.cm as cm
    base = (zoom(rgb) * 0.45).astype(np.uint8)
    a.imshow(base, interpolation="none", extent=[zc0, zc1, zr1, zr0])
    z = zoom(field)
    norm = np.clip((z + vmx) / (2 * vmx), 0, 1)
    rgba = cm.get_cmap(cmap)(np.nan_to_num(norm))
    rgba[..., 3] = np.where(np.isnan(z), 0.0, np.clip(np.abs(z) / vmx, 0, 1))  # alpha=|sal|
    a.imshow(rgba, interpolation="none", extent=[zc0, zc1, zr1, zr0])
    a.set_title(title, fontsize=9); a.set_xticks([]); a.set_yticks([])

overlay(ax[0, 1], sal_f, r"(b) $\partial$screen$/\partial$RIGHT (sampler)",
        "RdBu_r", vmax)
overlay(ax[1, 0], naive_f, r"(c) naive $\equiv$ 0 (vanishes)", "RdBu_r", vmax)

# (d) inverse bar chart (3 variants x 4 directions)
dirs = ["up", "down", "left", "right"]
x = np.arange(4); wbar = 0.26
show = ["STE", "relax_a6_T0.14", "naive"]      # soft-STE, soft, naive (per request)
nice = {"STE": "SOFT-STE", "relax_a6_T0.14": "soft", "naive": "naive"}
for i, v in enumerate(show):
    vals = [grads[v][d] for d in dirs]
    ax[1, 1].bar(x + (i - 1) * wbar, vals, wbar, label=nice.get(v, v))
ax[1, 1].axhline(0, color="k", lw=0.6)
ax[1, 1].set_xticks(x); ax[1, 1].set_xticklabels(dirs)
ax[1, 1].set_title(r"(d) inverse $\partial$(move right)$/\partial$joy", fontsize=9)
ax[1, 1].set_ylabel("gradient"); ax[1, 1].legend(fontsize=7, loc="upper left")

for a in (ax[0, 0],): a.set_xticks([]); a.set_yticks([])
fig.tight_layout()
fig.savefig(os.path.join(OUT, "si_joystick_gradient.png"), dpi=150)
fig.savefig(os.path.join(OUT, "si_joystick_gradient.pdf"))
print("wrote si_joystick_gradient.png/.pdf  | sampler max|sal|=%.2f  inverse right=%.2f"
      % (vmax, grads[variants[0]]["right"]))
