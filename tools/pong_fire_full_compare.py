"""Dump full RAM at frames 18, 19, 20, 21 from both jutari and xitari.
Find ALL bytes that differ + characterize the FIRE-handling divergence."""
import json, subprocess, numpy as np, sys
from pathlib import Path
REPO = Path("/Users/maier/Documents/code/UnderstandingVCS")

ACTIONS = REPO / "tools/breakout_video/output/pong_breakout_random_actions.txt"
ROM     = REPO / "xitari/roms/pong.bin"

actions = [int(l.strip()) for l in ACTIONS.read_text().splitlines() if l.strip() and not l.startswith("#")][:25]

# xitari trace
acts_path = "/tmp/pong_actions_25.txt"
with open(acts_path, "w") as f:
    for a in actions:
        f.write(f"{a}\n")
r = subprocess.run([str(REPO/"tools/trace_dump"), "--rom", str(ROM),
                    "--actions", acts_path, "--max-frames", "25"],
                   capture_output=True, text=True)
xi_rams = [np.frombuffer(bytes.fromhex(json.loads(l)["ram"]), dtype=np.uint8)
           for l in r.stdout.strip().splitlines()]

# jutari trace
import tempfile
trace_path = "/tmp/pong_actions_25_trace.jsonl"
with open(trace_path, "w") as f:
    for i, a in enumerate(actions):
        f.write(f'{{"frame": {i}, "action": {a}, "ram": ""}}\n')

j = subprocess.run(["julia", "--project=" + str(REPO/"jutari"),
                    str(REPO/"tools/jutari_trace_dump.jl"),
                    "--rom", str(ROM), "--trace", trace_path,
                    "--out", "/tmp/pong_25_jutari.jsonl"],
                   capture_output=True, text=True)
ju_rams = []
with open("/tmp/pong_25_jutari.jsonl") as f:
    for line in f:
        obj = json.loads(line)
        ju_rams.append(np.frombuffer(bytes.fromhex(obj["ram"]), dtype=np.uint8))

# Compare frame-by-frame around FIRE
for f in (18, 19, 20, 21):
    if f >= len(xi_rams) or f >= len(ju_rams): continue
    xi_r, ju_r = xi_rams[f], ju_rams[f]
    diffs = np.where(xi_r != ju_r)[0]
    print(f"\nframe {f} (action={actions[f]}): {len(diffs)} bytes differ")
    for off in diffs:
        print(f"  ${off:02x}: xi={int(xi_r[off]):#04x}  ju={int(ju_r[off]):#04x}")

# Also: what BYTES did each port change between frame 19 and frame 20?
print("\n=== changes during FIRE step (frame 19 -> 20) ===")
xi_changed = np.where(xi_rams[19] != xi_rams[20])[0]
ju_changed = np.where(ju_rams[19] != ju_rams[20])[0]
print(f"\nxitari changed {len(xi_changed)} bytes:")
for off in xi_changed:
    print(f"  ${off:02x}: {int(xi_rams[19][off]):#04x} -> {int(xi_rams[20][off]):#04x}")
print(f"\njutari changed {len(ju_changed)} bytes:")
for off in ju_changed:
    print(f"  ${off:02x}: {int(ju_rams[19][off]):#04x} -> {int(ju_rams[20][off]):#04x}")
