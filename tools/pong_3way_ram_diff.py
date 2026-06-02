"""Compare RAM per frame: jaxtari vs jutari vs xitari, pong random actions.
Find first divergence point and characterize how each port disagrees.

Usage:
    # First generate the jutari trace (one-time / refresh after a jutari change):
    python3 -c "
    with open('tools/breakout_video/output/pong_breakout_random_actions.txt') as f:
        actions = [int(l.strip()) for l in f if l.strip()]
    with open('/tmp/pong_actions_trace.jsonl', 'w') as f:
        for i, a in enumerate(actions[:200]):
            f.write(f'{{\"frame\": {i}, \"action\": {a}, \"ram\": \"\"}}\n')
    "
    julia --project=jutari tools/jutari_trace_dump.jl --rom xitari/roms/pong.bin \\
        --trace /tmp/pong_actions_trace.jsonl \\
        --out   /tmp/pong_jutari_rams.jsonl

    # Then run this probe from inside jaxtari/ (so .venv resolves):
    cd jaxtari && .venv/bin/python ../tools/pong_3way_ram_diff.py

Key finding (2026-06-02): the two ports are bit-identical through frame 58,
then jaxtari diverges from jutari at frame 59 (action=4=LEFT, 6 bytes diff:
$04, $08, $0c, $31, $36, $3b). Jutari then continues tracking xitari to
within ~4 bytes/frame, while jaxtari drifts to 15-18 bytes/frame. PXC2
(NOOP-10) misses this because random-action paths are never exercised.
Both ports also diverge from xitari at frame 20 (FIRE) by 2 bytes ($3f, $40
SWAPPED — xitari has 0xc0, 0x00; both ports have 0x00, 0xc0). See
`bug_fix_log.md` "Where we left off" section."""
import json, subprocess, numpy as np, sys
from pathlib import Path

REPO = Path("/Users/maier/Documents/code/UnderstandingVCS")
sys.path.insert(0, str(REPO / "jaxtari"))

from jaxtari.env.stella_environment import StellaEnvironment
from jaxtari.games.pong import PongRomSettings

ACTIONS = REPO / "tools/breakout_video/output/pong_breakout_random_actions.txt"
ROM     = REPO / "xitari/roms/pong.bin"
N       = 200

actions = [int(l.strip()) for l in ACTIONS.read_text().splitlines() if l.strip() and not l.startswith("#")][:N]

# jaxtari
rom = np.frombuffer(ROM.read_bytes(), dtype=np.uint8)
env = StellaEnvironment(rom, PongRomSettings())
env.reset(boot_noop_steps=60, boot_reset_steps=4)
jx_rams = []
for a in actions:
    env.step(int(a))
    jx_rams.append(np.asarray(env.get_ram(), dtype=np.uint8).copy())

# jutari (already captured above)
ju_rams = []
with open("/tmp/pong_jutari_rams.jsonl") as f:
    for line in f:
        obj = json.loads(line)
        ju_rams.append(np.frombuffer(bytes.fromhex(obj["ram"]), dtype=np.uint8))

# xitari
acts_path = "/tmp/pong_actions_plain.txt"
with open(acts_path, "w") as f:
    for a in actions:
        f.write(f"{a}\n")
r = subprocess.run([str(REPO/"tools/trace_dump"), "--rom", str(ROM),
                    "--actions", acts_path, "--max-frames", str(N)],
                   capture_output=True, text=True)
xi_rams = [np.frombuffer(bytes.fromhex(json.loads(l)["ram"]), dtype=np.uint8)
           for l in r.stdout.strip().splitlines()]

print(f"jaxtari: {len(jx_rams)} frames")
print(f"jutari : {len(ju_rams)} frames")
print(f"xitari : {len(xi_rams)} frames")

n = min(len(jx_rams), len(ju_rams), len(xi_rams))
# Per-frame: count divergences in each pair
print(f"\n{'frame':>5s} | jx==ju | jx-xi | ju-xi | action")
print("-" * 60)
for i in range(n):
    jx_ju = int((jx_rams[i] != ju_rams[i]).sum())
    jx_xi = int((jx_rams[i] != xi_rams[i]).sum())
    ju_xi = int((ju_rams[i] != xi_rams[i]).sum())
    if i < 30:
        print(f"{i:5d} | {jx_ju:>6d} | {jx_xi:>5d} | {ju_xi:>5d} | {actions[i]}")

# Find first frame jaxtari != jutari
print("\n=== first divergences ===")
for i in range(n):
    if not np.array_equal(jx_rams[i], ju_rams[i]):
        diffs = np.where(jx_rams[i] != ju_rams[i])[0]
        print(f"\nFIRST jaxtari!=jutari at frame {i} action={actions[i]}: {len(diffs)} bytes at offsets {diffs.tolist()[:15]}")
        for off in diffs[:10]:
            print(f"  ${off:02x}: jx={int(jx_rams[i][off]):#04x} ju={int(ju_rams[i][off]):#04x} xi={int(xi_rams[i][off]):#04x}")
        break

for i in range(n):
    if not np.array_equal(ju_rams[i], xi_rams[i]):
        diffs = np.where(ju_rams[i] != xi_rams[i])[0]
        print(f"\nFIRST jutari!=xitari at frame {i} action={actions[i]}: {len(diffs)} bytes at offsets {diffs.tolist()[:15]}")
        for off in diffs[:10]:
            print(f"  ${off:02x}: jx={int(jx_rams[i][off]):#04x} ju={int(ju_rams[i][off]):#04x} xi={int(xi_rams[i][off]):#04x}")
        break
