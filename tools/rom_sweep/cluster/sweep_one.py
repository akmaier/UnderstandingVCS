"""One-game jaxtari RAM+screen conformance vs precomputed xitari references.

Reuses tools/rom_sweep/sweep_jaxtari.py's _diff unchanged; only the xitari
reference is read from cached trace_dump JSONL (no xitari build on the cluster).
"""
import json, sys, time
from pathlib import Path
import numpy as np

REPO = Path(sys.argv[1]); XREF = Path(sys.argv[2]); OUT = Path(sys.argv[3]); idx = int(sys.argv[4])
sys.path.insert(0, str(REPO / "tools" / "rom_sweep"))
import sweep_jaxtari as sj

def cached(rom, n, screen):
    f = XREF / (f"{rom.stem}_screen60.jsonl" if screen else f"{rom.stem}_ram30.jsonl")
    out = []
    for line in f.read_text().strip().splitlines():
        o = json.loads(line)
        if o.get("boot_end"):
            continue
        if screen:
            out.append(np.frombuffer(bytes.fromhex(o["screen"]), dtype=np.uint8).reshape(o["h"], o["w"]))
        else:
            out.append(np.frombuffer(bytes.fromhex(o["ram"]), dtype=np.uint8))
    return out[:n]
sj._xitari_frames = cached
sj.JAXTARI_TIMEOUT = 6 * 3600   # cluster cores ~2x slower than the M1 (45 min default times out)

game = sj.NAMES[idx]
rom = sj.ROMS_DIR / f"{game}.bin"
res = {"game": game}
for kind, n, screen in (("ram", 30, False), ("screen", 60, True)):
    t0 = time.time()
    try:
        r = sj._diff(rom, n, screen)
    except Exception as e:
        r = (-3, -3, -1, f"ERROR {type(e).__name__}: {e}"[:300])
    res[kind] = list(r) + [round(time.time() - t0)]
    print(game, kind, res[kind], flush=True)
OUT.mkdir(parents=True, exist_ok=True)
(OUT / f"{game}.json").write_text(json.dumps(res))
