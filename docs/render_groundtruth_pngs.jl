# render_groundtruth_pngs.jl — one screenshot PNG per T3-labeled game for the
# "Ground Truth ROMs" website page. Renders each of the 54 XAI_LABELED games at
# the SHARED ANALYSIS STATE the ground truth is computed on: the seed=0
# random-action gameplay stream advanced to frame 90 (prefix=90, horizon=15).
#
# Uses the recommended composite path from tools/xai_study/common/gameplay_state.jl:
#   gameplay_actions(game; prefix=90, horizon=15, seed=0) -> actions
#   boot_gameplay(game, actions, 90)                      -> env AT frame 90
#   get_screen(env)                                       -> 210×160 palette idx
#
# Palette: xitari's ourNTSCPalette (Stella NTSC), lifted from
# tools/breakout_video/__init__.py (128 colours at even indices, odd carries the
# previous even entry). indices → RGB → PPM → PNG via macOS `sips` (2× nearest).
#
# SINGLE Julia process, no parallelism (MacBook is busy).

using JuTari
using JuTari.Env: env_step!, get_screen

const REPO = normpath(joinpath(@__DIR__, ".."))
include(joinpath(REPO, "tools", "xai_study", "common", "game_sets.jl"))
include(joinpath(REPO, "tools", "xai_study", "common", "gameplay_state.jl"))
using .GameplayState: gameplay_actions, boot_gameplay

const OUTDIR = joinpath(REPO, "docs", "assets", "groundtruth")
mkpath(OUTDIR)

# ---- Atari NTSC palette (xitari ourNTSCPalette) — 128 even entries -----------
const PALETTE_PAIRS = UInt32[
    0x000000, 0x4a4a4a, 0x6f6f6f, 0x8e8e8e, 0xaaaaaa, 0xc0c0c0, 0xd6d6d6, 0xececec,
    0x484800, 0x69690f, 0x86861d, 0xa2a22a, 0xbbbb35, 0xd2d240, 0xe8e84a, 0xfcfc54,
    0x7c2c00, 0x904811, 0xa26221, 0xb47a30, 0xc3903d, 0xd2a44a, 0xdfb755, 0xecc860,
    0x901c00, 0xa33915, 0xb55328, 0xc66c3a, 0xd5824a, 0xe39759, 0xf0aa67, 0xfcbc74,
    0x940000, 0xa71a1a, 0xb83232, 0xc84848, 0xd65c5c, 0xe46f6f, 0xf08080, 0xfc9090,
    0x840064, 0x97197a, 0xa8308f, 0xb846a2, 0xc659b3, 0xd46cc3, 0xe07cd2, 0xec8ce0,
    0x500084, 0x68199a, 0x7d30ad, 0x9246c0, 0xa459d0, 0xb56ce0, 0xc57cee, 0xd48cfc,
    0x140090, 0x331aa3, 0x4e32b5, 0x6848c6, 0x7f5cd5, 0x956fe3, 0xa980f0, 0xbc90fc,
    0x000094, 0x181aa7, 0x2d32b8, 0x4248c8, 0x545cd6, 0x656fe4, 0x7580f0, 0x8490fc,
    0x001c88, 0x183b9d, 0x2d57b0, 0x4272c2, 0x548ad2, 0x65a0e1, 0x75b5ef, 0x84c8fc,
    0x003064, 0x185080, 0x2d6d98, 0x4288b0, 0x54a0c5, 0x65b7d9, 0x75cceb, 0x84e0fc,
    0x004030, 0x18624e, 0x2d8169, 0x429e82, 0x54b899, 0x65d1ae, 0x75e7c2, 0x84fcd4,
    0x004400, 0x1a661a, 0x328432, 0x48a048, 0x5cba5c, 0x6fd26f, 0x80e880, 0x90fc90,
    0x143c00, 0x355f18, 0x527e2d, 0x6e9c42, 0x87b754, 0x9ed065, 0xb4e775, 0xc8fc84,
    0x303800, 0x505916, 0x6d762b, 0x88923e, 0xa0ab4f, 0xb7c25f, 0xccd86e, 0xe0ec7c,
    0x482c00, 0x694d14, 0x866a26, 0xa28638, 0xbb9f47, 0xd2b656, 0xe8cc63, 0xfce070,
]

# 256×3 RGB lookup: even idx = ourNTSCPalette entry; odd idx carries previous even.
function build_palette()
    pal = zeros(UInt8, 256, 3)
    for (i, rgb) in enumerate(PALETTE_PAIRS)
        idx = (i - 1) * 2            # 0-based even palette index
        r = UInt8((rgb >> 16) & 0xFF); g = UInt8((rgb >> 8) & 0xFF); b = UInt8(rgb & 0xFF)
        pal[idx + 1, 1] = r; pal[idx + 1, 2] = g; pal[idx + 1, 3] = b
        if idx + 1 < 256            # carry to odd entry
            pal[idx + 2, 1] = r; pal[idx + 2, 2] = g; pal[idx + 2, 3] = b
        end
    end
    return pal
end
const PAL = build_palette()

# screen (H×W palette indices) → binary PPM (P6) at `scale`× nearest-neighbor.
function write_ppm(path::AbstractString, screen::AbstractMatrix{<:Integer}; scale::Int = 2)
    H, W = size(screen)
    OW = W * scale; OH = H * scale
    open(path, "w") do io
        write(io, "P6\n$OW $OH\n255\n")
        buf = Vector{UInt8}(undef, OW * 3)
        for r in 1:H
            for _ in 1:scale
                k = 1
                for c in 1:W
                    idx = Int(screen[r, c]) & 0xFF
                    rr = PAL[idx + 1, 1]; gg = PAL[idx + 1, 2]; bb = PAL[idx + 1, 3]
                    for _ in 1:scale
                        buf[k] = rr; buf[k+1] = gg; buf[k+2] = bb; k += 3
                    end
                end
                write(io, buf)
            end
        end
    end
end

const PREFIX  = 90
const HORIZON = 15
const SEED    = 0

ok = String[]; failed = Tuple{String,String}[]
for game in XAI_LABELED
    try
        actions = gameplay_actions(game; prefix = PREFIX, horizon = HORIZON, seed = SEED)
        env = boot_gameplay(game, actions, PREFIX)
        screen = get_screen(env)                      # 210×160 palette indices
        ppm = joinpath(OUTDIR, game * ".ppm")
        png = joinpath(OUTDIR, game * ".png")
        write_ppm(ppm, screen; scale = 2)
        run(`sips -s format png $ppm --out $png`)     # PPM → PNG
        rm(ppm; force = true)
        push!(ok, game)
        println("OK   $game  $(size(screen,1))x$(size(screen,2)) -> $(basename(png))")
    catch err
        msg = sprint(showerror, err)
        push!(failed, (game, first(split(msg, '\n'))))
        println("FAIL $game  $(first(split(msg, '\n')))")
    end
end

println("\n==== SUMMARY ====")
println("wrote $(length(ok))/$(length(XAI_LABELED)) PNGs to $OUTDIR")
println("frame/state: seed=$SEED random-action gameplay stream, prefix=$PREFIX (frame 90), horizon=$HORIZON")
if !isempty(failed)
    println("FAILED:")
    for (g, m) in failed; println("  $g : $m"); end
end
