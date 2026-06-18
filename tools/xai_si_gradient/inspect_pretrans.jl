# Inspect the PRE-transition (classic-colour) SI state near 35 s to pick a scene
# matching the movie, and characterise the ~35 s transition (game event?).
# Run: cd jutari && julia --project=. ../tools/xai_si_gradient/inspect_pretrans.jl
using JuTari
using JuTari.Env: StellaEnvironment, env_reset!, env_step!, get_screen, get_ram
const NOOP = 0; const RIGHT = 3
rom_bytes = read(joinpath(@__DIR__, "..", "..", "xitari", "roms", "space_invaders.bin"))
outdir = joinpath(@__DIR__, "out"); _cflat(s) = vec(permutedims(s))

# invader formation pixel count (cols 40-130, rows 25-150) + lowest invader row
function inv_stats(s)
    band = s[25:150, 40:130]; rs = findall(!=(0), band)
    (length(rs), isempty(rs) ? 0 : maximum(c[1] for c in rs) + 24)
end
e = StellaEnvironment(rom_bytes); env_reset!(e; boot_noop_steps=60, boot_reset_steps=4)
println("frame  t(s)  bg   inv_px  inv_lowrow  cannon_band_px  cannon_colors")
for f in 1:2110
    env_step!(e, NOOP); s = get_screen(e)
    if f in (1500,1800,2080,2085,2090,2095,2100,2103,2106,2110)
        ip, il = inv_stats(s)
        cb = s[181:194, 26:60]; cc = sort(unique(Int.(cb[cb .!= 0])))
        println(lpad(f,5),"  ",lpad(round(f/60,digits=1),4),"  ",lpad(Int(s[60,3]),3),
                "  ",lpad(ip,5),"  ",lpad(il,6),"  ",lpad(count(!=(0),cb),6),"   ",cc)
    end
end
# dump a clean classic frame (2090) for the figure
e2 = StellaEnvironment(rom_bytes); env_reset!(e2; boot_noop_steps=60, boot_reset_steps=4)
for _ in 1:2090; env_step!(e2, NOOP); end
s90 = copy(get_screen(e2)); h,w = size(s90)
write(joinpath(outdir,"pre_scene.raw"), UInt8.(_cflat(s90)))
write(joinpath(outdir,"pre_scene.shape"), "$h $w")
# cannon footprint via finite difference: move RIGHT a few frames, diff bottom band
er = StellaEnvironment(rom_bytes); env_reset!(er; boot_noop_steps=60, boot_reset_steps=4)
for _ in 1:2090; env_step!(er, NOOP); end
base = copy(get_screen(er))
for _ in 1:9; env_step!(er, RIGHT); end
mv = get_screen(er)
chg = [(r,c) for r in 178:198 for c in 24:62 if base[r,c]!=mv[r,c]]
println("\nframe 2090 player-X=", Int(get_ram(e2)[29]))
if !isempty(chg)
    rs=[r for (r,c) in chg]; cs=[c for (r,c) in chg]
    println("cannon-motion bbox rows ", minimum(rs),"-",maximum(rs),
            " cols ", minimum(cs),"-",maximum(cs))
    # cannon colours at base in that bbox
    cols = sort(unique(Int(base[r,c]) for (r,c) in chg if base[r,c]!=0))
    println("colours of moving (cannon) pixels at base: ", cols)
end
