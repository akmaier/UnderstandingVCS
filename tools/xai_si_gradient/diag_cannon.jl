# Locate the player cannon on the real 35 s screen unambiguously: dump base,
# RIGHT-held (+N) and NOOP-held (+N) frames so we can see the sprite and the
# exact pixels that move. Run:
#   cd jutari && julia --project=. ../tools/xai_si_gradient/diag_cannon.jl
using JuTari
using JuTari.Env: StellaEnvironment, env_reset!, env_step!, get_screen, get_ram
const NOOP = 0; const RIGHT = 3
rom_bytes = read(joinpath(@__DIR__, "..", "..", "xitari", "roms", "space_invaders.bin"))
outdir = joinpath(@__DIR__, "out"); isdir(outdir) || mkpath(outdir)
_cflat(s) = vec(permutedims(s))
scene() = (e = StellaEnvironment(rom_bytes); env_reset!(e; boot_noop_steps = 60,
            boot_reset_steps = 4); for _ in 1:2100; env_step!(e, NOOP); end; e)
base = scene(); b = copy(get_screen(base)); h, w = size(b)
er = scene(); en = scene()
for _ in 1:24; env_step!(er, RIGHT); env_step!(en, NOOP); end
r = get_screen(er); n = get_screen(en)
for (nm, A) in (("base", b), ("right24", r), ("noop24", n))
    write(joinpath(outdir, "diag_$nm.raw"), UInt8.(_cflat(A)))
end
write(joinpath(outdir, "diag.shape"), "$h $w")
println("base player-X=", Int(get_ram(base)[29]), "  right24=",
        Int(get_ram(er)[29]), "  noop24=", Int(get_ram(en)[29]))
chg = findall(r .!= n)
if !isempty(chg)
    rs = [c[1] for c in chg]; cs = [c[2] for c in chg]
    println("right24 vs noop24 changed px=", length(chg),
            "  rows ", minimum(rs), "-", maximum(rs),
            "  cols ", minimum(cs), "-", maximum(cs))
end
