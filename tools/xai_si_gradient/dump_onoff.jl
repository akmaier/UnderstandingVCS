# Dump an OFF-phase (2100) and ON-phase (2106) SI frame at the 35 s scene so the
# cannon can be isolated by the flicker diff (both have player-X=35; the only
# bottom-band difference is the cannon sprite appearing).
# Run: cd jutari && julia --project=. ../tools/xai_si_gradient/dump_onoff.jl
using JuTari
using JuTari.Env: StellaEnvironment, env_reset!, env_step!, get_screen, get_ram
const NOOP = 0
rom_bytes = read(joinpath(@__DIR__, "..", "..", "xitari", "roms", "space_invaders.bin"))
outdir = joinpath(@__DIR__, "out"); isdir(outdir) || mkpath(outdir)
_cflat(s) = vec(permutedims(s))
e = StellaEnvironment(rom_bytes); env_reset!(e; boot_noop_steps = 60, boot_reset_steps = 4)
for _ in 1:2100; env_step!(e, NOOP); end
off = copy(get_screen(e)); h, w = size(off)
for _ in 1:6; env_step!(e, NOOP); end          # to frame 2106 (on-phase)
on = copy(get_screen(e))
write(joinpath(outdir, "onoff_off.raw"), UInt8.(_cflat(off)))
write(joinpath(outdir, "onoff_on.raw"),  UInt8.(_cflat(on)))
write(joinpath(outdir, "onoff.shape"), "$h $w")
println("dumped off(2100) + on(2106); player-X=", Int(get_ram(e)[29]))
