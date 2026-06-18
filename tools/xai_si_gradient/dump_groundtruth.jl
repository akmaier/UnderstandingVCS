# Ground-truth (HARD, faithful) saliency of joystick RIGHT on the real SI screen
# at 35 s: the finite-difference d screen / d right. This is what the gradient
# *should* recover. We accumulate |RIGHT - NOOP| framebuffer differences over a
# short hold so the slow (~1px / 3 frame) cannon motion shows up as a clean
# region. Dumps raw uint8 for the python NTSC decoder.
#
# Run: cd jutari && julia --project=. ../tools/xai_si_gradient/dump_groundtruth.jl

using JuTari
using JuTari.Env: StellaEnvironment, env_reset!, env_step!, get_screen, get_ram

const NOOP = 0
const RIGHT = 3
rom_bytes = read(joinpath(@__DIR__, "..", "..", "xitari", "roms", "space_invaders.bin"))
outdir = joinpath(@__DIR__, "out"); isdir(outdir) || mkpath(outdir)
_cflat(s) = vec(permutedims(s))                       # C-order for numpy

scene_env() = begin
    e = StellaEnvironment(rom_bytes)
    env_reset!(e; boot_noop_steps = 60, boot_reset_steps = 4)
    for _ in 1:2100; env_step!(e, NOOP); end
    e
end

base = scene_env(); base_scr = copy(get_screen(base))
h, w = size(base_scr)
er = scene_env(); en = scene_env()
sal = zeros(Float32, h, w)
for k in 1:14
    env_step!(er, RIGHT); env_step!(en, NOOP)
    global sal
    sal .+= Float32.(get_screen(er) .!= get_screen(en))
end

write(joinpath(outdir, "gt_base.raw"), UInt8.(_cflat(base_scr)))
write(joinpath(outdir, "gt_base.shape"), "$h $w")
write(joinpath(outdir, "gt_sal.raw"), _cflat(sal))     # Float32
println("dumped base ($h x $w) + saliency; sal max=", Int(maximum(sal)),
        " nz=", count(>(0f0), sal), "  player-X 35s=", Int(get_ram(base)[29]))
