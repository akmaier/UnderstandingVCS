# Is the 35 s visual change a frame-to-frame FLICKER or a one-time game-state
# transition? Sample a background pixel + the cannon-band fill across the video
# range (0-30 s) and around 35 s. Run:
#   cd jutari && julia --project=. ../tools/xai_si_gradient/check_flicker.jl
using JuTari
using JuTari.Env: StellaEnvironment, env_reset!, env_step!, get_screen, get_ram
const NOOP = 0
rom_bytes = read(joinpath(@__DIR__, "..", "..", "xitari", "roms", "space_invaders.bin"))
e = StellaEnvironment(rom_bytes); env_reset!(e; boot_noop_steps = 60, boot_reset_steps = 4)
bgpix(s) = Int(s[60, 3])                          # far-left background sample
cannon(s) = count(!=(0), s[181:194, 26:60])
# lowest invader row: invaders occupy the mid band; track the max row with the
# invader colour up high (col 40-130, rows 25-160)
function inv_low(s)
    band = s[25:160, 40:130]
    rs = findall(!=(0), band)
    isempty(rs) ? 0 : maximum(c[1] for c in rs) + 24
end
println("frame  t(s)   bgpix  cannon_px  invader_lowrow")
checks = vcat(collect(100:300:1700), [1800], collect(2095:1:2118))
prev = 0
for f in 1:2118
    env_step!(e, NOOP); s = get_screen(e)
    if f in checks
        println(lpad(f,5), "  ", lpad(round(f/60,digits=1),5), "   ",
                lpad(bgpix(s),5), "   ", lpad(cannon(s),6), "      ", inv_low(s))
    end
end
