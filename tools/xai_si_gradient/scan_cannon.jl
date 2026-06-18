# The SI player cannon flickers; find an on-phase frame near 35 s and pin its
# footprint. Reports, per frame around 2100, the nonzero pixels in the cannon
# band (rows 180-195, cols 25-60) so we can pick a frame where the cannon is
# drawn. Run: cd jutari && julia --project=. ../tools/xai_si_gradient/scan_cannon.jl
using JuTari
using JuTari.Env: StellaEnvironment, env_reset!, env_step!, get_screen, get_ram
const NOOP = 0
rom_bytes = read(joinpath(@__DIR__, "..", "..", "xitari", "roms", "space_invaders.bin"))
e = StellaEnvironment(rom_bytes); env_reset!(e; boot_noop_steps = 60, boot_reset_steps = 4)
for _ in 1:2095; env_step!(e, NOOP); end
for f in 2096:2116
    env_step!(e, NOOP); s = get_screen(e)
    band = s[181:195, 26:60]                       # cannon band (rows,cols 1-based)
    nz = findall(!=(0), band)
    if isempty(nz)
        println("frame $f: cannon band EMPTY  (player-X=", Int(get_ram(e)[29]), ")")
    else
        rs = [c[1] + 180 for c in nz]; cs = [c[2] + 25 for c in nz]
        vals = sort(unique(Int.(band[band .!= 0])))
        println("frame $f: cannon px=", length(nz), "  rows ", minimum(rs), "-",
                maximum(rs), " cols ", minimum(cs), "-", maximum(cs),
                " color=", vals, "  player-X=", Int(get_ram(e)[29]))
    end
end
