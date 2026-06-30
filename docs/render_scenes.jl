using JuTari
using JuTari.Env: StellaEnvironment, env_reset!, env_step!, get_screen
games = ["pong","breakout","space_invaders","seaquest","ms_pacman","qbert"]
outdir = ARGS[1]
for g in games
    rom = read(joinpath("tools","rom_sweep","roms", g*".bin"))
    e = StellaEnvironment(rom); env_reset!(e; boot_noop_steps=60, boot_reset_steps=4)
    for _ in 1:150; env_step!(e, 0); end
    s = get_screen(e); H,W = size(s)
    open(joinpath(outdir, "scene_"*g*".raw"),"w") do io
        write(io, UInt8.(vec(permutedims(s))))   # row-major H*W
    end
    println("rendered $g  $H x $W")
end
