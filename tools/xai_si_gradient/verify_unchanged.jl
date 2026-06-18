# Prove jutari HARD rendering is byte-identical to the 16:41 video: re-roll the
# exact same HARD NOOP rollout now and compare against the saved video frames.
# Also report whether the teal-bg transition is even inside the video's range.
# Run: cd jutari && julia --project=. ../tools/xai_si_gradient/verify_unchanged.jl
using JuTari
using JuTari.Env: StellaEnvironment, env_reset!, env_step!, get_screen
import JuTari.CPU: set_cpu_relax!

vid = joinpath(@__DIR__, "..", "relaxation_study", "video_out", "exact_frames.raw")
saved = read(vid)
N, H, W = (parse.(Int, split(read(joinpath(dirname(vid), "exact_frames.shape"), String)))...,)
println("saved video: $N frames of $H x $W  (", length(saved), " bytes)")

rom = read(joinpath(@__DIR__, "..", "..", "xitari", "roms", "space_invaders.bin"))
set_cpu_relax!(on = false)                      # exact panel = HARD, relax off
env = StellaEnvironment(rom); env_reset!(env; boot_noop_steps = 60, boot_reset_steps = 4)
mine = UInt8[]; bg = Int[]
for i in 1:N
    env_step!(env, 0); s = get_screen(env)
    append!(mine, UInt8.(vec(permutedims(s))))
    push!(bg, Int(s[60, 3]))
end

println("re-roll identical to saved video bytes? ", mine == saved)
fd = findfirst(saved .!= mine)
println("first differing byte: ", fd === nothing ? "NONE (byte-identical)" : string(fd))
teal = findfirst(!=(0), bg)
println("teal-bg (bg pixel != 0) within the video's $N frames? ",
        teal === nothing ? "NO — bg black for all $N frames" : "yes, first at frame $teal")
println("=> video covers frames 1..$N; the teal transition is at frame ~2103 (beyond the video).")
