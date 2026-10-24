# Opcode coverage of the 64 conformance ROMs along the conformance protocol
# (60-NOOP + 4-RESET boot, then the 3,600-step breakout_random_actions stream;
# frames 1-60 are the RAM/screen conformance window). No change to the jutari
# core: run_until_frame! is re-evaluated with a per-instruction opcode counter.
# Usage: cd jutari && julia --project=. ../tools/opcode_coverage.jl OUTDIR game1 game2 ...
using JuTari
const REPO = normpath(joinpath(@__DIR__, ".."))
src = read(joinpath(REPO, "tools", "jutari_screen_dump.jl"), String)
include_string(Main, src[1:findfirst("function main()", src)[1]-1])   # settings map only
using JuTari.Env: StellaEnvironment, env_reset!, env_step!
using JuTari.Cart: cart_peek_pure

const COUNTS = zeros(Int, 256)
const STATE = Ref(:boot)        # :boot | :window | :long
const C_BOOT = zeros(Int, 256); const C_WIN = zeros(Int, 256); const C_LONG = zeros(Int, 256)
cur() = STATE[] === :boot ? C_BOOT : STATE[] === :window ? C_WIN : C_LONG

@inline function _opc(bus, pc)
    a = pc & 0x1FFF
    (a & 0x1000) != 0 && return Int(cart_peek_pure(bus.cart, pc))
    ((a & 0x80) != 0 && (a & 0x200) == 0) && return Int(bus.ram[(a & 0x7F) + 1])
    return -1
end

@eval JuTari.ConsoleModule function run_until_frame!(console::Console)
    tia = console.bus.tia
    if tia.buffer_swap_pending
        tia.framebuffer, tia.framebuffer_prev = tia.framebuffer_prev, tia.framebuffer
        tia.buffer_swap_pending = false
    end
    start_frame = tia.frame
    for _ in 1:_UPDATE_INSTRUCTION_BUDGET
        o = Main._opc(console.bus, Int(console.cpu.PC) & 0xFFFF)
        o >= 0 && (Main.cur()[o + 1] += 1)
        step(console.cpu, console.bus)
        if console.bus.tia.frame != start_frame
            return console
        end
    end
    return console
end

actions = [parse(Int, strip(l)) for l in eachline(joinpath(REPO, "tools/breakout_video/output/breakout_random_actions.txt"))
           if !isempty(strip(l)) && !startswith(strip(l), "#")]
outdir = ARGS[1]
for g in ARGS[2:end]
    rom_path = joinpath(REPO, "tools/rom_sweep/roms", g * ".bin")
    fill!(C_BOOT, 0); fill!(C_WIN, 0); fill!(C_LONG, 0)
    env = StellaEnvironment(read(rom_path), _settings_for_rom(rom_path))
    STATE[] = :boot
    env_reset!(env; boot_noop_steps = 60, boot_reset_steps = 4)
    for (i, a) in enumerate(actions)
        STATE[] = i <= 60 ? :window : :long
        env_step!(env, a)
    end
    open(joinpath(outdir, g * ".txt"), "w") do io
        println(io, join(C_BOOT, ",")); println(io, join(C_WIN, ",")); println(io, join(C_LONG, ","))
    end
    println(g, " done")
end
