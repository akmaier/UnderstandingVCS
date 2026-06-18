# Phase 0b feasibility probe for the real-ROM SI joystick-gradient study.
# Answers, empirically and honestly:
#   (A) HARD ground truth: from the 35 s scene, does holding RIGHT vs NOOP for a
#       few frames actually move pixels on screen, and where? (finite difference)
#   (B) SOFT fidelity: promote the 35 s hard state into the differentiable SOFT
#       state, inject a continuous joystick at the SWCHA alias (bus.ram[1]); does
#       the soft rollout track at all, and does player-X (RAM offset $1C, idx 29)
#       respond? how fast / how divergent?
#   (C) SOFT gradient: can Zygote take d(player-X)/d(joystick) through soft_run for
#       a short horizon, and is it non-zero? (and the cost per step)
#
# NO changes to jutari. Run:
#   cd jutari && julia --project=. ../tools/xai_si_gradient/probe_soft.jl

using JuTari
using JuTari.Env: StellaEnvironment, env_reset!, env_step!, get_screen, get_ram
using JuTari.Diff: SoftCPUState, SoftBus, soft_step!, soft_run, soft_run!,
                   initial_soft_cpu_state
import Zygote

const NOOP = 0
const RIGHT = 3
const PLX = 29          # RAM array index of player-X (offset $1C, CPU $9C)
const SWCHA_IDLE = 0xFF
const SWCHA_RIGHT = 0x7F # P0 RIGHT = bit7, active-low

rom_bytes = read(joinpath(@__DIR__, "..", "..", "xitari", "roms", "space_invaders.bin"))
rom = Float32.(rom_bytes)

scene_env() = begin
    e = StellaEnvironment(rom_bytes)
    env_reset!(e; boot_noop_steps = 60, boot_reset_steps = 4)
    for _ in 1:2100; env_step!(e, NOOP); end          # to 35 s
    e
end

# (A) HARD finite-difference: how the screen reacts to RIGHT ------------------
println("=== (A) HARD finite difference: RIGHT vs NOOP from 35 s ===")
let
    base = scene_env(); base_scr = copy(get_screen(base))
    println("  35 s screen: size=", size(base_scr), "  nz=", count(!=(0), base_scr),
            "  player-X(\$9C)=", Int(get_ram(base)[PLX]))
    for k in (1, 2, 3, 4, 6, 8)
        er = scene_env(); en = scene_env()
        for _ in 1:k; env_step!(er, RIGHT); env_step!(en, NOOP); end
        sr = get_screen(er); sn = get_screen(en)
        diff = sr .!= sn
        rows = findall(any(diff, dims = 2)[:]); cols = findall(any(diff, dims = 1)[:])
        bbox = isempty(rows) ? "—" :
               "rows $(first(rows))–$(last(rows)), cols $(first(cols))–$(last(cols))"
        println("  +$k RIGHT: Δpx=", count(diff), "  plX=", Int(get_ram(er)[PLX]),
                " (noop ", Int(get_ram(en)[PLX]), ")   bbox: ", bbox)
    end
end

# (B) SOFT promotion + fidelity ----------------------------------------------
function promote(env)
    c = env.console.cpu
    st = SoftCPUState(Float32(c.A), Float32(c.X), Float32(c.Y), Float32(c.SP),
                      Float32(c.PC), Float32(c.P), Float32(c.cycles))
    bus = SoftBus(Float32.(get_ram(env)), copy(rom))
    return st, bus
end

println("\n=== (B) SOFT fidelity: promote 35 s state, inject joystick, soft_step! ===")
let
    base = scene_env()
    st0, bus0 = promote(base)
    println("  promoted PC=\$", uppercase(string(Int(st0.PC), base = 16)),
            "  player-X=", Int(round(bus0.ram[PLX])))
    for (lbl, inj) in (("noop", SWCHA_IDLE), ("right", SWCHA_RIGHT))
        st = deepcopy(st0); bus = deepcopy(bus0)
        bus.ram[1] = Float32(inj)             # SWCHA alias
        N = 6000; firstmove = -1; t0 = time()
        plx0 = bus.ram[PLX]
        for i in 1:N
            soft_step!(st, bus)
            bus.ram[1] = Float32(inj)         # keep joystick asserted each step
            if firstmove < 0 && abs(bus.ram[PLX] - plx0) > 0.5; firstmove = i; end
        end
        wall = time() - t0
        println("  $lbl: ", round(Int, N / wall), " steps/s  player-X ",
                Int(round(plx0)), "→", Int(round(bus.ram[PLX])),
                "  firstΔ@step=", firstmove,
                "  PC=\$", uppercase(string(Int(round(st.PC)), base = 16)))
    end
end

# (C) SOFT gradient smoke test (functional + Zygote) -------------------------
println("\n=== (C) SOFT gradient: d(player-X after N steps)/d(joystick) ===")
let
    base = scene_env()
    st0, bus0 = promote(base)
    ram0 = copy(bus0.ram)
    for N in (10, 50, 200)
        # objective: player-X after N functional soft steps, joystick = j at ram[1]
        function obj(j)
            ram = vcat(j, ram0[2:end])                 # inject at SWCHA alias
            bus = SoftBus(ram, copy(rom))
            _, bus2 = soft_run(st0, bus, N)
            return bus2.ram[PLX]
        end
        t0 = time(); val = obj(Float32(SWCHA_RIGHT)); tf = time() - t0
        t1 = time(); g = Zygote.gradient(obj, Float32(SWCHA_RIGHT))[1]; tg = time() - t1
        println("  N=$N: player-X=", round(val, digits = 3),
                "  d/dj=", g, "  (fwd ", round(tf, digits = 2), "s, grad ",
                round(tg, digits = 2), "s)")
    end
end

println("\ndone.")
