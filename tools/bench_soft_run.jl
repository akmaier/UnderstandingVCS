#!/usr/bin/env julia
# P9 — jutari throughput benchmark for SOFT-mode execution.
#
# Mirror of `bench_soft_run.py`. Loads a ROM, runs N SOFT-mode CPU
# instructions via `soft_run!`, and reports throughput. JSONL on
# stdout, summary on stderr.
#
# Usage
# -----
#
#     julia --project=jutari tools/bench_soft_run.jl \
#         --rom xitari/roms/pong.bin --steps 50000 --repeats 5
#
# Methodology
# -----------
# Julia JIT-compiles soft_step! per call shape; the first run includes
# specialisation cost, subsequent runs are cached. The summary takes
# the median over repeats 2..N, matching the jaxtari benchmark.

using JuTari
using JuTari.Diff: SoftBus, soft_run!, initial_soft_cpu_state

# Trivial JSON-line emitter — avoids pulling in JSON3 just for the
# benchmark. The output shape matches `bench_soft_run.py` byte-for-byte
# so downstream tooling can ingest both ports' results uniformly.
function _emit_jsonl(port, rom_label, steps, wall, repeat)
    println("{\"port\":\"$port\",\"rom\":\"$rom_label\",\"steps\":$steps," *
            "\"wall_s\":$(round(wall,digits=4))," *
            "\"steps_per_s\":$(round(steps/wall,digits=1)),\"repeat\":$repeat}")
end

function _main(argv)
    rom = ""; steps = 50_000; repeats = 5
    i = 1
    while i <= length(argv)
        a = argv[i]
        if a == "--rom"     && i + 1 <= length(argv); rom = argv[i + 1];     i += 2
        elseif a == "--steps"   && i + 1 <= length(argv); steps   = parse(Int, argv[i + 1]); i += 2
        elseif a == "--repeats" && i + 1 <= length(argv); repeats = parse(Int, argv[i + 1]); i += 2
        else;   println(stderr, "bench_soft_run.jl: unknown arg $a"); return 2
        end
    end
    isempty(rom) && (println(stderr,
        "usage: julia --project=jutari tools/bench_soft_run.jl --rom <path> [--steps N] [--repeats N]"); return 2)

    rom_bytes = read(rom)
    rom_vec = Float32.(rom_bytes)
    rom_label = basename(rom)

    wall_times = Float64[]
    for repeat in 0:(repeats - 1)
        bus = SoftBus(zeros(Float32, 128), rom_vec)
        state = initial_soft_cpu_state()
        t0 = time()
        soft_run!(state, bus, steps)
        wall = time() - t0
        push!(wall_times, wall)
        _emit_jsonl("jutari", rom_label, steps, wall, repeat)
    end

    if repeats > 1
        cached = wall_times[2:end]
        median_wall = sort(cached)[(length(cached) + 1) ÷ 2]
        median_sps  = steps / median_wall
        println(stderr, "\n[jutari summary] $rom_label × $steps steps × " *
                        "$(length(cached)) cached repeats: " *
                        "median $(round(median_wall, digits=3)) s " *
                        "($(round(Int, median_sps)) steps/s; " *
                        "first-run $(round(wall_times[1], digits=3)) s)")
    end
    return 0
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(_main(ARGS))
end
