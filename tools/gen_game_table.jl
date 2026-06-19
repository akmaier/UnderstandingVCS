#!/usr/bin/env julia
# Emit the per-game conformance table data for the supplement: short name,
# ROM size (bytes), detected mapper (jutari's own autodetect), and a short
# SHA-256. Run with the jutari project active, e.g.
#   julia --project=jutari tools/gen_game_table.jl tools/rom_sweep/roms tools/rom_sweep/manifest.txt
using JuTari
using SHA

const KIND_NAMES = Dict(
    0 => "2K", 1 => "4K", 2 => "F8", 3 => "F6", 4 => "F4",
    5 => "E0", 6 => "FE", 7 => "F8SC", 8 => "F6SC", 9 => "F4SC")

romdir   = ARGS[1]
manifest = ARGS[2]

shorts = String[]
for line in eachline(manifest)
    isempty(strip(line)) && continue
    push!(shorts, String(split(line, '\t')[1]))
end
sort!(shorts)

for s in shorts
    path = joinpath(romdir, s * ".bin")
    if !isfile(path)
        println(stderr, "MISSING $path"); continue
    end
    rom  = read(path)
    kind = KIND_NAMES[make_cart(rom).kind]
    h    = bytes2hex(sha256(rom))[1:12]
    println(join([s, string(length(rom)), kind, h], '\t'))
end
