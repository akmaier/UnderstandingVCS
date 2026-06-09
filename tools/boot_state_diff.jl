#!/usr/bin/env julia
"""
    boot_state_diff.jl

Phase B / task #80: localize the boot-end RAM divergence between
jutari and xitari for any ROM. After `env_reset!(env, 60, 4)` returns,
print the 128-byte RAM as hex — the matching xitari output comes from
`tools/trace_dump --max-frames 1` first JSON line (`"boot_end":true`).

Usage:
    julia --project=jutari tools/boot_state_diff.jl --rom xitari/roms/seaquest.bin

Output: a single hex string (256 chars) on stdout; that's the post-boot
RAM[0..127]. No actions applied yet.
"""

using JuTari
using JuTari.Env: StellaEnvironment, env_reset!
using JuTari.RomSettingsModule: GenericRomSettings, RomSettings
using JuTari.PaddleGames: BreakoutRomSettings, PongRomSettings
using JuTari.JoystickGames: PitfallRomSettings, EnduroRomSettings

# Same registry as jutari_trace_dump.jl so the RomSettings + starting
# actions match what xitari does.
const _SETTINGS_BY_BASENAME = Dict{String,Function}(
    "breakout.bin" => () -> BreakoutRomSettings(),
    "pong.bin"     => () -> PongRomSettings(),
    "pitfall.bin"  => () -> PitfallRomSettings(),
    "enduro.bin"   => () -> EnduroRomSettings(),
)
_settings_for_rom(rom_path) =
    haskey(_SETTINGS_BY_BASENAME, basename(rom_path)) ?
        _SETTINGS_BY_BASENAME[basename(rom_path)]() :
        GenericRomSettings()

function _hex_of(bytes::AbstractVector{UInt8})
    io = IOBuffer()
    for b in bytes
        print(io, string(b; base = 16, pad = 2))
    end
    return String(take!(io))
end

function main(argv = ARGS)
    rom_path = ""
    for i in 1:length(argv)
        if argv[i] == "--rom" && i < length(argv)
            rom_path = argv[i+1]
        end
    end
    isempty(rom_path) && error("--rom required")

    rom = read(rom_path)
    env = StellaEnvironment(rom, _settings_for_rom(rom_path))
    env_reset!(env; boot_noop_steps = 60, boot_reset_steps = 4)

    # 128-byte RAM via JuTari.Env.get_ram or directly from the bus
    ram = collect(UInt8.(env.console.bus.ram[1:128]))
    println(_hex_of(ram))
    return 0
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main())
end
