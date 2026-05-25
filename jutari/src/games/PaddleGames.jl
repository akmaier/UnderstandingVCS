"""
    PaddleGames

Minimal RomSettings stubs for the paddle-game ROMs we ship — they
override only `romsettings_uses_paddles` so `StellaEnvironment`
auto-translates LEFT/RIGHT actions into INPT0 dump-pot
paddle-position changes (xitari's `applyActionPaddles` semantic).
The full per-game scoring rules live on the jaxtari side
(`jaxtari/games/breakout.py` etc.) — porting them to jutari is a
separate task.

Mirror of jaxtari's `BreakoutRomSettings.uses_paddles() = True` /
`PongRomSettings.uses_paddles() = True`.
"""
module PaddleGames

using ..RomSettingsModule: RomSettings
import ..RomSettingsModule: romsettings_uses_paddles

export BreakoutRomSettings, PongRomSettings

"""
    BreakoutRomSettings

xitari stella.pro: `Cartridge.MD5 f34f08e5…`, `Controller.Left "PADDLES"`.
"""
mutable struct BreakoutRomSettings <: RomSettings
    BreakoutRomSettings() = new()
end

romsettings_uses_paddles(::BreakoutRomSettings) = true

"""
    PongRomSettings

The shipped `xitari/roms/pong.bin` is actually Video Olympics (Atari
1978, md5 60e0ea3c…). xitari stella.pro lists it as
`Controller.Left/Right "PADDLES"` with `Controller.SwapPaddles "YES"`.
"""
mutable struct PongRomSettings <: RomSettings
    PongRomSettings() = new()
end

romsettings_uses_paddles(::PongRomSettings) = true

end # module
