# ROM sweep — jutari SCREEN (framebuffer) bit-exactness vs xitari

Per-frame 210x160 palette-index diff, jutari `jutari_screen_dump.jl` vs xitari `trace_dump --screen`, breakout_random_actions stream, first **60** frames after the standard 60-NOOP+4-RESET boot. Same per-game RomSettings as the RAM sweep (so a divergence is a genuine render delta, not a settings/boot mismatch).

**Pixel-exact (0 px) over 60 frames: 44/64.**

| game | max px/frame | total px | first div frame | worst-frame rows |
|---|---|---|---|---|
| air_raid | 24 | 1440 | 1 | 219-223 |
| alien | **0 ✅** | 0 | — | — |
| amidar | 3 | 177 | 2 | 182-182 |
| assault | **0 ✅** | 0 | — | — |
| asterix | 1 | 60 | 1 | 190-190 |
| asteroids | **0 ✅** | 0 | — | — |
| atlantis | 24 | 1440 | 1 | 186-186 |
| bank_heist | **0 ✅** | 0 | — | — |
| battle_zone | **0 ✅** | 0 | — | — |
| beam_rider | **0 ✅** | 0 | — | — |
| berzerk | 21 | 21 | 42 | 180-181 |
| bowling | **0 ✅** | 0 | — | — |
| boxing | **0 ✅** | 0 | — | — |
| breakout | **0 ✅** | 0 | — | — |
| carnival | 4 | 240 | 1 | 27-27 |
| centipede | 3 | 180 | 1 | 193-193 |
| chopper_command | **0 ✅** | 0 | — | — |
| crazy_climber | **0 ✅** | 0 | — | — |
| defender | 9 | 540 | 1 | 183-183 |
| demon_attack | 3 | 180 | 1 | 15-15 |
| double_dunk | **0 ✅** | 0 | — | — |
| elevator_action | 16 | 160 | 41 | 73-74 |
| enduro | **0 ✅** | 0 | — | — |
| fishing_derby | **0 ✅** | 0 | — | — |
| freeway | **0 ✅** | 0 | — | — |
| frostbite | **0 ✅** | 0 | — | — |
| gopher | **0 ✅** | 0 | — | — |
| gravitar | **0 ✅** | 0 | — | — |
| hero | **0 ✅** | 0 | — | — |
| ice_hockey | 5 | 300 | 1 | 87-103 |
| jamesbond | 1 | 60 | 1 | 21-21 |
| journey_escape | 325 | 19191 | 1 | 13-196 |
| kangaroo | **0 ✅** | 0 | — | — |
| krull | **0 ✅** | 0 | — | — |
| kung_fu_master | **0 ✅** | 0 | — | — |
| montezuma_revenge | **0 ✅** | 0 | — | — |
| ms_pacman | **0 ✅** | 0 | — | — |
| name_this_game | 6 | 360 | 1 | 188-188 |
| pacman | **0 ✅** | 0 | — | — |
| phoenix | **0 ✅** | 0 | — | — |
| pitfall | **0 ✅** | 0 | — | — |
| pong | **0 ✅** | 0 | — | — |
| pooyan | 1 | 60 | 1 | 11-11 |
| private_eye | **0 ✅** | 0 | — | — |
| qbert | **0 ✅** | 0 | — | — |
| riverraid | **0 ✅** | 0 | — | — |
| road_runner | **0 ✅** | 0 | — | — |
| robotank | 241 | 14448 | 1 | 49-182 |
| seaquest | **0 ✅** | 0 | — | — |
| skiing | **0 ✅** | 0 | — | — |
| solaris | 2 | 120 | 1 | 11-11 |
| space_invaders | **0 ✅** | 0 | — | — |
| star_gunner | **0 ✅** | 0 | — | — |
| surround | **0 ✅** | 0 | — | — |
| tennis | **0 ✅** | 0 | — | — |
| time_pilot | **0 ✅** | 0 | — | — |
| tutankham | 80 | 4800 | 1 | 103-167 |
| up_n_down | 221 | 12257 | 1 | 5-203 |
| venture | **0 ✅** | 0 | — | — |
| video_pinball | **0 ✅** | 0 | — | — |
| videochess | **0 ✅** | 0 | — | — |
| wizard_of_wor | 3 | 180 | 1 | 164-164 |
| yars_revenge | **0 ✅** | 0 | — | — |
| zaxxon | **0 ✅** | 0 | — | — |
