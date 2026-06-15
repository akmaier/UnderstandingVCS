# ROM sweep — jutari SCREEN (framebuffer) bit-exactness vs xitari

Per-frame 210x160 palette-index diff, jutari `jutari_screen_dump.jl` vs xitari `trace_dump --screen`, breakout_random_actions stream, first **60** frames after the standard 60-NOOP+4-RESET boot. Same per-game RomSettings as the RAM sweep (so a divergence is a genuine render delta, not a settings/boot mismatch).

**Pixel-exact (0 px) over 60 frames: 29/64.**

| game | max px/frame | total px | first div frame | worst-frame rows |
|---|---|---|---|---|
| air_raid | n/a | — | — | PAL? xitari 250h vs jutari 210h |
| alien | **0 ✅** | 0 | — | — |
| amidar | 3 | 177 | 2 | 182-182 |
| assault | **0 ✅** | 0 | — | — |
| asterix | 1 | 60 | 1 | 190-190 |
| asteroids | **0 ✅** | 0 | — | — |
| atlantis | 24 | 1440 | 1 | 186-186 |
| bank_heist | **0 ✅** | 0 | — | — |
| battle_zone | 1112 | 66720 | 1 | 38-176 |
| beam_rider | 131 | 7860 | 1 | 191-209 |
| berzerk | 25 | 97 | 42 | 4-181 |
| bowling | 648 | 38880 | 1 | 0-4 |
| boxing | **0 ✅** | 0 | — | — |
| breakout | **0 ✅** | 0 | — | — |
| carnival | n/a | — | — | PAL? xitari 214h vs jutari 210h |
| centipede | 3 | 180 | 1 | 193-193 |
| chopper_command | 898 | 53880 | 1 | 198-203 |
| crazy_climber | 1592 | 95520 | 1 | 0-9 |
| defender | 9 | 540 | 1 | 183-183 |
| demon_attack | 3 | 180 | 1 | 15-15 |
| double_dunk | **0 ✅** | 0 | — | — |
| elevator_action | 742 | 30000 | 1 | 5-176 |
| enduro | **0 ✅** | 0 | — | — |
| fishing_derby | **0 ✅** | 0 | — | — |
| freeway | **0 ✅** | 0 | — | — |
| frostbite | 2736 | 164160 | 1 | 0-209 |
| gopher | **0 ✅** | 0 | — | — |
| gravitar | **0 ✅** | 0 | — | — |
| hero | 320 | 19200 | 1 | 0-9 |
| ice_hockey | 5 | 300 | 1 | 87-103 |
| jamesbond | 1 | 60 | 1 | 21-21 |
| journey_escape | n/a | — | — | PAL? xitari 230h vs jutari 210h |
| kangaroo | 9 | 540 | 1 | 2-3 |
| krull | **0 ✅** | 0 | — | — |
| kung_fu_master | **0 ✅** | 0 | — | — |
| montezuma_revenge | **0 ✅** | 0 | — | — |
| ms_pacman | 232 | 13920 | 1 | 1-169 |
| name_this_game | 6 | 360 | 1 | 188-188 |
| pacman | 3362 | 201720 | 1 | 0-189 |
| phoenix | **0 ✅** | 0 | — | — |
| pitfall | **0 ✅** | 0 | — | — |
| pong | **0 ✅** | 0 | — | — |
| pooyan | n/a | — | — | PAL? xitari 220h vs jutari 210h |
| private_eye | 2888 | 173280 | 1 | 0-209 |
| qbert | 7664 | 345224 | 2 | 34-205 |
| riverraid | **0 ✅** | 0 | — | — |
| road_runner | **0 ✅** | 0 | — | — |
| robotank | 1353 | 81168 | 1 | 0-182 |
| seaquest | **0 ✅** | 0 | — | — |
| skiing | **0 ✅** | 0 | — | — |
| solaris | 482 | 28920 | 1 | 0-11 |
| space_invaders | **0 ✅** | 0 | — | — |
| star_gunner | 45 | 2700 | 1 | 0-9 |
| surround | n/a | — | — | PAL? xitari 250h vs jutari 210h |
| tennis | **0 ✅** | 0 | — | — |
| time_pilot | **0 ✅** | 0 | — | — |
| tutankham | 80 | 4800 | 1 | 103-167 |
| up_n_down | 10838 | 639483 | 1 | 1-203 |
| venture | **0 ✅** | 0 | — | — |
| video_pinball | 66 | 3960 | 1 | 192-192 |
| videochess | **0 ✅** | 0 | — | — |
| wizard_of_wor | 3 | 180 | 1 | 164-164 |
| yars_revenge | **0 ✅** | 0 | — | — |
| zaxxon | **0 ✅** | 0 | — | — |
