# ROM sweep — jutari RAM bit-exactness vs xitari (NOOP, 30 frames)

Same paradigm as PXC1/PXC2: per-frame 128 B RIOT-RAM diff, jutari vs xitari trace_dump, NOOP from the standard ALE boot (60 NOOP + 4 RESET). `generic` = jutari ran with GenericRomSettings (no game-specific starting actions; divergence may be settings- not emulation-driven).

**Progress: 64/64 ROMs. Bit-exact (0 b/f): 44/64 completed.**

| game | settings | max RAM diff (b/f) | first div frame | status | secs |
|---|---|---|---|---|---|
| air_raid | generic | 55 | 0 | OK | 13 |
| alien | generic | **0 ✅** | — | OK | 13 |
| amidar | generic | 11 | 0 | OK | 13 |
| assault | generic | **0 ✅** | — | OK | 13 |
| asterix | generic | 63 | 0 | OK | 13 |
| asteroids | generic | **0 ✅** | — | OK | 12 |
| atlantis | generic | **0 ✅** | — | OK | 12 |
| bank_heist | generic | **0 ✅** | — | OK | 12 |
| battle_zone | generic | **0 ✅** | — | OK | 12 |
| beam_rider | generic | 30 | 0 | OK | 13 |
| berzerk | generic | **0 ✅** | — | OK | 11 |
| bowling | generic | **0 ✅** | — | OK | 11 |
| boxing | generic | **0 ✅** | — | OK | 11 |
| breakout | real | **0 ✅** | — | OK | 11 |
| carnival | generic | **0 ✅** | — | OK | 11 |
| centipede | generic | **0 ✅** | — | OK | 11 |
| chopper_command | generic | **0 ✅** | — | OK | 11 |
| crazy_climber | generic | **0 ✅** | — | OK | 11 |
| defender | generic | **0 ✅** | — | OK | 11 |
| demon_attack | generic | **0 ✅** | — | OK | 11 |
| double_dunk | generic | 64 | 0 | OK | 11 |
| elevator_action | generic | 98 | 0 | OK | 11 |
| enduro | real | **0 ✅** | — | OK | 11 |
| fishing_derby | generic | **0 ✅** | — | OK | 11 |
| freeway | generic | **0 ✅** | — | OK | 11 |
| frostbite | generic | 2 | 0 | OK | 11 |
| gopher | generic | 20 | 0 | OK | 11 |
| gravitar | generic | 94 | 0 | OK | 11 |
| hero | generic | **0 ✅** | — | OK | 11 |
| ice_hockey | generic | **0 ✅** | — | OK | 11 |
| jamesbond | generic | 83 | 0 | OK | 11 |
| journey_escape | generic | 24 | 0 | OK | 11 |
| kangaroo | generic | **0 ✅** | — | OK | 11 |
| krull | generic | **0 ✅** | — | OK | 11 |
| kung_fu_master | generic | **0 ✅** | — | OK | 11 |
| montezuma_revenge | generic | 84 | 0 | OK | 11 |
| ms_pacman | generic | **0 ✅** | — | OK | 11 |
| name_this_game | generic | **0 ✅** | — | OK | 11 |
| pacman | generic | **0 ✅** | — | OK | 10 |
| phoenix | generic | **0 ✅** | — | OK | 9 |
| pitfall | real | **0 ✅** | — | OK | 9 |
| pong | real | **0 ✅** | — | OK | 8 |
| pooyan | generic | **0 ✅** | — | OK | 9 |
| private_eye | generic | 22 | 0 | OK | 8 |
| qbert | generic | 56 | 0 | OK | 8 |
| riverraid | generic | **0 ✅** | — | OK | 9 |
| road_runner | generic | **0 ✅** | — | OK | 9 |
| robotank | generic | 81 | 0 | OK | 9 |
| seaquest | generic | **0 ✅** | — | OK | 8 |
| skiing | generic | 85 | 0 | OK | 8 |
| solaris | generic | **0 ✅** | — | OK | 8 |
| space_invaders | generic | **0 ✅** | — | OK | 8 |
| star_gunner | generic | **0 ✅** | — | OK | 8 |
| surround | generic | 15 | 0 | OK | 8 |
| tennis | generic | **0 ✅** | — | OK | 8 |
| time_pilot | generic | **0 ✅** | — | OK | 8 |
| tutankham | generic | 52 | 0 | OK | 8 |
| up_n_down | generic | 19 | 0 | OK | 9 |
| venture | generic | **0 ✅** | — | OK | 8 |
| video_pinball | generic | **0 ✅** | — | OK | 9 |
| videochess | generic | **0 ✅** | — | OK | 9 |
| wizard_of_wor | generic | **0 ✅** | — | OK | 8 |
| yars_revenge | generic | 18 | 0 | OK | 8 |
| zaxxon | generic | **0 ✅** | — | OK | 8 |
