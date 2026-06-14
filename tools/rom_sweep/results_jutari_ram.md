# ROM sweep — jutari RAM bit-exactness vs xitari (NOOP, 30 frames)

Same paradigm as PXC1/PXC2: per-frame 128 B RIOT-RAM diff, jutari vs xitari trace_dump, NOOP from the standard ALE boot (60 NOOP + 4 RESET). `generic` = jutari ran with GenericRomSettings (no game-specific starting actions; divergence may be settings- not emulation-driven).

**Progress: 64/64 ROMs. Bit-exact (0 b/f): 55/64 completed.**

| game | settings | max RAM diff (b/f) | first div frame | status | secs |
|---|---|---|---|---|---|
| air_raid | generic | 43 | 0 | OK | 8 |
| alien | generic | **0 ✅** | — | OK | 8 |
| amidar | generic | 11 | 0 | OK | 8 |
| assault | generic | **0 ✅** | — | OK | 8 |
| asterix | generic | **0 ✅** | — | OK | 8 |
| asteroids | generic | **0 ✅** | — | OK | 8 |
| atlantis | generic | **0 ✅** | — | OK | 8 |
| bank_heist | generic | **0 ✅** | — | OK | 8 |
| battle_zone | generic | **0 ✅** | — | OK | 8 |
| beam_rider | generic | **0 ✅** | — | OK | 8 |
| berzerk | generic | **0 ✅** | — | OK | 8 |
| bowling | generic | **0 ✅** | — | OK | 8 |
| boxing | generic | **0 ✅** | — | OK | 8 |
| breakout | real | **0 ✅** | — | OK | 8 |
| carnival | generic | **0 ✅** | — | OK | 8 |
| centipede | generic | **0 ✅** | — | OK | 8 |
| chopper_command | generic | **0 ✅** | — | OK | 8 |
| crazy_climber | generic | **0 ✅** | — | OK | 8 |
| defender | generic | **0 ✅** | — | OK | 8 |
| demon_attack | generic | **0 ✅** | — | OK | 8 |
| double_dunk | generic | **0 ✅** | — | OK | 8 |
| elevator_action | generic | 98 | 0 | OK | 8 |
| enduro | real | **0 ✅** | — | OK | 8 |
| fishing_derby | generic | **0 ✅** | — | OK | 8 |
| freeway | generic | **0 ✅** | — | OK | 8 |
| frostbite | generic | 2 | 0 | OK | 8 |
| gopher | generic | **0 ✅** | — | OK | 8 |
| gravitar | generic | 93 | 0 | OK | 8 |
| hero | generic | **0 ✅** | — | OK | 8 |
| ice_hockey | generic | **0 ✅** | — | OK | 8 |
| jamesbond | generic | **0 ✅** | — | OK | 8 |
| journey_escape | generic | **0 ✅** | — | OK | 8 |
| kangaroo | generic | **0 ✅** | — | OK | 7 |
| krull | generic | **0 ✅** | — | OK | 7 |
| kung_fu_master | generic | **0 ✅** | — | OK | 7 |
| montezuma_revenge | generic | **0 ✅** | — | OK | 7 |
| ms_pacman | generic | **0 ✅** | — | OK | 7 |
| name_this_game | generic | **0 ✅** | — | OK | 7 |
| pacman | generic | **0 ✅** | — | OK | 7 |
| phoenix | generic | **0 ✅** | — | OK | 8 |
| pitfall | real | **0 ✅** | — | OK | 8 |
| pong | real | **0 ✅** | — | OK | 8 |
| pooyan | generic | **0 ✅** | — | OK | 8 |
| private_eye | generic | **0 ✅** | — | OK | 8 |
| qbert | generic | 56 | 0 | OK | 8 |
| riverraid | generic | **0 ✅** | — | OK | 8 |
| road_runner | generic | **0 ✅** | — | OK | 8 |
| robotank | generic | 81 | 0 | OK | 8 |
| seaquest | generic | **0 ✅** | — | OK | 8 |
| skiing | generic | 85 | 0 | OK | 8 |
| solaris | generic | **0 ✅** | — | OK | 8 |
| space_invaders | generic | **0 ✅** | — | OK | 8 |
| star_gunner | generic | **0 ✅** | — | OK | 8 |
| surround | generic | 16 | 0 | OK | 8 |
| tennis | generic | **0 ✅** | — | OK | 8 |
| time_pilot | generic | **0 ✅** | — | OK | 8 |
| tutankham | generic | **0 ✅** | — | OK | 8 |
| up_n_down | generic | **0 ✅** | — | OK | 8 |
| venture | generic | **0 ✅** | — | OK | 8 |
| video_pinball | generic | **0 ✅** | — | OK | 8 |
| videochess | generic | **0 ✅** | — | OK | 8 |
| wizard_of_wor | generic | **0 ✅** | — | OK | 8 |
| yars_revenge | generic | **0 ✅** | — | OK | 8 |
| zaxxon | generic | **0 ✅** | — | OK | 8 |
