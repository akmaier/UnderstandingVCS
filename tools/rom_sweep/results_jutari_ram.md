# ROM sweep — jutari RAM bit-exactness vs xitari (NOOP, 30 frames)

Same paradigm as PXC1/PXC2: per-frame 128 B RIOT-RAM diff, jutari vs xitari trace_dump, NOOP from the standard ALE boot (60 NOOP + 4 RESET). `generic` = jutari ran with GenericRomSettings (no game-specific starting actions; divergence may be settings- not emulation-driven).

**Progress: 64/64 ROMs. Bit-exact (0 b/f): 62/64 completed.**

| game | settings | max RAM diff (b/f) | first div frame | status | secs |
|---|---|---|---|---|---|
| air_raid | generic | **0 ✅** | — | OK | 10 |
| alien | generic | **0 ✅** | — | OK | 10 |
| amidar | generic | **0 ✅** | — | OK | 10 |
| assault | generic | **0 ✅** | — | OK | 10 |
| asterix | generic | **0 ✅** | — | OK | 10 |
| asteroids | generic | **0 ✅** | — | OK | 10 |
| atlantis | generic | **0 ✅** | — | OK | 11 |
| bank_heist | generic | **0 ✅** | — | OK | 11 |
| battle_zone | generic | **0 ✅** | — | OK | 11 |
| beam_rider | generic | **0 ✅** | — | OK | 11 |
| berzerk | generic | **0 ✅** | — | OK | 11 |
| bowling | generic | **0 ✅** | — | OK | 11 |
| boxing | generic | **0 ✅** | — | OK | 11 |
| breakout | real | **0 ✅** | — | OK | 11 |
| carnival | generic | **0 ✅** | — | OK | 11 |
| centipede | generic | **0 ✅** | — | OK | 11 |
| chopper_command | generic | **0 ✅** | — | OK | 11 |
| crazy_climber | generic | **0 ✅** | — | OK | 11 |
| defender | generic | **0 ✅** | — | OK | 9 |
| demon_attack | generic | **0 ✅** | — | OK | 9 |
| double_dunk | generic | **0 ✅** | — | OK | 9 |
| elevator_action | generic | **0 ✅** | — | OK | 9 |
| enduro | real | **0 ✅** | — | OK | 9 |
| fishing_derby | generic | **0 ✅** | — | OK | 9 |
| freeway | generic | **0 ✅** | — | OK | 9 |
| frostbite | generic | **0 ✅** | — | OK | 9 |
| gopher | generic | **0 ✅** | — | OK | 9 |
| gravitar | generic | **0 ✅** | — | OK | 9 |
| hero | generic | **0 ✅** | — | OK | 9 |
| ice_hockey | generic | **0 ✅** | — | OK | 9 |
| jamesbond | generic | **0 ✅** | — | OK | 9 |
| journey_escape | generic | **0 ✅** | — | OK | 9 |
| kangaroo | generic | **0 ✅** | — | OK | 9 |
| krull | generic | **0 ✅** | — | OK | 9 |
| kung_fu_master | generic | **0 ✅** | — | OK | 9 |
| montezuma_revenge | generic | **0 ✅** | — | OK | 9 |
| ms_pacman | generic | **0 ✅** | — | OK | 9 |
| name_this_game | generic | **0 ✅** | — | OK | 9 |
| pacman | generic | **0 ✅** | — | OK | 9 |
| phoenix | generic | **0 ✅** | — | OK | 9 |
| pitfall | real | **0 ✅** | — | OK | 9 |
| pong | real | **0 ✅** | — | OK | 9 |
| pooyan | generic | **0 ✅** | — | OK | 9 |
| private_eye | generic | **0 ✅** | — | OK | 9 |
| qbert | generic | **0 ✅** | — | OK | 9 |
| riverraid | generic | **0 ✅** | — | OK | 9 |
| road_runner | generic | **0 ✅** | — | OK | 9 |
| robotank | generic | **0 ✅** | — | OK | 9 |
| seaquest | generic | **0 ✅** | — | OK | 9 |
| skiing | generic | 1 | 0 | OK | 9 |
| solaris | generic | **0 ✅** | — | OK | 9 |
| space_invaders | generic | **0 ✅** | — | OK | 9 |
| star_gunner | generic | **0 ✅** | — | OK | 9 |
| surround | generic | 7 | 0 | OK | 9 |
| tennis | generic | **0 ✅** | — | OK | 9 |
| time_pilot | generic | **0 ✅** | — | OK | 9 |
| tutankham | generic | **0 ✅** | — | OK | 9 |
| up_n_down | generic | **0 ✅** | — | OK | 9 |
| venture | generic | **0 ✅** | — | OK | 9 |
| video_pinball | generic | **0 ✅** | — | OK | 9 |
| videochess | generic | **0 ✅** | — | OK | 9 |
| wizard_of_wor | generic | **0 ✅** | — | OK | 8 |
| yars_revenge | generic | **0 ✅** | — | OK | 8 |
| zaxxon | generic | **0 ✅** | — | OK | 8 |
