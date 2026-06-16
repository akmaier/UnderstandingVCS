# ROM sweep — jutari RAM bit-exactness vs xitari (NOOP, 30 frames)

Same paradigm as PXC1/PXC2: per-frame 128 B RIOT-RAM diff, jutari vs xitari trace_dump, NOOP from the standard ALE boot (60 NOOP + 4 RESET). `generic` = jutari ran with GenericRomSettings (no game-specific starting actions; divergence may be settings- not emulation-driven).

**Progress: 64/64 ROMs. Bit-exact (0 b/f): 64/64 completed.**

| game | settings | max RAM diff (b/f) | first div frame | status | secs |
|---|---|---|---|---|---|
| air_raid | generic | **0 ✅** | — | OK | 14 |
| alien | generic | **0 ✅** | — | OK | 14 |
| amidar | generic | **0 ✅** | — | OK | 14 |
| assault | generic | **0 ✅** | — | OK | 14 |
| asterix | generic | **0 ✅** | — | OK | 14 |
| asteroids | generic | **0 ✅** | — | OK | 13 |
| atlantis | generic | **0 ✅** | — | OK | 12 |
| bank_heist | generic | **0 ✅** | — | OK | 11 |
| battle_zone | generic | **0 ✅** | — | OK | 11 |
| beam_rider | generic | **0 ✅** | — | OK | 12 |
| berzerk | generic | **0 ✅** | — | OK | 11 |
| bowling | generic | **0 ✅** | — | OK | 11 |
| boxing | generic | **0 ✅** | — | OK | 12 |
| breakout | real | **0 ✅** | — | OK | 12 |
| carnival | generic | **0 ✅** | — | OK | 11 |
| centipede | generic | **0 ✅** | — | OK | 12 |
| chopper_command | generic | **0 ✅** | — | OK | 12 |
| crazy_climber | generic | **0 ✅** | — | OK | 12 |
| defender | generic | **0 ✅** | — | OK | 12 |
| demon_attack | generic | **0 ✅** | — | OK | 12 |
| double_dunk | generic | **0 ✅** | — | OK | 12 |
| elevator_action | generic | **0 ✅** | — | OK | 12 |
| enduro | real | **0 ✅** | — | OK | 12 |
| fishing_derby | generic | **0 ✅** | — | OK | 12 |
| freeway | generic | **0 ✅** | — | OK | 11 |
| frostbite | generic | **0 ✅** | — | OK | 11 |
| gopher | generic | **0 ✅** | — | OK | 10 |
| gravitar | generic | **0 ✅** | — | OK | 10 |
| hero | generic | **0 ✅** | — | OK | 10 |
| ice_hockey | generic | **0 ✅** | — | OK | 11 |
| jamesbond | generic | **0 ✅** | — | OK | 12 |
| journey_escape | generic | **0 ✅** | — | OK | 12 |
| kangaroo | generic | **0 ✅** | — | OK | 12 |
| krull | generic | **0 ✅** | — | OK | 12 |
| kung_fu_master | generic | **0 ✅** | — | OK | 12 |
| montezuma_revenge | generic | **0 ✅** | — | OK | 12 |
| ms_pacman | generic | **0 ✅** | — | OK | 13 |
| name_this_game | generic | **0 ✅** | — | OK | 13 |
| pacman | generic | **0 ✅** | — | OK | 13 |
| phoenix | generic | **0 ✅** | — | OK | 13 |
| pitfall | real | **0 ✅** | — | OK | 13 |
| pong | real | **0 ✅** | — | OK | 13 |
| pooyan | generic | **0 ✅** | — | OK | 12 |
| private_eye | generic | **0 ✅** | — | OK | 12 |
| qbert | generic | **0 ✅** | — | OK | 12 |
| riverraid | generic | **0 ✅** | — | OK | 12 |
| road_runner | generic | **0 ✅** | — | OK | 12 |
| robotank | generic | **0 ✅** | — | OK | 12 |
| seaquest | generic | **0 ✅** | — | OK | 12 |
| skiing | generic | **0 ✅** | — | OK | 13 |
| solaris | generic | **0 ✅** | — | OK | 12 |
| space_invaders | generic | **0 ✅** | — | OK | 13 |
| star_gunner | generic | **0 ✅** | — | OK | 12 |
| surround | generic | **0 ✅** | — | OK | 13 |
| tennis | generic | **0 ✅** | — | OK | 12 |
| time_pilot | generic | **0 ✅** | — | OK | 12 |
| tutankham | generic | **0 ✅** | — | OK | 12 |
| up_n_down | generic | **0 ✅** | — | OK | 12 |
| venture | generic | **0 ✅** | — | OK | 12 |
| video_pinball | generic | **0 ✅** | — | OK | 12 |
| videochess | generic | **0 ✅** | — | OK | 12 |
| wizard_of_wor | generic | **0 ✅** | — | OK | 11 |
| yars_revenge | generic | **0 ✅** | — | OK | 11 |
| zaxxon | generic | **0 ✅** | — | OK | 11 |
