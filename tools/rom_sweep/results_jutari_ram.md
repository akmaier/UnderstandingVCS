# ROM sweep — jutari RAM bit-exactness vs xitari (NOOP, 30 frames)

Same paradigm as PXC1/PXC2: per-frame 128 B RIOT-RAM diff, jutari vs xitari trace_dump, NOOP from the standard ALE boot (60 NOOP + 4 RESET). `generic` = jutari ran with GenericRomSettings (no game-specific starting actions; divergence may be settings- not emulation-driven).

**Progress: 64/64 ROMs. Bit-exact (0 b/f): 62/64 completed.**

| game | settings | max RAM diff (b/f) | first div frame | status | secs |
|---|---|---|---|---|---|
| air_raid | generic | **0 ✅** | — | OK | 18 |
| alien | generic | **0 ✅** | — | OK | 18 |
| amidar | generic | **0 ✅** | — | OK | 18 |
| assault | generic | **0 ✅** | — | OK | 18 |
| asterix | generic | **0 ✅** | — | OK | 18 |
| asteroids | generic | **0 ✅** | — | OK | 18 |
| atlantis | generic | **0 ✅** | — | OK | 19 |
| bank_heist | generic | **0 ✅** | — | OK | 19 |
| battle_zone | generic | **0 ✅** | — | OK | 19 |
| beam_rider | generic | **0 ✅** | — | OK | 19 |
| berzerk | generic | **0 ✅** | — | OK | 18 |
| bowling | generic | **0 ✅** | — | OK | 19 |
| boxing | generic | **0 ✅** | — | OK | 19 |
| breakout | real | **0 ✅** | — | OK | 18 |
| carnival | generic | **0 ✅** | — | OK | 19 |
| centipede | generic | **0 ✅** | — | OK | 19 |
| chopper_command | generic | **0 ✅** | — | OK | 19 |
| crazy_climber | generic | **0 ✅** | — | OK | 19 |
| defender | generic | **0 ✅** | — | OK | 18 |
| demon_attack | generic | **0 ✅** | — | OK | 18 |
| double_dunk | generic | **0 ✅** | — | OK | 18 |
| elevator_action | generic | **0 ✅** | — | OK | 18 |
| enduro | real | **0 ✅** | — | OK | 18 |
| fishing_derby | generic | **0 ✅** | — | OK | 19 |
| freeway | generic | **0 ✅** | — | OK | 19 |
| frostbite | generic | **0 ✅** | — | OK | 18 |
| gopher | generic | **0 ✅** | — | OK | 18 |
| gravitar | generic | **0 ✅** | — | OK | 18 |
| hero | generic | **0 ✅** | — | OK | 19 |
| ice_hockey | generic | **0 ✅** | — | OK | 19 |
| jamesbond | generic | **0 ✅** | — | OK | 18 |
| journey_escape | generic | **0 ✅** | — | OK | 18 |
| kangaroo | generic | **0 ✅** | — | OK | 18 |
| krull | generic | **0 ✅** | — | OK | 18 |
| kung_fu_master | generic | **0 ✅** | — | OK | 19 |
| montezuma_revenge | generic | **0 ✅** | — | OK | 18 |
| ms_pacman | generic | **0 ✅** | — | OK | 18 |
| name_this_game | generic | **0 ✅** | — | OK | 18 |
| pacman | generic | **0 ✅** | — | OK | 18 |
| phoenix | generic | **0 ✅** | — | OK | 18 |
| pitfall | real | **0 ✅** | — | OK | 18 |
| pong | real | **0 ✅** | — | OK | 18 |
| pooyan | generic | **0 ✅** | — | OK | 18 |
| private_eye | generic | **0 ✅** | — | OK | 19 |
| qbert | generic | **0 ✅** | — | OK | 19 |
| riverraid | generic | **0 ✅** | — | OK | 18 |
| road_runner | generic | **0 ✅** | — | OK | 18 |
| robotank | generic | **0 ✅** | — | OK | 19 |
| seaquest | generic | **0 ✅** | — | OK | 18 |
| skiing | generic | 1 | 0 | OK | 19 |
| solaris | generic | **0 ✅** | — | OK | 19 |
| space_invaders | generic | **0 ✅** | — | OK | 19 |
| star_gunner | generic | **0 ✅** | — | OK | 18 |
| surround | generic | 7 | 0 | OK | 19 |
| tennis | generic | **0 ✅** | — | OK | 18 |
| time_pilot | generic | **0 ✅** | — | OK | 17 |
| tutankham | generic | **0 ✅** | — | OK | 18 |
| up_n_down | generic | **0 ✅** | — | OK | 18 |
| venture | generic | **0 ✅** | — | OK | 18 |
| video_pinball | generic | **0 ✅** | — | OK | 17 |
| videochess | generic | **0 ✅** | — | OK | 11 |
| wizard_of_wor | generic | **0 ✅** | — | OK | 11 |
| yars_revenge | generic | **0 ✅** | — | OK | 11 |
| zaxxon | generic | **0 ✅** | — | OK | 10 |
