# ROM sweep — jutari RAM bit-exactness vs xitari (NOOP, 30 frames)

Same paradigm as PXC1/PXC2: per-frame 128 B RIOT-RAM diff, jutari vs xitari trace_dump, NOOP from the standard ALE boot (60 NOOP + 4 RESET). `generic` = jutari ran with GenericRomSettings (no game-specific starting actions; divergence may be settings- not emulation-driven).

**Progress: 64/64 ROMs. Bit-exact (0 b/f): 64/64 completed.**

| game | settings | max RAM diff (b/f) | first div frame | status | secs |
|---|---|---|---|---|---|
| air_raid | generic | **0 ✅** | — | OK | 12 |
| alien | generic | **0 ✅** | — | OK | 13 |
| amidar | generic | **0 ✅** | — | OK | 13 |
| assault | generic | **0 ✅** | — | OK | 16 |
| asterix | generic | **0 ✅** | — | OK | 13 |
| asteroids | generic | **0 ✅** | — | OK | 12 |
| atlantis | generic | **0 ✅** | — | OK | 13 |
| bank_heist | generic | **0 ✅** | — | OK | 13 |
| battle_zone | generic | **0 ✅** | — | OK | 13 |
| beam_rider | generic | **0 ✅** | — | OK | 13 |
| berzerk | generic | **0 ✅** | — | OK | 12 |
| bowling | generic | **0 ✅** | — | OK | 12 |
| boxing | generic | **0 ✅** | — | OK | 13 |
| breakout | real | **0 ✅** | — | OK | 13 |
| carnival | generic | **0 ✅** | — | OK | 13 |
| centipede | generic | **0 ✅** | — | OK | 13 |
| chopper_command | generic | **0 ✅** | — | OK | 13 |
| crazy_climber | generic | **0 ✅** | — | OK | 13 |
| defender | generic | **0 ✅** | — | OK | 13 |
| demon_attack | generic | **0 ✅** | — | OK | 13 |
| double_dunk | generic | **0 ✅** | — | OK | 13 |
| elevator_action | generic | **0 ✅** | — | OK | 13 |
| enduro | real | **0 ✅** | — | OK | 13 |
| fishing_derby | generic | **0 ✅** | — | OK | 13 |
| freeway | generic | **0 ✅** | — | OK | 13 |
| frostbite | generic | **0 ✅** | — | OK | 13 |
| gopher | generic | **0 ✅** | — | OK | 13 |
| gravitar | generic | **0 ✅** | — | OK | 13 |
| hero | generic | **0 ✅** | — | OK | 13 |
| ice_hockey | generic | **0 ✅** | — | OK | 13 |
| jamesbond | generic | **0 ✅** | — | OK | 12 |
| journey_escape | generic | **0 ✅** | — | OK | 12 |
| kangaroo | generic | **0 ✅** | — | OK | 12 |
| krull | generic | **0 ✅** | — | OK | 12 |
| kung_fu_master | generic | **0 ✅** | — | OK | 13 |
| montezuma_revenge | generic | **0 ✅** | — | OK | 12 |
| ms_pacman | generic | **0 ✅** | — | OK | 13 |
| name_this_game | generic | **0 ✅** | — | OK | 13 |
| pacman | generic | **0 ✅** | — | OK | 13 |
| phoenix | generic | **0 ✅** | — | OK | 13 |
| pitfall | real | **0 ✅** | — | OK | 13 |
| pong | real | **0 ✅** | — | OK | 13 |
| pooyan | generic | **0 ✅** | — | OK | 13 |
| private_eye | generic | **0 ✅** | — | OK | 13 |
| qbert | generic | **0 ✅** | — | OK | 13 |
| riverraid | generic | **0 ✅** | — | OK | 13 |
| road_runner | generic | **0 ✅** | — | OK | 13 |
| robotank | generic | **0 ✅** | — | OK | 13 |
| seaquest | generic | **0 ✅** | — | OK | 12 |
| skiing | generic | **0 ✅** | — | OK | 12 |
| solaris | generic | **0 ✅** | — | OK | 12 |
| space_invaders | generic | **0 ✅** | — | OK | 12 |
| star_gunner | generic | **0 ✅** | — | OK | 12 |
| surround | generic | **0 ✅** | — | OK | 12 |
| tennis | generic | **0 ✅** | — | OK | 12 |
| time_pilot | generic | **0 ✅** | — | OK | 12 |
| tutankham | generic | **0 ✅** | — | OK | 13 |
| up_n_down | generic | **0 ✅** | — | OK | 13 |
| venture | generic | **0 ✅** | — | OK | 13 |
| video_pinball | generic | **0 ✅** | — | OK | 13 |
| videochess | generic | **0 ✅** | — | OK | 13 |
| wizard_of_wor | generic | **0 ✅** | — | OK | 13 |
| yars_revenge | generic | **0 ✅** | — | OK | 12 |
| zaxxon | generic | **0 ✅** | — | OK | 11 |
