# ROM sweep — jutari RAM bit-exactness vs xitari (NOOP, 30 frames)

Same paradigm as PXC1/PXC2: per-frame 128 B RIOT-RAM diff, jutari vs xitari trace_dump, NOOP from the standard ALE boot (60 NOOP + 4 RESET). `generic` = jutari ran with GenericRomSettings (no game-specific starting actions; divergence may be settings- not emulation-driven).

**Progress: 64/64 ROMs. Bit-exact (0 b/f): 64/64 completed.**

| game | settings | max RAM diff (b/f) | first div frame | status | secs |
|---|---|---|---|---|---|
| air_raid | generic | **0 ✅** | — | OK | 9 |
| alien | generic | **0 ✅** | — | OK | 10 |
| amidar | generic | **0 ✅** | — | OK | 10 |
| assault | generic | **0 ✅** | — | OK | 9 |
| asterix | generic | **0 ✅** | — | OK | 9 |
| asteroids | generic | **0 ✅** | — | OK | 9 |
| atlantis | generic | **0 ✅** | — | OK | 10 |
| bank_heist | generic | **0 ✅** | — | OK | 9 |
| battle_zone | generic | **0 ✅** | — | OK | 10 |
| beam_rider | generic | **0 ✅** | — | OK | 10 |
| berzerk | generic | **0 ✅** | — | OK | 10 |
| bowling | generic | **0 ✅** | — | OK | 10 |
| boxing | generic | **0 ✅** | — | OK | 10 |
| breakout | real | **0 ✅** | — | OK | 10 |
| carnival | generic | **0 ✅** | — | OK | 10 |
| centipede | generic | **0 ✅** | — | OK | 10 |
| chopper_command | generic | **0 ✅** | — | OK | 10 |
| crazy_climber | generic | **0 ✅** | — | OK | 10 |
| defender | generic | **0 ✅** | — | OK | 10 |
| demon_attack | generic | **0 ✅** | — | OK | 10 |
| double_dunk | generic | **0 ✅** | — | OK | 9 |
| elevator_action | generic | **0 ✅** | — | OK | 10 |
| enduro | real | **0 ✅** | — | OK | 10 |
| fishing_derby | generic | **0 ✅** | — | OK | 9 |
| freeway | generic | **0 ✅** | — | OK | 10 |
| frostbite | generic | **0 ✅** | — | OK | 10 |
| gopher | generic | **0 ✅** | — | OK | 10 |
| gravitar | generic | **0 ✅** | — | OK | 10 |
| hero | generic | **0 ✅** | — | OK | 10 |
| ice_hockey | generic | **0 ✅** | — | OK | 10 |
| jamesbond | generic | **0 ✅** | — | OK | 9 |
| journey_escape | generic | **0 ✅** | — | OK | 10 |
| kangaroo | generic | **0 ✅** | — | OK | 10 |
| krull | generic | **0 ✅** | — | OK | 10 |
| kung_fu_master | generic | **0 ✅** | — | OK | 10 |
| montezuma_revenge | generic | **0 ✅** | — | OK | 9 |
| ms_pacman | generic | **0 ✅** | — | OK | 9 |
| name_this_game | generic | **0 ✅** | — | OK | 10 |
| pacman | generic | **0 ✅** | — | OK | 10 |
| phoenix | generic | **0 ✅** | — | OK | 10 |
| pitfall | real | **0 ✅** | — | OK | 10 |
| pong | real | **0 ✅** | — | OK | 10 |
| pooyan | generic | **0 ✅** | — | OK | 10 |
| private_eye | generic | **0 ✅** | — | OK | 10 |
| qbert | generic | **0 ✅** | — | OK | 9 |
| riverraid | generic | **0 ✅** | — | OK | 10 |
| road_runner | generic | **0 ✅** | — | OK | 10 |
| robotank | generic | **0 ✅** | — | OK | 10 |
| seaquest | generic | **0 ✅** | — | OK | 10 |
| skiing | generic | **0 ✅** | — | OK | 10 |
| solaris | generic | **0 ✅** | — | OK | 10 |
| space_invaders | generic | **0 ✅** | — | OK | 10 |
| star_gunner | generic | **0 ✅** | — | OK | 10 |
| surround | generic | **0 ✅** | — | OK | 10 |
| tennis | generic | **0 ✅** | — | OK | 10 |
| time_pilot | generic | **0 ✅** | — | OK | 10 |
| tutankham | generic | **0 ✅** | — | OK | 10 |
| up_n_down | generic | **0 ✅** | — | OK | 10 |
| venture | generic | **0 ✅** | — | OK | 10 |
| video_pinball | generic | **0 ✅** | — | OK | 10 |
| videochess | generic | **0 ✅** | — | OK | 9 |
| wizard_of_wor | generic | **0 ✅** | — | OK | 9 |
| yars_revenge | generic | **0 ✅** | — | OK | 9 |
| zaxxon | generic | **0 ✅** | — | OK | 9 |
