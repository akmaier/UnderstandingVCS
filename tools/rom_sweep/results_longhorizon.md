# Long-horizon conformance sweep (problem games only)

Screen first-divergence jutari vs xitari via the 60fps comparison-video pipeline. The in-window sweeps (30f RAM / 60f screen) pass for all of these; divergence is post-window. `ju_frozen`>0 with `xi_frozen`=0 means jutari is stuck at game-over while xitari continues.

| game | first div frame | sec | ju_frozen | xi_frozen | diverging frames |
|---|---|---|---|---|---|
| asteroids | 194 | 3.2 | 0 | 359 | 167 |
| wizard_of_wor | 219 | 3.6 | 0 | 0 | 102 |
| berzerk | 581 | 9.7 | 0 | 0 | 167 |
| road_runner | 765 | 12.8 | 0 | 0 | 136 |
| montezuma_revenge | 867 | 14.4 | 2 | 0 | 174 |
| riverraid | 958 | 16.0 | 0 | 0 | 143 |
| space_invaders | 1092 | 18.2 | 0 | 0 | 169 |
| asterix | 1160 | 19.3 | 0 | 0 | 161 |
| pooyan | 1605 | 26.8 | 0 | 0 | 176 |
| kangaroo | 1720 | 28.7 | 81 | 0 | 81 |
| phoenix | 1743 | 29.1 | 0 | 0 | 58 |
| pacman | 1771 | 29.5 | 0 | 0 | 30 |
| ms_pacman | 1786 | 29.8 | 0 | 0 | 15 |
