---
id: P2-EX-N
title: <short imperative title>
epic: EX (<epic name>)
status: todo            # todo | in-sprint | in-progress | review | done | blocked
sprint:                 # set at Planning
owner:                  # agent slot, set at Planning
where: local            # local | cluster
depends_on: []          # list of item IDs
file_scope:             # EXACT paths this item may create/modify (the conflict key)
  - <path/glob>
estimate: M             # S | M | L
spec_ref: ../SPEC.md#<anchor>
---

## Goal
<one paragraph: exactly what to produce>

## Definition of Done
- [ ] runs to completion from this command: `<command>`
- [ ] result written to `<out path>` in the schema (SPEC §R)
- [ ] nothing outside `file_scope` changed; Paper-1 gates untouched/green
- [ ] committed + pushed to main (rebase-before-push); primary pulled ff-only
- [ ] `status: done`

## Notes / handoff
<anything Review needs; cross-refs to experiment_design.md sections>
