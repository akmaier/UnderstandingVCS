# P2 backlog

The work queue for Paper 2, driven by [`../SPEC.md`](../SPEC.md) under the process in
[`../../SCRUM.md`](../../SCRUM.md). One file per item: `<ID>.md` (schema in SCRUM §3;
copy [`TEMPLATE.md`](TEMPLATE.md)). The sprint board is [`BOARD.md`](BOARD.md).

## How it is used
- **Sprint 0 (E0-5) builds this dir:** an SM/agent reads `../SPEC.md` and emits one
  `<ID>.md` per task (IDs like `P2-E1-1`), fills `BOARD.md`, and assigns each a tentative
  sprint + **file-scope**, verifying that items sharing a sprint have **disjoint
  file-scopes** (SCRUM §0). Until Sprint 0 runs, this dir holds only this README +
  TEMPLATE + an empty BOARD.
- **Planning** sets `sprint`, `owner`, `status: in-sprint` (SM, on BOARD + item files).
- **Developers** edit **only the `status`** of items assigned to them (`in-progress` →
  `done`), in their own item file, in their own worktree. They never edit BOARD.
- **Review** reconciles statuses into BOARD, integrates shared files, adds discovered
  items (SM).

## Status lifecycle
`todo → in-sprint → in-progress → review → done` (or `blocked`, with a note).

## The one rule
No two items in the same sprint may list the same path in `file_scope`. If they would,
split the item or move it to another sprint (resolved at Planning/Review, never at runtime).
