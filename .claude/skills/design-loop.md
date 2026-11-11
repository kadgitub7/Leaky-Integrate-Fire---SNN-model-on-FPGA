# Design Loop Orchestrator

## Purpose

This skill orchestrates the full design team loop. Invoke it to run a complete cycle from objective through summary, or to resume a cycle that was interrupted. It coordinates the other skills in the correct order and enforces the protocol.

## Design Loop Protocol

### Standard Cycle (for each task/subtask)

```
User Request
  |
  v
[1] /objectives-engineer   — capture hard specs, confirm with user
  |
  v (user confirms)
[2] /architect-planner      — formulate approach, decompose steps, confirm with user
  |
  v (user confirms)
[3] Execute (Claude + /analog-specialist as needed)
  |  - Follow plan step by step
  |  - Consult analog specialist for circuit decisions
  |  - Mini-verify after each step before proceeding
  |
  v
[4] /verification-engineer  — design tests, run them, compare to OBJ criteria
  |
  v
[5] /design-reviewer        — audit all work products, check for errors
  |
  v (if APPROVED or APPROVED_WITH_CONCERNS)
[6] /integration-engineer   — check cross-block compatibility
  |
  v
[7] /program-manager        — synthesize status, present to user, recommend next step
  |
  v
User Decision -> next iteration or new objective
```

### If Design Reviewer returns BLOCK or REVISE:
- BLOCK: Halt. Present the critical finding to the user. Create an emergency sub-objective for the fix.
- REVISE: Return to step [2] with the reviewer's findings as input to replanning.

### Abbreviated Cycle (for small changes)
For changes that modify a single parameter, have a clear verification path, and do not cross block boundaries:

1. Confirm criteria from existing OBJ
2. Execute change + consult /analog-specialist
3. /verification-engineer runs targeted test
4. /design-reviewer quick check
5. /program-manager micro-summary

### Emergency Cycle (for critical bugs)
When a critical issue is found:

1. HALT current plan
2. /objectives-engineer captures the bug as a blocking objective
3. /analog-specialist provides root cause analysis
4. /architect-planner provides fix plan (shortest path)
5. Execute fix
6. /verification-engineer tests fix + regression
7. Resume original plan

## Log File Locations

All skills write to `design_logs/` in the project root:

```
design_logs/
  objectives/     — OBJ-*.json, active_objectives.json
  plans/          — PLAN-*.json, active_plan.json
  analog_kb/      — knowledge_base.json, circuit_catalog.json, design_rules.json, literature_database.json
  verification/   — VER-*.json, test_catalog.json
  reviews/        — REV-*.json, issue_tracker.json
  integration/    — INT-*.json, interface_register.json
  summaries/      — SUM-*.json, decision_log.json
```

## Log Record Standard Format

Every log record uses this envelope:
```json
{
  "id": "PREFIX-YYYY-MM-DD-NNN",
  "timestamp": "ISO-8601",
  "skill": "which_skill_created_this",
  "parent_objective": "OBJ-...",
  "parent_plan": "PLAN-...",
  "status": "draft|active|completed|superseded|rejected",
  "summary": "one-line human-readable",
  "body": { ... },
  "references": ["file paths or other record IDs consulted"],
  "supersedes": null
}
```

## Staleness Policy

- **Verification results** are stale if the circuit file they tested has been modified since the test ran. Must re-run.
- **Cache files** (.npz, .json in _cache/) are stale if the SPICE netlist or test script has been modified. Flag and regenerate.
- **Calibration data** (delay timer pipeline delays, SF calibration) is stale if delay_timer.spice or bias_gen.spice has changed.
- **Plans** are stale if the objective they reference has been superseded.

## How to Start

To begin a design cycle, the user can:
1. Describe what they want to work on (triggers /objectives-engineer)
2. Say "status" or "where are we?" (triggers /program-manager)
3. Say "review" (triggers /design-reviewer on recent work)
4. Name a specific skill to invoke directly

The orchestrator ensures skills run in the correct order and that log files are properly cross-referenced.
