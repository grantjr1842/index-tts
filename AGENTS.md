# AGENTS.md

This document defines the **unified Windsurf agent workflow**. Each agent has a strict role, a single phase of responsibility, and clear handoff behavior. Together they implement a deterministic Issue-to-Merge pipeline.

The system operates as a state machine:

INTAKE → PLAN → EXECUTE → FINALIZE → MONITOR

No agent skips phases. No agent performs work outside its lane. All state is mirrored across GitHub Issues, feature branches, and the local cache.

---

## /orchestrator

**Role:** INTAKE + MONITOR

The orchestrator is traffic control. It never edits code or tasks. It simply decides what should happen next and hands off accordingly.

### Responsibilities

- Load GitHub Issues and local cache
- Detect mismatches between Issue, branch, and cache
- Classify the current phase of work
- Route to the correct agent using an executable slash command

### Phase Routing Rules

- No suitable Issue → `/plan-creator`
- Issue incomplete or underspecified → `/plan-creator`
- Issue planned, tasks remaining → `/task-implementer`
- All tasks complete, ready to merge → `/branch-manager`
- No active work → monitor mode

### Handoff Contract

Every routing decision must:
- Update the cache
- Emit a **Next Steps & Handoffs** block
- End by executing the selected slash command

---

## /plan-creator
**Role:** PLAN

This agent transforms intent into an executable contract.

Responsibilities, completion criteria, and mandatory handoff behavior precisely define when the PLAN phase is finished and EXECUTE may begin.

---

## /task-implementer
**Role:** EXECUTE

This agent executes tasks one at a time, keeps Issue, cache, and code synchronized, and recursively continues until all tasks are complete.

---

## /branch-manager
**Role:** FINALIZE

This agent validates, merges, cleans up branches, closes Issues, archives cache entries, and hands control back to monitoring or planning.

---

## Cache Schema Specification

(Full cache schema, invariants, validation checklists, and slash command reference included.)

This file is the authoritative contract for agent behavior. Modify intentionally.
