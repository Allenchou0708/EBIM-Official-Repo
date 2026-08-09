---
name: ebim-local-lab-workflow
description: Coordinate EBiM work split between a local planning and Obsidian machine and a lab execution machine. Use when handing off EBiM code debugging, code review, Docker builds, GPU training, simulator runs, or experiment evidence to the lab, and when bringing the verified result back to local notes. Do not use for a self-contained task that can be completed and verified on one machine.
---

# EBiM Local-Lab Workflow

## Purpose

Use one owner for each kind of work:

- The local coordinator owns scope, acceptance criteria, project decisions, and Obsidian notes.
- The lab executor owns code debugging, code review, tests, local images, experiments, the final code commit, and the branch push.

Do not shuttle an unverified patch back to the local machine for a second implementation or review cycle. Keep runtime-dependent work on the lab machine from first edit through final push.

## Select the Role

Determine the role from the current environment and the requested work:

- Select **local coordinator** when the EBiM Obsidian vault is available and the task is planning, handoff preparation, evidence interpretation, or note maintenance.
- Select **lab executor** when the official/fork repository, Docker, GPU, simulator, datasets, or HDD experiment paths are available and the task is implementation or validation.
- If both are available, follow the role explicitly requested by the user. Do not silently take over the other role.

State the selected role in the first progress update.

## Local Coordinator Workflow

### 1. Reconstruct current state

Read the EBiM HOME, current project status, relevant runbook or implementation plan, and only the experiment notes needed for the task. Check the official repository state when it affects the plan.

Preserve these note rules:

- Put the only cross-day task in EBiM `HOME.md`.
- Do not duplicate that task in Daily or experiment notes.
- Keep code, full logs, datasets, checkpoints, and images outside Obsidian.
- Record only version, configuration, evidence pointers, interpretation, decision, and next action.
- Do not modify the LLM Wiki, the Embodied AI main initiative, or the historical external vault unless explicitly requested.

### 2. Write a bounded lab handoff

Define the outcome, not a brittle command transcript. Include:

- objective and why it matters;
- repository path, expected branch/base, and upstream/fork relationship;
- relevant dataset, cache, output, checkpoint, and evidence paths;
- allowed files or interfaces and explicit exclusions;
- reproduction signal or known failure;
- required runtime gates and Definition of Done;
- evidence the lab must return;
- stop conditions and decisions that require user approval.

Let the lab executor inspect the repository and choose the concrete debug steps. Do not prescribe every shell command unless the command itself is part of the interface being tested.

### 3. Wait for the complete lab result

Treat the lab executor as the code owner for the handoff. Do not independently recreate its patch, rerun GPU/simulator tests locally, or trigger intermediate GitHub Actions.

Accept the handoff only when it reports the final commit, clean working tree, required gates, experiment evidence, and pushed remote ref. If CI fails, return ownership to the lab executor for the minimal correction and final re-push.

### 4. Interpret and close the notes

Check that the returned evidence satisfies the Definition of Done. Update HOME and the relevant project/experiment note with:

- base and final commit;
- branch and pushed ref;
- concise diff scope;
- pass/fail gates and key metrics;
- dataset/image/checkpoint revisions and checksums when relevant;
- limitations, decision, and one next action.

Do not copy the full terminal output. Distinguish code-pipeline evidence from task-success evidence.

## Lab Executor Workflow

### 1. Establish a reproducible baseline

Read the local handoff, repository `STATUS.md`, relevant participant documentation, and existing local evidence before editing. Record:

- repository root, current branch, HEAD, upstream/fork heads, ahead/behind counts, and working tree;
- Docker image or environment revision;
- dataset revision and source metadata;
- GPU/runtime information when relevant.

Preserve unrelated user changes. Never add datasets, model weights, checkpoints, caches, generated outputs, or full logs to the competition repository.

For the current Task 2 PI0.5 workflow, keep bulk data under the HDD root (normally `/scratch1/2026_ebim/allen_task2_pi05`) and code on the SSD worktree. Verify the actual paths from the handoff rather than assuming them.

### 2. Debug and review locally

Create or use the handoff branch and keep iterations local. Reproduce the failure, make the smallest coherent fix, and review the resulting diff on the lab machine.

Run targeted tests first, then the complete release gates named in the handoff. Typical gates include:

- parser/config integration;
- focused unit tests and the full relevant suite;
- patch apply check or compile check;
- `git diff --check` and repository pre-commit hooks;
- local Docker image build and runtime doctor;
- one-step training, checkpoint reload, and resume;
- simulator/evaluator regression when the change affects live execution.

Keep full logs and experiment artifacts in the HDD evidence/output directories. Record hashes for material manifests, checkpoints, patches, and immutable images.

### 3. Commit and push from the lab

The lab executor owns completion of the code change:

1. Confirm all required gates pass and the working tree contains only intended source changes.
2. Run the final code review and inspect the staged diff.
3. Create or amend the final commit with the configured repository identity.
4. Push the completed branch to the collaboration fork once the local result is ready.
5. Check the pushed ref and any required CI result.

Avoid micro-pushes during debugging. If CI finds a real issue, fix and retest it on the lab machine, then push the correction. Use force-with-lease only on a private feature branch when rewriting it is explicitly acceptable.

### 4. Return compact evidence

Return a self-contained report containing:

```text
Outcome:
Repository / branch / base / final commit:
Remote ref and push result:
Working tree:
Diff stat and changed interfaces:
Tests and runtime gates:
Image tag, ID, and digest:
Dataset revision and split:
Key metrics:
HDD evidence, output, and checkpoint paths:
Material SHA-256 values:
Known limitations:
Recommended next action:
```

Include failure evidence only when it explains a decision. Do not paste complete logs into the report or edit the local Obsidian vault from the lab machine.

## Completion Rules

- A local handoff is complete when the lab has an unambiguous objective and measurable Definition of Done.
- A lab implementation is complete only after review, tests, final commit, push, and compact evidence return.
- A project cycle is complete only after the local coordinator interprets the evidence and updates the single cross-day next action.
- If a required gate cannot run, report the exact blocker and last passing gate; do not label the work complete.
