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

## Competition-Speed Policy

Optimize for finishing a working robotics competition entry before the deadline. This is not a security-hardening or adversarial network-defense project.

- Do not add speculative guards, redundant validations, generalized abstractions, or failure branches without evidence that they address a current competition blocker.
- Prefer the smallest direct implementation that can be tested in the simulator. Reuse official interfaces and existing repository behavior.
- Keep only essential physical safety: shadow-before-publish, no unintended base/spine command, bounded actuator commands, emergency stop, and publisher-contention checks when commands can move the robot.
- Do not turn ordinary file, configuration, or provenance handling into a security protocol. A Git commit, dataset revision, image digest, or checkpoint path is normally sufficient identification.
- Do not calculate or report SHA-256 hashes by default. Use a checksum only when the user explicitly requests it, corruption is suspected, or a submission/download interface specifically requires integrity verification. Never produce several equivalent hashes for the same run.
- Use one minimal smoke test that exercises the changed path. Add a second targeted test only for a distinct high-risk interface. Do not run broad suites, repeated resume tests, clean-clone rehearsals, or GitHub Actions during iteration unless the change reaches a release/submission boundary or a failure makes them necessary.
- Stop expanding infrastructure once the current gate is measurable. Move quickly to simulator evidence and task success.

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
- the smallest decisive runtime gate and Definition of Done;
- evidence the lab must return;
- stop conditions and decisions that require user approval.

Let the lab executor inspect the repository and choose the concrete debug steps. Do not prescribe every shell command unless the command itself is part of the interface being tested.

### 3. Wait for the complete lab result

Treat the lab executor as the code owner for the handoff. Do not independently recreate its patch, rerun GPU/simulator tests locally, or trigger intermediate GitHub Actions.

Accept the handoff when it reports the final commit, intended diff, the agreed minimal gate, experiment evidence, and pushed remote ref. Require CI only when it is already part of the release path or the user asks for it. If required CI fails, return ownership to the lab executor for the minimal correction and final re-push.

### 4. Interpret and close the notes

Check that the returned evidence satisfies the Definition of Done. Update HOME and the relevant project/experiment note with:

- base and final commit;
- branch and pushed ref;
- concise diff scope;
- pass/fail gates and key metrics;
- dataset/image/checkpoint identifiers needed to reproduce the result;
- limitations, decision, and one next action.

Do not copy the full terminal output. Distinguish code-pipeline evidence from task-success evidence.

## Lab Executor Workflow

### 1. Establish a reproducible baseline

Read the local handoff, repository `STATUS.md`, relevant participant documentation, and existing local evidence before editing. Record only what is needed to avoid working on the wrong code or data:

- repository root, current branch, HEAD, and working tree;
- Docker image/environment and dataset/checkpoint paths used by the run;
- GPU/runtime information only when it can explain the current failure.

Do not repeatedly fetch every remote, print long commit graphs, inventory the whole machine, or re-verify unchanged data on every iteration.

Preserve unrelated user changes. Never add datasets, model weights, checkpoints, caches, generated outputs, or full logs to the competition repository.

For the current Task 2 PI0.5 workflow, keep bulk data under the HDD root (normally `/scratch1/2026_ebim/allen_task2_pi05`) and code on the SSD worktree. Verify the actual paths from the handoff rather than assuming them.

### 2. Debug and review locally

Create or use the handoff branch and keep iterations local. Reproduce the failure, make the smallest coherent fix, and review the resulting diff on the lab machine.

Run the smallest test that can disprove the fix. A normal debug cycle is: reproduce once, patch, run one focused smoke, then run the simulator or training gate that matters. Possible targeted gates include:

- one parser/config or focused unit test;
- one-step training only when the training path changed;
- checkpoint load/inference only when serialization or inference changed;
- simulator/evaluator smoke when live execution changed;
- full pre-commit or broader suites only immediately before a release/submission push, or when requested.

Keep useful logs and experiment artifacts in the HDD evidence/output directories. Do not preserve duplicate logs, failed scratch outputs, or routine hashes unless they are needed to diagnose the result.

### 3. Commit and push from the lab

The lab executor owns completion of the code change:

1. Confirm the agreed minimal gate passes and the working tree contains only intended source changes.
2. Run the final code review and inspect the staged diff.
3. Create or amend the final commit with the configured repository identity.
4. Push the completed branch to the collaboration fork once the local result is ready.
5. Check the pushed ref. Check CI only when required for this boundary.

Avoid micro-pushes during debugging. If CI finds a real issue, fix and retest it on the lab machine, then push the correction. Use force-with-lease only on a private feature branch when rewriting it is explicitly acceptable.

### 4. Return compact evidence

Return a self-contained report containing:

```text
Outcome:
Repository / branch / base / final commit:
Remote ref and push result:
Working tree:
Changed interfaces:
Minimal smoke/runtime gate:
Dataset, image, checkpoint, and output identifiers actually used:
Key result or failure metric:
Known limitations:
Recommended next action:
```

Include failure evidence only when it explains a decision. Do not paste complete logs into the report or edit the local Obsidian vault from the lab machine.

## Completion Rules

- A local handoff is complete when the lab has an unambiguous objective and measurable Definition of Done.
- A lab implementation is complete after a focused review, the agreed minimal gate, final commit, push, and compact evidence return. Do not delay simulator progress for optional hardening.
- A project cycle is complete only after the local coordinator interprets the evidence and updates the single cross-day next action.
- If a required gate cannot run, report the exact blocker and last passing gate; do not label the work complete.
