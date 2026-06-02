# GitHub-Ready Prototype Documentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prepare OpenDistillation for a GitHub push and the first prototype implementation without building the training pipeline.

**Architecture:** Keep this change documentation-only. The README is the public front door, `docs/` contains aligned product and prototype specs, and `.github/ISSUE_TEMPLATE/` contains structured issue forms for later GitHub work.

**Tech Stack:** Markdown, GitHub issue forms, Git, future Colab notebook, future Python helpers.

---

### Task 1: Rewrite The Public README

**Files:**
- Modify: `README.md`

- [x] **Step 1: Explain the project in the first screen**

Make the README open with the personal model factory positioning while clearly narrowing v0 to the notes / school model.

- [x] **Step 2: Separate current status from future plans**

List what exists now and what does not exist yet so the repo does not overclaim.

- [x] **Step 3: Define v0 scope**

State the TXT/MD-only Colab prototype path and list excluded work such as SaaS, Mac app, accounts, billing, and cloud backend.

### Task 2: Align Product Docs

**Files:**
- Modify: `START_HERE.md`
- Modify: `docs/product-vision.md`
- Modify: `docs/roadmap.md`
- Modify: `docs/current-decisions.md`
- Modify: `docs/first-demo-flow.md`
- Modify: `docs/agent-handoff.md`
- Modify: `docs/next-goal-prompt.md`
- Modify: `notebooks/README.md`

- [x] **Step 1: Use one product promise everywhere**

Use the personal model factory framing as the long-term public promise, and the notes / school model as the v0 promise.

- [x] **Step 2: Specify the v0 Colab flow**

Write the step-by-step notebook flow from upload through local-run guidance.

- [x] **Step 3: Update the next-goal prompt**

Point the next goal at creating the notebook skeleton, not broad training work.

### Task 3: Add Prototype Planning Docs

**Files:**
- Create: `docs/first-demo-implementation-plan.md`
- Create: `docs/github-issue-plan.md`
- Modify: `docs/github-launch-checklist.md`

- [x] **Step 1: Add first-demo implementation plan**

Define milestones for skeleton, loading/chunking, dataset generation, short training, comparison, export, and public launch.

- [x] **Step 2: Add starter issue plan**

List labels, milestones, and starter issues to create once a GitHub remote exists.

- [x] **Step 3: Update launch checklist**

Separate documentation readiness from prototype readiness and leave license/remote tasks unchecked.

### Task 4: Add GitHub Templates

**Files:**
- Delete: `.github/ISSUE_TEMPLATE/docs-task.md`
- Delete: `.github/ISSUE_TEMPLATE/prototype-task.md`
- Create: `.github/ISSUE_TEMPLATE/docs-task.yml`
- Create: `.github/ISSUE_TEMPLATE/prototype-task.yml`
- Create: `.github/ISSUE_TEMPLATE/research-decision.yml`
- Create: `.github/ISSUE_TEMPLATE/config.yml`
- Create: `.github/PULL_REQUEST_TEMPLATE.md`

- [x] **Step 1: Replace Markdown templates with issue forms**

Use GitHub issue-form YAML with `name`, `description`, and `body` fields.

- [x] **Step 2: Add a research decision form**

Make model/backend/package decisions explicit before implementation.

- [x] **Step 3: Add a pull request checklist**

Require docs honesty, verification, and artifact/secret checks.

### Task 5: Verify And Commit

**Files:**
- Modify: `.gitignore`
- Create: `LICENSE`
- Review: all changed files

- [x] **Step 1: Extend ignore rules**

Ensure common model outputs and experiment folders are ignored.

- [x] **Step 2: Validate YAML**

Parse `.github/ISSUE_TEMPLATE/*.yml` locally.

- [x] **Step 3: Search for secrets and artifacts**

Check for `.env`, key files, generated datasets, checkpoints, model weights, and GGUF files.

- [x] **Step 4: Review diff**

Inspect `git diff --stat` and `git diff`.

- [x] **Step 5: Commit locally**

Stage only intended files and commit with message `docs: prepare GitHub-ready prototype plan`.
