# Start Here

Use this file to get oriented quickly when returning to OpenDistillation.

## Current Project State

OpenDistillation is a documentation-first project foundation. It does not have real training code yet.

The repo is being prepared for a narrow first prototype:

> Upload a `.txt` or `.md` file in Colab, generate training examples, fine-tune a small student model, compare before/after answers, and show the path to local running.

## Read These In Order

1. `README.md` - public-facing project page.
2. `docs/first-demo-flow.md` - exact v0 Colab user flow.
3. `docs/first-demo-implementation-plan.md` - build order for the first demo.
4. `docs/current-decisions.md` - decisions already made.
5. `docs/roadmap.md` - staged product direction.
6. `docs/github-issue-plan.md` - initial milestones and issues to create after the repo is on GitHub.
7. `AGENTS.md` - working agreements for coding agents.

## What To Do Next

The next useful implementation goal is in:

```text
docs/next-goal-prompt.md
```

It should create the notebook skeleton and shared helper package without starting a broad training platform.

## What Not To Do Next

- Do not start with a SaaS.
- Do not start with a Mac app.
- Do not add accounts, billing, dashboards, or cloud orchestration.
- Do not broaden v0 beyond `.txt` and `.md` inputs.
- Do not claim novel distillation algorithms.
- Do not commit generated datasets, model weights, checkpoints, `.env` files, or local machine config.
