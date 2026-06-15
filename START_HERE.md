# Start Here

Use this file to get oriented quickly when returning to OpenDistillation.

## Current Project State

OpenDistillation is an open-source personal model factory concept for the AI PC and AI phone era.

The long-term direction is multiple small personal models, such as notes/school, coding, writing, work, and phone models.

The current prototype is much narrower:

> One Colab-first notes / school model prototype for `.txt` and `.md` notes.

It has a runnable notebook, helper package, deterministic mock teacher with varied question styles, a deterministic dataset quality report, a fact-ledger quality gate, a six-row-per-fact label/value local signal, an opt-in local Qwen real teacher path, a bounded optional TRL/PEFT LoRA training entry point, runtime readiness messages, and optional multi-question before/after quality report wiring. The latest GPU evidence is a small positive signal, not a solved model-quality result: the 2026-06-15 six-row label/value fact-ledger T4 smoke changed all 8 trained answers and improved exact held-out fact hits from 0/8 to 1/8, but still missed 7/8 checked facts. It does not yet have model export, local runtime instructions, useful held-out fact learning, or multiple model profiles.

## Read These In Order

1. `README.md` - public-facing project page.
2. `docs/product-vision.md` - personal model factory vision.
3. `docs/first-demo-flow.md` - exact v0 notes-model Colab flow.
4. `docs/current-decisions.md` - decisions already made.
5. `docs/roadmap.md` - staged product direction.
6. `docs/next-goal-prompt.md` - recommended next goal.
7. `AGENTS.md` - working agreements for coding agents.

## What To Do Next

The next useful implementation goal is in:

```text
docs/next-goal-prompt.md
```

It should improve the local fact-ledger label/value disambiguation signal before another GPU run. The six-row T4 smoke proved one held-out exact fact can be learned, but the adapter still swapped labels and values or invented values for 7/8 facts. The next work should stay local until the train/eval split, preview, and readiness report show a clearer way to teach each label's exact value without broadening v0.

## What Not To Do Next

- Do not start with a SaaS.
- Do not start with a Mac app.
- Do not add accounts, billing, dashboards, or cloud orchestration.
- Do not broaden v0 beyond `.txt` and `.md` notes input.
- Do not build multiple model profiles before the notes model works.
- Do not claim novel distillation algorithms.
- Do not commit generated datasets, model weights, checkpoints, `.env` files, or local machine config.
