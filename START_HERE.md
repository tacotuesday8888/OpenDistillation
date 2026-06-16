# Start Here

Use this file to get oriented quickly when returning to OpenDistillation.

## Current Project State

OpenDistillation is an open-source personal model factory concept for the AI PC and AI phone era.

The long-term direction is multiple small personal models, such as notes/school, coding, writing, work, and phone models.

The current prototype is much narrower:

> One Colab-first notes / school model prototype for `.txt` and `.md` notes.

It has a runnable notebook, helper package, deterministic mock teacher with varied question styles, a deterministic dataset quality report, a fact-ledger quality gate, a six-row-per-fact label/value local signal, same-chunk label/value disambiguation rows, an opt-in local Qwen real teacher path, a bounded optional TRL/PEFT LoRA training entry point, runtime readiness messages, and optional multi-question before/after quality report wiring. The latest GPU evidence is a small positive signal, not a solved model-quality result: the 2026-06-15 six-row label/value fact-ledger T4 smoke changed all 8 trained answers and improved exact held-out fact hits from 0/8 to 1/8, but still missed 7/8 checked facts. The new disambiguation rows have not been GPU-tested yet. It does not yet have model export, local runtime instructions, useful held-out fact learning, or multiple model profiles.

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

It should run and document exactly one bounded Colab T4 smoke for the new local disambiguation rows: 8 facts, 48 fact-ledger train rows, 8 held-out eval rows, 30-step `Qwen/Qwen2.5-0.5B-Instruct` LoRA, and exact fact-hit scoring. Treat it as useful only if trained exact held-out fact hits improve beyond 1/8 without leakage or unscored answers.

## What Not To Do Next

- Do not start with a SaaS.
- Do not start with a Mac app.
- Do not add accounts, billing, dashboards, or cloud orchestration.
- Do not broaden v0 beyond `.txt` and `.md` notes input.
- Do not build multiple model profiles before the notes model works.
- Do not claim novel distillation algorithms.
- Do not commit generated datasets, model weights, checkpoints, `.env` files, or local machine config.
