# Start Here

Use this file to get oriented quickly when returning to OpenDistillation.

## Current Project State

OpenDistillation is an open-source personal model factory concept for the AI PC and AI phone era.

The long-term direction is multiple small personal models, such as notes/school, coding, writing, work, and phone models.

The current prototype is much narrower:

> One Colab-first notes / school model prototype for `.txt` and `.md` notes.

It has a runnable notebook, helper package, deterministic mock teacher with varied question styles, a deterministic dataset quality report, a fact-ledger quality gate, a six-row-per-fact label/value local signal, same-chunk label/value disambiguation rows, anti-invention known-values rows, local fact-miss diagnostics and training-signal context reports for changed-but-wrong answers, an internal anti-invention T4 smoke preflight manifest, an opt-in local Qwen real teacher path, a bounded optional TRL/PEFT LoRA training entry point, runtime readiness messages, and optional multi-question before/after quality report wiring. The latest GPU evidence is a narrow pass, not useful note learning: the 2026-06-16 manifest-gated anti-invention T4 smoke changed all 8 trained answers and improved exact held-out facts from base 0/8 to trained 2/8, beating the previous best 1/8, but it still missed 6/8 facts and every miss was an invented-value failure. It does not yet have model export, local runtime instructions, reliable held-out fact learning, or multiple model profiles.

## Read These In Order

1. `README.md` - public-facing project page.
2. `docs/product-vision.md` - personal model factory vision.
3. `docs/first-demo-flow.md` - exact v0 notes-model Colab flow.
4. `docs/current-decisions.md` - decisions already made.
5. `docs/roadmap.md` - staged product direction.
6. `docs/next-goal-prompt.md` - recommended next goal.
7. `AGENTS.md` - working agreements for coding agents.

## What To Do Next

Use the local fact-miss context report to make one targeted data-signal change before another GPU run. The anti-invention signal beat the previous best by reaching 2/8 exact held-out hits, but the model still invented wrong numbers, times, identifiers, or names for the missed facts. The next data change should directly reduce those failures, then the next bounded T4 smoke should require at least 3/8 exact held-out fact hits and at most 5/8 invented-value misses.

## What Not To Do Next

- Do not start with a SaaS.
- Do not start with a Mac app.
- Do not add accounts, billing, dashboards, or cloud orchestration.
- Do not broaden v0 beyond `.txt` and `.md` notes input.
- Do not build multiple model profiles before the notes model works.
- Do not claim novel distillation algorithms.
- Do not commit generated datasets, model weights, checkpoints, `.env` files, or local machine config.
