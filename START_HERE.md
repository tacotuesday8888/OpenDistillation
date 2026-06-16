# Start Here

Use this file to get oriented quickly when returning to OpenDistillation.

## Current Project State

OpenDistillation is an open-source personal model factory concept for the AI PC and AI phone era.

The long-term direction is multiple small personal models, such as notes/school, coding, writing, work, and phone models.

The current prototype is much narrower:

> One Colab-first notes / school model prototype for `.txt` and `.md` notes.

It has a runnable notebook, helper package, deterministic mock teacher with varied question styles, a deterministic dataset quality report, a fact-ledger quality gate, a six-row-per-fact label/value local signal, same-chunk label/value disambiguation rows, anti-invention known-values rows, local fact-miss diagnostics for changed-but-wrong answers, an internal anti-invention T4 smoke preflight manifest, an opt-in local Qwen real teacher path, a bounded optional TRL/PEFT LoRA training entry point, runtime readiness messages, and optional multi-question before/after quality report wiring. The latest GPU evidence is a failure: the 2026-06-16 same-chunk disambiguation T4 smoke changed all 8 trained answers but scored 0/8 exact held-out facts, below the previous best 1/8. It does not yet have model export, local runtime instructions, useful held-out fact learning, or multiple model profiles.

## Read These In Order

1. `README.md` - public-facing project page.
2. `docs/product-vision.md` - personal model factory vision.
3. `docs/first-demo-flow.md` - exact v0 notes-model Colab flow.
4. `docs/current-decisions.md` - decisions already made.
5. `docs/roadmap.md` - staged product direction.
6. `docs/next-goal-prompt.md` - recommended next goal.
7. `AGENTS.md` - working agreements for coding agents.

## What To Do Next

Run the anti-invention smoke preflight, then run one bounded Colab T4 smoke only if the manifest is ready. The current anti-invention signal targets the 2026-06-16 invented-value failure by replacing the risky swapped-value correction row with a known-values-only same-chunk row that lists real note values and explicitly says not to invent number, time, identifier, name, or color substitutes. Local verification now checks the exact sample contract before GPU time: 8 facts, 48 train rows, 8 held-out eval rows, 8 disambiguation rows, 8 known-values-only rows, zero leakage, six SFT preview rows, and a pass condition of at least 2/8 trained exact fact hits to beat the previous best 1/8.

## What Not To Do Next

- Do not start with a SaaS.
- Do not start with a Mac app.
- Do not add accounts, billing, dashboards, or cloud orchestration.
- Do not broaden v0 beyond `.txt` and `.md` notes input.
- Do not build multiple model profiles before the notes model works.
- Do not claim novel distillation algorithms.
- Do not commit generated datasets, model weights, checkpoints, `.env` files, or local machine config.
