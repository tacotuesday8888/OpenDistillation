# AGENTS.md

## Project Mission

OpenDistillation is an open-source, productized workflow for building small personal models for the AI PC and AI phone era.

The long-term direction is a personal model factory:

- Notes / school models.
- Coding models.
- Writing models.
- Work models.
- Phone-local models.

The first product path is much narrower:

> Build a tiny local notes model from TXT/MD notes.

Do not turn this into a broad research framework or multi-profile platform before the first notes-model demo works.

## Current Stage

The project has a Colab-first prototype skeleton.

It can load and chunk TXT/MD notes, validate a JSONL dataset schema, generate deterministic mock QA examples, and prepare an optional short TRL/PEFT LoRA training run. It does not have real teacher-model calls, a verified Colab GPU training run, export, before/after comparison, or multiple model profiles yet.

The next major milestones are one real teacher-generation path for the notes model and a Colab GPU smoke test for the optional training cell, not a Mac app, SaaS, phone app, or full training platform.

## Product Defaults

Use these defaults unless the user explicitly changes direction:

- Public positioning: personal model factory for the AI PC and AI phone era.
- First model type: notes / school model.
- Future model types: coding, writing, work, and phone models, only after the notes model works.
- Main first-run experience: Colab notebook.
- Secondary interface: thin local CLI.
- Initial input: `.txt` and `.md` notes only.
- Initial teacher path: deterministic mock teacher, then one real open-source teacher path.
- Initial student target: around 0.5B-1.5B parameters.
- Initial training method: response distillation / SFT.
- Advanced training method: experimental logits distillation only where technically feasible.
- Initial export target: GGUF or a clear documented path toward GGUF.
- Local running target: llama.cpp and/or Ollama-style local runtime.

## What To Avoid

- Do not start with a closed SaaS.
- Do not start with a Mac app.
- Do not start with a phone app.
- Do not build a giant research framework first.
- Do not build coding, writing, work, phone, or multi-profile flows in v0.
- Do not claim novel algorithms unless the code actually implements them.
- Do not overbuild PDF parsing, arbitrary document ingestion, dashboards, accounts, billing, or cloud training in v0.
- Do not commit generated models, datasets, checkpoints, API keys, `.env` files, or local machine config.

## Communication Style

The user is exploring and is not deeply technical.

- Explain choices in plain language.
- Be honest when an idea is mostly product packaging versus real technical novelty.
- Prefer concrete next steps over abstract strategy.
- Do not flatter weak ideas; improve them.
- When something is risky or too broad, say so clearly and offer a narrower path.

## Development Workflow

- Inspect the current repo before making changes.
- Keep changes small and intentional.
- Before committing, run `git status`, inspect the diff, and check for secrets.
- Commit only intended files.
- Prefer documentation and prototype clarity before deep infrastructure.
- If using version-sensitive packages, check current official docs before locking choices.

## Near-Term Priority

The next useful work is:

1. Keep the personal model factory positioning clear.
2. Keep v0 scoped to the notes / school model.
3. Choose one real teacher-generation path for TXT/MD notes.
4. Keep the deterministic mock teacher as a safe fallback.
5. Smoke-test the existing Qwen2.5-0.5B TRL/PEFT training path in Colab before adding new model profiles.
