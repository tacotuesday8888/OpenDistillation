# AGENTS.md

## Project Mission

OpenDistillation is an open-source, productized tool for helping people turn their own documents or task knowledge into a small local model.

The first product promise is:

> Upload docs. Distill a tiny local model. Run it locally.

Do not turn this into a broad research framework before the first clear demo works.

## Current Stage

The project is in foundation mode.

The next major milestone is a Colab-first prototype, not a Mac app, SaaS, or full training platform.

## Product Defaults

Use these defaults unless the user explicitly changes direction:

- Public positioning: open-source productized tool.
- Main first-run experience: Colab notebook.
- Secondary interface: thin local CLI.
- Initial input: `.txt` and `.md` files only.
- Initial teacher path: open-source teacher model, remote if that improves beginner success.
- Initial student target: around 0.5B-1.5B parameters.
- Initial training method: response distillation / SFT.
- Advanced training method: experimental logits distillation only where technically feasible.
- Initial export target: GGUF or a clear documented path toward GGUF.
- Local running target: llama.cpp and/or Ollama-style local runtime.

## What To Avoid

- Do not start with a closed SaaS.
- Do not start with a Mac app.
- Do not build a giant research framework first.
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

1. Polish README and launch narrative.
2. Define the exact Colab v0 flow.
3. Create a minimal notebook skeleton.
4. Create a minimal Python package skeleton only when it supports the notebook.
5. Validate the first flow with the smallest possible teacher/student path.
