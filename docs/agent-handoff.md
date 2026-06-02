# Agent Handoff

## Project Root

`/Users/langqi/Developer/Projects/OpenDistillation`

## Current State

OpenDistillation is GitHub-ready at the documentation and prototype-skeleton level. It does not have real training code yet.

The repo now contains:

- Public README.
- Product vision for a personal model factory.
- Roadmap.
- Exact v0 notes-model Colab flow.
- Runnable v0 skeleton notebook.
- Minimal Python helpers for TXT/MD loading, chunking, dataset validation, and mock teacher generation.
- First-demo implementation plan.
- GitHub issue forms.
- Starter milestone and issue plan.
- Ignore rules for secrets, generated datasets, checkpoints, and model artifacts.
- Apache-2.0 license.

## Product Direction

Long-term direction:

> OpenDistillation is a personal model factory for the AI PC and AI phone era.

Future model types can include notes/school, coding, writing, work, and phone models.

Near-term direction:

> Build the notes / school model path first.

The first implementation surface is a Colab notebook. The CLI comes later as a thin wrapper. Do not start a SaaS, Mac app, account system, billing flow, cloud backend, or broad multi-profile system.

## Next Recommended Work

Use `docs/next-goal-prompt.md`.

The next task should choose and test the first real teacher path for the notes model. It should not implement real model training until the teacher and dataset path are validated.

## Important Guardrails

- Do not imply unbuilt features already work.
- Do not commit generated datasets, checkpoints, model weights, `.env` files, keys, or local machine config.
- Check current official docs before choosing model, training, Colab, Hugging Face, Unsloth, PEFT, TRL, llama.cpp, Ollama, or GitHub-specific behavior.
- Keep v0 to `.txt` and `.md` notes.
- Prefer one reliable default path over many options.
- Keep real engines behind the helper interfaces so the notebook flow does not need to be rewritten.
- Do not build coding, writing, work, phone, or multi-profile flows until the notes model path works end to end.
