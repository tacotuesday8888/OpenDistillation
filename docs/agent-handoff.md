# Agent Handoff

## Project Root

`/Users/langqi/Developer/Projects/OpenDistillation`

## Current State

OpenDistillation is GitHub-ready at the documentation and prototype-skeleton level, with bounded optional training and comparison entry points. A clean GitHub-opened Colab T4 runtime completed one optional training/comparison smoke test from the sample notes path.

The repo now contains:

- Public README.
- Product vision for a personal model factory.
- Roadmap.
- Exact v0 notes-model Colab flow.
- Runnable v0 skeleton notebook.
- Minimal Python helpers for TXT/MD loading, chunking, dataset validation, and mock teacher generation.
- Optional TRL `SFTTrainer` + PEFT LoRA training engine for `Qwen/Qwen2.5-0.5B-Instruct`, skipped by default in the notebook.
- Optional before/after comparison engine for one generated question, skipped by default in the notebook.
- Runtime readiness helpers for optional Colab training dependencies, CUDA checks, and common setup failure messages.
- Manual Colab GPU smoke-test checklist.
- Smoke-test results file recording the first real Colab T4 blockers, the recovered-runtime pass, and one clean GitHub-opened T4 training/comparison pass.
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

The next task should choose and test the first real teacher-generation path for the notes model while keeping the deterministic mock teacher as the fallback.

## Important Guardrails

- Do not imply unbuilt features already work.
- Do not commit generated datasets, checkpoints, model weights, `.env` files, keys, or local machine config.
- Check current official docs before choosing model, training, Colab, Hugging Face, Unsloth, PEFT, TRL, llama.cpp, Ollama, or GitHub-specific behavior.
- Keep v0 to `.txt` and `.md` notes.
- Prefer one reliable default path over many options.
- Keep real engines behind the helper interfaces so the notebook flow does not need to be rewritten.
- Do not build coding, writing, work, phone, or multi-profile flows until the notes model path works end to end.
