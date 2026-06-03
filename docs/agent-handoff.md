# Agent Handoff

## Project Root

`/Users/langqi/Developer/Projects/OpenDistillation`

## Current State

OpenDistillation is GitHub-ready at the documentation and prototype level, with a safe mock teacher default, an opt-in local real teacher path, and bounded optional training and comparison entry points. A clean GitHub-opened Colab T4 runtime completed one optional training/comparison smoke test from the sample notes path, a later T4 verification completed the real-teacher path from sample notes through before/after comparison, and the first uploaded-notes rehearsal has one `.txt` pass and one `.md` pass through the default mock-teacher path.

The repo now contains:

- Public README.
- Product vision for a personal model factory.
- Roadmap.
- Exact v0 notes-model Colab flow.
- Runnable v0 prototype notebook.
- Minimal Python helpers for TXT/MD loading, chunking, dataset validation, mock teacher generation, and opt-in real teacher generation.
- Optional `HuggingFaceLocalTeacherEngine` using `Qwen/Qwen2.5-1.5B-Instruct`, disabled by default with `RUN_REAL_TEACHER = False`.
- Optional TRL `SFTTrainer` + PEFT LoRA training engine for `Qwen/Qwen2.5-0.5B-Instruct`, skipped by default in the notebook.
- Optional before/after comparison engine for one generated question, skipped by default in the notebook.
- Runtime readiness helpers for optional Colab training dependencies, CUDA checks, and common setup failure messages.
- Manual Colab GPU smoke-test checklist.
- Smoke-test results file recording the first real Colab T4 blockers, the recovered-runtime pass, one clean GitHub-opened T4 training/comparison pass, one real-teacher end-to-end T4 verification, the first uploaded `.txt` rehearsal pass, the first uploaded `.md` rehearsal pass, and earlier historical upload-control blockers.
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

The next task should prepare the first public demo release candidate: tighten the demo script, issue/milestone status, launch checklist, and narrow GGUF/local-runtime handoff plan without implementing export or broadening beyond TXT/MD notes.

## Important Guardrails

- Do not imply unbuilt features already work.
- Do not commit generated datasets, checkpoints, model weights, `.env` files, keys, or local machine config.
- Check current official docs before choosing model, training, Colab, Hugging Face, Unsloth, PEFT, TRL, llama.cpp, Ollama, or GitHub-specific behavior.
- Keep v0 to `.txt` and `.md` notes.
- Prefer one reliable default path over many options.
- Keep real engines behind the helper interfaces so the notebook flow does not need to be rewritten.
- Do not build coding, writing, work, phone, or multi-profile flows until the notes model path works end to end.
