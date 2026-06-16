# Agent Handoff

## Project Root

`/Users/langqi/Developer/Projects/OpenDistillation`

## Current State

OpenDistillation is GitHub-ready at the documentation and prototype level, with a safe mock teacher default, deterministic dataset quality reporting, an opt-in local real teacher path, fact-ledger train/eval checks, bounded optional training and comparison entry points, local exact-miss diagnostics, and an internal anti-invention T4 smoke preflight manifest. The latest GPU learning evidence failed: the 2026-06-16 same-chunk disambiguation fact-ledger Colab T4 smoke changed all 8 trained-adapter answers but scored base 0/8 and trained 0/8 exact held-out facts, below the previous best 1/8. The current flow now targets that diagnosed invented-value failure with a verified local anti-invention row-signal contract before the next GPU run.

The repo now contains:

- Public README.
- Product vision for a personal model factory.
- Roadmap.
- Exact v0 notes-model Colab flow.
- Runnable v0 prototype notebook.
- Minimal Python helpers for TXT/MD loading, chunking, dataset validation, dataset quality reporting, mock teacher generation, and opt-in real teacher generation.
- Optional `HuggingFaceLocalTeacherEngine` using `Qwen/Qwen2.5-1.5B-Instruct`, disabled by default with `RUN_REAL_TEACHER = False`.
- Optional TRL `SFTTrainer` + PEFT LoRA training engine for `Qwen/Qwen2.5-0.5B-Instruct`, skipped by default in the notebook.
- Optional multi-question before/after quality report with exact-miss diagnostics, skipped by default in the notebook.
- Fact-ledger quality gate with 8 sample facts, 48 train rows in the current flow, 8 held-out eval rows, 8 same-chunk disambiguation rows, 8 known-values-only anti-invention rows, leakage checks, exact expected-term checks, anti-invention known-values coverage, and exact fact-hit scoring.
- Internal anti-invention smoke preflight script and manifest builder that pins the next sample-notes T4 contract before GPU time.
- Runtime readiness helpers for optional Colab training dependencies, CUDA checks, and common setup failure messages.
- Manual Colab GPU smoke-test checklist.
- Smoke-test results file recording the first real Colab T4 blockers, the recovered-runtime pass, one clean GitHub-opened T4 training/comparison pass, one real-teacher end-to-end T4 verification, the uploaded-notes rehearsals, the earlier multi-question quality smokes, the failed 0/8 fact-ledger T4 smokes, the weak 1/8 six-row label/value T4 smoke, and the failed 0/8 same-chunk disambiguation T4 smoke.
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

The next task is one bounded anti-invention Colab T4 smoke using the preflight manifest from `scripts/prepare_anti_invention_smoke.py`. The run should use the same 30-step TRL/PEFT LoRA setup and sample notes contract, then record whether trained exact held-out fact hits reach at least 2/8. If trained answers change but exact hits stay at 0/8 or 1/8, record that as failed learning, not progress.

## Important Guardrails

- Do not imply unbuilt features already work.
- Do not commit generated datasets, checkpoints, model weights, `.env` files, keys, or local machine config.
- Check current official docs before choosing model, training, Colab, Hugging Face, Unsloth, PEFT, TRL, llama.cpp, Ollama, or GitHub-specific behavior.
- Keep v0 to `.txt` and `.md` notes.
- Prefer one reliable default path over many options.
- Keep real engines behind the helper interfaces so the notebook flow does not need to be rewritten.
- Do not build coding, writing, work, phone, or multi-profile flows until the notes model path works end to end.
