# GitHub Launch Checklist

Use this before making the repository public or inviting outside contributors.

## Repository Setup

- [ ] Choose repository visibility.
- [ ] Add a GitHub remote.
- [x] Choose a permissive open-source license.
- [x] Add `LICENSE`.
- [x] Keep generated datasets, checkpoints, model weights, and secrets ignored.
- [x] Add structured GitHub issue forms.
- [ ] Create labels from `docs/github-issue-plan.md` after the remote exists.
- [ ] Create milestones from `docs/github-issue-plan.md` after the remote exists.

## Project Clarity

- [x] README explains the value in the first screen.
- [x] README has an honest current status section.
- [x] README defines the narrow notes-model v0 scope.
- [x] README reports the verified clean Colab T4 smoke test without claiming model quality.
- [x] README labels coding, writing, work, and phone models as future directions.
- [x] Product vision, roadmap, and first-demo flow agree with each other.
- [x] First-demo implementation plan exists.

## Prototype Readiness

- [x] Create `notebooks/opendistillation_v0_demo.ipynb`.
- [x] Validate upload and preview for `.txt` and `.md`.
- [x] Implement simple chunking.
- [x] Define dataset schema.
- [x] Document engine integration points.
- [x] Add deterministic mock teacher path.
- [ ] Choose real teacher path.
- [x] Choose student model and training backend.
- [x] Add a bounded optional short training path.
- [x] Add before/after comparison wiring.
- [x] Add optional dependency install switch and runtime readiness messages.
- [x] Add manual Colab GPU smoke-test checklist.
- [x] Smoke-test short training in Colab with GPU.
- [x] Smoke-test before/after comparison in Colab with GPU.
- [ ] Verify or honestly defer GGUF export.

## Launch Copy

Short version:

```text
OpenDistillation is a personal model factory for the AI PC and AI phone era.

The first path is intentionally small: a Colab notes-model flow for TXT/MD notes, mock QA data, and a clear path toward a tiny local model.
```

Longer version:

```text
OpenDistillation is an open-source workflow for building small personal models for specific parts of life: notes, coding, writing, work, and eventually phone-local routines.

The first milestone is not a broad platform. It is one notes / school model path: TXT/MD input, generated question-answer data, one small optional Qwen/LoRA training path, and an honest path toward llama.cpp or Ollama-style local running.
```
