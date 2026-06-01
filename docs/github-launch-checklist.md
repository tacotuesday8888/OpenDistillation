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
- [x] README defines the narrow v0 scope.
- [x] README does not imply the training pipeline already exists.
- [x] Product vision, roadmap, and first-demo flow agree with each other.
- [x] First-demo implementation plan exists.

## Prototype Readiness

- [ ] Create `notebooks/opendistillation_v0_demo.ipynb`.
- [ ] Validate upload and preview for `.txt` and `.md`.
- [ ] Implement simple chunking.
- [ ] Define dataset schema.
- [ ] Choose teacher path.
- [ ] Choose student model and training backend.
- [ ] Add a short training path.
- [ ] Add before/after comparison.
- [ ] Verify or honestly defer GGUF export.

## Launch Copy

Short version:

```text
OpenDistillation turns your docs into a tiny local model.

Upload notes in Colab, generate training examples, fine-tune a small student model, and run the result locally.
```

Longer version:

```text
OpenDistillation is an open-source project for making personal model distillation feel like a product workflow instead of a pile of disconnected training tools.

The first milestone is intentionally small: one Colab notebook, TXT/MD input, generated question-answer data, one small student model, and a clear path toward llama.cpp or Ollama-style local running.
```
