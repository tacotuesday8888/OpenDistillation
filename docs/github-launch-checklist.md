# GitHub Launch Checklist

Use this before making the repository public.

## Repo Setup

- [ ] Choose repository visibility: private for drafting, public for launch.
- [ ] Choose license: Apache-2.0 or MIT.
- [ ] Confirm `README.md` is public-facing and not just internal notes.
- [ ] Confirm `START_HERE.md` is useful for future agents and collaborators.
- [ ] Confirm no `.env`, API keys, generated models, checkpoints, or datasets are committed.

## Project Clarity

- [ ] README explains the value in the first screen.
- [ ] README has an honest current status section.
- [ ] README has a clear v0 flow.
- [ ] README does not imply unbuilt features already work.
- [ ] Roadmap matches the actual project direction.

## First Issues

Create issues for:

- [ ] Refine public README.
- [ ] Choose teacher and student model defaults.
- [ ] Specify generated dataset schema.
- [ ] Build Colab notebook skeleton.
- [ ] Implement document loading for TXT/MD.
- [ ] Implement document chunking.
- [ ] Implement teacher QA generation.
- [ ] Implement dataset preview and save.
- [ ] Implement first student training path.
- [ ] Document local export/run path.

## Launch Copy

Draft one short launch message:

```text
OpenDistillation is an open-source project for turning your docs into a tiny local model.

The goal: upload notes, generate distillation data, train a small student model, and run it locally.

Fine-tuning has Unsloth. Personal model distillation has OpenDistillation.
```
