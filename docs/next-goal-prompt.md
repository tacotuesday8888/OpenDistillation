# Recommended Next Goal Prompt

Use this as the next `/goal` after the uploaded `.txt` rehearsal pass and `.md` upload blocker are documented:

```text
/goal Finish the public OpenDistillation Colab uploaded-notes rehearsal by unblocking the uploaded `.md` file path. Work in /Users/langqi/Developer/Projects/OpenDistillation on latest main. Keep scope narrow: TXT/MD notes only, one notes/school model only, Colab-first, MockTeacherEngine as the safe default, optional local Qwen real teacher, optional short TRL/PEFT LoRA training, and optional before/after comparison. The uploaded `.txt` path already passed once at commit 6f7d9c66cacb07dc82571abb85b3232285f6961c with setup ready, install skipped, teacher succeeded, dataset saved, training skipped, and comparison skipped in `/tmp/opendistillation_status.jsonl`. Do not rerun TXT unless needed as a control. Use Chrome and Computer for Colab/browser/native picker control; do not ask me for screenshots when those plugins can inspect the page. First determine why `/private/tmp/opendistillation-upload-rehearsal-notes.md` could not be attached through the Colab `files.upload()` widget even though the `.txt` file worked. Then run the default CPU-safe flow with one uploaded `.md` file through validation, chunking, mock QA rows, dataset save, training skipped, and comparison skipped. Confirm the status log records setup/install-skipped/teacher-success/dataset-saved/training-skipped/comparison-skipped for the `.md` run, keep generated datasets/model artifacts/secrets out of git, update docs with verified versus deferred status, run local verification, review the diff for secrets/artifacts, commit, and push. Do not build SaaS, Mac app, phone app, accounts, backend, GGUF export, local runtime packaging, multiple profiles, coding model, writing model, or broad training platform.
```

## Why This Goal

The sample-notes path, optional training path, and one real-teacher T4 wiring path have passed. The uploaded `.txt` path also passed once from a GitHub-opened Colab notebook at commit `6f7d9c66cacb07dc82571abb85b3232285f6961c`.

The remaining public-demo risk is narrower now: uploaded `.md` attachment. The `.md` run reached the actual Colab upload widget, but the native Open dialog did not expose a selectable Markdown file row and the Open button stayed disabled; Chrome's file-chooser listener also returned no usable chooser object for the Colab output iframe.

## Done Means

- The sample-notes default remains safe and CPU-runnable.
- One uploaded `.md` note works through the default mock-teacher path.
- The `.md` run records `OD_STATUS` markers in `/tmp/opendistillation_status.jsonl` for setup, install skipped, teacher success, dataset saved, training skipped, and comparison skipped.
- Any copy changes keep the project scoped to the notes / school model.
- No generated datasets, model artifacts, checkpoints, secrets, or local config are committed.

## Do Not Use This Goal For

- Improving model quality.
- Expanding beyond TXT/MD notes.
- Adding coding, writing, work, phone, or multi-profile flows.
- Implementing GGUF export or local runtime.
- Claiming benchmark results.
