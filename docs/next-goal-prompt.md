# Recommended Next Goal Prompt

Use this as the next `/goal` after the first uploaded-notes rehearsal attempt is documented:

```text
/goal Unblock and rerun the first public OpenDistillation Colab uploaded-notes rehearsal. Work in /Users/langqi/Developer/Projects/OpenDistillation on latest main. Keep scope narrow: TXT/MD notes only, one notes/school model only, Colab-first, MockTeacherEngine as the safe default, optional local Qwen real teacher, optional short TRL/PEFT LoRA training, and optional before/after comparison. Use Chrome as the first Colab control path and Computer as the fallback if Chrome cannot attach files to the Colab upload widget. Do not ask me for screenshots when those plugins can inspect the page. First verify that the callable browser-control path can attach a local file to Colab's `files.upload()` iframe; if Chrome file upload is blocked, follow the Chrome plugin file-upload guidance exactly or use an actually callable Computer control tool. Then run the default CPU-safe flow with one uploaded .txt file and one uploaded .md file, confirm the status log records setup/install-skipped/teacher-success/dataset-saved/training-skipped/comparison-skipped markers for both, keep generated datasets/model artifacts/secrets out of git, update docs with verified versus deferred status, run local verification, review the diff for secrets/artifacts, commit, and push. Do not build SaaS, Mac app, phone app, accounts, backend, GGUF export, local runtime packaging, multiple profiles, coding model, writing model, or broad training platform.
```

## Why This Goal

The sample-notes path, optional training path, and one real-teacher T4 wiring path have passed. The hardened notebook also printed setup and install-skip status markers in Colab at commit `7113f87fa915b789cc77bbfb423b405defd9b5ec`.

The remaining public-demo risk is still the uploaded-notes path. The latest attempt reached Colab's upload iframe, but Chrome file-chooser events timed out before the `.txt` file could be attached, and Computer Use did not expose a callable control tool in that thread.

## Done Means

- The sample-notes default remains safe and CPU-runnable.
- One uploaded `.txt` note and one uploaded `.md` note work through the default mock-teacher path.
- Each run records `OD_STATUS` markers in `/tmp/opendistillation_status.jsonl` for setup, install skipped, teacher success, dataset saved, training skipped, and comparison skipped.
- Any copy changes keep the project scoped to the notes / school model.
- No generated datasets, model artifacts, checkpoints, secrets, or local config are committed.

## Do Not Use This Goal For

- Improving model quality.
- Expanding beyond TXT/MD notes.
- Adding coding, writing, work, phone, or multi-profile flows.
- Implementing GGUF export or local runtime.
- Claiming benchmark results.
