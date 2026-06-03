# Recommended Next Goal Prompt

Use this as the next `/goal` after the public Colab demo hardening commit lands:

```text
/goal Rehearse the first public OpenDistillation Colab demo with uploaded notes. Work in /Users/langqi/Developer/Projects/OpenDistillation on latest main. Keep scope narrow: TXT/MD notes only, one notes/school model only, Colab-first, MockTeacherEngine as the safe default, optional local Qwen real teacher, optional short TRL/PEFT LoRA training, and optional before/after comparison. Use Chrome as the first Colab control path and Computer as the fallback if Chrome cannot read or operate Colab. Do not ask the user for screenshots when those plugins can inspect the page. Do not build SaaS, Mac app, phone app, accounts, backend, GGUF export, local runtime packaging, multiple profiles, coding model, writing model, or broad training platform. Run the default CPU-safe flow with one uploaded .txt file and one uploaded .md file, confirm the status log records setup/teacher/dataset/training-skipped/comparison-skipped markers, tighten confusing beginner-facing text if needed, keep generated datasets/model artifacts/secrets out of git, update docs with verified versus deferred status, run local verification, review the diff for secrets/artifacts, commit, and push.
```

## Why This Goal

The sample-notes path, optional training path, and one real-teacher T4 wiring path have passed. The remaining public-demo risk is simpler: a beginner may upload their own `.txt` or `.md` notes and hit confusing wording or an output-frame issue.

The new status log makes that easier to verify without broadening v0.

## Done Means

- The sample-notes default remains safe and CPU-runnable.
- One uploaded `.txt` note and one uploaded `.md` note work through the default mock-teacher path.
- The run records `OD_STATUS` markers in `/tmp/opendistillation_status.jsonl`.
- Any copy changes keep the project scoped to the notes / school model.
- No generated datasets, model artifacts, checkpoints, secrets, or local config are committed.

## Do Not Use This Goal For

- Improving model quality.
- Expanding beyond TXT/MD notes.
- Adding coding, writing, work, phone, or multi-profile flows.
- Implementing GGUF export or local runtime.
- Claiming benchmark results.
