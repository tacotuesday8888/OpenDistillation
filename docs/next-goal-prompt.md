# Recommended Next Goal Prompt

Use this as the next `/goal` after the optional real teacher implementation is committed:

```text
/goal Smoke-test and harden the optional real teacher path for the OpenDistillation v0 notes model. Work in /Users/langqi/Developer/Projects/OpenDistillation on the current branch. Keep the product scope narrow: TXT/MD notes only, one notes/school model only, Colab-first, deterministic MockTeacherEngine as the safe fallback. Use Chrome or Computer if needed to run the GitHub-opened Colab notebook. Do not build a SaaS, Mac app, phone app, account system, cloud backend, multi-profile system, GGUF export, local runtime, or real large training pipeline. Run `RUN_REAL_TEACHER = True` with the optional Hugging Face dependencies in a clean Colab GPU runtime, record whether `Qwen/Qwen2.5-1.5B-Instruct` loads and produces valid rows, capture exact failure messages if it does not, keep generated datasets/model artifacts/secrets out of git, update README/docs/current-decisions.md/docs/first-demo-flow.md/docs/colab-smoke-test-results.md/docs/github-issue-plan.md with verified versus unverified status, run local verification, review the diff for secrets/artifacts, commit locally, and push.
```

## Why This Goal

The first optional local real teacher path now exists behind `TeacherEngine`: `HuggingFaceLocalTeacherEngine` using `Qwen/Qwen2.5-1.5B-Instruct`. The next useful product risk is proving whether that teacher path actually works in a clean Colab GPU runtime and whether the generated rows are useful enough to train from.

## Done Means

- `RUN_REAL_TEACHER = True` is attempted from a clean GitHub-opened Colab GPU runtime.
- The docs say plainly whether the real teacher loaded, generated valid rows, failed, or remained unverified.
- The first generated rows are inspected and recorded without committing generated datasets.
- `MockTeacherEngine` remains available and safe.
- No-download tests still cover the teacher path by using fake dependencies.
- Real training, GGUF export, local runtime, and future model profiles remain deferred.
- No generated datasets, model artifacts, checkpoints, secrets, or local config are committed.

## Do Not Use This Goal For

- Expanding beyond TXT/MD notes.
- Adding coding, writing, work, phone, or multi-profile flows.
- Building account systems, cloud services, or paid API infrastructure.
- Implementing GGUF export or local runtime.
- Claiming model quality from a tiny smoke run.
