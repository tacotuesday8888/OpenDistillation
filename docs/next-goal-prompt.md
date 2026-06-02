# Recommended Next Goal Prompt

Use this as the next `/goal` after the v0 demo hardening update is committed:

```text
/goal Run and record the manual Colab GPU smoke test for the existing OpenDistillation v0 optional training and before/after comparison path. Work in /Users/langqi/Developer/Projects/OpenDistillation on the current branch. Use docs/colab-smoke-test-checklist.md. Do not change the product scope. Open the notebook from GitHub in Colab, choose a GPU runtime, install the optional Hugging Face packages with INSTALL_TRAINING_DEPS = True, run the default TXT/MD notes flow, set RUN_TRAINING = True, record package install success, GPU name, model download status, training start, adapter output path, before/after comparison output, runtime, memory failures, and any exact errors. Update README/docs/current-decisions.md/docs/first-demo-implementation-plan.md/docs/github-issue-plan.md with verified versus unverified status. Do not implement a real teacher path, GGUF export, SaaS, Mac app, phone app, account system, cloud backend, multiple profiles, coding model, writing model, or work model. Do not commit generated datasets, adapters, checkpoints, model weights, caches, secrets, .env files, or local machine config. Verify tests and notebook JSON, review the diff for secrets/generated files, commit locally, and push.
```

## Why This Goal

The optional training and comparison path is wired but not yet proven in a real Colab GPU runtime. That is the biggest remaining demo risk, so verify it or document the exact failure before adding a real teacher path.

## Done Means

- `docs/colab-smoke-test-checklist.md` has concrete results.
- The optional dependency install result is recorded.
- The runtime check prints the GPU name or the exact failure is recorded.
- Training either creates an adapter under `outputs/notes-lora/adapter` or records the exact failure.
- Before/after comparison either prints both answers or records the exact failure.
- Docs clearly say what is verified and what remains unverified.
- Real teacher generation and export remain deferred.
- No coding, writing, work, phone, or multi-profile implementation is added.
- No generated datasets, model artifacts, checkpoints, secrets, or local config are committed.

## Do Not Use This Goal For

- Implementing a new training pipeline.
- Real teacher generation.
- GGUF export implementation.
- Coding model implementation.
- Writing model implementation.
- Work model implementation.
- Phone model implementation.
- SaaS, Mac app, account system, billing, or cloud backend.
