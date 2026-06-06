# Recommended Next Goal Prompt

Use this as the next `/goal` after the product-core design in `docs/superpowers/specs/2026-06-06-personal-model-learning-core-design.md`:

```text
/goal Implement OpenDistillation's fact-ledger train/eval learning loop for the v0 notes model. Work in /Users/langqi/Developer/Projects/OpenDistillation on latest origin/main. Keep v0 narrow: TXT/MD notes only, one notes/school model only, Colab-first, MockTeacherEngine as the safe default, optional local/open-source Qwen teacher, optional short TRL/PEFT LoRA training, and honest before/after quality report. Do not build SaaS, Mac app, phone app, accounts, backend, GGUF export, local runtime packaging, multiple profiles, coding model, writing model, work model, phone model, broad benchmark suite, Unsloth migration, bitsandbytes migration, or a larger training platform.

Starting evidence: the 2026-06-06 google-colab-cli T4 smoke at commit 0797ed21682960acc8e462db1d793ba357689258 trained a 30-step `Qwen/Qwen2.5-0.5B-Instruct` TRL/PEFT LoRA adapter from 24 mock rows. The adapter changed all four held-out sample-fact answers, but base and trained answers both hit 0/4 expected facts. Treat this as a failed learning signal, not a success.

Design to follow: `docs/superpowers/specs/2026-06-06-personal-model-learning-core-design.md`. The next implementation should build the data/eval loop before changing training knobs.

Task: create a fact-ledger-centered notes dataset path. The system should extract or define atomic facts from TXT/MD chunks, generate training rows and separate held-out eval questions from those facts, keep the current simple training JSONL schema, save any needed internal metadata as a sidecar manifest, and block training/eval question leakage. Add deterministic quality checks for exact duplicate and near-duplicate train/eval questions, expected-term coverage, source chunk IDs, and unsupported or too-short answers where practical.

Quality gate: the next bounded sample should prove zero train/eval question leakage before training. A later Colab T4 run should only be called useful if the trained adapter improves exact fact hits over the base model on held-out paraphrase questions. Changed answers alone are not enough.

Safe defaults must remain off in committed notebook code: `INSTALL_TRAINING_DEPS = False`, `RUN_REAL_TEACHER = False`, and `RUN_TRAINING = False`. If you rerun Colab, use google-colab-cli first and keep the smoke bounded.

Docs to keep aligned: README.md, docs/current-decisions.md, docs/roadmap.md, docs/first-demo-flow.md, docs/colab-smoke-test-results.md, docs/dataset-schema.md, and this file.

Verification: run focused unit tests, notebook JSON validation, default local notebook smoke path if touched, git diff --check, secret scan, artifact/model/data scan, and git status check. Commit and push only intended source/docs changes. Do not commit generated datasets, adapters, checkpoints, model files, secrets, or local config.
```

## Why This Goal

The project now has a clear design direction:

> Build a fact-ledger train/eval loop before trying more training tweaks.

The latest T4 run proved that the adapter can move, but not that it learned the notes. The next useful work is to make the dataset and evaluation honest enough that a future training run means something.

## Done Means

- Latest main is inspected before changing anything.
- The implementation follows the product-core design spec.
- TXT/MD notes remain the only v0 input.
- Training rows and eval questions are separated before training.
- Train/eval question leakage checks exist and are tested.
- Exact fact-hit scoring or expected-term checks exist for the sample quality gate.
- Safe committed defaults stay off.
- Generated datasets, adapters, model artifacts, checkpoints, secrets, and local config stay out of git.
- Unit tests, notebook JSON validation if touched, `git diff --check`, secret scan, artifact scan, and git status are checked.
- Changes are committed and pushed.

## Do Not Use This Goal For

- Expanding beyond TXT/MD notes.
- Adding coding, writing, work, phone, or multi-profile flows.
- Implementing GGUF export or local runtime.
- Migrating to Unsloth or bitsandbytes.
- Claiming benchmark results.
