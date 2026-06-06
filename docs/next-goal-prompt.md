# Recommended Next Goal Prompt

Use this as the next `/goal` after the 30-step sample-fact Colab T4 result:

```text
/goal Improve the OpenDistillation notes-model learning signal after the measured 30-step sample-fact T4 smoke failed to recall the held-out facts. Work in /Users/langqi/Developer/Projects/OpenDistillation on latest origin/main. Keep v0 narrow: TXT/MD notes only, one notes/school model only, Colab-first, MockTeacherEngine as the safe default, optional local Qwen real teacher, optional short TRL/PEFT LoRA training, and bounded before/after quality report. Do not build SaaS, Mac app, phone app, accounts, backend, GGUF export, local runtime packaging, multiple profiles, coding model, writing model, work model, phone model, paid APIs, broad benchmark suite, Unsloth migration, bitsandbytes migration, or larger training platform.

Starting evidence: the 2026-06-06 google-colab-cli T4 smoke at commit 0797ed21682960acc8e462db1d793ba357689258 used `Tesla T4`, torch 2.11.0+cu128, transformers 4.57.6, datasets 5.0.0, trl 0.29.1, peft 0.18.1, and accelerate 1.13.0. It generated 24 mock rows from 4 sample-note chunks, passed dataset quality with 24/24 valid rows, 4/4 chunk coverage, zero duplicate/near-duplicate questions, zero answer-length warnings, trained a 30-step Qwen2.5-0.5B LoRA adapter, and ran 4 held-out sample-fact comparisons.

Measured result: all four trained-adapter answers changed, but the adapter did not answer the held-out facts better. Base fact hits were 0/4 and trained-adapter fact hits were 0/4. The trained answers were wrong or hallucinated: `Echo` instead of `Glass Harbor`, `remember_samples_notes` instead of `copper-lantern-47`, `H2O` instead of `llama-harbor-alpha`, and `emerald green` / `violet` instead of `4:17 PM` / `ultramarine`. Treat this as a failed learning signal, not a success.

Task: find the smallest notes-only change that makes the sample-fact learning signal more likely to work, then rerun local verification. Prefer improving the generated teacher targets, prompt wording, held-out question alignment, or bounded training settings before adding new libraries or larger systems. Keep safe committed defaults off: `INSTALL_TRAINING_DEPS = False`, `RUN_REAL_TEACHER = False`, and `RUN_TRAINING = False`. If you rerun Colab, use google-colab-cli first and keep the smoke bounded.

Docs to keep aligned: README.md, docs/current-decisions.md, docs/roadmap.md, docs/first-demo-flow.md, docs/colab-smoke-test-results.md, and this file.

Verification: run unit tests, notebook JSON validation, default local notebook smoke path, git diff --check, secret scan, artifact/model/data scan, and git status check. Commit and push only intended source/docs changes. Do not commit generated datasets, adapters, checkpoints, model files, secrets, or local config.
```

## Why This Goal

The project now has a real measured T4 result:

> A 30-step adapter can change answers, but it still does not recall the sample facts.

The next useful work is not export or a new app. It is improving the narrow notes-model learning signal until the sample smoke can show useful note-grounded answers.

## Done Means

- Latest main is inspected before changing anything.
- The measured 30-step T4 failure remains documented honestly.
- The change stays inside the notes-only v0 path.
- Safe committed defaults stay off.
- Generated datasets, adapters, model artifacts, checkpoints, secrets, and local config stay out of git.
- Unit tests, notebook JSON validation, default local notebook smoke path, `git diff --check`, secret scan, artifact scan, and git status are checked.
- Changes are committed and pushed.

## Do Not Use This Goal For

- Expanding beyond TXT/MD notes.
- Adding coding, writing, work, phone, or multi-profile flows.
- Implementing GGUF export or local runtime.
- Adding Hugging Face Evaluate, LightEval, Unsloth, or bitsandbytes.
- Claiming benchmark results.
