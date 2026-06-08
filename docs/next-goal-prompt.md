# Recommended Next Goal Prompt

Use this as the next `/goal` after the 2026-06-08 fact-ledger Colab T4 smoke is documented:

```text
/goal Diagnose OpenDistillation's fact-ledger learning signal locally before spending more GPU.

Context:
Work in /Users/langqi/Developer/Projects/OpenDistillation on latest origin/main. Keep v0 narrow: TXT/MD notes only, one notes/school model only, Colab-first, MockTeacherEngine as the safe default, optional local/open-source Qwen teacher, optional short TRL/PEFT LoRA training, and honest before/after quality reporting. Do not build SaaS, Mac app, phone app, accounts, backend, GGUF export, local runtime packaging, multiple profiles, coding model, writing model, work model, phone model, broad benchmark suite, Unsloth migration, bitsandbytes migration, or a larger training platform.

Starting evidence:
The local fact-ledger quality gate is hardened and passes on the committed sample notes: 8 facts, 24 fact-ledger train rows, 8 held-out eval rows, 8/8 train coverage, 8/8 eval coverage, zero exact train/eval leaks, zero near-duplicate/token-overlap leaks, and zero missing expected terms.

The 2026-06-08 google-colab-cli T4 smoke used commit 479b110773ad9d3382523a4d98c5cca1645e0cdd, Tesla T4, `Qwen/Qwen2.5-0.5B-Instruct`, TRL/PEFT LoRA, 30 training steps, the 24 fact-ledger train rows, and the 8 held-out fact-ledger eval rows. The adapter changed all 8 answers, but base and trained answers both hit 0/8 exact expected facts. Treat that as a failed learning signal, not a success.

Task:
Find the smallest local product-layer change that makes the next GPU smoke worth running. Start by inspecting the generated fact-ledger train rows, held-out eval rows, instruction wording, response wording, sidecar metadata, scoring, and the exact SFT text formatting passed to TRL. Determine whether the failure is likely caused by weak answer targets, eval wording mismatch, too little row variety, bad prompt formatting, label/value ambiguity, scoring mismatch, or a training path issue that can be caught locally.

Implementation priorities:
- Keep the public JSONL schema stable: `instruction`, `response`, `source_chunk_id`.
- Keep fact metadata in the internal sidecar.
- Improve fact-ledger train/eval row wording only if it makes the checked facts clearer without copying eval questions into train rows.
- Add local tests that would have caught the weak learning signal or bad formatting before a Colab run.
- Improve the local report so a non-technical user can see why a fact-ledger split is safe for training but still may fail model learning.
- Do not change GPU training knobs as the first fix.
- Do not add dependencies unless a proven open-source tool clearly improves the local core without making v0 heavier.

GPU rule:
Do not run Colab or GPU training in this goal by default. If local evidence shows a clear fix and a single bounded T4 rerun is justified, stop and ask for explicit approval before spending compute.

Safe defaults:
Committed notebook defaults must remain off: `INSTALL_TRAINING_DEPS = False`, `RUN_REAL_TEACHER = False`, and `RUN_TRAINING = False`.

Docs to keep aligned:
README.md, docs/current-decisions.md, docs/open-source-tool-strategy.md, docs/roadmap.md, docs/first-demo-flow.md, docs/colab-smoke-test-results.md, docs/dataset-schema.md, notebooks/README.md, and this file.

Verification:
Run unit tests, notebook JSON validation if touched, confirm no committed notebook outputs, default local notebook smoke if notebook flow changes, git diff --check, secret scan, artifact/model/data scan, and git status check. Commit and push only intended source/docs changes. Do not commit generated datasets, adapters, checkpoints, model files, secrets, or local config.

Finish:
Final response must include commit hash if any, plain-English diagnosis, files changed, verification results, whether Colab/GPU was avoided, and the next recommended bounded GPU-smoke condition.
```

## Why This Goal

The fact-ledger data split is now safe enough to test, but the T4 run still scored 0/8 exact facts after training. Another GPU run without a local diagnosis would likely spend compute to reproduce the same failure.

## Done Means

- Latest main is inspected before changing anything.
- The exact fact-ledger train/eval rows are reviewed locally.
- The SFT input text format is inspected or tested without downloading large models.
- Any code change improves the data/eval/reporting layer, not custom GPU training code.
- Train/eval leakage remains blocked.
- Exact expected-term scoring remains strict.
- Safe committed notebook defaults stay off.
- Generated datasets, adapters, model artifacts, checkpoints, secrets, and local config stay out of git.
- Unit tests, notebook checks when relevant, `git diff --check`, secret scan, artifact scan, and git status are checked.
- Changes are committed and pushed if docs or code changed.

## Do Not Use This Goal For

- Running another Colab training job without explicit approval.
- Expanding beyond TXT/MD notes.
- Adding coding, writing, work, phone, or multi-profile flows.
- Implementing GGUF export or local runtime.
- Migrating to Unsloth or bitsandbytes.
- Claiming benchmark results.
