# Recommended Next Goal Prompt

Use this as the next `/goal` after the first local quality loop is implemented and documented:

```text
/goal Run and document OpenDistillation's first Colab GPU quality smoke test without broadening v0. Work in /Users/langqi/Developer/Projects/OpenDistillation on latest main. Keep v0 narrow: TXT/MD notes only, one notes/school model only, Colab-first, MockTeacherEngine as the safe default, optional local Qwen real teacher, optional short TRL/PEFT LoRA training, and bounded before/after quality report. Do not build SaaS, Mac app, phone app, accounts, backend, GGUF export, local runtime packaging, multiple profiles, coding model, writing model, work model, phone model, paid APIs, broad benchmark suite, or larger training platform. Starting evidence: local default path now produces 16 varied mock-teacher rows from sample notes, a deterministic dataset quality report with 4/4 chunk coverage and zero quality issues, training skipped, and model quality skipped. Run a small Colab GPU quality smoke test only if the environment is available: install the bounded Hugging Face package set, keep notes scope tiny, optionally use the local Qwen teacher if it stays stable, run the short TRL/PEFT LoRA path, and capture the multi-question model-quality report. Record exact OD_STATUS evidence, package versions, runtime, adapter path, comparison answers, and whether the trained adapter is visibly better, unchanged, or worse. If Colab cannot be controlled, document the blocker honestly. Keep generated datasets, adapters, model artifacts, checkpoints, secrets, and local config out of git. Update docs, run local verification, review the diff for secrets/artifacts, commit, and push.
```

## Why This Goal

The repo now has a first quality loop that runs locally without model downloads: better mock-teacher row variety, deterministic dataset-quality checks, and a bounded multi-question comparison helper. That proves the report wiring, not model usefulness.

The next risk is evidence. A beginner needs to see whether the optional adapter actually learns note-grounded answers after a tiny Colab run.

## Done Means

- The sample-notes default remains safe and CPU-runnable.
- One Colab GPU quality smoke run is attempted with the new report shape.
- The run records dataset quality separately from model quality.
- The model-quality section compares up to three generated questions when training creates an adapter.
- Docs clearly say whether the trained adapter looked better, unchanged, worse, or unverified.
- No generated datasets, model artifacts, checkpoints, secrets, or local config are committed.

## Do Not Use This Goal For

- Expanding beyond TXT/MD notes.
- Adding coding, writing, work, phone, or multi-profile flows.
- Implementing GGUF export or local runtime.
- Adding Hugging Face Evaluate, LightEval, Unsloth, or bitsandbytes unless there is a specific measured reason.
- Claiming benchmark results.
