# Recommended Next Goal Prompt

Use this as the next `/goal` after the first Colab GPU quality smoke is implemented and documented:

```text
/goal Improve OpenDistillation's first notes-model learning signal without broadening v0. Work in /Users/langqi/Developer/Projects/OpenDistillation on latest main. Keep v0 narrow: TXT/MD notes only, one notes/school model only, Colab-first, MockTeacherEngine as the safe default, optional local Qwen real teacher, optional short TRL/PEFT LoRA training, and bounded before/after quality report. Do not build SaaS, Mac app, phone app, accounts, backend, GGUF export, local runtime packaging, multiple profiles, coding model, writing model, work model, phone model, paid APIs, broad benchmark suite, Unsloth migration, bitsandbytes migration, or larger training platform. Starting evidence: the first Tesla T4 Colab quality smoke at commit 276a8d3 produced 16 mock-teacher rows from sample notes, dataset quality passed with 16/16 schema-valid rows, 4/4 chunk coverage, zero duplicate or near-duplicate questions, zero answer-length warnings, a 3-step Qwen2.5-0.5B LoRA adapter, and a 3-question before/after report. All trained-adapter answers were identical to the base-model answers, so wiring is verified but useful learning is not. Diagnose why the adapter did not visibly change answers using the existing helpers and docs, then make the smallest notes-only improvement to the teacher rows, sample notes, training settings, or comparison prompts that could produce a more meaningful quality signal. Keep committed safe defaults unchanged: INSTALL_TRAINING_DEPS = False, RUN_REAL_TEACHER = False, RUN_TRAINING = False. If a second Colab smoke is run, keep it bounded for T4 and record exact OD_STATUS evidence, package versions, runtime, adapter path, comparison answers, and whether answers are better, unchanged, or worse. Keep generated datasets, adapters, model artifacts, checkpoints, secrets, and local config out of git. Update docs, run local verification, review the diff for secrets/artifacts, commit, and push.
```

## Why This Goal

The repo now has a first quality loop that runs locally and once in Colab on a Tesla T4: better mock-teacher row variety, deterministic dataset-quality checks, short LoRA training, and a bounded multi-question comparison helper. That proves the report wiring, not model usefulness.

The next risk is learning signal. A beginner needs to see whether the optional adapter can produce more note-grounded answers after a tiny Colab run, or the demo should clearly explain why the current short run is only a wiring check.

## Done Means

- The sample-notes default remains safe and CPU-runnable.
- The first unchanged Colab result is used as the baseline.
- Any implementation change stays inside the notes/school model path.
- Dataset quality remains recorded separately from model quality.
- The model-quality section still compares up to three generated questions when training creates an adapter.
- Docs clearly say whether the trained adapter looked better, unchanged, or worse after any new smoke run.
- No generated datasets, model artifacts, checkpoints, secrets, or local config are committed.

## Do Not Use This Goal For

- Expanding beyond TXT/MD notes.
- Adding coding, writing, work, phone, or multi-profile flows.
- Implementing GGUF export or local runtime.
- Adding Hugging Face Evaluate, LightEval, Unsloth, or bitsandbytes unless there is a specific measured reason.
- Claiming benchmark results.
