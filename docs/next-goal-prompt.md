# Recommended Next Goal Prompt

Use this as the next `/goal` after the adapter-disabled Colab GPU quality smoke is implemented and documented:

```text
/goal Improve OpenDistillation's first notes-model answer quality without broadening v0. Work in /Users/langqi/Developer/Projects/OpenDistillation on latest main. Keep v0 narrow: TXT/MD notes only, one notes/school model only, Colab-first, MockTeacherEngine as the safe default, optional local Qwen real teacher, optional short TRL/PEFT LoRA training, and bounded before/after quality report. Do not build SaaS, Mac app, phone app, accounts, backend, GGUF export, local runtime packaging, multiple profiles, coding model, writing model, work model, phone model, paid APIs, broad benchmark suite, Unsloth migration, bitsandbytes migration, or larger training platform. Starting evidence: the first Tesla T4 Colab quality smoke at commit 276a8d3 produced 16 mock-teacher rows, a clean dataset-quality report, a 3-step Qwen2.5-0.5B LoRA adapter, and unchanged base/trained answers. Follow-up diagnosis fixed the comparison path by generating base answers with the PEFT adapter disabled. The second T4 smoke at commit 6a98c92599d1defa2b4a61510f7372f399f5fd87 used the fixed comparison path, produced 16/16 valid rows with 4/4 chunk coverage, trained another 3-step adapter, and made all three trained answers change; however, the answers were generic or hallucinated, with overlap deltas +0.031, +0.027, and -0.026, so useful note learning is still not proven. Make the smallest notes-only improvement likely to improve the actual answer content: stronger note-grounded mock teacher targets, clearer reference answers, better comparison prompts, a tiny held-out notes question set, slightly more bounded training steps, or a sample-notes tweak that gives the model concrete facts to learn. Keep committed safe defaults unchanged: INSTALL_TRAINING_DEPS = False, RUN_REAL_TEACHER = False, RUN_TRAINING = False. If another Colab smoke is run, keep it bounded for T4 and record package versions, runtime, dataset quality values, training steps, adapter path, comparison answers, overlap signals, and whether answers are better, unchanged, or worse. Keep generated datasets, adapters, model artifacts, checkpoints, secrets, and local config out of git. Update docs, run local verification, review the diff for secrets/artifacts, commit, and push.
```

## Why This Goal

The repo now has a first quality loop that runs locally and twice in Colab on a Tesla T4: better mock-teacher row variety, deterministic dataset-quality checks, short LoRA training, and a bounded multi-question comparison helper. The first Colab report proved wiring but had identical answers; the second proved the fixed comparison can see adapter-side movement. Neither proved useful note learning.

The next risk is answer content. A beginner needs to see whether the optional adapter can produce more note-grounded answers after a tiny Colab run, or the demo should clearly explain why the current short run is only a wiring check.

## Done Means

- The sample-notes default remains safe and CPU-runnable.
- The unchanged first Colab result and changed-but-not-improved second Colab result are used as baselines.
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
