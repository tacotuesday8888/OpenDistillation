# Recommended Next Goal Prompt

Use this as the next `/goal` after Colab GPU access is available again:

```text
/goal Finish the OpenDistillation sample-fact Colab T4 quality smoke and document the measured answer result. Work in /Users/langqi/Developer/Projects/OpenDistillation on latest origin/main. Keep v0 narrow: TXT/MD notes only, one notes/school model only, Colab-first, MockTeacherEngine as the safe default, optional local Qwen real teacher, optional short TRL/PEFT LoRA training, and bounded before/after quality report. Do not build SaaS, Mac app, phone app, accounts, backend, GGUF export, local runtime packaging, multiple profiles, coding model, writing model, work model, phone model, paid APIs, broad benchmark suite, Unsloth migration, bitsandbytes migration, or larger training platform.

Starting evidence: the first Tesla T4 Colab quality smoke at commit 276a8d3 produced 16 mock-teacher rows, a clean dataset-quality report, a 3-step Qwen2.5-0.5B LoRA adapter, and unchanged base/trained answers. Follow-up diagnosis fixed the comparison path by generating base answers with the PEFT adapter disabled. The second T4 smoke at commit 6a98c92599d1defa2b4a61510f7372f399f5fd87 used the fixed comparison path, produced 16/16 valid rows with 4/4 chunk coverage, trained another 3-step adapter, and made all three trained answers change; however, the answers were generic or hallucinated, with overlap deltas +0.031, +0.027, and -0.026, so useful note learning was still not proven.

Current local experiment: latest main has fact-rich `examples/sample-notes.md`, fact-aware mock rows for simple `Label: value` chunks, `build_sample_fact_comparison_rows(...)`, four held-out sample-fact questions, notebook sample setting `examples_per_chunk = 6`, notebook optional training setting `SFTLoRAConfig(max_steps=30)`, and notebook comparison setting `COMPARISON_MAX_EXAMPLES = 4`. Local verification produced 4 chunks, 24 mock rows, 24 schema-valid rows, 4/4 chunk coverage, 0 duplicate questions, 0 near-duplicate questions, 0 very short answers, 0 very long answers, 0 dataset-quality issues, and 4 held-out sample-fact questions. Safe committed defaults are still `INSTALL_TRAINING_DEPS = False`, `RUN_REAL_TEACHER = False`, and `RUN_TRAINING = False`.

The latest 2026-06-03 Colab attempt did not reach notebook execution. The official Colab CLI authenticated and successfully ran a CPU VM probe, but `--gpu T4` failed before code execution while assigning the runtime. Chrome control then operated the Colab UI, selected T4 GPU, saved the runtime type, and clicked "Connect T4"; Colab showed "Cannot connect to GPU backend" and "You cannot currently connect to a GPU due to usage limits in Colab." Do not invent Colab evidence.

Use the Colab CLI first if it can allocate T4 cleanly, then @Chrome/@Computer as fallback for authenticated Colab control. If Colab runs, use a T4 runtime and live-only settings `INSTALL_TRAINING_DEPS = True`, `USE_SAMPLE_NOTES = True`, `RUN_REAL_TEACHER = False`, and `RUN_TRAINING = True`. Record package versions, runtime/GPU, dataset quality values, training steps, adapter path, adapter file list, comparison questions, base answers, trained-adapter answers, overlap values/deltas, and honest judgment: better, unchanged, or worse. If the adapter answers are still generic or hallucinated, say that plainly.

If Colab still is not feasible, update docs with the exact blocker and leave answer quality unverified. Keep generated datasets, adapters, model artifacts, checkpoints, secrets, and local config out of git. Run local verification, review the diff for secrets/artifacts, commit, and push.
```

## Why This Goal

The implementation now has a sharper local learning-signal setup, but the core question is still external:

> Does a short T4 LoRA run answer the held-out sample-fact questions better than the base model?

The answer can be better, unchanged, or worse. The project needs measured evidence, not optimism.

## Done Means

- Latest main is inspected before changing anything.
- The safe committed defaults stay off.
- The Colab run either completes with full measured evidence, or the exact blocker is documented.
- `docs/colab-smoke-test-results.md`, `docs/current-decisions.md`, README-facing docs, and this file agree on the result.
- No generated datasets, model artifacts, checkpoints, secrets, or local config are committed.
- Unit tests, notebook JSON validation, default local notebook smoke path, `git diff --check`, secret scan, artifact scan, and git status are checked.
- Changes are committed and pushed.

## Do Not Use This Goal For

- Expanding beyond TXT/MD notes.
- Adding coding, writing, work, phone, or multi-profile flows.
- Implementing GGUF export or local runtime.
- Adding Hugging Face Evaluate, LightEval, Unsloth, or bitsandbytes.
- Claiming benchmark results.
