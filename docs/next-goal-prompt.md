# Recommended Next Goal Prompt

Use this as the next `/goal` after the local scoring-identity and SFT-preview fix lands:

```text
/goal Strengthen OpenDistillation's fact-ledger label/value learning signal locally before another GPU smoke.

Context:
Work in /Users/langqi/Developer/Projects/OpenDistillation on latest origin/main. Keep v0 narrow: TXT/MD notes only, one notes/school model only, Colab-first, MockTeacherEngine as the safe default, optional local/open-source Qwen teacher, optional short TRL/PEFT LoRA training, and honest before/after quality reporting. Do not build SaaS, Mac app, phone app, accounts, backend, GGUF export, local runtime packaging, multiple profiles, coding model, writing model, work model, phone model, broad benchmark suite, Unsloth migration, bitsandbytes migration, or a larger training platform.

Evidence:
Two bounded Colab T4 fact-ledger runs failed exact learning. The latest value-first run used 8 facts, 24 train rows, 8 held-out eval rows, zero train/eval leakage, `Qwen/Qwen2.5-0.5B-Instruct`, and 30 TRL/PEFT LoRA steps. It changed 8/8 answers, but base and trained answers both hit 0/8 exact facts.

The follow-up local audit fixed a reporting risk: exact expected terms now travel with comparison examples by fact/question identity even when comparison rows are reordered for source-chunk coverage. The notebook also prints a bounded SFT preview showing exact prompt/completion pairs before training. That preview shows the current sample teaches each fact with only three rows.

Task:
Do local product-layer work only unless a very small local data fix clearly justifies exactly one bounded T4 rerun. Strengthen the fact-ledger train rows so the tiny adapter sees more explicit `Label: value` bindings before training. Do not change public JSONL schema: it remains `instruction`, `response`, `source_chunk_id`.

Implementation priorities:
- Add tests showing the train rows include enough explicit canonical `Label: value` bindings per fact.
- Decide and document the new per-fact row count for the next smoke. The likely safer default is six rows per fact, not three, while keeping the run small.
- Keep answer-only exact value rows, but make more rows include both the label and exact value in the response.
- Keep held-out eval wording separated from train wording and keep leakage checks strict.
- Ensure the SFT preview/report makes the stronger label/value signal obvious before any GPU run.
- Update docs with the local diagnosis and the next GPU-smoke condition.

GPU rule:
Do not run Colab/GPU in this goal unless the local code/data change is complete, tests pass, the notebook safe defaults remain off, and the docs state exactly what one bounded rerun would test. If those conditions are met and non-interactive `google-colab-cli` still works, one T4 smoke may be considered; otherwise stop after local verification.

Preserve:
Notebook defaults stay `INSTALL_TRAINING_DEPS = False`, `RUN_REAL_TEACHER = False`, and `RUN_TRAINING = False`. Fact metadata stays internal/sidecar. Do not commit generated datasets, adapters, checkpoints, model files, secrets, `.env` files, API keys, or local config.

Docs to keep aligned:
README.md, docs/current-decisions.md, docs/roadmap.md, docs/first-demo-flow.md, docs/colab-smoke-test-results.md, docs/dataset-schema.md, notebooks/README.md, and this file.

Verification:
Run unit tests, notebook JSON validation if touched, confirm no committed notebook outputs, local notebook smoke if notebook flow changes, git diff --check, secret scan, artifact/model/data scan, staged diff inspection, and git status. Commit and push only intended docs/source changes.

Finish:
Final response must include commit hash if any, diagnosis, files changed, verification results, whether Colab/GPU was avoided, and the exact next GPU-smoke condition if one exists.
```

## Why This Goal

The value-first rows were cleaner, but the tiny adapter still did not bind labels to values after 30 steps. The local SFT preview shows a concrete weakness: only three rows per fact. The next change should strengthen the supervised signal before spending more GPU.

## Done Means

- Local tests prove stronger label/value binding in train rows.
- Public JSONL schema stays unchanged.
- Train/eval leakage checks still pass.
- The SFT preview makes the exact training signal visible.
- Safe committed notebook defaults stay off.
- Generated datasets, adapters, model artifacts, checkpoints, secrets, and local config stay out of git.
- Unit tests, notebook checks when relevant, `git diff --check`, secret scan, artifact scan, staged diff inspection, and git status are checked.
- Changes are committed and pushed if docs or code changed.

## Do Not Use This Goal For

- Running another GPU smoke before a local data-signal fix lands.
- Expanding beyond TXT/MD notes.
- Adding coding, writing, work, phone, or multi-profile flows.
- Implementing GGUF export or local runtime.
- Migrating to Unsloth or bitsandbytes.
- Claiming benchmark results.
