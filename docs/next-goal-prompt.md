# Recommended Next Goal Prompt

Use this as the next `/goal` after the six-row label/value fact-ledger branch lands:

```text
/goal Run one bounded OpenDistillation fact-ledger T4 quality smoke for the six-row label/value signal.

Context:
Work in /Users/langqi/Developer/Projects/OpenDistillation on the current fact-ledger branch. Keep v0 narrow: TXT/MD notes only, one notes/school model only, Colab-first, MockTeacherEngine as the safe default, optional local/open-source Qwen teacher, optional short TRL/PEFT LoRA training, and honest before/after quality reporting. Do not build SaaS, Mac app, phone app, accounts, backend, GGUF export, local runtime packaging, multiple profiles, coding model, writing model, work model, phone model, broad benchmark suite, Unsloth migration, bitsandbytes migration, or a larger training platform.

Evidence:
Two bounded Colab T4 fact-ledger runs failed exact learning. The latest value-first run used 8 facts, 24 train rows, 8 held-out eval rows, zero train/eval leakage, `Qwen/Qwen2.5-0.5B-Instruct`, and 30 TRL/PEFT LoRA steps. It changed 8/8 answers, but base and trained answers both hit 0/8 exact facts.

The follow-up local work fixed a reporting risk: exact expected terms now travel with comparison examples by fact/question identity even when comparison rows are reordered for source-chunk coverage. The notebook also prints a bounded SFT preview showing exact prompt/completion pairs before training. The current branch now implements the six-row label/value local signal, so the sample uses 8 facts, 48 fact-ledger train rows, and 8 held-out eval rows before any GPU work starts.

Task:
Run exactly one bounded Colab T4 quality smoke for the current six-row label/value signal if the local checks pass and non-interactive `google-colab-cli` is usable. Do not change training knobs, broaden scope, or edit product surfaces. Do not change public JSONL schema: it remains `instruction`, `response`, `source_chunk_id`.

Exact GPU smoke:
- 8 extracted facts.
- 48 fact-ledger train rows.
- 8 held-out eval rows.
- Zero exact train/eval leaks and zero near-duplicate/token-overlap leaks before training.
- 30-step `Qwen/Qwen2.5-0.5B-Instruct` TRL/PEFT LoRA on Colab T4.
- Exact fact-hit scoring attached by fact/question identity, not row position.

Quality rule:
Treat the smoke as failed if answers change but still miss the expected facts. Changed wording is not useful learning unless exact held-out fact hits improve over the base model.

Preserve:
Notebook defaults stay `INSTALL_TRAINING_DEPS = False`, `RUN_REAL_TEACHER = False`, and `RUN_TRAINING = False`. Fact metadata stays internal/sidecar. Do not commit generated datasets, adapters, checkpoints, model files, secrets, `.env` files, API keys, or local config.

Docs to keep aligned:
README.md, START_HERE.md, docs/current-decisions.md, docs/agent-handoff.md, docs/roadmap.md, docs/first-demo-flow.md, docs/colab-smoke-test-results.md, docs/dataset-schema.md, notebooks/README.md, and this file.

Verification:
Before the T4 run, run unit tests, notebook JSON validation, confirm safe notebook defaults, `git diff --check`, secret scan, artifact/model/data scan, and git status. After the T4 run, update only the evidence docs with the exact result, including raw base/trained fact-hit counts and whether the adapter changed answers.

Finish:
Final response must include commit hash if any, diagnosis, files changed, verification results, GPU runtime used or blocker, exact base/trained fact-hit counts, and whether the result passed or failed.
```

## Why This Goal

The value-first rows were cleaner, but the tiny adapter still did not bind labels to values after 30 steps. This branch has now strengthened the supervised signal locally from three rows per fact to six. The next evidence must be a bounded GPU check, not more speculation.

## Done Means

- Local tests still prove stronger label/value binding in train rows.
- Public JSONL schema stays unchanged.
- Train/eval leakage checks still pass.
- The SFT preview makes the exact training signal visible.
- Safe committed notebook defaults stay off.
- Generated datasets, adapters, model artifacts, checkpoints, secrets, and local config stay out of git.
- The bounded T4 smoke either records improved exact fact hits or honestly records failure.
- Unit tests, notebook checks, `git diff --check`, secret scan, artifact scan, staged diff inspection, and git status are checked.
- Changes are committed and pushed if docs or code changed.

## Do Not Use This Goal For

- More local row redesign before the six-row signal is tested once.
- Expanding beyond TXT/MD notes.
- Adding coding, writing, work, phone, or multi-profile flows.
- Implementing GGUF export or local runtime.
- Migrating to Unsloth or bitsandbytes.
- Claiming benchmark results.
