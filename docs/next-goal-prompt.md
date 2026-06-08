# Recommended Next Goal Prompt

Use this as the next `/goal` after the value-first fact-ledger local fix lands:

```text
/goal Run one bounded value-first fact-ledger Colab T4 smoke for OpenDistillation.

Context:
Work in /Users/langqi/Developer/Projects/OpenDistillation on latest origin/main. Keep v0 narrow: TXT/MD notes only, one notes/school model only, Colab-first, MockTeacherEngine as the safe default, optional local/open-source Qwen teacher, optional short TRL/PEFT LoRA training, and honest before/after quality reporting. Do not build SaaS, Mac app, phone app, accounts, backend, GGUF export, local runtime packaging, multiple profiles, coding model, writing model, work model, phone model, broad benchmark suite, Unsloth migration, bitsandbytes migration, or a larger training platform.

Starting evidence:
The 2026-06-08 fact-ledger Colab T4 smoke used 8 facts, 24 fact-ledger train rows, 8 held-out eval rows, zero train/eval leakage, and a 30-step `Qwen/Qwen2.5-0.5B-Instruct` TRL/PEFT LoRA adapter. It changed all 8 answers, but base and trained answers both hit 0/8 exact expected facts. Treat that as failed learning.

Follow-up local diagnosis found no obvious TRL/PEFT formatting bug: current TRL docs support conversational prompt/completion SFT rows with completion-only loss, and current PEFT docs support `disable_adapter()` for base-vs-adapter comparison. The local product-layer fix now makes fact-ledger train targets value-first, uses direct exact-recall held-out eval wording, records `row_style` and `value` in the sidecar, and makes the notebook use fact-ledger train/eval rows after a passing gate.

Task:
Run exactly one bounded GPU quality smoke for the revised value-first fact-ledger rows. Use the committed sample notes, not uploaded notes. Train only on the 24 fact-ledger train rows and score only on the 8 held-out fact-ledger eval rows. Changed answers alone do not count; exact expected-term hits are the main signal.

Run requirements:
- Use google-colab-cli first.
- Use Chrome or Computer Use only if CLI auth/runtime control is blocked and the fallback is non-interactive.
- Use a T4 GPU runtime if available.
- Keep the run bounded: 8 facts, 24 train rows, 8 held-out eval rows, `Qwen/Qwen2.5-0.5B-Instruct`, short TRL/PEFT LoRA.
- Before training, print fact count, train/eval coverage, exact leak count, near/token-overlap leak count, and expected-term count.
- Record package versions, GPU type, runtime, training steps, adapter path, base answers, trained answers, exact fact-hit counts, and final judgment: better, unchanged, or worse.
- If Colab GPU quota, auth, sandbox, CLI, browser, or runtime control blocks the run, do not retry wastefully. Document the blocker honestly and do useful local verification instead.

Safe defaults:
Committed notebook defaults must remain off: `INSTALL_TRAINING_DEPS = False`, `RUN_REAL_TEACHER = False`, and `RUN_TRAINING = False`. Enable installs/training only in the live Colab runtime or temporary run script.

Docs to keep aligned:
README.md, docs/current-decisions.md, docs/open-source-tool-strategy.md, docs/roadmap.md, docs/first-demo-flow.md, docs/colab-smoke-test-results.md, docs/dataset-schema.md, notebooks/README.md, and this file.

Verification:
Run unit tests, notebook JSON validation, confirm no committed notebook outputs, default local notebook smoke if notebook flow changes, git diff --check, secret scan, artifact/model/data scan, staged diff inspection, and git status. Commit and push only intended docs/source changes. Do not commit generated datasets, adapters, checkpoints, model files, secrets, or local config.

Finish:
Final response must include commit hash if any, whether Colab ran, GPU type if known, fact-ledger summary, training steps, base/trained exact fact-hit counts, judgment, blockers if any, files changed, verification results, and the next recommended goal.
```

## Why This Goal

The local data/eval split now gives the tiny adapter a clearer target than the failed 0/8 run. The next useful evidence is a single bounded GPU smoke, not more local speculation or broader product work.

## Done Means

- Latest main is inspected before running anything.
- The revised value-first fact-ledger train rows are the training source.
- The held-out fact-ledger eval rows are the comparison source.
- Train/eval leakage is checked before training.
- Exact expected-term hits are reported for base and trained answers.
- Useful learning is judged only by held-out fact-hit improvement.
- Safe committed notebook defaults stay off.
- Generated datasets, adapters, model artifacts, checkpoints, secrets, and local config stay out of git.
- Unit tests, notebook checks, `git diff --check`, secret scan, artifact scan, staged diff inspection, and git status are checked.
- Changes are committed and pushed if docs or code changed.

## Do Not Use This Goal For

- Expanding beyond TXT/MD notes.
- Adding coding, writing, work, phone, or multi-profile flows.
- Implementing GGUF export or local runtime.
- Migrating to Unsloth or bitsandbytes.
- Claiming benchmark results.
